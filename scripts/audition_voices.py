"""Synthesise one line in every Timbre v3.0 voice, aged and over room tone, so a human can pick.

Gnani publishes gender and language per voice but not age, and none of the male voices reads
as a man in his seventies. So this script does three things from one API call per voice:

* writes the voice as Timbre returned it,
* writes it through each local ageing preset, and
* lays a synthetic room-tone bed under each one.

Ageing and ambience are pure DSP (`app.core.audio.aging`, `app.core.audio.ambience`), so the
whole grid costs the credits of a single synthesis per voice. Thirty-three clips, six
requests.

Judge the aged clips, not the raw ones. The raw clip is there to show what the ageing is
doing, but nobody will ever hear it on a call.

Output WAVs get headers we write ourselves: Timbre's `container=wav` responses declare a
0xFFFFFFFF data size, which makes every player misreport the duration.

    python scripts/audition_voices.py --male --aged        # Dada's audition
    python scripts/audition_voices.py --male --aged --ambience fan-and-street
    python scripts/audition_voices.py --male --aged --voices Avinash,Prakash --presets classic
    python scripts/audition_voices.py                      # the seven female voices, raw
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

import httpx
import numpy as np

from app.core.audio.aging import AGING_PRESETS, PRESETS_BY_NAME, AgingPreset, peak_normalise
from app.core.audio.ambience import AMBIENCE_BY_NAME, AmbienceBed
from app.core.audio.framing import float32_to_pcm, pcm_to_float32, pcm_to_wav
from app.core.cost import BillingPool
from app.core.tags import ELDER_TAGS, billable_characters, sanitize_tags
from app.infra.budget import BudgetGuard
from app.infra.db import create_engine, create_session_factory
from app.infra.logging import configure_logging
from app.infra.redis_bus import create_redis
from app.infra.settings import Settings, get_settings
from app.infra.tts_cache import CachingTTS
from app.providers.base import TtsRequest
from app.providers.gnani_timbre import GnaniTimbreTTS

# Timbre v3.0 voices, from docs.gnani.ai/api/TTS/available-voices. All support all ten
# languages, so the only variable being auditioned is how the voice itself sounds.
FEMALE_VOICES = ("Shivani", "Lakshmi", "Kalpana", "Chetana", "Vasudha", "Riyaa", "Tarika")
MALE_VOICES = ("Avinash", "Shubham", "Prakash", "Devansh", "Yashraj", "Harshad")

#: One line for every voice, so the comparison is fair. It is Dada in character — the
#: glasses, the employee number, the Railways — and it exercises the four tags he is allowed
#: to use. `<slow>` goes near the front because a Timbre tag applies from its position to
#: the end of the input, so one tag slows everything after it.
AUDITION_LINE = (
    "Haan beta, <slow> ek minute. <coughs> Mera chashma kahan gaya. "
    "<pauses> Aap apna employee number bataiye na, main likh leta hoon. "
    "<sigh> Railway mein aise hi hota tha."
)
SAMPLE_RATE = 24000

#: Silence around each clip, so the bed is audible on its own before the first word. The
#: live path uses zero: there the bed already runs continuously between utterances.
LEAD_MS = 450.0

#: Which ageing preset the ambience comparison uses, so the three beds are judged against a
#: constant voice treatment.
AMBIENCE_REFERENCE_PRESET = "classic"


def _write(path: Path, samples: np.ndarray, sample_rate: int) -> float:
    """Write float32 samples as a WAV with a header we trust. Returns duration in seconds."""
    pcm = float32_to_pcm(samples.astype(np.float32))
    path.write_bytes(pcm_to_wav(pcm, sample_rate))
    return len(pcm) / 2 / sample_rate


def _default_out(male: bool, aged: bool) -> Path:
    if aged:
        return Path("var/auditions/aged-male") if male else Path("var/auditions/aged")
    return Path("var/auditions")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--male", action="store_true", help="audition the male voices instead")
    parser.add_argument(
        "--aged",
        action="store_true",
        help="also write every ageing preset: pitch down, slower tempo, tremble, phone band",
    )
    parser.add_argument(
        "--ambience",
        default="ceiling-fan",
        help=f"room tone under the aged clips, or 'none'. One of {sorted(AMBIENCE_BY_NAME)}",
    )
    parser.add_argument("--voices", help="comma-separated subset, e.g. Avinash,Prakash")
    parser.add_argument("--presets", help="comma-separated subset of ageing preset names")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--text", default=AUDITION_LINE)
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)
    if not settings.gnani_api_key:
        print("GNANI_API_KEY is not set in .env", file=sys.stderr)
        return 2

    if args.voices:
        voices = tuple(v.strip() for v in args.voices.split(",") if v.strip())
    else:
        voices = MALE_VOICES if args.male else FEMALE_VOICES

    presets: tuple[AgingPreset, ...] = ()
    if args.aged:
        if args.presets:
            try:
                presets = tuple(PRESETS_BY_NAME[n.strip()] for n in args.presets.split(","))
            except KeyError as exc:
                print(
                    f"unknown ageing preset {exc}; have {sorted(PRESETS_BY_NAME)}", file=sys.stderr
                )
                return 2
        else:
            presets = AGING_PRESETS

    bed: AmbienceBed | None = None
    if args.aged and args.ambience != "none":
        if args.ambience not in AMBIENCE_BY_NAME:
            print(
                f"unknown ambience {args.ambience!r}; have {sorted(AMBIENCE_BY_NAME)}",
                file=sys.stderr,
            )
            return 2
        bed = AmbienceBed(AMBIENCE_BY_NAME[args.ambience], SAMPLE_RATE)

    out = args.out or _default_out(args.male, args.aged)
    out.mkdir(parents=True, exist_ok=True)
    rooms = out / "_rooms"
    if bed is not None:
        rooms.mkdir(exist_ok=True)

    # The persona is only allowed six of the thirteen tags. Running the audition line through
    # the same filter the persona output will go through means the line cannot quietly use a
    # tag the character is forbidden, and proves the filter on real text.
    text, dropped = sanitize_tags(args.text, ELDER_TAGS)
    if dropped:
        print(f"Stripped tags not allowed for an elder persona: {', '.join(dropped)}")

    chars = billable_characters(text)
    estimate = settings.cost_model().price_tts(chars * len(voices))
    print(f"Auditioning {len(voices)} voices on {settings.tts_model}")
    print(f'Line ({chars} billable chars incl. tags): "{text}"')
    if presets:
        print(f"Ageing presets (local DSP, no extra credits): {', '.join(p.name for p in presets)}")
    if bed is not None:
        layers = ", ".join(f"{n} {w:g}" for n, w in bed.preset.layers())
        print(f"Room tone: {bed.preset.name} at {bed.preset.level_db:g} dB  [{layers}]")
    print(
        f"Estimated cost: {estimate.credits:.3f} credits"
        f"{' (rate is a placeholder)' if estimate.estimated else ''}\n"
    )
    return asyncio.run(_run(settings, voices, presets, bed, out, rooms, text))


async def _run(
    settings: Settings,
    voices: tuple[str, ...],
    presets: tuple[AgingPreset, ...],
    bed: AmbienceBed | None,
    out: Path,
    rooms: Path,
    text: str,
) -> int:
    ok: list[str] = []
    failed: list[str] = []
    fastest: tuple[str, float] | None = None
    dsp_ms: list[float] = []
    reference: np.ndarray | None = None

    # The guard is wired in even for a one-off script: an audition is real spend, and
    # unrecorded spend makes the end-of-project credit numbers a guess.
    engine = create_engine(settings)
    sessions = create_session_factory(engine)
    redis = create_redis(settings)
    budget = BudgetGuard(redis, sessions, settings)

    async with httpx.AsyncClient() as client:
        # Through the cache, like everything else that spends credits. Re-running an audition
        # to fix a bug in the local DSP must not pay Timbre a second time for identical text.
        tts = CachingTTS(
            GnaniTimbreTTS(
                api_key=settings.gnani_api_key,
                sse_url=settings.gnani_tts_sse_url,
                model=settings.tts_model,
                language=settings.tts_language,
                client=client,
                usage=budget,
                read_timeout=60.0,
            ),
            settings.tts_cache_dir,
            usage=budget,
        )
        for index, voice in enumerate(voices):
            request = TtsRequest(
                text=text,
                voice=voice,
                language=settings.tts_language,
                model=settings.tts_model,
                sample_rate=SAMPLE_RATE,
            )
            started = time.perf_counter()
            first_byte_ms = 0.0
            pcm = bytearray()
            try:
                async for chunk in tts.synthesize(request):
                    if not pcm:
                        first_byte_ms = (time.perf_counter() - started) * 1000
                    pcm.extend(chunk.pcm)
            except Exception as exc:  # one bad voice must not abort the whole audition
                print(f"  {voice:10} FAILED  {type(exc).__name__}: {exc}")
                failed.append(voice)
                continue

            raw = pcm_to_float32(bytes(pcm))
            seconds = _write(out / f"{voice.lower()}.raw.wav", raw, SAMPLE_RATE)
            print(
                f"  {voice:10} first audio {first_byte_ms:6.0f} ms   {seconds:4.1f}s   "
                f"{voice.lower()}.raw.wav  (unprocessed reference)"
            )
            ok.append(voice)
            if reference is None:
                reference = raw
            if fastest is None or first_byte_ms < fastest[1]:
                fastest = (voice, first_byte_ms)

            for preset in presets:
                began = time.perf_counter()
                # A different point in the bed loop per clip, so the fan is not in the same
                # phase on all 24 files and the comparison is not accidentally rigged.
                aged = preset.apply(
                    raw,
                    SAMPLE_RATE,
                    bed=bed,
                    bed_offset=index * SAMPLE_RATE * 2,
                    lead_ms=LEAD_MS,
                )
                elapsed_ms = (time.perf_counter() - began) * 1000
                dsp_ms.append(elapsed_ms)
                aged_seconds = _write(out / f"{voice.lower()}.{preset.name}.wav", aged, SAMPLE_RATE)
                print(
                    f"             {preset.name:11} {aged_seconds:4.1f}s   "
                    f"{elapsed_ms:5.1f} ms DSP   {voice.lower()}.{preset.name}.wav"
                )

    # One voice, one ageing preset, every bed: so the room can be chosen separately from
    # the voice instead of being confounded with it.
    if reference is not None and bed is not None and AMBIENCE_REFERENCE_PRESET in PRESETS_BY_NAME:
        preset = PRESETS_BY_NAME[AMBIENCE_REFERENCE_PRESET]
        print(f"\nRoom tone options on {ok[0]} + {preset.name}, in {rooms}/:")
        for name, ambience in AMBIENCE_BY_NAME.items():
            other = AmbienceBed(ambience, SAMPLE_RATE)
            _write(
                rooms / f"{name}.wav",
                preset.apply(reference, SAMPLE_RATE, bed=other, lead_ms=LEAD_MS),
                SAMPLE_RATE,
            )
            # The bare bed too: easier to tell a fan from traffic without a voice over it.
            # Peak-normalised rather than scaled by a constant: the loop is unit RMS, so its
            # peak is several times 1.0 and any fixed gain clips it into a crackle.
            _write(
                rooms / f"{name}.bed-only.wav",
                peak_normalise(other.take(SAMPLE_RATE * 4), 0.6),
                SAMPLE_RATE,
            )
            print(f"  {name:16} {ambience.level_db:6.1f} dB   {name}.wav + {name}.bed-only.wav")

    status = await budget.status(BillingPool.GNANI)
    await redis.aclose()
    await engine.dispose()

    print(f"\n{len(ok)}/{len(voices)} voices synthesised into {out}/")
    print(f"Timbre: {tts.misses} synthesised, {tts.hits} served from cache")
    if fastest is not None:
        print(f"Fastest to first audio: {fastest[0]} at {fastest[1]:.0f} ms")
    if dsp_ms:
        # This number decides whether ageing can run in the live path or has to be baked in
        # at cache-write time. A few milliseconds per utterance is free; a few hundred is not.
        print(
            f"Ageing + room tone: {len(dsp_ms)} clips, "
            f"{min(dsp_ms):.0f}-{max(dsp_ms):.0f} ms each, 0 extra credits"
        )
    print(f"\nListen to all of them:\n  open {out}/")
    print(
        f"\nGnani pool now at {status.spent:.4f} credits of a "
        f"{status.stop_at:g} hard stop"
        f"{' (estimated rates)' if status.estimated else ''}"
    )
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
