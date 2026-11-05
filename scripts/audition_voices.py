"""Synthesise one line per Timbre v3.0 voice and render it through the full call chain.

Gnani publishes gender and language per voice but not age, and none of the male voices reads
as a man in his seventies. Nor does any TTS output sound like a phone call. So this script
does two separable things:

* **one API call per voice**, and
* a grid of local renders of that one call.

Everything after synthesis is pure DSP — ageing, room reverb, room tone, and the phone line
itself (`app/core/audio/`) — so the whole grid costs the credits of a single synthesis per
voice.

Two ways to use it:

* `--ladder` renders one voice through every stage of the realism ladder, cleanest first, so
  you can hear what each stage contributes. This is how you choose a profile.
* `--aged` renders every voice at one profile, so you can compare voices on equal terms.
  This is how you choose a voice.

Output WAVs get headers we write ourselves: Timbre's `container=wav` responses declare a
0xFFFFFFFF data size, which makes every player misreport the duration.

    python scripts/audition_voices.py --male --ladder            # Devansh through the ladder
    python scripts/audition_voices.py --male --aged              # all six at one profile
    python scripts/audition_voices.py --male --ladder --voices Prakash --room quiet-room
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

import httpx
import numpy as np

from app.core.audio.ambience import AMBIENCE_BY_NAME, AmbienceBed
from app.core.audio.call_profile import (
    CALL_PROFILES,
    DEFAULT_ELDER_PROFILE,
    GRANDSON_PROFILE,
    PROFILES_BY_NAME,
    CallProfile,
)
from app.core.audio.framing import float32_to_pcm, pcm_to_float32, pcm_to_wav
from app.core.cost import BillingPool
from app.core.tags import ELDER_TAGS, GRANDSON_TAGS, billable_characters, sanitize_tags
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

#: Chosen by ear from the first audition. The ladder defaults to this one voice because
#: rendering six voices through six profiles is 36 clips, which nobody listens to.
LEAD_VOICE = "Devansh"

#: One line per voice, so the comparison is fair. It is Dada in character — the glasses, the
#: employee number, the Railways — and it exercises the four tags he is allowed to use.
#: `<slow>` goes near the front because a Timbre tag applies from its position to the end of
#: the input, so one tag slows everything after it.
AUDITION_LINE = (
    "Haan beta, <slow> ek minute. <coughs> Mera chashma kahan gaya. "
    "<pauses> Aap apna employee number bataiye na, main likh leta hoon. "
    "<sigh> Railway mein aise hi hota tha."
)
#: The grandson's line, for choosing his voice. Judging him on Dada's line would mislead:
#: `<slow>` drags out everything after it, and the whole point of the grandson is that he is
#: crisp where his grandfather is slow. He gets only the four tags he is allowed, and barely
#: needs them.
GRANDSON_LINE = (
    "Ek second, main baat karta hoon. <clears throat> Main bank mein operations mein hoon. "
    "<pauses> Aap ye bataiye, ye notice kis department se aaya hai, aur kiske authority par? "
    "<sigh> Dadaji ne jo likha hai, naam match nahi kar raha."
)
SAMPLE_RATE = 24000

#: Silence around each clip, so the room is audible on its own before the first word. The
#: live path uses zero: there the bed already runs continuously between utterances.
LEAD_MS = 600.0


def _write(path: Path, samples: np.ndarray, sample_rate: int) -> float:
    """Write float32 samples as a WAV with a header we trust. Returns duration in seconds."""
    pcm = float32_to_pcm(samples.astype(np.float32))
    path.write_bytes(pcm_to_wav(pcm, sample_rate))
    return len(pcm) / 2 / sample_rate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--male", action="store_true", help="audition the male voices")
    parser.add_argument(
        "--ladder",
        action="store_true",
        help=f"render one voice (default {LEAD_VOICE}) through every realism profile",
    )
    parser.add_argument(
        "--aged",
        action="store_true",
        help="render every voice through one profile, to compare voices on equal terms",
    )
    parser.add_argument(
        "--handoff",
        action="store_true",
        help="render Dada and then the grandson back to back in one clip, with the room tone "
        "running continuously across the join - the handoff as the scammer hears it",
    )
    parser.add_argument(
        "--grandson",
        action="store_true",
        help="audition the grandson instead: his own line, his tag vocabulary, unaged, on "
        "the same room and line as the elder",
    )
    parser.add_argument(
        "--profile",
        default=DEFAULT_ELDER_PROFILE,
        help=f"profile used by --aged. One of {[p.name for p in CALL_PROFILES]}",
    )
    parser.add_argument(
        "--room",
        default=None,
        help=f"override the room tone on every profile. One of {sorted(AMBIENCE_BY_NAME)}",
    )
    parser.add_argument(
        "--bed-db",
        default=None,
        help="comma-separated bed levels in dB under the voice, e.g. -22,-18,-15,-12,-9. "
        "Renders one clip per level so how far the voice sits under the room can be chosen "
        "by ear. This is the knob that masks synthesis artefacts.",
    )
    parser.add_argument("--voices", help="comma-separated subset, e.g. Devansh,Prakash")
    parser.add_argument("--out", type=Path, default=Path("var/auditions/aged-male"))
    parser.add_argument("--text", default=AUDITION_LINE)
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)
    if not settings.gnani_api_key:
        print("GNANI_API_KEY is not set in .env", file=sys.stderr)
        return 2
    if not any((args.ladder, args.aged, args.bed_db, args.grandson, args.handoff)):
        print(
            "nothing to do: pass --ladder, --aged, --bed-db, --grandson or --handoff",
            file=sys.stderr,
        )
        return 2
    if args.profile not in PROFILES_BY_NAME:
        print(f"unknown profile {args.profile!r}; have {sorted(PROFILES_BY_NAME)}", file=sys.stderr)
        return 2
    if args.room is not None and args.room not in AMBIENCE_BY_NAME:
        print(f"unknown room {args.room!r}; have {sorted(AMBIENCE_BY_NAME)}", file=sys.stderr)
        return 2

    if args.voices:
        voices = tuple(v.strip() for v in args.voices.split(",") if v.strip())
    elif args.aged or args.grandson or args.handoff:
        voices = MALE_VOICES if args.male else FEMALE_VOICES
    else:
        voices = (LEAD_VOICE,)

    levels: tuple[float, ...] = ()
    if args.bed_db:
        try:
            levels = tuple(float(value) for value in args.bed_db.split(","))
        except ValueError:
            print(f"--bed-db wants numbers, got {args.bed_db!r}", file=sys.stderr)
            return 2
        if any(level > 0.0 for level in levels):
            print("--bed-db values are dB *under* the voice, so they must be <= 0", file=sys.stderr)
            return 2

    ladder = tuple(_with_room(p, args.room) for p in CALL_PROFILES) if args.ladder else ()
    chosen = _with_room(PROFILES_BY_NAME[args.profile], args.room) if args.aged else None
    if args.handoff:
        args.out.mkdir(parents=True, exist_ok=True)
        return asyncio.run(_handoff(settings, voices, args.room, args.out))

    grandson = _with_room(GRANDSON_PROFILE, args.room) if args.grandson else None
    sweep: tuple[CallProfile, ...] = ()
    if levels:
        base = _with_room(PROFILES_BY_NAME[args.profile], args.room)
        if base.ambience is None:
            print(f"profile {base.name!r} has no room to set a level on", file=sys.stderr)
            return 2
        # Renamed by level, so the filenames say which is which without a lookup.
        sweep = tuple(
            base.with_room(base.ambience, level_db=level).model_copy(
                update={"name": f"{base.name}-bed{level:+.0f}dB"}
            )
            for level in levels
        )

    # Each persona is allowed only part of the thirteen-tag catalogue. Running the audition
    # line through the same filter the persona output will go through means the line cannot
    # quietly use a tag the character is forbidden, and proves the filter on real text.
    allowed = GRANDSON_TAGS if args.grandson else ELDER_TAGS
    if args.grandson and args.text == AUDITION_LINE:
        args.text = GRANDSON_LINE
    text, dropped = sanitize_tags(args.text, allowed)
    if dropped:
        print(f"Stripped tags not allowed for an elder persona: {', '.join(dropped)}")

    chars = billable_characters(text)
    estimate = settings.cost_model().price_tts(chars * len(voices))
    print(f"Auditioning {len(voices)} voice(s) on {settings.tts_model}: {', '.join(voices)}")
    print(f'Line ({chars} billable chars incl. tags): "{text}"')
    if ladder:
        print(f"Realism ladder: {' -> '.join(p.name for p in ladder)}")
    if chosen is not None:
        print(f"Per-voice profile: {chosen.name}")
    if grandson is not None:
        print(f"Grandson profile: {_describe(grandson)}")
    if args.room:
        print(f"Room overridden to: {args.room}")
    if sweep:
        joined = ", ".join(f"{value:+.0f} dB" for value in levels)
        print(f"Bed level sweep on {args.profile}: {joined}")
    print(
        f"Estimated cost: {estimate.credits:.3f} credits"
        f"{' (rate is a placeholder)' if estimate.estimated else ''}"
        "  (renders are local DSP and cost nothing)\n"
    )

    args.out.mkdir(parents=True, exist_ok=True)
    return asyncio.run(_run(settings, voices, ladder, chosen, grandson, sweep, args.out, text))


def _with_room(profile: CallProfile, room: str | None) -> CallProfile:
    """Swap the room tone without touching the voice or the line."""
    if room is None or profile.ambience is None:
        return profile
    return profile.model_copy(update={"ambience": AMBIENCE_BY_NAME[room]})


def _describe(profile: CallProfile) -> str:
    parts = [f"age={profile.aging.name}", f"line={profile.telephony.name}"]
    if profile.ambience is not None:
        parts.append(f"room={profile.ambience.name}@{profile.ambience.level_db:+.0f}dB")
    else:
        parts.append("room=none")
    parts.append("reverb" if profile.acoustics else "dry")
    parts.append(f"peak={profile.output_peak:.2f}")
    return " ".join(parts)


async def _run(
    settings: Settings,
    voices: tuple[str, ...],
    ladder: tuple[CallProfile, ...],
    chosen: CallProfile | None,
    grandson: CallProfile | None,
    sweep: tuple[CallProfile, ...],
    out: Path,
    text: str,
) -> int:
    ok: list[str] = []
    failed: list[str] = []
    fastest: tuple[str, float] | None = None
    renders = 0
    render_ms: list[float] = []
    # Beds are expensive to build and identical between renders of the same room, so build
    # each one once. This is exactly what a live call does for the whole call.
    beds: dict[str, AmbienceBed] = {}

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
            print(f"  {voice:10} first audio {first_byte_ms:6.0f} ms   {seconds:4.1f}s   raw")
            ok.append(voice)
            if fastest is None or first_byte_ms < fastest[1]:
                fastest = (voice, first_byte_ms)

            wanted = (
                *ladder,
                *((chosen,) if chosen is not None else ()),
                *((grandson,) if grandson is not None else ()),
                *sweep,
            )
            for profile in wanted:
                if profile.ambience is not None and profile.ambience.name not in beds:
                    beds[profile.ambience.name] = AmbienceBed(profile.ambience, SAMPLE_RATE)
                bed = beds.get(profile.ambience.name) if profile.ambience else None
                result = profile.render(
                    raw,
                    SAMPLE_RATE,
                    bed=bed,
                    # A different point in the loop per voice, so the fan is not in the same
                    # phase on every clip and the comparison is not accidentally rigged.
                    bed_offset=index * SAMPLE_RATE * 3,
                    lead_ms=LEAD_MS,
                )
                name = f"{voice.lower()}.{profile.name}.wav"
                _write(out / name, result.audio, SAMPLE_RATE)
                renders += 1
                render_ms.append(result.total_ms)
                loss = f" {result.dropped_packets} pkt lost" if result.dropped_packets else ""
                stages = " ".join(f"{k} {v:.0f}ms" for k, v in result.stage_ms.items())
                print(
                    f"             {profile.name:12} {result.seconds:4.1f}s  "
                    f"[{stages}]{loss}   {name}"
                )
                print(f"             {'':12} {_describe(profile)}")

    status = await budget.status(BillingPool.GNANI)
    await redis.aclose()
    await engine.dispose()

    print(f"\n{len(ok)}/{len(voices)} voices synthesised, {renders} renders into {out}/")
    print(f"Timbre: {tts.misses} synthesised, {tts.hits} served from cache")
    if fastest is not None and len(voices) > 1:
        print(f"Fastest to first audio: {fastest[0]} at {fastest[1]:.0f} ms")
    if render_ms:
        # This number decides whether the chain can run in the live path or has to be baked
        # in at cache-write time. A few milliseconds per utterance is free; hundreds are not.
        print(f"Render: {min(render_ms):.0f}-{max(render_ms):.0f} ms per clip, 0 extra credits")
    print(f"\nListen to all of them:\n  open {out}/")
    print(
        f"\nGnani pool now at {status.spent:.4f} credits of a "
        f"{status.stop_at:g} hard stop"
        f"{' (estimated rates)' if status.estimated else ''}"
    )
    return 0 if not failed else 1


async def _handoff(settings: Settings, voices: tuple[str, ...], room: str | None, out: Path) -> int:
    """Render Dada, then the grandson, as one continuous call.

    This is the only way to judge whether two personas are distinguishable, and the only way
    to judge it at all if they share a Timbre voice. Listening to two clips in a file browser
    does not answer the question - the scammer hears one unbroken line.

    The room tone is continuous across the join because the bed is read positionally: the
    second utterance starts reading the loop exactly where the first stopped. Rendering each
    utterance from bed offset zero would restart the fan mid-call, which is the giveaway that
    `AmbienceBed.take` taking an offset exists to prevent.
    """
    elder_profile = _with_room(PROFILES_BY_NAME[DEFAULT_ELDER_PROFILE], room)
    grandson_profile = _with_room(GRANDSON_PROFILE, room)
    elder_text, _ = sanitize_tags(AUDITION_LINE, ELDER_TAGS)
    grandson_text, _ = sanitize_tags(GRANDSON_LINE, GRANDSON_TAGS)

    print(f"Elder    : {_describe(elder_profile)}")
    print(f"Grandson : {_describe(grandson_profile)}")
    if elder_profile.aging.name == grandson_profile.aging.name:
        print("WARNING: both personas use the same ageing; the handoff will be inaudible")

    engine = create_engine(settings)
    sessions = create_session_factory(engine)
    redis = create_redis(settings)
    budget = BudgetGuard(redis, sessions, settings)
    bed = AmbienceBed(elder_profile.ambience, SAMPLE_RATE) if elder_profile.ambience else None

    async with httpx.AsyncClient() as client:
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
        for voice in voices:
            pieces: list[np.ndarray] = []
            offset = 0
            for profile, line, who in (
                (elder_profile, elder_text, "dada"),
                (grandson_profile, grandson_text, "grandson"),
            ):
                pcm = bytearray()
                async for chunk in tts.synthesize(
                    TtsRequest(
                        text=line,
                        voice=voice,
                        language=settings.tts_language,
                        model=settings.tts_model,
                        sample_rate=SAMPLE_RATE,
                    )
                ):
                    pcm.extend(chunk.pcm)
                result = profile.render(
                    pcm_to_float32(bytes(pcm)),
                    SAMPLE_RATE,
                    bed=bed,
                    bed_offset=offset,
                    lead_ms=LEAD_MS,
                )
                pieces.append(result.audio)
                offset += result.audio.size
                print(
                    f"  {voice:10} {who:9} {result.seconds:4.1f}s  "
                    f"{result.dropped_packets} pkt lost"
                )
            path = out / f"{voice.lower()}.handoff.wav"
            seconds = _write(path, np.concatenate(pieces), SAMPLE_RATE)
            print(f"  {voice:10} {'joined':9} {seconds:4.1f}s  {path.name}")

    status = await budget.status(BillingPool.GNANI)
    await redis.aclose()
    await engine.dispose()
    print(f"\nTimbre: {tts.misses} synthesised, {tts.hits} served from cache")
    print(f"\nListen:\n  open {out}/")
    print(
        f"\nGnani pool now at {status.spent:.4f} credits of a {status.stop_at:g} hard stop"
        f"{' (estimated rates)' if status.estimated else ''}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
