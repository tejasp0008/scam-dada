"""Synthesise the same Dadi line in every Timbre v3.0 voice, so a human can pick one.

Gnani publishes gender and language per voice but not age, so which voice sounds like a
72-year-old grandmother is an empirical question. This script answers it with audio rather
than with a guess.

Output WAVs get headers we write ourselves: Timbre's own `container=wav` responses declare
a 0xFFFFFFFF data size, which makes every player misreport the duration.

    python scripts/audition_voices.py            # the seven v3.0 female voices
    python scripts/audition_voices.py --male     # the six v3.0 male voices, for Dada
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

import httpx

from app.core.audio.framing import pcm_to_wav
from app.core.cost import BillingPool
from app.infra.logging import configure_logging
from app.infra.settings import get_settings
from app.providers.base import TtsRequest
from app.providers.gnani_timbre import GnaniTimbreTTS

# Timbre v3.0 voices, from docs.gnani.ai/api/TTS/available-voices. All support all ten
# languages, so the only variable being auditioned is how old and warm each one sounds.
FEMALE_VOICES = ("Shivani", "Lakshmi", "Kalpana", "Chetana", "Vasudha", "Riyaa", "Tarika")
MALE_VOICES = ("Avinash", "Shubham", "Prakash", "Devansh", "Yashraj", "Harshad")

#: One line, used for every voice so the comparison is fair. It exercises the three audio
#: tags the persona leans on, and reads like someone stalling without understanding why.
AUDITION_LINE = (
    "Haan beta, <sigh> zara dheere boliye na. "
    "Mera kaan thoda kamzor hai. <coughs> Aap kaun bol rahe hain?"
)
SAMPLE_RATE = 24000


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--male", action="store_true", help="audition the male voices instead")
    parser.add_argument("--out", type=Path, default=Path("var/auditions"))
    parser.add_argument("--text", default=AUDITION_LINE)
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)
    if not settings.gnani_api_key:
        print("GNANI_API_KEY is not set in .env", file=sys.stderr)
        return 2

    voices = MALE_VOICES if args.male else FEMALE_VOICES
    cost = settings.cost_model()
    estimate = cost.price_tts(len(args.text) * len(voices))
    print(f"Auditioning {len(voices)} voices on {settings.tts_model}")
    print(f'Line ({len(args.text)} billable chars incl. tags): "{args.text}"')
    print(
        f"Estimated cost: {estimate.credits:.3f} credits"
        f"{' (rate is a placeholder)' if estimate.estimated else ''}\n"
    )

    args.out.mkdir(parents=True, exist_ok=True)
    results: list[tuple[str, float, float, Path | None, str]] = []

    async with httpx.AsyncClient() as client:
        tts = GnaniTimbreTTS(
            api_key=settings.gnani_api_key,
            sse_url=settings.gnani_tts_sse_url,
            model=settings.tts_model,
            language=settings.tts_language,
            client=client,
            read_timeout=60.0,
        )
        for voice in voices:
            request = TtsRequest(
                text=args.text,
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
                results.append((voice, 0.0, 0.0, None, f"{type(exc).__name__}"))
                continue

            seconds = len(pcm) / 2 / SAMPLE_RATE
            path = args.out / f"{voice.lower()}.wav"
            path.write_bytes(pcm_to_wav(bytes(pcm), SAMPLE_RATE))
            print(
                f"  {voice:10} OK      first audio {first_byte_ms:6.0f} ms   "
                f"{seconds:4.1f}s   {path}"
            )
            results.append((voice, first_byte_ms, seconds, path, "ok"))

    ok = [r for r in results if r[4] == "ok"]
    print(f"\n{len(ok)}/{len(voices)} voices synthesised into {args.out}/")
    if ok:
        fastest = min(ok, key=lambda r: r[1])
        print(f"Fastest to first audio: {fastest[0]} at {fastest[1]:.0f} ms")
        print(f"\nListen to all of them:\n  open {args.out}/")
    print(
        f"\nSpend recorded against the {BillingPool.GNANI.value} pool; "
        f"run with BUDGET_ENABLED=true to have the guard enforce the ceiling."
    )
    return 0 if len(ok) == len(voices) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
