"""Hit every real provider once, end to end, and report latency and cost.

This is the Phase 1 acceptance check: audio in, transcript out, reply generated, audio back,
with the budget guard engaged and every credit accounted for. It spends real credits, so it
is never part of `make test`.

    make smoke

What it proves, in order:
  1. Timbre v3.0 synthesises Hinglish with inline audio tags.
  2. Prisma v3.0 transcribes that audio over the streaming WebSocket, at real-time cadence,
     in exactly 1024-byte frames.
  3. The LLM answers in character, streamed, and returns validated JSON.
  4. The fallback switches models when the primary is unavailable.
  5. The TTS cache serves a repeat without spending anything.
  6. The budget guard refuses a call that would breach its ceiling.
"""

from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass, field

import httpx
from pydantic import BaseModel, Field

from app.core.audio.framing import FrameBuffer, pcm_to_wav
from app.core.cost import BillingPool
from app.infra.budget import BudgetGuard
from app.infra.db import create_engine, create_session_factory
from app.infra.logging import configure_logging
from app.infra.redis_bus import create_redis
from app.infra.settings import get_settings
from app.infra.tts_cache import CachingTTS
from app.providers.base import (
    BudgetExceeded,
    ChatMessage,
    Role,
    SttConfig,
    SttProcessing,
    SttSpeechStart,
    SttTranscript,
    TtsRequest,
)
from app.providers.factory import build_llm_provider, build_stt_provider, build_tts_provider
from app.providers.names import ProviderName

#: Short, so the real-time cadence of the STT leg does not make the smoke test slow.
SMOKE_LINE = "Haan beta, <sigh> kaun bol raha hai?"
SMOKE_RATE = 16000
SCAMMER_LINE = "Sir main cyber cell se bol raha hoon, aapke naam par parcel pakda gaya hai."
# Deliberately a condensed stand-in for app/prompts/dada.v1.md, not the file itself: the
# smoke test measures the provider leg, and a 4.5 KB system prompt would make the measured
# time-to-first-token a measurement of prompt length instead.
DADA_SYSTEM = (
    "You are Dada, a 74-year-old retired Indian Railways clerk on a phone call. Speak "
    "Hinglish in Latin script. Be hard of hearing, slow and suspicious. Ask for the "
    "caller's employee number. One or two short sentences. Never threaten, insult, or "
    "claim to be police."
)


class StageAssessment(BaseModel):
    """Minimal schema, purely to exercise `complete_json` against a real model."""

    stage: str = Field(default="hook")
    pressure: float = Field(default=0.0, ge=0.0, le=1.0)


@dataclass
class Step:
    name: str
    ok: bool = False
    detail: str = ""
    timings: dict[str, float] = field(default_factory=dict)


def row(step: Step) -> str:
    mark = "PASS" if step.ok else "FAIL"
    timings = "  ".join(f"{k} {v:.0f}ms" for k, v in step.timings.items())
    return f"  [{mark}] {step.name:28} {timings:38} {step.detail}"


async def main() -> int:
    # Deliberately one linear function: the value of a smoke test is that it reads in the
    # same order the system runs, so a failure points straight at a stage.
    settings = get_settings()
    configure_logging(settings.log_level)
    if not settings.gnani_api_key:
        print("GNANI_API_KEY is not set in .env", file=sys.stderr)
        return 2
    if not settings.llm_api_key or not settings.llm_model:
        print("LLM_API_KEY and LLM_MODEL must be set in .env", file=sys.stderr)
        return 2

    out_dir = settings.var_dir / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    steps: list[Step] = []

    engine = create_engine(settings)
    sessions = create_session_factory(engine)
    redis = create_redis(settings)
    budget = BudgetGuard(redis, sessions, settings)
    http = httpx.AsyncClient()

    before = await budget.spent_credits(BillingPool.GNANI)
    print(f"Gnani credits spent before this run: {before:.4f}")
    print(
        f"Rates confirmed: {budget.cost.rates_confirmed} "
        f"(credit figures below are {'real' if budget.cost.rates_confirmed else 'estimates'})\n"
    )

    try:
        # 1. Timbre -------------------------------------------------------------
        raw_tts = build_tts_provider(
            ProviderName.GNANI,
            client=http,
            api_key=settings.gnani_api_key,
            sse_url=settings.gnani_tts_sse_url,
            model=settings.tts_model,
            language=settings.tts_language,
            usage=budget,
        )
        tts = CachingTTS(raw_tts, settings.tts_cache_dir, usage=budget)
        request = TtsRequest(
            text=SMOKE_LINE,
            voice="Chetana",
            language=settings.tts_language,
            model=settings.tts_model,
            sample_rate=SMOKE_RATE,
        )
        step = Step("timbre v3.0 synthesis")
        started = time.perf_counter()
        pcm = bytearray()
        first_byte = 0.0
        try:
            # Deliberately the uncached provider: a smoke test served from a previous run's
            # cache would report 25 ms and prove nothing about the API being reachable.
            async for chunk in raw_tts.synthesize(request):
                if not pcm:
                    first_byte = (time.perf_counter() - started) * 1000
                pcm.extend(chunk.pcm)
            seconds = len(pcm) / 2 / SMOKE_RATE
            (out_dir / "synthesised.wav").write_bytes(pcm_to_wav(bytes(pcm), SMOKE_RATE))
            step.ok = len(pcm) > 0
            step.timings = {
                "first byte": first_byte,
                "total": (time.perf_counter() - started) * 1000,
            }
            step.detail = f"{seconds:.2f}s audio, {len(SMOKE_LINE)} billable chars"
        except Exception as exc:
            step.detail = f"{type(exc).__name__}: {exc}"
        steps.append(step)

        # 2. The same text again, from cache ------------------------------------
        step = Step("tts cache miss then hit")
        try:
            # Start cold so the result does not depend on whether an earlier run cached it.
            tts.path_for(request).unlink(missing_ok=True)
            cold = bytearray()
            async for chunk in tts.synthesize(request):
                cold.extend(chunk.pcm)
            after_miss = tts.hits

            started = time.perf_counter()
            warm = bytearray()
            async for chunk in tts.synthesize(request):
                warm.extend(chunk.pcm)
            hit_ms = (time.perf_counter() - started) * 1000

            identical = bytes(warm) == bytes(cold)
            step.ok = identical and tts.hits == after_miss + 1 and tts.misses >= 1
            step.timings = {"cached replay": hit_ms}
            step.detail = f"{tts.misses} miss / {tts.hits} hit, byte-identical: {identical}"
        except Exception as exc:
            step.detail = f"{type(exc).__name__}: {exc}"
        steps.append(step)

        # 3. Prisma streaming ---------------------------------------------------
        step = Step("prisma v3.0 streaming stt")
        if pcm:
            stt = build_stt_provider(
                ProviderName.GNANI,
                api_key=settings.gnani_api_key,
                ws_url=settings.gnani_stt_ws_url,
                model=settings.stt_model,
                usage=budget,
            )
            config = SttConfig(
                model=settings.stt_model,
                language_code=settings.stt_language_code,
                sample_rate=SMOKE_RATE,
                output_format=settings.stt_format,
                idle_timeout_sec=30,
            )
            try:
                session = await stt.open_session(config)
                transcripts: list[str] = []
                provider_latencies: list[int] = []
                event_kinds: list[str] = []
                last_endpoint_at = 0.0
                finalise_ms: float | None = None

                async def pump() -> None:
                    buffer = FrameBuffer()
                    frames = buffer.push(bytes(pcm))
                    tail = buffer.flush()
                    if tail:
                        frames.append(tail)
                    for frame in frames:
                        await session.send_audio(frame)
                        # Real-time cadence: the docs say not to burst.
                        await asyncio.sleep((len(frame) // 2) / SMOKE_RATE)
                    await session.commit()

                pump_task = asyncio.create_task(pump())
                try:
                    # Drain until the stream goes quiet rather than stopping at the first
                    # segment: Prisma splits on its own VAD, so one spoken line can arrive
                    # as several transcripts, and stopping early understates what it heard.
                    async with asyncio.timeout(45):
                        async for event in session.events():
                            event_kinds.append(event.kind.value)
                            if isinstance(event, (SttProcessing, SttSpeechStart)):
                                last_endpoint_at = time.perf_counter()
                            if isinstance(event, SttTranscript) and event.text:
                                transcripts.append(event.text)
                                if event.provider_latency_ms:
                                    provider_latencies.append(event.provider_latency_ms)
                                # Measured from the last endpointing signal the server
                                # gave us. Prisma was not observed to emit `processing`, so
                                # in practice this is speech_start -> transcript: an upper
                                # bound on recognition latency, not the endpointing delay.
                                if last_endpoint_at:
                                    elapsed = (time.perf_counter() - last_endpoint_at) * 1000
                                    finalise_ms = max(finalise_ms or 0.0, elapsed)
                                if pump_task.done():
                                    break
                except TimeoutError:
                    pass
                finally:
                    pump_task.cancel()
                    await session.aclose()

                heard = " ".join(transcripts)
                step.ok = bool(transcripts)
                step.timings = {"provider-reported": float(max(provider_latencies, default=0))}
                if finalise_ms is not None:
                    step.timings["speech_start->text"] = finalise_ms
                step.detail = (
                    f"{len(transcripts)} segment(s): {heard!r} | events: "
                    f"{'>'.join(dict.fromkeys(event_kinds))}"
                    if transcripts
                    else f"no transcript; events seen: {'>'.join(dict.fromkeys(event_kinds))}"
                )
            except Exception as exc:
                step.detail = f"{type(exc).__name__}: {exc}"
        else:
            step.detail = "skipped: no audio from step 1"
        steps.append(step)

        # 4. LLM dialogue + JSON ------------------------------------------------
        llm = build_llm_provider(
            ProviderName.OPENAI,
            client=http,
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            fallback_model=settings.llm_fallback_model,
            fallback_reasoning_effort=settings.llm_fallback_reasoning_effort,
            first_token_timeout=settings.llm_first_token_timeout_s,
            request_timeout=settings.llm_request_timeout_s,
            usage=budget,
        )
        messages = [
            ChatMessage(role=Role.SYSTEM, content=DADA_SYSTEM),
            ChatMessage(role=Role.USER, content=SCAMMER_LINE),
        ]
        step = Step(f"llm dialogue ({settings.llm_model})")
        started = time.perf_counter()
        try:
            pieces: list[str] = []
            ttft = 0.0
            async for piece in llm.stream_chat(messages, max_tokens=80):
                if not pieces:
                    ttft = (time.perf_counter() - started) * 1000
                pieces.append(piece)
            reply = "".join(pieces).strip()
            step.ok = bool(reply)
            step.timings = {"first token": ttft, "total": (time.perf_counter() - started) * 1000}
            step.detail = f"{reply[:58]!r}"
        except Exception as exc:
            step.detail = f"{type(exc).__name__}: {exc}"
        steps.append(step)

        step = Step(f"llm json ({settings.analysis_model})")
        started = time.perf_counter()
        try:
            assessment = await llm.complete_json(
                [
                    ChatMessage(
                        role=Role.SYSTEM,
                        content=(
                            "Classify the scam stage. Reply with JSON only: "
                            '{"stage": one of hook|authority|fear|isolation|payment, '
                            '"pressure": 0.0-1.0}'
                        ),
                    ),
                    ChatMessage(role=Role.USER, content=SCAMMER_LINE),
                ],
                StageAssessment,
                model=settings.analysis_model,
            )
            step.ok = True
            step.timings = {"total": (time.perf_counter() - started) * 1000}
            step.detail = f"stage={assessment.stage} pressure={assessment.pressure}"
        except Exception as exc:
            step.detail = f"{type(exc).__name__}: {exc}"
        steps.append(step)

        # 5. The fallback, forced ----------------------------------------------
        step = Step("llm fallback switches")
        try:
            forced = build_llm_provider(
                ProviderName.OPENAI,
                client=http,
                base_url=settings.llm_base_url,
                api_key=settings.llm_api_key,
                model="model-that-does-not-exist",
                fallback_model=settings.llm_fallback_model,
                fallback_reasoning_effort=settings.llm_fallback_reasoning_effort,
                first_token_timeout=settings.llm_first_token_timeout_s,
                request_timeout=settings.llm_request_timeout_s,
                usage=budget,
            )
            started = time.perf_counter()
            text = "".join([p async for p in forced.stream_chat(messages, max_tokens=60)]).strip()
            step.ok = bool(text) and "`" not in text
            step.timings = {"total": (time.perf_counter() - started) * 1000}
            step.detail = f"recovered on {settings.llm_fallback_model}: {text[:40]!r}"
        except Exception as exc:
            step.detail = f"{type(exc).__name__}: {exc}"
        steps.append(step)

        # 6. The budget guard actually refuses ---------------------------------
        step = Step("budget guard refuses")
        try:
            tight = BudgetGuard(
                redis,
                sessions,
                settings.model_copy(update={"gnani_credit_stop_at": 0.0}),
            )
            try:
                await tight.check("smoke", tight.cost.price_tts(1000))
                step.detail = "guard did NOT refuse a call past its ceiling"
            except BudgetExceeded as exc:
                step.ok = True
                step.detail = str(exc)[:70]
        except Exception as exc:
            step.detail = f"{type(exc).__name__}: {exc}"
        steps.append(step)

        # Report ---------------------------------------------------------------
        after = await budget.spent_credits(BillingPool.GNANI)
        avoided = await budget.avoided_credits()
        status = await budget.status(BillingPool.GNANI)

        print("Results")
        print("-" * 118)
        for step in steps:
            print(row(step))
        print("-" * 118)
        passed = sum(1 for s in steps if s.ok)
        print(f"{passed}/{len(steps)} checks passed")
        print(
            f"\nGnani credits this run : {after - before:.4f}"
            f"{'  (estimated)' if not budget.cost.rates_confirmed else ''}"
        )
        print(f"Avoided by caching     : {avoided:.4f}")
        print(
            f"Gnani spent / hard stop: {status.spent:.4f} / "
            f"{status.stop_at if status.stop_at is not None else 'none'}"
        )
        print(f"Artifacts              : {out_dir}/")
        return 0 if passed == len(steps) else 1
    finally:
        await http.aclose()
        await redis.aclose()
        await engine.dispose()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
