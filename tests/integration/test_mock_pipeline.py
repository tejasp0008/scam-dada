"""A whole turn through all three mock providers.

Not the orchestrator (that arrives in phase 2) but the path it will drive: browser-shaped
audio chunks, re-framed to Prisma's contract, transcribed, answered by the LLM and
synthesised. It proves the ports fit together before any real adapter exists.
"""

from __future__ import annotations

import numpy as np

from app.core.audio.framing import FrameBuffer, float32_to_pcm, resample_linear
from app.core.audio.vad import EnergyVad
from app.providers.base import (
    ChatMessage,
    Role,
    SttConfig,
    SttTranscript,
    TtsRequest,
)
from app.providers.mock import MockLLMProvider, MockSTTProvider, MockTTSProvider

BROWSER_RATE = 48000
PIPELINE_RATE = 16000


def browser_chunk(seconds: float, amplitude: float) -> bytes:
    """Audio as a browser would hand it over: 48 kHz float frames, arbitrary length."""
    t = np.arange(int(BROWSER_RATE * seconds), dtype=np.float32) / BROWSER_RATE
    return float32_to_pcm((np.sin(2 * np.pi * 220.0 * t) * amplitude).astype(np.float32))


async def test_one_turn_from_browser_audio_to_synthesised_reply() -> None:
    stt = MockSTTProvider(frames_per_utterance=5)
    llm = MockLLMProvider()
    tts = MockTTSProvider()
    vad = EnergyVad(speech_frames_to_open=2)

    session = await stt.open_session(
        SttConfig(model="mock", language_code="hi-IN", sample_rate=PIPELINE_RATE)
    )
    buffer = FrameBuffer()

    # 1. Resample and re-frame browser audio, and watch it with the local VAD.
    raw = browser_chunk(seconds=0.6, amplitude=0.4)
    samples = resample_linear(
        np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0,
        BROWSER_RATE,
        PIPELINE_RATE,
    )
    speech_detected = False
    frames_sent = 0
    for frame in buffer.push(float32_to_pcm(samples)):
        assert len(frame) == session.frame_bytes
        speech_detected |= vad.is_speech(frame)
        await session.send_audio(frame)
        frames_sent += 1

    assert frames_sent > 0, "0.6s of 48 kHz audio must produce whole 16 kHz frames"
    assert speech_detected, "the local VAD must hear speech, otherwise barge-in can never fire"

    # 2. Pull the transcript the provider produced.
    await session.commit()
    transcript: SttTranscript | None = None
    async for event in session.events():
        if isinstance(event, SttTranscript):
            transcript = event
            break
    assert transcript is not None
    assert transcript.text
    assert transcript.is_final
    assert transcript.detected_language_code == "hi-IN"
    await session.aclose()

    # 3. Answer in character, streamed.
    tokens = [
        token
        async for token in llm.stream_chat(
            [
                ChatMessage(role=Role.SYSTEM, content="You are Dadi."),
                ChatMessage(role=Role.USER, content=transcript.text),
            ]
        )
    ]
    reply = "".join(tokens).strip()
    assert len(tokens) > 1, "the dialogue path must stream, so TTS can start mid-reply"
    assert reply

    # 4. Synthesise it.
    chunks = [
        chunk
        async for chunk in tts.synthesize(
            TtsRequest(
                text=reply,
                voice="Shivani",
                language="hi-en",
                model="timbre-v3.0",
                sample_rate=24000,
            )
        )
    ]
    assert chunks
    assert chunks[-1].is_final
    assert sum(len(c.pcm) for c in chunks) > 0
    assert tts.requests[0].text == reply


async def test_the_scripted_call_runs_to_the_end_of_the_script() -> None:
    """A multi-turn call must not stall once the script is exhausted."""
    stt = MockSTTProvider(frames_per_utterance=2)
    session = await stt.open_session(
        SttConfig(model="mock", language_code="indic-auto", sample_rate=PIPELINE_RATE)
    )
    transcripts: list[str] = []
    for _ in range(8):
        await session.send_audio(b"\x00" * session.frame_bytes)
        await session.commit()
    await session.aclose()
    async for event in session.events():
        if isinstance(event, SttTranscript):
            transcripts.append(event.text)
    assert len(transcripts) == 5, "the default script has five utterances and no more"
    assert len(set(transcripts)) == 5, "each utterance is delivered exactly once"
