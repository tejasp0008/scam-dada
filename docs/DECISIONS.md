# Decisions

One entry per choice that is expensive to reverse. Dated, with the reasoning, so the
architecture can be defended (or revisited) months later.

---

## 0001 — Config travels as WebSocket upgrade headers, and frames are exactly 1024 bytes
*2026-11-04 · Phase 0 · irreversible-ish: the whole audio path assumes it*

Gnani's Prisma streaming API takes all its configuration as WS upgrade headers and requires
raw PCM s16le mono in frames of **exactly 1024 bytes** (512 samples), sent at real-time
cadence. This is not a preference we get to have.

**Decision.** 16 kHz is the pipeline's internal rate end to end. The browser worklet
captures at the device rate (usually 48 kHz), downsamples to 16 kHz and emits exact
512-sample frames. `FrameBuffer` carries the remainder between pushes rather than padding
or dropping it.

**Why 16 kHz rather than 48 kHz.** Prisma resamples 44.1/48 kHz internally, but the
1024-byte frame rule still applies, so 48 kHz would mean 3× the WebSocket messages for
audio the model downsamples anyway. 16 kHz is also closer to the telephone audio the model
was trained on.

**Consequence.** `app/core/audio/framing.py` is pure and heavily tested, because an
off-by-one in frame slicing shows up as a subtly garbled transcript, hours later, with no
error anywhere.

---

## 0002 — The STT port yields an event union, not a transcript stream
*2026-11-04 · Phase 0 · changes every consumer if revisited*

Prisma streaming returns `is_final=true` on **every** transcript — it has no partial
hypotheses — and signals turn boundaries through separate `speech_start` and `processing`
events.

**Decision.** `SttSession.events()` yields `SttEvent`, a union of `ready`, `speech_start`,
`processing`, `transcript` and `error`, rather than the transcript-only iterator the brief
suggested.

**Why.** A transcript-only stream throws away the only endpointing signal the API gives us.
Without `processing` there is no cue to start a filler phrase, and without `speech_start`
there is no confirmation for barge-in. The usual trick of endpointing on stabilising
partials is simply not available here.

---

## 0003 — The local energy VAD survives even though Prisma ships a VAD
*2026-11-04 · Phase 0*

Prisma does server-side VAD with configurable thresholds and reports `speech_start`.

**Decision.** Keep a local `EnergyVad` over the inbound frames anyway. It exists **only**
to cut barge-in; Prisma's events remain the authority on turn boundaries.

**Why.** Barge-in cannot afford a network round trip. By the time the server reports that
the scammer started talking, the persona has already talked over them for 100-300 ms, which
is exactly what makes a voice agent feel robotic. Hysteresis (N loud frames to open, M quiet
to close) is what stops a line click from cancelling the persona mid-word.

---

## 0004 — TTS yields raw PCM, not encoded audio
*2026-11-04 · Phase 0*

Timbre can return WAV, MP3 or OGG Opus, over REST, SSE or WebSocket, with audio **base64
encoded inside JSON** on the streaming transports. Neither streaming transport documents a
cancel or interrupt message.

**Decision.** `TTSProvider.synthesize` yields `AudioChunk` carrying raw PCM s16le. The
adapter requests `container=raw`, `encoding=pcm_s16le` and decodes the base64 itself. Since
no cancel exists, barge-in is implemented as: stop browser playback immediately, then
abandon and close the TTS stream.

**Why.** The browser must be able to discard queued audio the instant the scammer
interrupts, and that is only cheap with unframed PCM in a ring buffer. Container headers
mid-stream, or an MP3 decode, both get in the way of dropping audio instantly.

---

## 0005 — One event schema serves the dashboard, the database and the evidence pack
*2026-11-04 · Phase 0 · the most expensive thing here to change later*

**Decision.** `app/core/events.py` defines eight frozen Pydantic events behind a
discriminated union. The same JSON is the dashboard WebSocket payload, the `events.payload`
JSONB column and the evidence pack's input. Events carry a per-call monotonic `seq`.

**Why.** Three schemas for the same facts drift, and the drift shows up as a dashboard that
disagrees with the PDF. One schema also means a finished call replays into the live UI with
no second code path, and a dashboard that reconnects can ask for "everything after seq N".
`extra="forbid"` makes a typo'd field a loud failure instead of a silently missing panel.

**Cost.** Adding a field to an event is a schema change in one place, but renaming one
breaks persisted history. Accepted: the alternative is worse.

---

## 0006 — Rotating STT sessions, hidden behind the port
*2026-11-04 · Phase 0 decision, Phase 2 implementation*

Prisma closes a streaming session at **15 minutes** (WS close 1001). The product's entire
purpose is keeping a scammer on the line for longer than that.

**Decision.** `services/stt_session.py` opens a successor socket at ~13 minutes, waits for a
silence window, swaps, and drains the old one. The orchestrator sees one continuous
transcript stream and knows nothing about it. `x-idle-timeout-sec` is set high (900) so a
scammer pausing to think does not drop the session.

**Why.** Discovering this during a demo rehearsal, as a call that dies at minute 15 for no
visible reason, is the worst possible time to learn it.

---

## 0007 — The budget guard counts raw units, not credits
*2026-11-04 · Phase 0*

Gnani's docs publish no credit or rupee conversion anywhere; the limits page says rates live
in the account console. The programme grants 5,000 credits of unstated denomination.

**Decision.** `api_usage` and the guard track raw units per provider — TTS billable
characters (tags included: the docs say tag characters are billable), STT audio seconds, LLM
tokens — against a configurable ceiling per unit type. Cache hits are recorded as zero-unit
rows rather than skipped.

**Why.** Inventing a cost model would produce exactly the kind of fabricated number the
brief forbids. Recording cache hits is what turns "caching saves credits" into a measured
figure.

---

## 0008 — Evon is not reachable, so the LLM is always OpenAI-compatible
*2026-11-04 · Phase 0*

Gnani's own FAQ says Evon v3.3 is not on the API platform — open weights by request on
Hugging Face (Apache 2.0, 30B MoE). The Agent Builder's `GET /platform/v1/agents/config/llm_models`
returns Claude, GPT-4.1-mini, GPT-4o-mini and Gnani's "Pampa" models, no Evon, and it is a
selector for Agent Builder agents rather than a completions endpoint.

**Decision.** `LLMProvider` with a single OpenAI-compatible adapter (configurable base URL).
Pointing it at Evon later, self-hosted or otherwise, is one environment variable.

**Why.** Serving a 30B MoE locally is not available to this project in contest week, and the
brief's instruction not to invent API details rules out writing an Evon client against no
documentation. The README states this plainly rather than implying Evon is in use.

---

## 0009 — Prisma runs in `transliterate`, so the pipeline speaks Latin Hinglish
*2026-11-04 · measured, not assumed*

Prisma v3.0 offers four output formats. All four were run against the same 2.7 s clip, in
which Timbre said *"Haan beta, &lt;sigh&gt; kaun bol raha hai?"*:

| `format` | transcript | notes |
|---|---|---|
| `verbatim` | `बहन बेटा, कौन बोल रहा है?` | Devanagari; "Haan" misheard as "बहन" (bahan) |
| `transcribe` | `कौन बोल रहा है?` | Devanagari; **dropped the opening two words** |
| `codemix` | `बहन बेटा, कौन बोल रहा है?` | identical to verbatim on this clip |
| `transliterate` | `Haan beta, kaun bol raha hai?` | **exact, in Latin script** |

**Decision.** `STT_FORMAT=transliterate`, exposed as `Settings.stt_format`.

**Why.** Three reasons, in order of weight. The LLM persona prompt is written in Latin
Hinglish, so a Devanagari transcript forces a script mismatch between what the scammer said
and how Dadi is told to speak. The dashboard transcript is read by judges who may not read
Devanagari. And `transcribe`, the format used in the docs' own quick-start example, silently
dropped the first two words of a four-word phrase, which in this product is the difference
between hearing the scammer's opening claim and missing it.

**Caveat.** `transliterate` returns no `detected_language_code` (the docs say so, and the run
confirmed `None`). Language detection therefore has to come from a second signal if we ever
need it; nothing currently does.

---

## 0010 — Dialogue and analysis run on different models, chosen by measured TTFT
*2026-11-04 · measured, not assumed*

The Groq account has no Llama models. Available chat models were benchmarked on a real Dadi
system prompt plus a scammer line, streaming, measuring time to first token:

| model | TTFT | reply quality |
|---|---|---|
| `qwen/qwen3.8-27b` | **181 ms** | in character, fluent Hinglish |
| `allam-2-7b` | 164 ms | fastest, but confused the roles — it is an Arabic/English model |
| `openai/gpt-oss-120b` (`reasoning_effort: low`) | 422 ms | good reply, emitted a stray markdown fence |
| `openai/gpt-oss-20b` (`reasoning_effort: low`) | 584 ms | in character |
| `openai/gpt-oss-20b` (default) | — | **returned empty content**: the whole budget went to reasoning tokens |

**Decision.** `LLM_MODEL=qwen/qwen3.8-27b` for dialogue, `LLM_ANALYSIS_MODEL=openai/gpt-oss-120b`
for analysis.

**Why.** Dialogue is latency-critical — every millisecond before the first token is silence
the scammer hears — and Qwen was 2.3× faster to first token than the next usable option while
also producing the best Hinglish. Analysis is a background JSON extraction that must not
block a reply, so a slower reasoning model is the right trade there.

**Consequence for the adapter.** The gpt-oss default returning empty content is a trap: the
OpenAI-compatible adapter must pass arbitrary extra parameters (`reasoning_effort`) from
config, and `complete_json` must treat an empty completion as a failure to retry rather than
as valid output.

---

## 0011 — Timbre's WAV header cannot be trusted; we use raw PCM and write our own
*2026-11-04 · measured, not assumed*

A Timbre REST response with `container=wav` was inspected byte by byte:

```
file size on disk   : 128792 bytes
RIFF chunk declares : 4294967303 bytes
data chunk declares : 4294967295 bytes   (0xFFFFFFFF)
data bytes actually : 128748 bytes
duration per header : 89478.49 s         <- what any WAV reader believes
duration from bytes : 2.68 s             <- the truth
```

Timbre ships a streaming placeholder size (`0xFFFFFFFF`) in a non-streaming response. Python's
`wave` module reports the clip as 24.8 hours long. Prisma transcribed it correctly regardless,
so the bytes are fine — only the header lies.

**Decision.** Request `container=raw` with `encoding=pcm_s16le` everywhere, which decision 0004
already required for barge-in. Wherever a real WAV file is needed — the demo replay fixture,
the ffmpeg highlight cut, anything an operator opens — we write the header ourselves from the
byte count.

**Why it matters beyond tidiness.** Duration is load-bearing in three places: the
"scammer minutes wasted" counter, the STT seconds the budget guard charges, and any ffmpeg
operation in Phase 5. All three would have been wrong by four orders of magnitude, and the
counter is a number shown to judges.
