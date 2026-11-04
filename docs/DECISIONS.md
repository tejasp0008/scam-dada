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
*2026-11-04 · Phase 0 · **superseded by 0013**, once the dashboard showed credits are denominated in rupees*

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

---

## 0012 — Prisma does not emit `processing`, so filler timing comes from the local VAD
*2026-11-04 · measured against the live streaming API*

The docs list `processing` among the server events ("speech end detected; transcription
started"), and decision 0002 assumed it as the cue to start a filler phrase while the LLM
thinks. A real streaming session, at real-time cadence, with
`model=prisma-v3.0 format=transliterate language_code=indic-auto`, emitted:

```
ready > speech_start > transcript
```

No `processing`, across both transcript segments. The smoke script now prints the observed
event sequence on every run so this stays visible rather than becoming folklore.

**Consequence.** There is no server-side signal that the speaker has *stopped*, only that
they *started* (`speech_start`) and, later, what they said (`transcript`). So the filler
phrase cannot be scheduled from a server event. In Phase 2 it fires from the local
`EnergyVad` closing its gate after `silence_frames_to_close`, which is the only
end-of-speech signal we actually have. This makes the local VAD load-bearing for turn-taking
as well as for barge-in — a bigger role than decision 0003 gave it.

**Also measured.** Recognition latency can therefore only be bounded, not isolated: the
smoke script reports `speech_start -> transcript` (1208 ms on a 3.3 s clip) alongside the
`latency` figure Prisma itself reports per segment (69 ms). The two measure different things
and the gap is mostly the speaker still talking, which is why the honest label is
`speech_start->text` rather than anything that implies endpointing delay.

---

## 0013 — Spend is tracked in credits, with raw units kept alongside
*2026-11-04 · supersedes 0007*

Decision 0007 capped raw units because the docs publish no conversion. The usage dashboard
then showed what the docs do not: credits are denominated in rupees (5,000 granted, ₹0.17
spent over six requests, so 1 credit = ₹1), and **STT is billed in minutes, rounded up** —
each ~2.7 s call was logged as 0.05 minutes.

**Decision.** `app/core/cost.py` prices every call: raw measurement → billed units →
credits. The guard's ceiling is in credits, defaulting to a 5,000 total with a hard stop at
4,500 used. Two independent pools are tracked, because they are two wallets: Gnani's
programme credits, and the LLM vendor's own billing.

**Three things this gets right that a naive implementation would not:**

*Rounding is modelled, not divided.* 2.7 s of audio costs the same as 3.0 s. Treating STT
as per-second would have understated a chatty call by over 10%, and the error compounds
across hundreds of short utterances in a long scambait. The increment and the minimum are
both config values.

*Raw units are stored next to credits.* `api_usage` keeps `raw_units` (seconds, characters,
tokens) as well as `billed_units` and `credits`. The rates are currently **placeholders
inferred from a six-request sample**, so when the real rates arrive every historical credit
figure can be re-derived from the measurements rather than being lost. Until
`RATES_CONFIRMED=true`, every charge carries `estimated=True` and the smoke script says so
in its header — no inferred number can be mistaken for a fact.

*The pools do not mix.* Groq spend hitting the Gnani ceiling would hide a real limit and
stop a call for the wrong reason. The LLM pool has no credit rate yet, so it is constrained
by its token ceiling instead.

**Cost.** This replaced the `api_usage` schema outright (migration 0002) rather than
evolving it: an integer `units` column cannot hold 0.05 minutes, and the table held only
throwaway smoke rows.

---

## 0014 — A fallback model, but only before the first token
*2026-11-04 · Phase 1*

`qwen/qwen3.8-27b` is a Groq **preview** model. It is the fastest to first token by 2.3×
and the best at Hinglish, so it is worth running, but a preview model can be withdrawn or
throttled without notice — and the dialogue path is the one place where a 404 means the
scammer hears silence.

**Decision.** `FallbackLLM` wraps the primary and switches to `openai/gpt-oss-120b` when the
primary errors, stalls past `LLM_FIRST_TOKEN_TIMEOUT_S` (3 s), or yields an empty stream.
The fallback is configured with `reasoning_effort: low`, a plain-text system instruction,
and backtick stripping.

**The constraint that shapes it: fallback only before the first token.** Once a piece of
text has been handed to the caller, the persona has begun speaking that sentence. Switching
then would splice the first half of Qwen's reply onto the second half of a different model's
reply, mid-utterance. So a failure after the first token propagates instead, and the
orchestrator degrades to a cached filler. There is a contract test pinning exactly this.

**Why the fallback needs its own configuration.** Measured, not assumed: `gpt-oss` on
default settings returns HTTP 200 with no content at all — the whole budget goes to
reasoning tokens — and with effort set low it still occasionally emits markdown fences into
text that is about to be spoken aloud. An empty completion is therefore treated as a
failure, and backticks are stripped from the stream.

## 0015 — The lead persona is Dada, not Dadi
*2026-11-05 · Phase 1.5*

The project was built around an elderly woman, "Dadi". None of Timbre's seven female v3.0
voices read as elderly even aged in post, and cloning a real elderly woman's voice is out of
reach — there is no consenting speaker to record, and recording one without consent is not
something this project will do for a contest entry.

**Decision.** The lead persona is **Dada**: Hariprasad, 74, a retired Indian Railways clerk.
The relay is Dada → grandson. The project is renamed Scam Dada.

**What was deliberately *not* renamed.** The Postgres role and database are still `dadi` and
`scamdadi`. Renaming them means `docker compose down -v` and a fresh volume, which would
destroy the `api_usage` rows — the only record of what this project has actually spent, and
the evidence behind every credit figure in the README. A cosmetic rename is not worth
turning a measurement back into a guess. The working directory is still `Scam Dadi` for the
same class of reason: it is one `mv` whenever it is convenient, and it breaks nothing.

Past entries in this log still say "Dadi" where they describe what was decided at the time.
A decision log that is edited to agree with the present is not a log.

## 0016 — Timbre tag scope is sticky, and unknown tags are billed
*2026-11-05 · Phase 1.5*

Two facts from docs.gnani.ai/api/TTS/tts-audio-tags that change how prompts are written:

**Tags are not events; they are modes.** "A tag applies from its position to the end of the
input (or until the next tag). There is no closing tag." One `<slow>` at the front of an
utterance therefore slows the entire utterance for six billable characters. That is the
cheapest elderly-voice effect available and it is now how the Dada prompt and the audition
line are written. The flip side: a single stray `<excited>` poisons everything after it,
which is an argument for a small vocabulary per persona rather than the full catalogue.

**Unknown tags are stripped silently but still billed.** An LLM that writes `<cough>` for
`<coughs>` is charged eight characters for nothing. `app/core/tags.py` holds the 13-tag
catalogue and a per-persona allowed set (6 tags for Dada, 4 for the grandson, laughter
excluded for both because mockery is ruled out by the ethics block), and strips anything
else before synthesis.

**An unresolved contradiction in the docs.** The billing section states that
`Hello! <giggle> Great to hear from you.` "is billed as 42 characters — the full string, tag
included". That string is 39 characters. Either the example is wrong or Timbre adds ~3
characters per request, which would make TTS estimates ~8% low on short utterances.
`billable_characters` is plain `len` and a test pins the discrepancy. Resolving it needs one
real invoice line, which is on the list of numbers requested from the usage dashboard.

## 0017 — Dada's voice is aged in DSP, and the room is synthetic
*2026-11-05 · Phase 1.5*

Timbre publishes gender and language per voice but not age, and none of the six male v3.0
voices sounds older than about 45. Two gaps had to be closed locally.

**Ageing** (`app/core/audio/aging.py`): pitch down by resampling, tempo restored
independently by a WSOLA time-stretch, 5 Hz amplitude and pitch tremble, then a 300–3400 Hz
telephone band-pass. The band-pass does most of the perceptual work — it removes the
studio-clean low end that identifies TTS — which is why `phone-only` exists as a preset: it
is the control that shows how much of the effect is the ageing and how much is the channel.
The time-stretch is WSOLA rather than plain overlap-add because splicing a voiced vowel at
arbitrary phase cancels the fundamental and warbles.

**Room tone** (`app/core/audio/ambience.py`): a ceiling fan (noise chopped at the blade-pass
rate), 50 Hz mains harmonics, distant traffic and a broadband floor, generated rather than
sampled. Digital silence between words is the loudest tell that a voice is synthetic.

**Three consequences worth recording:**

1. **The bed is mixed before the band-pass, not after.** A real microphone hears the room and
   the carrier then filters voice and room together. Mixing after would put 50 Hz hum on a
   line that cannot carry it. A pleasant side effect: the hum fundamental is removed exactly
   as a real line removes it, and only its upper harmonics survive.
2. **The bed is a loop read positionally, not a per-utterance effect.** Room noise that
   starts and stops with each reply tells the listener precisely when the machine is and is
   not speaking. `AmbienceBed.take(count, offset)` exists so one offset runs for a whole call.
3. **Everything is deterministic — no clock-seeded RNG anywhere.** The TTS cache is keyed by
   content hash; a bed that differed per run would make every cache entry a different room.
   The ambience presets carry a fixed seed and the tremble uses two incommensurate sinusoids
   instead of noise.

**Measured cost.** 4–136 ms per clip for 10–13 s of audio, i.e. roughly 1.2% of real time;
the WSOLA stretch is ~90% of it. Cheap enough to run in the live path, *if* the stretch is
reworked to be chunk-wise with one frame of lookahead (~43 ms at 24 kHz). Until then ageing
runs per-utterance and adds its cost before the first byte, which is why the elder's filler
phrases will be aged once at cache-write time rather than on every playback.

**No credit cost.** All of it is local DSP applied to one synthesis per voice, so the
24-clip audition grid cost exactly six requests.

## 0018 — Voice cloning is available, but it cannot be combined with audio tags
*2026-11-05 · Phase 1.5*

Investigated because a consenting elderly male relative may be available to record.

**It works, and it is reachable.** Two steps: `POST /api/v1/tts/voice-clone/embeddings` with
5–30 s of clean reference audio as `multipart/form-data` returns a `speaker_embedding`
(`embedding` string, `shape [1, 768]`, `dtype torch.bfloat16`); that object is then passed to
the ordinary `/api/v1/tts/inference`, `/tts/sse` or `/tts` WebSocket endpoints. There is no
server-side voice ID — the caller caches the embedding.

**But it is a different model.** The cloned-voice request uses `vachana-vc-v1`, which "is
automatically set when `speaker_embedding` is provided". Audio tags are documented as
"supported on `timbre-v3.0` **only**". So cloning costs the entire expressive layer:
`<slow>`, `<coughs>`, `<sigh>` and `<pauses>` — the four tags Dada's prompt is built on, and
the ones doing the most work in the audition clips.

**Two further unknowns.** The cloned-voice request body has no `language` field at all,
unlike Timbre's, so Hinglish handling for a cloned voice is undocumented. And the docs state
no consent requirement, which means the obligation is entirely ours rather than enforced.

**Decision.** Do not clone for the demo. The ageing chain already produces an elderly voice
from a tagged Timbre voice, and trading a certain loss (tags) for an uncertain gain
(timbre fidelity, unknown Hinglish) is the wrong way round this close to a deadline. The
experiment is cheap to run later — one embedding call plus one synthesis, and the ageing
chain applies to a cloned voice unchanged — so it is a post-submission question.

**If it is ever run:** a voice embedding of an identifiable living person is personal
biometric data. It never goes in the repo, it goes in a gitignored file referenced from
`.env`, written consent is recorded alongside it, and the fact that a cloned elderly voice
has obvious misuse potential is reason enough to keep it out of a public artifact.
