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

## 0019 — Three stages, in the order physics puts them
*2026-11-05 · Phase 1.5*

The first ageing attempt was ageing plus a band-pass, and it still sounded like filtered TTS.
Band-limiting is only the first of the things a phone call does to audio, and a voice that
has been through a 300–3400 Hz filter and nothing else is a studio recording with its ends
trimmed. Four cues were missing, and `app/core/audio/` is now split by where in the physical
chain each one belongs.

**1. The voice** (`aging.py`) — pitch, tempo, tremble, and now:

* **Aperiodic jitter.** The tremble was two incommensurate sinusoids, which never repeat
  audibly but are still far too even to be a body — it read as a tremolo pedal. The modulator
  is now 60% seeded random wander and 40% sinusoid.
* **An audible inhale** before the first word. Synthetic speech begins from absolute
  stillness, which nothing alive does. A listener never notices the breath; they notice its
  absence.

**2. The room** (`ambience.py`) — the bed, plus a short reverb. The reverb matters more than
expected: without it the voice is dry and the bed sits *behind* it, which reads as "a
recording with noise added" rather than as one place. Reverb is what glues them. 280 ms,
16% wet — low, because a phone mic is held close to the mouth, so the direct sound dominates
even in a live room.

**3. The line** (`telephony.py`) — the part that was missing entirely:

* **Handset colour.** A cheap electret capsule has a presence bump around 1–2 kHz.
* **Carrier AGC.** Telephony compresses hard, which is why phone speech sounds flat and
  close. An envelope-based compressor with a floor, so it chases speech and not the room tone.
* **8 kHz narrowband.** Down to the G.711 rate and back. Not the same as band-limiting at
  24 kHz: it brings the aliasing a real codec brings.
* **mu-law companding.** Quantising to 8 bits on a log curve puts more quantisation noise on
  loud samples than quiet ones, and that level-dependent grain is the single most recognisable
  artefact of a landline. Verified by test: the error on a full-scale tone is >5× the error on
  a quiet one.
* **Packet loss with concealment.** A lost VoIP frame is replaced by a fading repeat of the
  previous one, which is what a decoder does. A hole of true silence reads as an edit; a
  repeat reads as a bad line.

**`call_profile.py` exists because the order is audible, not for tidiness.** Three ways to get
it wrong, each of which was easy to write and wrong to hear:

* Room tone mixed *after* the carrier filter carries 50 Hz hum down a line that cannot pass
  it. There is a test that fails on exactly this.
* Reverb applied *after* the bed reverberates the ceiling fan — but the fan does not echo, it
  *is* the echo. It is already a diffuse recording of a whole room.
* Compression applied *after* companding pulls quantisation noise up into the gaps between
  words instead of leaving it buried under speech.

**The ladder.** Six profiles, cleanest first: `studio`, `aged-dry`, `landline`,
`living-room`, `mobile`, `bad-line`. Each adds one thing, so the audition demonstrates what
every stage contributes rather than only presenting a finished result. `studio` is a true
control — no ageing, no room, no carrier filter.

**A bug this found.** An empty utterance rendered a 440 ms breath and a burst of room tone
out of nowhere: a failed synthesis would have made the persona audibly gasp and say nothing.
Both stages now short-circuit on empty input.

**Measured.** 0–145 ms per clip for a 13-second render, the WSOLA stretch still ~90% of it.
Bed construction is 7–15 ms and happens once per call, not per utterance. Zero credits: the
whole six-profile ladder rendered from one cached synthesis, and the run that produced it
spent nothing at all.

## 0020 — The voice sits low against the room, and the room is on the landline
*2026-11-05 · Phase 1.5*

Two requests, one mechanism.

**"In the landline voice add street and fan and room background."** `living-room` already was
exactly that — the landline channel plus `fan-and-street` plus reverb — so the bed at −22 dB
was evidently too quiet to register as present. The `landline` rung now carries the room
itself, the old `living-room` name is gone, and a bare-channel rung survives as
`landline-bare` so the ladder still isolates one change per step. `landline` is the default
elder profile.

**"Loudness of the person should be low so that it is difficult to detect whether it is AI."**
Correct instinct, and worth being precise about which knob does it:

* **Absolute output level does almost nothing.** The receiving handset's AGC lifts the voice
  and the room back up together, so the ratio survives the channel and the loudness does not.
  `output_peak` exists and is set to 0.62 on the persona profiles, but it is close to
  cosmetic; a test asserts that changing it leaves the voice-to-room ratio untouched.
* **The voice-to-room ratio is the knob that works.** A voice low against a fan and a street
  gives a listener far less clean silence in which to hear a synthesis artefact. The persona
  profiles moved from −22 dB to **−15 dB** (`PERSONA_BED_DB`).

**The dial is not calibrated to the output, and only the measurement is worth quoting.**
Nominal `level_db` is set against the voice *before* the line. The carrier AGC then lifts
quiet passages — which are bed-only — toward its target, so the delivered ratio is several dB
tighter than the nominal figure. Measured on the Devansh render, speech frames against the
lead-in:

| nominal `level_db` | delivered speech-to-room |
|---|---|
| −22 dB | 17.2 dB |
| −18 dB | 13.2 dB |
| **−15 dB (default)** | **10.2 dB** |
| −12 dB | 7.1 dB |
| −9 dB | 5.8 dB |

Monotonic but offset by 5–7 dB, which is the AGC doing exactly what an AGC does. Three tests
pin this: that the delivered ratio is below nominal, that it still moves the right way, and
that the default stays in a 6–16 dB band.

**Where the limit is, and it is not an audio limit.** Past roughly −10 dB nominal the scammer
starts saying "hello, I cannot hear you". That *raises* hang-up risk, which is the thing
feature 3 exists to minimise, so burying the voice further trades one detection risk for a
worse one. −15 dB is the chosen compromise and `--bed-db` exists to revisit it by ear.

**This does not affect transcription.** Our STT transcribes the scammer's audio, not our own
output, so a closer bed costs nothing in accuracy. The one exception is the browser simulator
when a speaker feeds the microphone, where our bed would reach our own VAD — a reason to use
headphones for the demo recording, noted rather than engineered around.

**Also fixed here:** bed generation is now decoupled from bed level. `AmbiencePreset.
generator_key()` excludes `level_db` and `AmbienceBed.mix_into` takes the level as an
argument, so an eleven-clip level sweep builds one 12-second loop instead of eleven.

## 0021 — Dada is Devansh on a bad line, and three bugs that choice exposed
*2026-11-05 · Phase 1.5*

Chosen by ear: **Devansh**, `bad-line` — the `frail` voice on `mobile-poor`, in a room with a
fan and a street. It is the aggressive end of the ladder and the trade is worth stating: a bad
line masks synthesis artefacts *and* forces the scammer to repeat himself, which is the entire
point of a scambait, but it also hands him a reason to say he will call back on a better line.
Hang-up risk is feature 3's problem; this is the input it has to work with.

Switching the default surfaced three things, two of them bugs that would have reached the demo.

**1. The grandson was on a different carrier.** `GRANDSON_PROFILE` was written out separately
and happened to say `landline` while Dada moved to `mobile-poor`. The two personas are in one
room passing one handset — neither the room nor the carrier can change when a phone changes
hands, and a connection that audibly improves mid-call is a tell in the one moment that is
meant to be seamless. The grandson is now *derived* from the elder's profile
(`_ELDER.model_copy(update={"aging": ...})`), so the only thing that can differ is the larynx,
and a test compares the whole object rather than field by field so a stage added later cannot
diverge either. `test_both_personas_are_in_the_same_room_on_the_same_line` caught this the
moment the default changed, which is the entire reason it exists.

**2. The bed level meant something different for every voice.** It was referenced to the
voice's overall RMS, which includes the pauses — so a voice that pauses less got a louder bed.
Measured across six voices reading the *same* grandson line at the *same* nominal level, the
delivered ratio ranged from **1.1 dB to 6.3 dB**. A dial that means six different things makes
"choose a level by ear" meaningless. `speech_rms` now measures only the frames above 10% of
the envelope peak, and the bed is referenced to that.

**3. An equal nominal level is not an equal delivered level across lines.** `mobile-poor`
compresses at 0.85, and a compressor lifts the bed-only passages toward its target, so:

| nominal | delivered on `landline` | delivered on `bad-line` |
|---|---|---|
| −15 dB | 8.2 dB | **3.7 dB** |
| −18 dB | — | 6.5 dB |
| −22 dB | — | **10.5 dB** |
| −26 dB | — | 14.5 dB |

At 3.7 dB the voice is genuinely hard work to follow, and `bad-line` already masks heavily
through packet loss, hard compression and mu-law quantisation — the bed does not need to do
that job twice. `BAD_LINE_BED_DB` is −22 dB, which lands at 10.5 dB delivered, level with the
cleaner profiles. **The delivered ratio is the thing held constant; the nominal number is an
implementation detail of whichever line it is on**, and a test asserts the three persona
profiles land within 6 dB of each other.

**Also added: the app refuses to start if both personas have the same voice.** A `.env` where
`PERSONA_ELDER_VOICE` and `PERSONA_GRANDSON_VOICE` match is not a degraded demo, it is a
broken one — the relay works *because* the scammer hears someone else pick up the phone. It is
an easy mistake to make by copying a line, so `Settings` validates it, case- and
whitespace-insensitively, and fails at startup rather than halfway through a recorded call.

**The grandson's audition uses his own line,** not Dada's. Judging a 27-year-old on a line
that opens with `<slow>` would have been judging the tag, not the voice.

## 0022 — Both personas are Devansh, pushed in opposite directions
*2026-11-05 · Phase 1.5*

Chosen by ear: Devansh for the grandson too. That collided with the startup guard added in
0021, which refused one voice for both personas. **The guard was too blunt and has been
replaced**, because "different Timbre voice" is the wrong test for "the scammer can tell
somebody else picked up the phone".

**Sharing a voice is legitimate, and arguably better.** Relatives share vocal timbre — a
grandfather and grandson who sound related is more plausible than two unrelated strangers
living in one house. What matters is not the catalogue entry but the *rendered* voice.
`relay_collision` now refuses only the case that is genuinely broken: the same voice with the
same ageing, which is one man pretending to hand the phone to himself.

**But the gap as chosen was too narrow, and that needed measuring rather than assuming.**
Controlled comparison — the same cached synthesis through both profiles, so only the
processing differs:

| | median F0 | duration |
|---|---|---|
| as the grandson (`none`) | 145.9 Hz | 14.3 s |
| as Dada (`frail`) | 133.3 Hz | 16.7 s |

That is **1.56 semitones** and 16% slower. Audible, but 1.6 semitones is comfortably inside
one speaker's own range, and a scammer ten minutes into a call has had a lot of exposure to
the voice. (A first attempt measured F0 on the actual handoff clip and got Dada *higher* than
the grandson — the two halves read different sentences, and intonation swamped a 9% shift.
Only the controlled comparison answers the question.)

**So the grandson is pushed the other way.** A new `young` preset — `pitch=1.05`,
`speed=1.08`, no tremble, no breath — which is not an ageing preset and says so. Remeasured:

| | median F0 | duration |
|---|---|---|
| grandson (`young`) | 151.9 Hz | 13.3 s |
| Dada (`frail`) | 133.3 Hz | 16.7 s |

**+2.26 semitones, grandson 20% faster.** Two semitones is roughly where two adult male
voices stop sounding like one voice, so this clears it rather than sitting on it. The
`AgingPreset.pitch` ceiling moved from 1.0 to 1.15 to allow it, documented as being for
distinguishing a younger relative and not for ageing anyone.

**`--handoff` renders the thing that actually answers the question.** Dada's line and the
grandson's line back to back in one clip, with the room tone continuous across the join
because the bed is read positionally — `bed_offset` for the second utterance is the sample
count of the first. Two clips side by side in a file browser do not answer it; the scammer
hears one unbroken line. Verified numerically: the step across the join is 0.0034 against a
99.9th-percentile step of 0.112, so there is no click, and the bed sits at 0.0219 before the
join and 0.0187 after.

Rendering the handoff cost nothing — both lines were already cached.

## 0023 — Timbre allows one synthesis at a time, and abandoning a stream counts
*2026-11-06 · Phase 2*

Found on the first real call through the orchestrator, not in the documentation:

```
429 RATE_LIMITED Only 1 active request is allowed at a time.
Wait for your current request to finish before sending another.
```

Every reply after the first failed to synthesise. The persona went silent for four turns
while the transcript kept scrolling, which is the worst possible failure mode — it looks like
the model stopped working.

**The cause was barge-in.** `PersonaSpeaker.stop` cancels the playback task, but cancelling
a task does not close the HTTP stream the cancelled coroutine was iterating; the generator's
cleanup ran whenever the garbage collector got to it. Upstream the request was still
*active*, so the next synthesis was refused. Barge-in is frequent, so this hit almost every
call.

**Two fixes, at the two layers that own the problem.**

* `GnaniTimbreTTS` holds an `asyncio.Lock` for the whole stream. The limit is the vendor's
  and applies across every call this process handles, so serialising in the adapter is
  right; serialising in the caller would only have covered one call at a time.
* `PersonaSpeaker` wraps the stream in `contextlib.aclosing`, so a cancelled synthesis closes
  deterministically during unwinding rather than eventually.

**And a port change this justifies.** `TTSProvider.synthesize` now returns an
`AsyncGenerator` rather than an `AsyncIterator`. That is not tidying: a bare `AsyncIterator`
makes no promise that the stream can be closed, and here the caller *must* be able to close
it. The type now states the requirement that the vendor imposes.

**The cost.** TTS concurrency on this account is one, so two simultaneous calls would queue
behind each other. Fine for a demo and for the contest; it is a real ceiling for anything
beyond that, and it belongs in the limitations list rather than in a footnote.

## 0024 — Phase 2: what the persona loop actually does
*2026-11-06 · Phase 2*

Verified on real providers end to end: five scammer turns, five persona replies, a handoff,
barge-in, and 45 seconds of persona audio. Measured per-turn latency from the caller
stopping speaking, over two consecutive live calls:

| leg | range |
|---|---|
| end of speech → transcript | 448–565 ms |
| LLM first token | 243–706 ms |
| TTS first audio | 54–860 ms |
| filler played | every turn |

**Three bugs the live calls found that the test suite could not.**

1. **The orchestrator hardcoded `prisma-v3.0`, `timbre-v3.0` and `hi-en`.** Two things wrong
   with that, and the second cost a debugging session: a vendor model name in a service
   module breaks the rule that vendors are named only inside `app.providers`, and hardcoding
   the STT config meant `STT_FORMAT` was ignored — so the first real call came back in
   Devanagari instead of the transliterated Hinglish that DECISIONS 0009 chose on measured
   evidence. The model names and the `SttConfig` now come from settings, built in the
   transport layer.
2. **After a handoff, the grandson spoke his grandfather's lines.** The relay shares one
   history, which is the point — he has to know what Dada was told. But Dada's replies sit in
   that history as `assistant` turns, so to the model they read as *its own* earlier words,
   and it continued them: his first line was Dada's line about looking for his glasses. A
   handoff now appends a system note naming whose words those were, placed last because the
   instruction that has to win is the one closest to the generation.
3. **Replies arrived with line breaks and a leading `<pauses>`.** Both are billed, neither is
   audible, and the leading silence tag landed immediately after the filler phrase had
   already covered that beat — dead air bolted onto the one place in a call where dead air is
   most expensive. `tidy_for_speech` normalises whitespace and drops a leading silence tag,
   keeping one mid-utterance where it is doing real work.

**The filler starts on the transcript, not on the VAD closing.** DECISIONS 0012 reasoned that
the local VAD closing its gate should cue the filler, since it is the only end-of-speech
signal there is. The code does not do that, deliberately: the gate needs 12 consecutive
silent frames (384 ms), and a mid-sentence pause longer than that is ordinary in slow
Hinglish speech, so firing there would make the persona talk over the scammer regularly. The
filler therefore starts when the transcript lands, which still covers the LLM and TTS legs —
about 1.3 s of the wait — but not the 450–565 ms of transcription. Starting it earlier is
worth revisiting with a shorter hysteresis or Prisma's own `speech_start`/silence timing.

**`stt_final_ms` is measurable only because the local VAD closes first.** When Prisma's
server-side VAD is faster than our hysteresis the transcript arrives before the gate closes
and the number is `None` rather than zero. Reported as `None`, never as `0`.

**Groq's input-token limit is a real constraint on call length.** The system prompt is ~6.2 KB,
about 1600 input tokens, sent every turn. The free tier allows 7000 input tokens per minute —
roughly four turns a minute — and a live call tripped it, at which point the fallback
switched to `openai/gpt-oss-120b` before the first token exactly as DECISIONS 0014 intended.
That is the first time the fallback has fired outside a contract test. Trimming the system
prompt is the obvious optimisation and has not been done yet.

**A demo-day trap worth writing down.** `docker compose` publishes the `api` container on
port 8000 as well, so a locally run `uvicorn` and a stale container both answer there — and
`/readyz` was reporting mock providers from the container while the real server ran beside
it. Stop the container, or use a different port.

## 0025 — Phase 3: two readings of the transcript, and which one wins
*2026-11-06 · Phase 3*

The intelligence layer runs twice over every transcript and reports both answers.

**The deterministic pass** (`app/core/extract.py`, `stages.py`, `risk.py`) is regex and
arithmetic. Everything it reports is reproducible, auditable and free, and lands with
`source=REGEX` at full confidence. The dashboard renders from this alone if there is no model
at all.

**The model pass** (`app/services/analysis.py`) adds what regex cannot see and lands with
`source=LLM` and its own confidence. On a live call it reconstructed `officer@demo` from the
transcript "fake dot officer at d e m o" and `CYB001` from "C Y B triple zero one" — both
spoken letter by letter, both invisible to a pattern.

**Where they disagree about the stage, the disagreement is recorded rather than resolved.**
The deterministic reading is the one shown; the model's is carried alongside so the dashboard
can display a conflict. An invented agreement would be worse than a visible disagreement. The
model may advance the stage but never drag it backwards, and the pressure reading takes the
higher of the two, because under-reporting pressure is the expensive direction — it decides
whether a retention hook is thrown.

**Analysis never touches the dialogue path.** It runs as its own task per turn; the reply does
not wait for it, does not change because of it, and does not fail when it fails. A dead model
costs dashboard detail and nothing else. Three tests assert that the persona still replies
with analysis switched off entirely.

### What a live call found that the test suite could not

**The canary detector did not detect a canary.** The worst possible outcome for the most
distinctive feature in the project. The call was leaking one by construction: `susila.9030@demo`
had been issued to an earlier call, and the scammer read it aloud. Prisma returned it as
"susila dot nine zero three zero at d e m o", so exact matching found nothing.

A scammer reading from a lead list *says* the identifier. So the transcript is now also
matched in a de-spoken form — "at" becomes `@`, "dot" becomes `.`, spelled-out digits become
digits, and "double"/"triple" expand. Re-verified live: a canary issued to one call was caught
in a later one and attributed back to the call that issued it. The exact transcript that was
missed is pinned as a test, and so is the inverse — that an ordinary Hinglish sentence
containing "at", "do" and "teen" still matches nothing, because that rewriting is aggressive
enough to be a false-positive machine if it were not checked.

**Four precision bugs, all false positives, all visible in one transcript.**

* `case_number = "NIKAL"` from "warrant nikal jayega", and `case_number = "FAKE"` from "case
  number fake C Y B...". A case number now has to contain a digit; every real one does, and
  an all-letters token after the keyword is just the next word of the sentence.
* `suspecting_a_bot` scored 0.4 because the pattern contained the bare word "fake" — which
  came from our own synthetic case-number prefix, on a caller who had suspected nothing. The
  pattern now requires the accusation ("aap bot ho", "yeh recording hai"), not the word.
* Three organisation rows for one claim: "Delhi cyber crime branch" matched the cell, the
  branch and the police patterns. The more specific label now subsumes the others.
* The badge never reached `payment` on "chhota amount bhej dijiye", because the pattern
  required the verb before the noun. It is now order-independent — and that was the single
  line in the call that most needed to move the badge.

**An ellipsis is hesitation, not punctuation.** `take_sentence` treated `...` as a sentence
end, so the orchestrator spoke "Inspector..." as a complete utterance. Then the terse reply
entered the conversation history, the model imitated it, and the next four turns were
two-word fragments. Trailing off is most of what this persona does, so a run of two or more
dots no longer ends a sentence; `?!` still does. The prompts also now state a floor of eight
words and forbid opening a reply with an audio tag.

**Analysis competes with dialogue for the same token budget, and dialogue loses audibly.**
Both share one Groq account at 8000 tokens a minute: roughly 1600 input tokens of dialogue
plus 1200 of analysis per turn. Running both every turn throttled the *dialogue* model, which
the caller hears, while the analysis failures were invisible. Analysis now runs every second
caller turn — always the first, so the dashboard is never blank through the opening — because
stage and risk move slowly enough that nothing a human would notice is lost.

**Entities are written to their own table as well as the event log.** Redundant with the JSONB
payload on purpose: the evidence pack needs "every identifier from this call" and "every call
that mentioned this handle", and neither is a reasonable question to ask of a JSONB column.

**A note on test environments.** SQLite does not enforce foreign keys unless asked, so a test
that provokes a write failure with one passes here for the wrong reason and proves nothing
about Postgres. The recorder's failure test injects the failure instead.
