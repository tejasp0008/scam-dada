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

## 0026 — Phase 4: the control room and the evidence pack
*2026-11-06 · Phase 4*

**One set of numbers, three readers.** The dashboard, the REST API and the evidence pack all
render the same `CallReport`, built by the same code. A dashboard that disagreed with the PDF
it generated would be worse than either on its own.

**The summary is recomputed, not read from a column.** `calls` does carry
`scammer_seconds_wasted`, written at hang-up, but a live call has not been written yet and a
crashed one never will be. Recomputing from turns and events means the figure is the same for
a call in progress, a finished call and a call whose process died — and is always derivable
from the evidence rather than from a field somebody remembered to set.

**The arithmetic is in `app/core/report.py`, not in SQL.** "No hardcoded results" means a
number has to come from a real run; it also has to be reproducible *from that run's data*. So
`seconds_wasted`, the percentiles and the turn counts are pure functions a test calls
directly. `seconds_wasted` spans the whole call rather than summing turn durations, because
the silences between turns are the scammer's time too and they are most of what a stalling
persona produces.

**A field that can only be zero was removed.** `CallSummary` had `barge_ins`, and nothing
populated it: barge-in happens, is logged, and is visible live in the simulator, but it is
not an event and not a column. Always-zero reads as "never happened" rather than "not
measured", which is exactly the kind of number the honesty rule exists to prevent. Emitting a
barge-in event is the right fix and is not done. A test asserts the field is absent.

**Unmeasured latency legs are reported, not omitted.** A leg missing from the table reads as
fast; a leg with `samples: 0` and no numbers reads as never measured, which is the truth —
`stt_final_ms` is often unmeasurable because Prisma's own VAD can beat our hysteresis
(DECISIONS 0024). The pack says so in words underneath the table.

**Percentiles are nearest-rank, not interpolated.** With the handful of turns a demo call
produces, an interpolated p95 invents a number that no turn actually took.

**The evidence pack is HTML first, PDF from the same HTML.** Two reasons rather than one: the
HTML needs nothing installed, so a machine that cannot render a PDF still produces the pack;
and a PDF rendered from the page the operator already looked at cannot disagree with it.

**Autoescaping in that template is load-bearing, not hygiene.** Every string in a report came
out of a stranger's mouth by way of a transcriber, and the pack is a document people forward.
Unescaped transcript text would be a stored injection in the one artefact meant to be passed
around. The API test seeds a turn containing `<script>` and asserts it comes out escaped.

**WeasyPrint fails with `OSError`, not `ImportError`.** Found by installing it: with the
package present but Pango absent the import raises
`OSError: cannot load library 'libgobject-2.0-0'`, which a handler catching only
`ImportError` turns into a 500 instead of an explanation. The route now answers 503 with the
two things an operator needs — the extra to install and the system library behind it — and
points at the HTML pack, which needs neither. `brew install pango` makes the PDF work on
macOS; verified, 17 KB, text confirmed present in the rendered output.

**Verified on real recorded calls**, not fixtures: 49 s wasted across six caller turns and
six persona turns, both personas, stage timeline `authority → fear → payment`, peak pressure
0.85, six identifiers with their provenance, and one identity leak attributed back to the
call that issued it. Latency over six turns: transcript median 519 ms, first token 273 ms
(p95 1394), first audio 472 ms.

## 0027 — The demo page, and the token budget that makes a live call possible
*2026-11-07 · demo hardening*

**Groq's rate limits are per model, not per account.** The two 429s on 2026-11-06 quoted
different limits for different models — 7000 ITPM for `qwen/qwen3.8-27b`, 8000 TPM for
`openai/gpt-oss-120b`. So the single most effective change was not a prompt edit at all:
`LLM_ANALYSIS_MODEL=openai/gpt-oss-120b` gives the analysis pass its own budget and stops it
competing with the dialogue the caller can actually hear.

**Prompt sizes, measured with `tiktoken` (`cl100k_base`), not estimated:**

| | before | after |
|---|---|---|
| Dada system prompt | 1641 | **1016** (−38%) |
| Grandson system prompt | 1316 | **876** (−33%) |
| Analysis prompt | 686 | **605** |
| History ceiling | ~1000 | **~550** (4000 → 2200 chars) |

What went was repetition, not content: all eight ethics constraints, every character detail
and every speech rule survive, and the per-persona tests still assert the six named
constraints, the eight-word floor and the no-opening-tag rule. The digest pin on
`ethics.v1.md` failed the moment the file changed, which is exactly what it is for — the new
digest was recorded deliberately rather than the test being loosened.

Trimming the **history** mattered as much as trimming the prompt: at 4000 characters it cost
roughly as many tokens per turn as the system message. Five exchanges is ample for staying in
character, and the scammer's opening pitch has been extracted by the analysis layer by then.

Per dialogue turn is now ~1566 tokens against qwen's own 7000 ITPM: about four turns a
minute, one every 13 seconds. A natural cycle with Dada is ~16 seconds — his reply is 10 of
them — so a live conversation now fits inside the budget where before it did not.

**Fillers now cover the whole wait, not the first two seconds.** This needed the transport
understood properly. A cached filler is served from disk and `speak` returns as soon as the
bytes reach the socket, so the audio is *queued*, not played — the browser's ring buffer does
the pacing. Queue three at once and the persona stalls three times and then answers; queue
one and wait, and the line goes silent the moment it ends if the model is slower than one
phrase. So the loop paces itself by the audio it just handed over: hand over a filler, wait
almost that long, reach for another only if the reply still is not ready. Capped at three.

`filler_played` stays false when the model answered instantly, because reporting a filler
that never played would be the same class of fabricated number as the barge-in count removed
in Phase 4.

### The page

`/demo` is one screen for the camera: microphone, who-is-speaking and the echo warning on
the left, the dashboard on the right. It reuses the simulator's transport and the dashboard's
event handling rather than adding a third — the same `/ws/call` socket, the same worklets,
the same event payloads. What is new is presentation.

**Where the visual direction came from.** Three families of reference, and what was taken:

* **SOC and threat consoles** — layered dark surfaces built up from a near-black base rather
  than inverted light ones (`#0a0c10` → `#12161d` → `#1a1f29`), and red and amber reserved
  strictly for severity so a red pixel always means something.
* **Call-centre wallboards** — six to eight metrics at most, because a board showing
  everything stops being read, and the number you act on largest and highest. "Readable from
  across a room" is the same constraint as "readable in a phone-sized video".
* **The Bloomberg terminal** — tabular numerals everywhere a number changes, because
  proportional digits make a live counter visibly jitter as it ticks; in-place updates rather
  than re-layouts; and a brief colour wash on a changed value, which draws the eye without
  moving anything.

**Motion is decoration, and every state it conveys is also stated in text.** So it is
switched off wholesale under `prefers-reduced-motion`, and under `?still=1` — which exists
because headless Chrome does not advance CSS transitions, so a screenshot without it captures
half-faded panels and an unlit stage ladder. That was discovered by taking the screenshot.

**`/demo?replay=<call_id>` plays a finished call back through the same panels.** This is the
replay mode deferred from Phase 2, and it earns its place three times over: rehearsing the
visuals without talking, showing the page on a machine with no microphone, and proving the
panels cannot drift from the record because they are driven by it. It falls back to
reconstructing a stream from the report when Redis has expired the event log.

**The echo warning is not decorative.** On speakers, the persona's own room tone reaches the
microphone and trips the barge-in detector, cutting Dada off mid-sentence — the single most
likely way a live demo goes wrong. The page inspects the track label and
`echoCancellation` setting and says which of the three situations it is in.

## 0028 — Four things a live human call exposed
*2026-11-07 · demo hardening*

A real conversation through a microphone broke things that six scripted calls had not.

**1. The transcript contained words the caller never heard.** `PersonaSpeaker.speak` returned
only a latency, so an utterance cut off by barge-in was indistinguishable from a finished
one, and the orchestrator recorded the full sentence either way. On a live call three
utterances were cut and all three went into the record as spoken. That is the evidence pack
telling a story the audio does not support, which is the one thing it must never do. `speak`
now returns a `Spoken` with `completed`, and an interrupted sentence is logged and dropped
rather than recorded. The rest of that reply is abandoned too: the caller has moved on.

**2. The grandson did most of the talking.** He took the phone at turn 9 of a live call and
still had it at turn 21 — and when the scammer said "aap apne dada ji se baat karwaiye", the
system could not comply, because the relay refused to go backwards. DECISIONS 0015 had
reasoned that returning a caller to a voice he has already heard breaks the illusion. The
opposite is true: refusing the most natural thing in a family is what breaks it, and the
grandson's own prompt already promises he puts his grandfather back on.

So `hand_back` exists, and who holds the phone is decided by three triggers in priority
order: **the caller's own words** ("put me on to your grandfather" is the clearest handoff
signal there is), then **the grandson's turn limit** — he is an interlude, two turns — then
the lead's floor, raised from 4 to 6. The elder does the baiting, so the phone belongs with
him by default. A test asserts he ends up with more turns than the grandson.

**3. Dada was slow but not *exhausting*.** The prompt described an old man; it did not
describe a tactic. It now does, in the one section that matters: never say no, agree and then
undo it, make him repeat himself, re-ask what he has already answered, answer a different
question from the one asked, be delighted by him, take his urgency personally and become
slower — and never look deliberate, because the moment he thinks the delay is a tactic he
hangs up. Merged with the old stalling section, which said much the same thing less usefully.

**4. The line was digitally silent whenever nobody spoke.** Real rooms do not stop. The
comfort-noise loop from Phase 2 is now wired in: it reads the *same* bed at the *same* offset
the utterances are mixed over, so switching between speaking and not speaking does not
restart the fan. Measured on a live call: 49% room tone, 48% speech, 2% true silence. There
is a switch, on the page and over the socket as `room-tone:on|off`, because on some setups
the bed is the thing feeding back into the microphone.

### Two things that only appeared once those were fixed

**The dialogue model was returning empty completions.** `qwen/qwen3.8-27b returned no content
(reasoning-only completion)`, repeatedly, once the persona prompt grew denser. A reasoning
model given a thick instruction set spends its budget thinking and emits nothing; every such
turn costs a second model call, and the reply lands after the caller has started talking
again. `reasoning_effort` was configured for the *fallback* only — DECISIONS 0014 found this
exact failure on gpt-oss and fixed it in one place. It is now set for the dialogue model too.

**The scripted driver could not tell room tone from speech,** so after the bed was added it
never saw a quiet line, waited out its full timeout every turn and talked over the reply
anyway. It now thresholds on RMS — the bed sits near 0.012 and speech near 0.07. And its
quiet threshold went from 0.8 s to 2.0 s, because a reply is spoken sentence by sentence and
the gap *inside* one reply is about a second: at 0.8 s the driver decided Dada had finished
after his first sentence. Barge-ins on the final run: **zero**, with every caller turn
answered, against three and three missing replies before.

Also fixed in passing: `PersonaSpeaker` and `CallOrchestrator` took their clock from
`asyncio.get_event_loop().time`, which is deprecated in that form and meant neither could be
constructed outside a running loop. Both use `time.monotonic`, which is what the
measurements wanted anyway.

## 0029 — "The text is generated but no voice comes"
*2026-11-07 · demo hardening*

Reported after a live call, immediately following the room-tone change. Four bugs, each one
hidden behind the last, and all four share a single root cause: **`speaking` was taken to
mean "the caller is hearing audio", and it never did.** It means "synthesising, or writing
bytes to a socket" — milliseconds — while the audio those bytes become plays for ten seconds.

**1. Room tone drowned the speech.** `_play` hands a whole sentence to the transport almost
instantly, so `speaking` went false while ten seconds of audio were still queued in the
browser. The comfort-noise loop saw a free line and began stacking room tone behind it. The
ring buffer holds eight seconds and `enqueue` **dropped silently** once full — so the *next*
reply was discarded before it was ever played. Text in the transcript, nothing in the ear.

`transport_busy` now tracks when the audio will actually have finished, and the loop waits
for that. The worklet also reports its drops instead of swallowing them: a buffer that
quietly discards a whole reply is indistinguishable from a server that never sent one, and
that cost real debugging time.

**2. Barge-in had never worked during playback.** `stop` returned early when the playback
task was done, and the orchestrator only tried to barge in when `speaking` was true — both
of which stop being true the moment the bytes are written. So interrupting Dada worked only
during synthesis, the one part of a reply nobody hears. It now triggers on
`speaking or transport_busy`, and `stop` flushes whenever the transport is still playing.

**3. Every sentence after the first cut off the one before it.** `speak` began with
`await self.stop()`, which flushed the transport and counted an interruption. Saying the
next sentence is not an interruption; `_supersede` does that part without flushing, so
consecutive sentences play back to back as intended. The "barge-in count" was, in part, a
sentence count.

**4. Ending a call counted as a barge-in.** Teardown called `stop`, so every call that ever
finished reported one phantom interruption. `abandon` exists for that, and the counter now
means exactly one thing: the caller cut in. It reaches the dashboard and the evidence pack,
so it has to.

**The scripted driver had the same bug as the server**, for the same reason: it waited on
when audio last *arrived*. It now tracks when the persona's audio will have finished playing
and waits for that as well as for two seconds of quiet.

Measured on the verification call afterwards: **zero barge-ins, all six caller turns
answered with full multi-sentence replies**, 27.9 s of speech and 47.5 s of room tone across
78 seconds, 2.5 s of true silence, no talk-over, nothing clipped. Before the fix the same
script produced five barge-ins and truncated replies.

## 0030 — Barge-in was firing on the persona's own voice
*2026-11-08 · demo hardening*

Reported as "still cannot listen to Dada's voice". The server log settled it in one look:
twelve `persona spoke` events with real durations — 2680 ms, 3080 ms, 5192 ms — and **four
barge-ins** in forty seconds. The audio was being generated, sent, and then flushed out of
the browser's buffer before it could play.

**The cause is the fix from 0029.** Barge-in now works during playback, as it always should
have. On speakers, the microphone hears the persona, the energy VAD opens on it, and the
system concludes the caller is interrupting — so it flushes everything queued, which is
heard as the persona having no voice at all. Making barge-in correct is what exposed the
echo, and the two are in genuine tension: perfect interruption and perfect echo immunity
cannot both be had from one microphone with no reference signal.

**A higher bar while the persona is audible.** The VAD's 3 frames above 0.015 RMS is right
for "is the caller talking" and far too eager for "is the caller talking *over us*". Barge-in
now needs 8 consecutive frames — 256 ms of sustained speech — above 0.06 RMS, four times the
VAD threshold. No echo of a single word survives that; a real interruption clears it easily.
Both numbers are config, and barge-in has an off switch on the page and over the socket.

Headphones remain the actual answer, which is what the echo warning on the page says. This
makes speakers survivable rather than good.

**Also fixed, found by a test that started failing:** a turn where nothing reached the caller
was still emitting a latency event. Reporting "reply first token 300 ms" for speech that was
never heard puts a timing on the dashboard for an utterance that does not exist — the same
mistake as recording the text, which 0029 fixed. Latency is now emitted only when the
persona actually spoke.

Verification after the change: zero barge-ins on the scripted call, every turn answered.

## 0031 — The persona mirrored the caller's language
*2026-11-08 · demo hardening*

Reported as "the agent is answering in English, not Hindi". A two-case probe against the
real model settled it: the same prompt, the same persona, the caller's language the only
difference.

```
CALLER: Hello sir, this is Inspector Sharma from Delhi cyber crime branch.
DADA:   Hello? Hello? Wait, who is speaking? ... My grandson is sleeping, please speak clearly.

CALLER: Hello sir, main Delhi cyber crime branch se inspector Sharma bol raha hoon.
DADA:   Kya? Sharma? Main samajh nahi paaya, aap Delhi se ho?
```

Nothing was misconfigured. **Language mirroring is driven by the most recent message in the
request**, and the Hinglish rule sat at line 49 of the persona prompt under `## Speech` — the
furthest message in the request from the generation. On the scripted call the scammer speaks
Hinglish, so the mirroring agreed with the instruction and the bug was invisible for a week;
it only appeared on the live mic, where the caller is whoever picked up.

**Three changes, each verified against the live model:**

1. **The rule moves to the top of each persona prompt**, as its own `## Language` section
   stating that the caller's language never changes the persona's.
2. **A 38-token system note in last position on every dialogue turn.** This is the same
   remedy the handoff note needed in 0019, for the same reason, so the two now share one
   message: anything that must hold for *this* reply — whose words the history holds, an
   armed bait hook, the language to answer in — is joined into a single trailing system note,
   with the language line last of all because it is the instruction the model drops first.
3. **The grandson needed a reason, not an instruction.** With the rule at the top and the
   note appended he still answered in fluent English four turns out of four: a 27-year-old
   who works at a bank plausibly speaks English, and the model preferred that inference to
   the rule. Dada complies because his block gives a character reason — his English was never
   good. The grandson's now gives one too: he is standing in his own house with his
   grandfather beside him, who has to follow what is being said. That, plus four example
   lines in his register, holds.

**Cost.** Dada's system prompt goes 1287 → 1365 tokens and the grandson's 877 → 997, plus 38
per turn for the note. Against Groq's 7000 ITPM that is roughly 800 extra input tokens over a
ten-turn call — real, and the reason it is written down rather than absorbed quietly.

## 0032 — "Latency has increased a lot" was reply length, not latency
*2026-11-08 · demo hardening*

Reported straight after 0031. The first thing worth recording is that **the running server
had never loaded 0031**: started 00:05:20, no `--reload`, prompts behind an `lru_cache`. So
whatever was being felt, that commit had not caused it. Measured anyway, three runs each
against the live model:

| | input tokens | first token |
|---|---|---|
| old prompt, no note | 1406 | 260 ms |
| new prompt + note | 1522 | 274 ms |
| fallback model | 1522 | 503 ms |

**14 ms.** Pipeline latency across the recorded calls is unchanged too — STT ~520 ms, LLM
first token ~270 ms, TTS first byte ~700 ms, before and after.

**What actually changed was how long the persona talks.** Reply length went from a 94-character
median to 134, and the scripted call from 78 s to 86 s with 76.9 s of persona speech across
20 utterances — about 12.8 s per turn. First token at 270 ms is irrelevant if the caller then
waits twelve seconds for a turn, and on a live microphone that *is* the latency he feels. The
bigger `## Language` section had pushed the brevity rule in `## Speech` far enough from the
generation to be dropped — the same failure as 0031, caused by the fix for 0031.

**Two corrections, and the first one was wrong.** Cutting `max_reply_sentences` to 2 and
bounding characters at 130 produced 40-character stubs — the persona prompt is right that a
two-word reply reads as a broken machine rather than a slow man. Worse, the next run with the
cap restored was *shorter still*, a 33-character median:

> **Mechanical truncation feeds back through the transcript.** Only what was spoken is
> recorded (0029), which is correct for the evidence pack — but the dialogue history is the
> same list, so a truncated reply becomes an example the model then imitates. One clipped
> turn collapses the rest of the call. A ceiling alone does the same thing on its own, because
> the model trends to the floor of whatever range it is given.

So length is asked for as a **range** — "fifteen to twenty-five words, fewer sounds like a
dropped line, more and he never gets a turn" — in the turn note where form rules survive, and
`max_reply_sentences` stays at 3 with the character bound raised to 220 as a safety valve for
a runaway reply rather than a routine cut.

**Also fixed:** one turn's entire reply was `<pauses>`. A tag-only utterance is billed in
full, arrives as a breath with no words around it, and reads on the transcript as a reply the
caller never heard. `has_speech()` now drops those before synthesis, in the loop rather than
in `_speak`, because `_speak` returning false means "cut off" and would abandon the rest of
the turn.

**Measured, same six-line script each time:**

| run | median reply | min | max | call |
|---|---|---|---|---|
| baseline, before 0031 | 94 ch | 39 | 110 | 78 s |
| after 0031 | 134 ch | 98 | 164 | 86 s |
| 2-sentence cap | 40 ch | 8 | 55 | 59 s |
| 3 sentences, ceiling only | 33 ch | 20 | 92 | 66 s |
| **3 sentences, range note** | **97 ch** | **49** | **123** | **81 s** |

Back to the baseline length with tighter variance, zero barge-ins, the UPI handle and the
`payment` stage still captured. The turn note costs 64 tokens.

**One number worth knowing before the recording:** the dialogue model's limit is 7000 input
tokens per minute and a turn costs ~1520, so the primary sustains about **four and a half
dialogue turns per minute**. Past that it returns 429 and every turn goes to the fallback,
which is 503 ms to first token instead of 274 ms. That ceiling is unrelated to these
commits — it is the reason the fallback exists.

## 0033 — Acceptance at the payment stage, to collect a payee
*2026-11-08 · intelligence*

Every hook in `BAIT_HOOKS` is a **delay**, armed by hang-up risk, and three of the seven are
for the payment stage — a fixed deposit he cannot break, a pension that arrives on the first,
an account at another bank. They waste his time well and collect nothing. The call ends with
his claimed rank, his claimed department and a case number he made up: no identifier that
leads anywhere.

**The destination account does lead somewhere.** Unlike a rank or a case number it is a real
registered thing, and a UPI handle, an account with an IFSC, or the name a mule account is
held in is what a bank or the cybercrime portal can act on. So it is worth a tactic of its
own.

**A second hook list with a different trigger.** `PAYMENT_ELICITATION` is armed on *reaching*
the payment stage, whether or not he is impatient, and it is checked before the delay hooks.
The difference is recoverability: a forgotten bank account is just as good four turns later,
while a destination account is only ever on the table while he still expects money to arrive
in it. A high risk score at the payment stage is a reason to elicit sooner rather than to
stall — he stays for the same reason he called.

**Acceptance, never completion.** The persona agrees in principle and then needs one more
thing before it can happen: a slip to carry to the counter, a name the clerk will ask for, a
fallback account because a transfer once went astray, a reference to write so nothing is
lost. This keeps the property the delay hooks have, that nothing is promised outright — a
promise he can hold the persona to is the point at which he stops talking and starts waiting.
Here the asking *is* the tactic, because every question is a precondition he has to satisfy
before his money can arrive. Each of the four asks for a different field, so firing them in
sequence assembles a payee rather than asking the same thing four ways.

The lead-in differs by trigger, which is the part that is easy to miss: telling the model
"the caller is close to ending the call" in front of an elicitation hook makes the persona
hurry, and hurrying is what loses the details.

**Nothing is paid and no identifier of ours is improvised.** The canary identity exists for
the one case where the persona must hand something over, and it is issued deliberately rather
than invented mid-sentence.

**Extraction had to keep up**, or eliciting a number nobody captures is theatre. Two new
types, `account_number` and `payee_name`, and the regex has two properties worth recording:

* **A keyword is the whole pattern.** A bare digit run is the loosest thing in a transcript —
  it is also phone numbers, case numbers, amounts, dates and badge numbers — so a number is
  an account number only if he said it was one. A ten-digit run starting 6-9 is dropped,
  because "account se linked number" is said on every call and `phone` already reports it. A
  genuine ten-digit account in that range is lost; every major Indian bank issues eleven or
  more, so that is the cheaper error.
* **It is read aloud, not written.** Prisma transcribes a dictated number as *words* — "zero
  zero one one two two" — which no digit pattern matches. `app/core/spoken.py` now holds the
  digit-word table that `canary.despoken` owned, because an account number is not a canary
  concern, and `digit_run` stops at the first token that is not part of a number so the
  keyword cannot swallow the clause after it. `digit_run_after` steps over the few words
  between the keyword and the digits, with a small limit for the same reason.

**Measured on the scripted call**, which now continues past the UPI line so there is something
to elicit. Entities went from 6 to 9 and the full payee was assembled:

| | |
|---|---|
| `account_number` | `000011223344` (regex, from dictated speech) |
| `ifsc` | `FAKE0000777` (llm) |
| `payee_name` | `Suresh Kumar` (llm) |
| `upi_id` | `fake.officer@demo` (llm) |

Three hooks armed, all on `trigger=payment_stage`, and the persona asked for it in character:
"Arre, yeh toh adhoora hai. Poora number batao, aur IFSC code bhi." Then, on the holder's
name: "Suresh Kumar? Dada ne likha tha Ramesh ji ka naam" — the hook landing on top of the
grandfather's-notes-do-not-match tactic, which is the behaviour the prompt already had.

The scripted values stay unusable by construction: the account begins `0000`, which no Indian
bank issues, and the IFSC bank code is `FAKE`, exactly as the canary builds them.
