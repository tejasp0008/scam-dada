# Scam Dadi — Implementation Plan (v1, awaiting approval)

Written 2026-11-04. Source of truth for scope until you amend it.

---

## 0. The headline: the timeline in the brief does not exist

The brief says "demo-ready by Nov 8, 2026; final submission Nov 10, 2026" and lays out
seven phases. **Today is Nov 4, 2026.** Gnani's own page confirms submissions close
Nov 10, 11:59 PM IST. That is **4 days to demo-ready, 6 days to submit** — not the 3–4
weeks the phase list implies.

So this plan is not the brief's plan compressed. It is a re-prioritised plan built around
what the contest actually scores. Per gnani.ai/internship, the award categories include
*Mixed Vibes* (best code-mixed voice experience), *Jugaad Genius* (most creative use of
the models), *Demo Day Drop* (best 60-second demo video) and *Crowd Favourite* (public
votes), and **posts must be live before you submit**.

That reorders things sharply:

- A **live Hinglish persona loop with barge-in** is the entry. It wins *Mixed Vibes* and
  *Jugaad Genius* on its own. Everything else is support.
- The **control room dashboard** is what the 60-second video films. It is worth more than
  the evidence-pack PDF, which nobody will watch.
- **ShorBench, the bandit and the highlight reel earn nothing** by Nov 10 and cost days.

### Proposed cuts (your call — this is the main thing I need a decision on)

| Brief item | Proposal | Why |
|---|---|---|
| Phase 6 ShorBench + bandit | **Cut the bandit. Keep a text-only scammer sim** as a dev tool | A text-only Dadi-vs-scammer loop costs ~2h, burns zero audio credits, and lets us tune persona prompts without talking to the mic 50 times. The bandit needs hundreds of runs to mean anything and would be a fabricated result by Nov 10. |
| Highlight reel (ffmpeg waveform video + burned subtitles) | **Cut** | 1–1.5 days of ffmpeg fiddling for an artifact the judges never open. If Nov 9 is calm, a plain audio-only clip cut at LLM-chosen timestamps is a 2h add-on. |
| Evidence pack PDF | **Keep, last** | Cheap with WeasyPrint, and it is the "intelligence" payoff that makes the project look like a product. Phase 5, Nov 8–9. |
| Family relay (Dadi → Dada → grandson) | **Keep** | Genuinely the best 10 seconds of the demo video and nearly free once the persona loop exists — it is a prompt + voice swap on shared history. |
| Canary identities + cross-call match | **Keep** | Cheap, pure-Python, unit-testable, and the single most interview-defensible idea in the brief. |
| Guardian Mode | **Keep as specified** (state + handoff trigger only) | Already scoped to a trigger. |
| Human vs robocall check | **Keep** | ~1h: a barge-in probe plus a timing assertion. |
| Postgres + Alembic + Redis | **Keep** | You asked for them and they are defensible in interviews. Mitigation: **one** baseline migration covering the whole schema, no incremental migrations this week. |

### Revised calendar

| Day | Work | Gate |
|---|---|---|
| **Nov 4 (today, remaining)** | Phase 0 scaffold + Phase 1 provider adapters + smoke script | A real round trip: mic WAV → Prisma transcript → LLM reply → Timbre audio, with units logged |
| **Nov 5** | Phase 2 persona loop: orchestrator, state machine, energy VAD, barge-in, filler cache, Dadi prompt, browser simulator page | You hold a multi-turn Hinglish conversation with Dadi in the browser and can interrupt her |
| **Nov 6** | Phase 3 intelligence: regex extractors, LLM analysis JSON, stages, pressure, hang-up risk + bait hooks, canaries, family relay, robocall check | All events fire and persist in a simulated call |
| **Nov 7** | Phase 4 control room + demo hardening | Dashboard updates live, no refresh |
| **Nov 8** | **Freeze. Record the 60s video.** Phase 5 evidence pack if green | Demo-ready |
| **Nov 9** | README, architecture diagram, LinkedIn/X post, buffer for what broke | Posts live |
| **Nov 10** | Submit with hours to spare | Submitted |

Phases 0 and 1 run together today because they are the only parts with unknown external
behaviour. Everything after them is code I can write without surprises; the API
integration is where a day can vanish. If the adapters are not working tonight, that is
the signal to tell you we are cutting deeper.

---

## 1. What the Gnani docs actually say

I read docs.gnani.ai rather than assuming. Facts the design now depends on, with the
page they came from:

**Auth and bases** (`api/introduction/quick-start`)
- Header `X-API-Key-ID: <key>` on REST and on the WS upgrade. No token exchange.
- STT REST `https://api.vachana.ai/stt/v3` · STT realtime `wss://api.vachana.ai/stt/v3/stream`
- TTS REST `https://api.vachana.ai/api/v1/tts/inference` · SSE `.../tts/sse` · realtime `wss://api.vachana.ai/api/v1/tts`

**Prisma streaming STT** (`api/STT/stt-websocket`, `api/STT/limits`)
- Config travels as **WS upgrade headers**, not query params.
- Audio must be **raw PCM s16le mono in frames of exactly 1024 bytes** (512 samples), sent
  at real-time cadence. Sample rates 8000/16000/44100/48000.
- Models `prisma-v3.0` / `prisma-v2.5` (default). v3.0 gives `indic-auto`, a 1–4 language
  candidate list, `keyterms` (≤200), and `format=codemix|transliterate`.
- Server-side VAD is configurable per connection: `x-vad-threshold` (0.7), `x-min-silence-ms`
  (500), `x-min-speech-ms`, `x-max-speech-s`.
- Server events: `connected`, `speech_start`, `processing`, `transcript`, `settings_updated`, `error`.
- **`transcript.is_final` is always `true` — there are no partial hypotheses.**
- **Hard 15-minute session cap** (close 1001). Idle timeout default 60s, configurable 5–900s.

**Timbre TTS** (`api/TTS/tts-websocket`, `api/TTS/tts-audio-tags`, `api/TTS/available-voices`)
- Models `timbre-v2.5` / `timbre-v3.0`. **Audio tags only work on v3.0**; v2.5 strips them silently.
- The 13 tags: `<laugh> <giggle> <sigh> <exhales> <sniffles> <yawns> <coughs> <clears throat>
  <whispers> <shouting> <excited> <slow> <pauses>`. Case-sensitive, no closing tag, each
  applies until the next tag or end of input. Unknown tags are stripped silently.
  **Tag characters count as billable characters.**
- v3.0 voices (all 10 languages): F — Shivani, Lakshmi, Kalpana, Chetana, Vasudha, Riyaa,
  Tarika · M — Avinash, Shubham, Prakash, Devansh, Yashraj, Harshad.
- `language` accepts **`hi-en`** (Hinglish) and `auto`.
- `speed` 0.85–1.15 only.
- `audio_config`: `sample_rate`, `num_channels`, `sample_width`, `encoding`
  (`linear_pcm|pcm_s16le|pcm_mulaw|pcm_alaw|oggopus`), `container` (`wav|raw|mp3|ogg|mulaw|alaw`).
- Realtime WS and SSE both return **base64 audio inside JSON**, not binary frames.
- **No documented cancel or interrupt message** on either transport.

**Errors** (`api/introduction/errors`)
- Retry only connection errors, 429, 500, 503, 504 — ~3 attempts from 0.5s, doubling.
  Other 4xx "fail the same way". 402 means out of credit.
- Shapes differ per surface (`{success, error:{type,message}}` for STT REST,
  `{detail:{error_code,...}}` for auth/rate, `{success, message, status_code}` for TTS).
  The adapters normalise all of these into our own exception types.

### Three doc findings that change the architecture

**1. Evon is not callable.** Gnani's FAQ says Evon v3.3 "isn't on the API platform yet —
open weights, request them on Hugging Face" (Apache 2.0, 30B MoE). The Agent Builder's
`GET /platform/v1/agents/config/llm_models` lists Claude, GPT-4.1-mini, GPT-4o-mini and
Gnani's "Pampa" models — **no Evon** — and it is only a selector for Agent Builder agents,
not a chat-completions endpoint. A 30B MoE is also not something we serve on a laptop this
week. So: the `LLMProvider` protocol with an OpenAI-compatible adapter is the whole answer,
exactly as the brief anticipated, and the README will state plainly that Evon was not
reachable and where the swap point is. **I need to know which OpenAI-compatible endpoint
you have access to.**

**2. The 15-minute STT cap fights the product.** The entire point is keeping a scammer on
the line for 20+ minutes, and Prisma closes the socket at 15. Design answer: `SttSession`
is a rotating wrapper that opens a successor socket at ~13 minutes, waits for a silence
window from the VAD, swaps, and drains the old one — invisible to the orchestrator, which
only sees one transcript stream. Also set `x-idle-timeout-sec` high (900) so a thinking
scammer does not drop the call. This is a real constraint I would otherwise have hit on
Nov 7 during the demo rehearsal.

**3. No partial transcripts and no TTS cancel reshape turn-taking.** Because
`is_final` is always true, we cannot do the usual "endpoint on stabilising partials". Turn
boundaries come from Prisma's own VAD events (`speech_start` → `processing` → `transcript`).
Barge-in therefore cannot wait for the server: it runs off **our local energy VAD** on the
inbound frames, which (a) kills browser playback immediately, (b) abandons the TTS stream,
since no cancel message exists. `speech_start` is used as the slower confirmation. This is
why the brief's energy VAD stays even though Prisma ships a VAD — they do different jobs,
and the plan says so in `DECISIONS.md`.

### What I did not find, and will not invent

- No published credit/pricing units anywhere in the docs — the limits page says rates live
  in your account. The budget guard therefore counts **raw units** (TTS billable characters,
  STT audio seconds, LLM tokens) and holds a configurable ceiling per unit; it does not
  pretend to know rupees or credits until you tell me the conversion.
- No numeric rate limit or concurrency cap ("a per-organisation cap applies").
- No voice age metadata, so which v3.0 voice sounds like a 70-year-old is an empirical
  question → `scripts/audition_voices.py`, one short line per voice, ~13 calls, then we listen.

---

## 2. Repository layout

The brief's layout, with four changes, each justified.

```
scam-dadi/
  app/
    api/
      routes/            health.py  calls.py  dashboard.py  artifacts.py
      ws/                call_ws.py          # browser mic in, persona audio out
                         dashboard_ws.py     # Redis fan-out to the control room
      deps.py                                # Depends() wiring, the only composition root
    core/                                    # pure Python, zero I/O, mypy strict
      models.py          Call, Turn, Speaker, Persona, Entity, Identity
      events.py          the 8 event types + the DomainEvent union
      state.py           CallState enum + transition table + illegal-transition error
      audio/             framing.py  resample.py  vad.py      # <- change 1
      personas.py        persona registry: prompt file, voice, language, speed
      tactics.py         stalling tactic catalogue + selection
      risk.py            hang-up risk scoring from transcript signals
      extract.py         regex layer: UPI, phone, IFSC, amounts (lakh/crore), names
      canary.py          identity generation + fuzzy cross-call matching
      stages.py          hook/authority/fear/isolation/payment + pressure score
    services/
      orchestrator.py    one session object per call; owns the loop
      stt_session.py     15-minute rotation wrapper
      turn_taking.py     barge-in, filler scheduling, sentence-chunked TTS handoff
      analysis.py        background analysis every N turns, never blocks a reply
      evidence.py        HTML -> PDF pack
      scammer_sim.py     text-only sim (the surviving slice of ShorBench)
    providers/
      base.py            STTProvider / TTSProvider / LLMProvider Protocols + errors
      gnani_prisma.py  gnani_timbre.py  openai_compatible.py  mock.py  factory.py
    infra/
      settings.py  db.py  models_sql.py  redis_bus.py  tts_cache.py
      budget.py    logging.py
    web/
      templates/   static/js/ (mic-worklet.js, play-worklet.js, dashboard.js)  static/css/
    prompts/       dadi.v1.md  dada.v1.md  grandson.v1.md  analysis.v1.md  fillers.v1.yaml
  scripts/         smoke.py  audition_voices.py  pregen_fillers.py  seed.py
  tests/           unit/  integration/  contract/  fixtures/     # <- change 3
  migrations/      one baseline
  docs/            architecture.md (Mermaid)  DECISIONS.md       # <- change 4
  docker-compose.yml  Dockerfile  pyproject.toml  .env.example  Makefile  README.md
```

**Change 1 — `core/audio/` (framing, resampling, energy VAD) lives in `core`, not a new
top-level package.** Frame slicing to exactly 1024 bytes, 48k→16k downsampling and energy
VAD are pure functions over numpy arrays with no I/O, which is precisely the `core`
contract. Keeping them there means the hairiest logic in the project — the part most likely
to be subtly wrong about frame boundaries — is unit-testable with no mocks at all.

**Change 2 — `services/stt_session.py` and `services/turn_taking.py` are split out of the
orchestrator.** The brief hands the orchestrator eight responsibilities. Session rotation
and barge-in/filler timing are the two with real state machines inside them; left inline
they make `orchestrator.py` a 600-line file nobody can review, and they are the two things
I most want to test in isolation.

**Change 3 — `tests/contract/` as its own directory**, since the brief asks for contract
tests that must never run live in CI. A separate directory gets a separate marker and a
separate `make` target, so "never live in CI" is enforced by configuration rather than by
remembering.

**Change 4 — `docs/DECISIONS.md`.** An ADR-lite log: one short entry per hard-to-reverse
call (frame contract, transport choice, event schema, why the energy VAD survives
alongside Prisma's). You asked for decisions to be flagged before they are committed to,
and in an interview this file is the artifact that proves you drove the architecture.

Everything else is the brief's layout unchanged. `shorbench/` is gone as a package; its
surviving slice is `services/scammer_sim.py` plus a script, which is honest about it being
a dev tool rather than a benchmark.

---

## 3. Module responsibilities

### `core` (pure, mypy strict, no imports from FastAPI/SQLAlchemy/Redis/providers)

- **`state.py`** — `RINGING, GUARDIAN, HANDOFF_PENDING, HUMAN_CHECK, PERSONA_ACTIVE,
  ROBOCALL_DROPPED, ENDED` with an explicit frozen transition table. `transition()` returns
  the new state plus a `StateChangeEvent`, or raises `IllegalTransition`. No I/O, so the
  whole table is a parametrised test.
- **`events.py`** — the eight Pydantic events from the brief, each with `call_id`, `seq`,
  `ts`, plus a discriminated union for the WS payload and the replay log. **This is the
  schema I most want you to look at before I build on it**, because it is simultaneously
  the WS contract, the DB payload and the evidence-pack input.
- **`audio/vad.py`** — energy VAD behind a `Vad` Protocol with `EnergyVad` as the first
  implementation; swapping in Silero later is a constructor change.
- **`extract.py`** — regex-only, deterministic, returns `Entity(type, value, span, source="regex")`.
  Handles `name@bank` UPI handles, Indian mobile patterns, IFSC, and amounts written as
  "do lakh", "2.5 lakh", "₹50,000", "paanch hazaar".
- **`canary.py`** — generates one obviously-fake identity per call (name, a UPI handle on a
  `@demo` suffix, an address in a fictional locality) and does token-level fuzzy matching of
  a new transcript against previously issued identities. Never emits anything that could
  collide with a real identifier.
- **`risk.py` / `tactics.py` / `stages.py`** — scoring and selection as pure functions of
  transcript history, so every number in the dashboard is reproducible in a test.

### `providers`

`base.py` defines the three Protocols and the error hierarchy
(`ProviderError` → `AuthError`, `RateLimited`, `OutOfCredit`, `UpstreamTimeout`,
`SynthesisFailed`, `TranscriptionFailed`). Each adapter normalises Gnani's four different
error shapes into these, and only retryable ones get tenacity. `factory.py` reads settings
and returns the triple; it is the **only** place a vendor name appears outside its own
module, which is what makes the Sarvam/BharatGen adapters you want later a config change.

`mock.py` is deterministic and offline: scripted transcripts on a timer, TTS that returns
silence of a length proportional to the text, and an LLM that replays canned Hinglish turns.
The entire integration suite and the whole dashboard can be developed against it at zero cost.

### `services/orchestrator.py`

One object per call. Owns: the rotating STT session, local VAD, turn-taking, persona
selection, filler playback, the dialogue LLM call, the TTS stream, and latency
instrumentation (`stt_final_ms`, `llm_first_token_ms`, `tts_first_byte_ms` per turn).
Dialogue and analysis are separate calls on separate tasks; analysis failure is caught and
logged and never reaches the caller's ear. All CPU-bound work goes through
`asyncio.to_thread`.

Two latency tricks, because a scammer hangs up on dead air:
1. A cached filler fires the moment `transcript` lands, while the LLM is still thinking.
2. The LLM reply is streamed and **cut at the first sentence boundary**, which goes to TTS
   before the rest of the reply exists. This is the single biggest win available and it is
   why the orchestrator streams rather than awaits.

### `infra`

`tts_cache.py` keys on `sha256(text|model|voice|language|speed|audio_config)` → raw PCM on
disk, and logs every hit as a zero-unit row so "credits saved by caching" is a **real
measured number** for the README rather than a claim. `budget.py` records
`(provider, unit_type, units, call_id)` to `api_usage` and raises `BudgetExceeded` before
the call goes out once a configurable ceiling is reached. `logging.py` emits JSON with
`call_id` bound via contextvar.

### Data model

The brief's seven tables, with three additions: `calls` gains
`final_state`/`persona_sequence`/`ended_reason`; `api_usage` gains a nullable `call_id` and
a `cache_hit` flag; `turns` gains the three latency columns so the latency panel survives a
restart. One Alembic baseline.

---

## 4. Testing

- **Unit (core):** the full transition table including every illegal edge; extractor cases
  per entity type including Hinglish amounts; canary generation invariants (every field
  visibly fake) and fuzzy-match true/false positives; risk monotonicity; tactic selection.
- **Integration:** a scripted call end-to-end through the orchestrator on mock providers,
  asserting the **exact event sequence**, plus a barge-in case and a robocall-dropped case.
- **Contract:** adapters against sample payloads captured from the doc pages into
  `tests/fixtures/` — the frame-size rule, the base64-in-JSON decode, each error shape.
  Marked `contract`, excluded from the default run.
- **`make smoke`:** one real call to each of the three providers, prints TTFB/latency and
  the units consumed, and asserts the budget guard trips when the ceiling is lowered.

`ruff`, `mypy` (strict on `core` and `providers`) and `pytest` gate every phase.

---

## 5. Risks

| Risk | Mitigation |
|---|---|
| **Evon unavailable** | OpenAI-compatible adapter; README states it plainly. Needs your endpoint today. |
| **Credits run dry mid-week** | Budget guard + TTS cache + pre-generated fillers + mock providers for all dev. Audition voices once, cache the output. |
| **TTS TTFB too slow for natural turns** | Measure SSE vs realtime WS in `make smoke` tonight; the Protocol hides the transport so switching is a config change. Fillers + sentence chunking absorb the rest. |
| **Browser audio is the classic time sink** | Lock the frame contract on day 1 (16kHz s16le mono, 512-sample frames out of the mic worklet; raw PCM back in for instant cut-off). Resampling is tested in `core` where it is pure. |
| **15-minute STT cap** | Rotating session, built in Phase 2 not discovered in Phase 4. |
| **Local tooling** | `ffmpeg` is not installed on this machine and WeasyPrint needs pango/cairo. Both are Phase 5 only; `brew install ffmpeg pango` when we get there, or run that job in the container. Default `python3` here is 3.9 — we use `uv` pinned to 3.11. |
| **Demo depends on live acting** | See open question 5: a replayable synthetic-scammer WAV makes the 60s video reproducible. |

## 6. Where I will hold you to the brief

- **Synthetic data only.** Every generated identifier will be visibly fake, and the canary
  generator has a test asserting it. Phone numbers only from reserved/fictional ranges,
  UPI handles only on `@demo`. No recorded real calls in fixtures — the scammer-side audio
  will be Timbre-synthesised from scripts we write.
- **No hardcoded results.** Any number in the README or dashboard comes from a logged run,
  stated with its sample size. With six days that means "n=8 calls", not "87% accuracy".
- **Ethics in the prompts.** Dadi never threatens, insults, claims to be police, calls back,
  or tries to identify anyone. These go in `dadi.v1.md` as hard constraints, and the
  persona prompts get a test that the constraint block is present and unmodified.
- Nothing in the brief conflicts with any of this. The one tension is the timeline, which
  §0 addresses.

---

## 7. Open questions

**Blocking Phase 1 (today):**

1. **Gnani API key** — put it in `.env` as `GNANI_API_KEY`. I will never print or log it.
2. **Which OpenAI-compatible LLM endpoint do you have?** (OpenAI, Groq, OpenRouter,
   Together, a local Ollama…) Dialogue needs fast first tokens; this choice visibly changes
   how alive Dadi feels. I need base URL + model name + where the key lives.
3. **Credit reality check** — open app.gnani.ai and tell me the balance and, if shown, what
   the 5,000 programme credits buy per STT second / TTS character. Until then the budget
   guard caps raw units with a conservative default.

**Needed by Nov 5, not blocking tonight:**

4. **Languages.** Hinglish (`hi-en` TTS, Prisma `hi-IN,en-IN`) as the primary, which targets
   *Mixed Vibes*? Or one regional language as well, which opens *Main Character*? My
   recommendation: Hinglish primary, with Prisma on `indic-auto` so a scammer switching to
   Kannada mid-call still transcribes — that demonstrates range without a second persona.
5. **Demo input mode.** Live mic only, or also a replay mode that streams a synthetic
   scammer WAV into the pipeline? **I strongly recommend the replay mode.** Your 60-second
   video is a scored category; filming it live means re-acting a scam call until the take is
   clean, and a mic glitch on Nov 8 is fatal. A scripted synthetic scammer gives a
   reproducible demo and doubles as an integration fixture. ~2 hours.
6. **Telephony.** Confirming the brief's implication: browser only, no real phone leg, no
   Twilio/Exotel. Everything in §1–5 assumes that.
7. **Docker Desktop running?** Compose brings up Postgres and Redis; the dev loop runs
   uvicorn on the host for fast reloads.
8. **Sample rate.** 16 kHz default, with 8 kHz configurable to mimic phone audio? The 8 kHz
   path is also the honest way to claim anything about the *Noise Canceller* category.

**Flagged as hard to reverse — say no now or I build on them:**

- The event schema in `core/events.py` (WS contract + DB payload + evidence input at once).
- The provider Protocol signatures, especially TTS yielding raw PCM chunks rather than
  encoded audio, and STT yielding only finals because Prisma has no partials.
- The 1024-byte/512-sample frame contract and 16 kHz as the pipeline's internal rate.
- Rotating STT sessions as a wrapper rather than orchestrator-visible reconnects.

---

## 8. What happens on approval

I start Phase 0 + Phase 1 immediately and stop when the smoke script round-trips real
audio with logged costs, or when something in the Gnani API contradicts the docs — in which
case I come back with the discrepancy rather than guessing a field.

---

## 9. Decisions recorded 2026-11-04

Approved by Tejas at plan review:

1. **Scope:** all §0 cuts approved. Bandit and ShorBench-as-a-package are out; a text-only
   scammer sim survives as a prompt-tuning dev tool. The ffmpeg highlight reel is out.
   Family relay, canaries, dashboard and the evidence-pack PDF (last) are in.
2. **LLM:** Groq/Cerebras-class fast-inference endpoint via the OpenAI-compatible adapter.
   Awaiting base URL + model name + key. Evon stays documented as unreachable.
3. **Demo input:** both live mic and a replay mode streaming a Timbre-synthesised scammer
   WAV through the pipeline, so the 60-second video is reproducible.
4. **Languages:** Hinglish primary (Timbre `language=hi-en`), Prisma listening on
   `indic-auto` so a mid-call switch to another Indian language still transcribes.
