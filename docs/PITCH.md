# Scam Dada — Solution Pitch

**One line.** When a scam call is flagged, Scam Dada answers it as a 74-year-old retired
Railways clerk, keeps the scammer talking in Hinglish, and turns the call he is wasting into
a structured evidence pack — on Gnani's Prisma and Timbre speech models.

**The problem.** India's dominant phone-fraud script is impersonation of authority: a "cyber
crime branch" officer, a parcel held at customs, a case number, a verification payment. Every
defence a household has is reactive. Hanging up forfeits the two things worth taking — the
scammer's time, his only scarce resource, and his payment rails, which he will volunteer to
anyone he believes is about to pay.

## The solution

| | |
|---|---|
| **Waste the scarce resource** | A slow, cooperative, endlessly confused elderly persona. Seven retention tactics, chosen by a hang-up-risk score built from named signals — two of which no keyword list can see: the caller's replies shortening, and recency. |
| **Collect while stalling** | A deterministic regex pass and a model pass read every turn for UPI handles, destination accounts, IFSC, case numbers, claimed ranks and Hinglish amounts (`do lakh`). Dictated digits included — Prisma transcribes `zero zero one one` as words. |
| **Elicit where it works** | Reaching the `payment` stage arms an elicitation hook: the persona accepts, then needs one thing first — a name for the bank clerk, a fallback account in case the transfer fails. Nothing is ever paid. |
| **Prove it afterwards** | A per-call HTML/PDF evidence pack: minutes wasted, every identifier with the machine that found it, stage timeline, identity leaks, latency, full transcript. |
| **Link repeat callers** | Each call issues one provably-unusable fictional identity. When a later caller reads it back — even digit by digit — the two calls are linked. |

## Why it is hard, and what that cost

No TTS catalogue holds a convincing elderly Indian voice, and a scammer hangs up on latency.
Both were measured, not asserted.

- **The voice is made, not chosen.** Pitch shift, WSOLA tempo, tremble with aperiodic jitter,
  a breath before the first word, a synthetic room (ceiling fan, mains hum), then a narrowband
  mu-law line with packet loss. Voice-to-room ratio is equalised across six line profiles so
  the *delivered* ratio holds at ~10.5 dB measured.
- **The reply starts before it is finished.** Generation is cut at the first sentence boundary
  so synthesis begins mid-stream, and a filler covers every turn. Across two live calls:
  transcript 448–565 ms, first token 243–706 ms, first audio 54–860 ms.
- **Capacity was measured, not guessed.** Token caps are per model, so dialogue, its fallback
  and the analyst run on three; the rate-limit header is read off every response, so the next
  turn moves models *before* a refusal. Writing the JSON shape into the analysis prompt cut a
  pass from 3315 tokens to 1281 — a 61% saving no amount of trimming would have found.

## Evidence

One recorded live call, read back through the API: **49.1 s of caller time consumed** over 6
turns, stages `authority → fear → payment`, peak pressure 0.85, **6 identifiers with
provenance** (2 regex, 4 model), 1 identity leak attributed to the call that issued the
canary. Dialogue spend fell 1773 → 1498 tokens a turn, peak per-model load 111% of the cap →
78%. **1219 offline tests + 74 contract tests, `ruff` and `mypy` clean.** Every figure in the
repository comes from a logged run and is quoted with its sample size.

## Safety, and what is not built

No real phone number, account, Aadhaar, PAN or recording exists anywhere in the project, and
tests assert that generated identities are visibly fake. The persona never threatens,
insults, claims to be police, offers to call back, or tries to identify anyone; those
constraints sit in one shared file whose sha256 is pinned by a test, so weakening them fails
the suite. No persona may even laugh — mockery is ruled out. The system deters and documents;
it does not retaliate.

Not built: the robocall-versus-human check and Guardian Mode's handoff trigger are modelled
in the state machine but not driven, and there is no telephony leg — input is a live
microphone or a replayed synthetic scammer. Gnani's Evon LLM is not reachable on the API
platform, so the LLM sits behind a port; pointing it at Evon is one environment variable.

**Stack.** Prisma v3.0 · Timbre v3.0 with inline audio tags · OpenAI-compatible LLM behind a
port · FastAPI · Postgres · Redis · a pure-Python core with no I/O. Architecture and every
hard-to-reverse trade-off: `README.md` and `docs/DECISIONS.md` (34 entries).
