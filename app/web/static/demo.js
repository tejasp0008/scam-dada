// Scam Dada — live control room.
//
// Reuses the simulator's transport and the dashboard's event handling rather than inventing
// a third: the same /ws/call socket carries PCM and control, the same capture and playback
// worklets do the audio, and the same event payloads drive the panels. What is new here is
// only the presentation and the "who is speaking" state.

const STAGES = ["hook", "authority", "fear", "isolation", "payment"];
const STAGE_BLURB = {
  hook: "making contact with a pretext",
  authority: "establishing he must be obeyed",
  fear: "manufacturing consequences",
  isolation: "cutting you off from help",
  payment: "asking for money or a code",
};
const ENTITY_LABELS = {
  upi_id: "UPI handle",
  phone: "Phone number",
  ifsc: "Bank IFSC",
  amount: "Amount",
  officer_name: "Claimed name",
  organisation: "Claimed org",
  case_number: "Case number",
};

// Mic level above which we call it speech, matched to the server's energy VAD threshold so
// the indicator agrees with what the orchestrator actually decided.
const SPEECH_RMS = 0.015;
const QUIET_MS = 450;

const el = (id) => document.getElementById(id);

let ctx = null;
let stream = null;
let socket = null;
let capture = null;
let playback = null;

let startedAt = 0;
let ended = true;
let turns = 0;
let seenEntities = new Set();
let stageReached = -1;
let quietTimer = null;
let micSpeaking = false;

/* ---------- who is speaking ---------- */

function setWho(who) {
  const box = el("speaking");
  box.dataset.who = who;
  el("who-text").textContent =
    who === "you" ? "you, the scammer" : who === "persona" ? "Dada" : "nobody";

  const chip = el("chip-state");
  const text = el("state-text");
  chip.className = "chip " + (who === "persona" ? "live" : who === "you" ? "caller" : "");
  text.textContent = ended
    ? "standing by"
    : who === "persona"
      ? "Dada speaking"
      : who === "you"
        ? "listening"
        : "on the line";
}

function animateBars(level) {
  const bars = el("bars").children;
  // Hidden rather than flattened when idle: five 4px stubs read as a row of stray dots,
  // which is worse than nothing on a still frame.
  el("bars").style.opacity = level > 0.001 ? "1" : "0";
  for (let i = 0; i < bars.length; i += 1) {
    // Deterministic per-bar shaping rather than random, so it reads as a level meter
    // rather than as decoration.
    const weight = [0.55, 0.85, 1, 0.8, 0.5][i];
    bars[i].style.height = `${Math.max(4, Math.min(20, level * 60 * weight))}px`;
  }
}

/* ---------- call lifecycle ---------- */

async function start() {
  el("btn-start").disabled = true;
  ended = false;
  turns = 0;
  seenEntities = new Set();
  stageReached = -1;
  resetPanels();

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      // The browser's own processing fights ours: AGC flattens the level the energy VAD
      // thresholds on, and noise suppression eats the scammer's room.
      audio: { echoCancellation: true, noiseSuppression: false, autoGainControl: false },
    });
  } catch (err) {
    el("state-text").textContent = "microphone refused";
    el("btn-start").disabled = false;
    return;
  }

  const track = stream.getAudioTracks()[0];
  el("mic-device").textContent = track ? track.label.slice(0, 26) || "default" : "default";
  checkEcho(track);

  ctx = new AudioContext();
  await ctx.audioWorklet.addModule("/static/capture-worklet.js");
  await ctx.audioWorklet.addModule("/static/playback-worklet.js");

  const proto = location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(`${proto}://${location.host}/ws/call`);
  socket.binaryType = "arraybuffer";

  socket.onopen = () => {
    const source = ctx.createMediaStreamSource(stream);
    capture = new AudioWorkletNode(ctx, "capture-processor");
    capture.port.onmessage = (event) => onMicFrame(event.data);
    source.connect(capture);
    capture.connect(ctx.destination);

    playback = new AudioWorkletNode(ctx, "playback-processor", { outputChannelCount: [1] });
    // A buffer that silently swallows a reply is indistinguishable from a server that never
    // sent one. Surface it rather than letting it hide again.
    playback.port.onmessage = (event) => {
      if (event.data && event.data.type === "dropped") {
        console.warn(`playback buffer full; dropped ${event.data.samples} samples`);
        el("state-text").textContent = "audio dropped — buffer full";
      }
    };
    playback.connect(ctx.destination);

    startedAt = Date.now();
    el("btn-stop").disabled = false;
    el("btn-room-tone").disabled = false;
    setWho("idle");
    tick();
  };

  socket.onmessage = (event) => {
    if (typeof event.data === "string") return onControl(JSON.parse(event.data));
    onPersonaAudio(event.data);
  };
  socket.onclose = () => finish("call ended");
  socket.onerror = () => { el("state-text").textContent = "connection error"; };
}

function onMicFrame(frame) {
  if (socket && socket.readyState === WebSocket.OPEN) socket.send(frame.buffer);

  let sum = 0;
  for (let i = 0; i < frame.length; i += 4) {
    const v = frame[i] / 32768;
    sum += v * v;
  }
  const rms = Math.sqrt(sum / (frame.length / 4));
  el("mic-level").style.width = `${Math.min(100, rms * 320)}%`;

  const loud = rms >= SPEECH_RMS;
  if (loud && !micSpeaking) {
    micSpeaking = true;
    if (el("speaking").dataset.who !== "persona") setWho("you");
  }
  if (loud) {
    animateBars(rms);
    clearTimeout(window.__micQuiet);
    window.__micQuiet = setTimeout(() => {
      micSpeaking = false;
      animateBars(0);
      if (el("speaking").dataset.who === "you") setWho("idle");
    }, QUIET_MS);
  }
}

function onPersonaAudio(buffer) {
  const pcm = new Int16Array(buffer);
  const samples = new Float32Array(pcm.length);
  for (let i = 0; i < pcm.length; i += 1) samples[i] = pcm[i] / 32768;
  playback.port.postMessage({
    type: "audio",
    samples: resample(samples, 24000, ctx.sampleRate),
  });

  setWho("persona");
  animateBars(0.35);
  clearTimeout(quietTimer);
  // The server hands over a whole sentence at once, so "still speaking" is measured from
  // the last byte plus roughly how long it takes to play.
  quietTimer = setTimeout(() => {
    animateBars(0);
    setWho(micSpeaking ? "you" : "idle");
  }, Math.max(QUIET_MS, (pcm.length / 24000) * 1000 + 250));
}

function resample(samples, from, to) {
  if (from === to) return samples;
  const ratio = from / to;
  const out = new Float32Array(Math.floor(samples.length / ratio));
  for (let i = 0; i < out.length; i += 1) {
    const pos = i * ratio;
    const index = Math.floor(pos);
    const frac = pos - index;
    const next = index + 1 < samples.length ? samples[index + 1] : samples[index];
    out[i] = samples[index] + (next - samples[index]) * frac;
  }
  return out;
}

let roomTone = true;

function setRoomToneUi(enabled) {
  roomTone = enabled;
  el("btn-room-tone").classList.toggle("on", enabled);
  el("room-tone-text").textContent = `Room tone: ${enabled ? "on" : "off"}`;
}

function toggleRoomTone() {
  if (!socket || socket.readyState !== WebSocket.OPEN) return;
  // Optimistic, then corrected by the server's acknowledgement, so the lamp responds the
  // instant it is pressed on camera rather than after a round trip.
  setRoomToneUi(!roomTone);
  socket.send(`room-tone:${roomTone ? "on" : "off"}`);
}

function stop() {
  if (socket && socket.readyState === WebSocket.OPEN) socket.send("hangup");
  finish("you hung up");
}

function finish(reason) {
  if (ended) return;
  ended = true;
  el("state-text").textContent = reason;
  el("chip-state").className = "chip";
  el("btn-stop").disabled = true;
  el("btn-room-tone").disabled = true;
  el("btn-start").disabled = false;
  setWho("idle");
  animateBars(0);
  if (stream) stream.getTracks().forEach((t) => t.stop());
  if (ctx) ctx.close();
}

/* ---------- control messages ---------- */

function onControl(message) {
  if (message.type === "ready") {
    el("chip-call").textContent = message.call_id.slice(0, 8);
    if (typeof message.room_tone === "boolean") setRoomToneUi(message.room_tone);
    if (message.canary) {
      el("canary-name").textContent = message.canary.full_name;
      el("canary-handle").textContent = message.canary.upi_handle;
    }
  } else if (message.type === "flush") {
    playback.port.postMessage({ type: "flush" });
    clearTimeout(quietTimer);
    animateBars(0);
    setWho("you");
    addTurn("system", "barge-in", "You cut in — Dada stopped mid-sentence.");
  } else if (message.type === "event") {
    onEvent(message.event);
  } else if (message.type === "error") {
    addTurn("system", "error", message.message);
  } else if (message.type === "room-tone") {
    setRoomToneUi(message.enabled);
  } else if (message.type === "ended") {
    finish(message.reason || "call ended");
  }
}

function onEvent(event) {
  if (event.type === "transcript") {
    turns += 1;
    bump("t-turns", turns);
    addTurn(
      event.persona ? "persona" : "caller",
      event.persona ? event.persona : "you",
      event.text,
    );
  } else if (event.type === "stage") {
    setStage(event.stage);
    setGauge("pressure", event.pressure_score);
  } else if (event.type === "risk") {
    setGauge("risk", event.hangup_risk);
    bump("t-risk", `${Math.round(event.hangup_risk * 100)}%`);
    el("tile-risk").classList.toggle("hot", event.hangup_risk >= 0.45);
    el("t-risk-sub").textContent =
      event.hangup_risk >= 0.45
        ? event.tactic
          ? `baiting him: ${event.tactic.replace(/_/g, " ")}`
          : "he is close to leaving"
        : "he is not going anywhere";
    el("sig-risk").textContent = (event.signals || []).length
      ? `Signals: ${event.signals.join(", ")}`
      : "";
  } else if (event.type === "entity") {
    addEntity(event);
  } else if (event.type === "canary_match") {
    showLeak(event);
  } else if (event.type === "latency") {
    el("l-stt").textContent = fmtMs(event.stt_final_ms);
    el("l-llm").textContent = fmtMs(event.llm_first_token_ms);
    el("l-tts").textContent = fmtMs(event.tts_first_byte_ms);
    el("l-filler").textContent = event.filler_played ? "yes" : "not needed";
  } else if (event.type === "persona_handoff") {
    addTurn(
      "system",
      "handoff",
      `${event.from_persona} passes the phone to ${event.to_persona} — ${event.reason}.`,
    );
  }
}

function fmtMs(value) {
  // null means never measurable on this turn, which is different from fast.
  return typeof value === "number" ? `${value}` : "—";
}

/* ---------- panels ---------- */

function resetPanels() {
  el("transcript").innerHTML = '<p class="empty">Listening…</p>';
  el("entities").innerHTML =
    '<p class="empty">Nothing yet. UPI handles, names, case numbers and amounts land here.</p>';
  el("leak-slot").innerHTML = "";
  ["t-turns", "t-entities"].forEach((id) => (el(id).textContent = "0"));
  el("t-risk").textContent = "0%";
  el("t-wasted").textContent = "0:00";
  el("t-wasted-sub").textContent = "live";
  el("chip-stage").textContent = "stage —";
  buildLadder();
  setGauge("pressure", 0);
  setGauge("risk", 0);
  el("sig-risk").textContent = "";
  el("sig-pressure").textContent = "";
}

function buildLadder() {
  const ladder = el("ladder");
  ladder.innerHTML = "";
  STAGES.forEach((stage, index) => {
    const rung = document.createElement("div");
    rung.className = "rung" + (index === STAGES.length - 1 ? " last" : "");
    rung.id = `rung-${stage}`;
    rung.textContent = stage;
    ladder.append(rung);
  });
}

function setStage(stage) {
  const index = STAGES.indexOf(stage);
  if (index < 0 || index <= stageReached) return;
  stageReached = index;
  STAGES.forEach((name, i) => {
    const rung = el(`rung-${name}`);
    if (rung) rung.classList.toggle("on", i <= index);
  });
  el("chip-stage").textContent = `stage ${stage}`;
  el("chip-stage").className = "chip" + (stage === "payment" ? " danger" : "");
  addTurn("system", "stage", `He has reached ${stage} — ${STAGE_BLURB[stage]}.`);
}

function setGauge(name, value) {
  if (typeof value !== "number") return;
  const pct = Math.round(value * 100);
  el(`g-${name}`).textContent = `${pct}%`;
  el(`bar-${name}`).style.width = `${pct}%`;
  if (name === "pressure") el("sig-pressure").textContent = "";
}

function bump(id, value) {
  const node = el(id);
  if (String(node.textContent) === String(value)) return;
  node.textContent = value;
  const tile = node.closest(".tile");
  if (!tile) return;
  tile.classList.remove("flash");
  // Reflow so the animation restarts even when the class was already there.
  void tile.offsetWidth;
  tile.classList.add("flash");
}

function addTurn(cls, who, text) {
  const box = el("transcript");
  if (box.querySelector(".empty")) box.innerHTML = "";
  const div = document.createElement("div");
  div.className = `turn ${cls}`;
  const label = document.createElement("div");
  label.className = "who";
  label.textContent = who;
  const body = document.createElement("div");
  body.className = "text";
  body.textContent = text;
  div.append(label, body);
  box.append(div);
  box.scrollTop = box.scrollHeight;
}

function addEntity(event) {
  const key = `${event.entity_type}:${event.value}`;
  if (seenEntities.has(key)) return;
  seenEntities.add(key);
  bump("t-entities", seenEntities.size);

  const box = el("entities");
  if (box.querySelector(".empty")) box.innerHTML = "";
  const row = document.createElement("div");
  row.className = `entity ${event.source}`;
  const kind = document.createElement("span");
  kind.className = "kind";
  kind.textContent = ENTITY_LABELS[event.entity_type] || event.entity_type;
  const value = document.createElement("span");
  value.className = "val";
  value.textContent = event.value;
  const src = document.createElement("span");
  src.className = "src";
  // Which machine found it, and how sure. Never collapse these into one number.
  src.textContent =
    event.source === "llm" ? `model ${Math.round(event.confidence * 100)}%` : "regex";
  row.append(kind, value, src);
  box.prepend(row);
}

function showLeak(event) {
  const box = el("leak-slot");
  const div = document.createElement("div");
  div.className = "leak-alert";
  div.innerHTML =
    '<div class="head"><span>&#9888;</span> identity leak detected</div>' +
    `<div class="what"></div><div class="why"></div>`;
  div.querySelector(".what").textContent = event.matched_text;
  div.querySelector(".why").textContent =
    `This was issued to call ${event.issued_call_id.slice(0, 8)} and nobody else. ` +
    `He just read it out, so both calls are working from the same lead list. ` +
    `Matched on ${event.matched_field}, confidence ${Math.round(event.score * 100)}%.`;
  box.prepend(div);
  addTurn("system", "leak", `He quoted ${event.matched_text}, issued to an earlier call.`);
}

/* ---------- echo warning ---------- */

function checkEcho(track) {
  const box = el("echo-warn");
  const settings = track && track.getSettings ? track.getSettings() : {};
  const label = (track && track.label ? track.label : "").toLowerCase();
  const likelyHeadset = /head|ear|airpod|buds|hands?free|usb/.test(label);

  if (likelyHeadset) {
    box.className = "warn-box ok";
    box.innerHTML = "<b>Headset detected.</b> Barge-in will respond to you, not to Dada.";
  } else if (settings.echoCancellation === false) {
    box.className = "warn-box";
    box.innerHTML =
      "<b>Echo cancellation is off and no headset was detected.</b> Dada's room tone will " +
      "reach your microphone and cut him off. Use headphones.";
  } else {
    box.className = "warn-box";
    box.innerHTML =
      "<b>No headset detected.</b> Browser echo cancellation is on, which helps, but on " +
      "speakers Dada's room tone can still trip the barge-in detector. Headphones are safer.";
  }
}

/* ---------- clock ---------- */

function tick() {
  if (ended) return;
  const secs = Math.floor((Date.now() - startedAt) / 1000);
  el("t-wasted").textContent = `${Math.floor(secs / 60)}:${String(secs % 60).padStart(2, "0")}`;
  el("t-wasted-sub").textContent = "and counting";
  setTimeout(tick, 250);
}

/* ---------- replay ---------- */
//
// `/demo?replay=<call_id>` plays a finished call's stored events back through the same panels
// at their original spacing. Three uses, and the third is why it exists at all:
// rehearsing the visuals without talking, showing the page with real data on a machine with
// no microphone, and proving that what the dashboard renders live is exactly what was
// persisted - the panels cannot drift from the record if they are driven by it.
//
// `?speed=` compresses the gaps; `?instant=1` skips the timing entirely, which is what the
// screenshot run uses.

async function replay(callId, speed, instant) {
  ended = false;
  resetPanels();
  el("btn-start").disabled = true;
  el("chip-call").textContent = `${callId.slice(0, 8)} · replay`;
  // These are medians across the replayed call, not one turn, and the heading has to say so.
  document.querySelector("#l-stt").closest(".card").querySelector("h3").textContent =
    "Latency, median across call";

  let report;
  try {
    report = await (await fetch(`/api/calls/${callId}`)).json();
  } catch (err) {
    el("state-text").textContent = "no such call";
    el("btn-start").disabled = false;
    return;
  }

  const canary = report.canary_issued || {};
  if (canary.full_name) el("canary-name").textContent = canary.full_name;
  if (canary.upi_handle) el("canary-handle").textContent = canary.upi_handle;

  const body = await (await fetch(`/api/calls/${callId}/events?after_seq=-1`)).json();
  // Stored events, if Redis still holds them; otherwise rebuild a stream from the report so
  // an old call still replays.
  const events = body.events && body.events.length ? body.events : fromReport(report);
  if (!events.length) {
    el("state-text").textContent = "nothing recorded for that call";
    return;
  }

  el("state-text").textContent = "replaying";
  el("chip-state").className = "chip live";
  startedAt = Date.now();
  tick();

  const first = new Date(events[0].ts).getTime();
  for (const event of events) {
    if (!instant) {
      const offset = (new Date(event.ts).getTime() - first) / Math.max(0.1, speed);
      await sleep(Math.min(4000, Math.max(0, offset - (Date.now() - startedAt))));
    }
    if (event.type === "transcript") setWho(event.persona ? "persona" : "you");
    onEvent(event);
  }

  const secs = Math.round(report.summary.seconds_wasted);
  el("t-wasted").textContent = `${Math.floor(secs / 60)}:${String(secs % 60).padStart(2, "0")}`;
  el("t-wasted-sub").textContent = "recorded call";
  ended = true;
  setWho("idle");
  el("state-text").textContent = "replay finished";
  el("chip-state").className = "chip";
  el("btn-start").disabled = false;
}

function fromReport(report) {
  // A minimal stream for a call whose Redis log has expired: the transcript in order, then
  // the terminal readings. Enough to render, and honest about being a reconstruction.
  const out = [];
  let seq = 0;
  const ts = report.summary.started_at;
  for (const turn of report.turns) {
    out.push({
      type: "transcript", seq: seq++, ts, call_id: report.summary.call_id,
      text: turn.text, persona: turn.persona, speaker: turn.speaker,
    });
  }
  for (const entity of report.entities) {
    out.push({ type: "entity", seq: seq++, ts, ...entity });
  }
  for (const leak of report.leaks) {
    out.push({ type: "canary_match", seq: seq++, ts, ...leak });
  }
  if (report.summary.stage_reached) {
    out.push({
      type: "stage", seq: seq++, ts, stage: report.summary.stage_reached,
      confidence: 0, pressure_score: report.summary.peak_pressure,
    });
  }
  out.push({
    type: "risk", seq: seq++, ts, hangup_risk: report.summary.peak_hangup_risk,
    signals: [], tactic: null,
  });
  // The report carries latency medians even when the event log has expired. Showing the
  // median and saying so beats an empty panel, which reads as "we did not measure".
  const legs = {};
  for (const leg of report.latency || []) legs[leg.name] = leg.median_ms;
  out.push({
    type: "latency", seq: seq++, ts, turn_seq: 0,
    stt_final_ms: legs.stt_final_ms ?? null,
    llm_first_token_ms: legs.llm_first_token_ms ?? null,
    tts_first_byte_ms: legs.tts_first_byte_ms ?? null,
    filler_played: true,
  });
  return out;
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/* ---------- boot ---------- */

el("btn-start").onclick = start;
el("btn-stop").onclick = stop;
el("btn-room-tone").onclick = toggleRoomTone;
buildLadder();
animateBars(0);

const params = new URLSearchParams(location.search);
if (params.get("still") === "1") document.body.classList.add("still");
if (params.get("replay")) {
  replay(
    params.get("replay"),
    Number(params.get("speed") || 1),
    params.get("instant") === "1",
  );
}

fetch("/readyz")
  .then((r) => r.json())
  .then((body) => {
    const p = body.providers || {};
    el("chip-providers").textContent = `${p.stt || "?"} · ${p.tts || "?"} · ${p.llm || "?"}`;
    if ((p.stt || "").startsWith("mock")) {
      el("chip-providers").className = "chip danger";
      el("chip-providers").textContent += " (mock!)";
    }
  })
  .catch(() => { el("chip-providers").textContent = "providers unknown"; });
