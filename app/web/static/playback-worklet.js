// Persona playback through a ring buffer, so a barge-in can drop queued audio instantly.
//
// The obvious approach - one AudioBufferSourceNode per chunk - cannot be interrupted: once
// scheduled, a node plays. With a ring buffer, "stop talking" is one message that moves the
// read pointer to the write pointer, and the persona goes quiet within a render quantum
// (about 3 ms) instead of whenever its queue happens to run dry.
//
// Capacity is eight seconds, which is far more than the server ever buffers; it exists so a
// scheduling hiccup under-runs silently rather than glitching.

const CAPACITY = 8 * 48000;

class PlaybackProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = new Float32Array(CAPACITY);
    this.read = 0;
    this.write = 0;
    this.underruns = 0;
    this.dropped = 0;
    this.port.onmessage = (event) => {
      const message = event.data;
      if (message.type === "audio") {
        this.enqueue(message.samples);
      } else if (message.type === "flush") {
        // Barge-in. Everything already handed to us is abandoned; it has been paid for
        // either way, because Timbre documents no cancel.
        this.read = this.write;
      }
    };
  }

  enqueue(samples) {
    for (let i = 0; i < samples.length; i += 1) {
      const next = (this.write + 1) % CAPACITY;
      if (next === this.read) {
        // Full. Dropping is still better than overwriting audio that has not played, but
        // it must not be silent about it: a buffer quietly swallowing a whole reply is
        // indistinguishable from the server never sending one, and that cost real
        // debugging time once already.
        this.dropped += samples.length - i;
        this.port.postMessage({ type: "dropped", samples: this.dropped });
        return;
      }
      this.buffer[this.write] = samples[i];
      this.write = next;
    }
  }

  get queued() {
    return (this.write - this.read + CAPACITY) % CAPACITY;
  }

  process(_inputs, outputs) {
    const output = outputs[0][0];
    if (!output) return true;
    for (let i = 0; i < output.length; i += 1) {
      if (this.read === this.write) {
        output[i] = 0;
        this.underruns += 1;
      } else {
        output[i] = this.buffer[this.read];
        this.read = (this.read + 1) % CAPACITY;
      }
    }
    return true;
  }
}

registerProcessor("playback-processor", PlaybackProcessor);
