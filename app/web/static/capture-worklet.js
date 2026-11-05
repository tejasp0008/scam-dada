// Microphone capture: resample to 16 kHz and emit exactly 1024-byte frames.
//
// Prisma requires frames of exactly 1024 bytes (512 samples at 16 kHz, 32 ms) and rejects
// anything else, so the framing happens here rather than on the server: a worklet runs on
// the audio thread in 128-sample quanta, and anywhere else would mean an extra buffer and
// extra latency on the path that decides how fast barge-in feels.
//
// The browser gives us whatever rate the hardware runs at - 44.1 or 48 kHz, usually - so
// the ratio is computed at construction rather than assumed.

const TARGET_RATE = 16000;
const FRAME_SAMPLES = 512;

class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / TARGET_RATE;
    this.frame = new Int16Array(FRAME_SAMPLES);
    this.filled = 0;
    // Fractional read position into the incoming block, carried across blocks so the
    // resampler does not click once every 128 samples.
    this.position = 0;
    this.tail = 0;
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel) return true;

    // Linear interpolation. Not a polyphase filter, and the server-side note about that
    // applies here too: the ASR is trained on telephone audio and this is not the
    // bottleneck. Correctness that matters is the frame size, not the filter.
    while (this.position < channel.length) {
      const index = Math.floor(this.position);
      const frac = this.position - index;
      const current = channel[index];
      const next = index + 1 < channel.length ? channel[index + 1] : this.tail;
      const sample = current + (next - current) * frac;

      const clamped = Math.max(-1, Math.min(1, sample));
      this.frame[this.filled++] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;

      if (this.filled === FRAME_SAMPLES) {
        // Copied, not transferred: the same Int16Array is reused for the next frame.
        this.port.postMessage(this.frame.slice());
        this.filled = 0;
      }
      this.position += this.ratio;
    }
    this.position -= channel.length;
    this.tail = channel[channel.length - 1];
    return true;
  }
}

registerProcessor("capture-processor", CaptureProcessor);
