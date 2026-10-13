"""
Qwen3-TTS-12Hz-1.7B-Base — Voice Cloner on Modal (L4 GPU)

Architecture:
  - Model weights are downloaded ONCE into a Modal Volume (`qwen3-tts-weights`)
    and reused by every container afterwards.
  - The web server / frontend does NOT wait for the model. The model is loaded
    in a background thread at container startup, so the page is served instantly
    and the UI just polls /status until the model is warm.
  - Long scripts are split into sentence-level chunks server-side; each chunk is
    synthesized separately (reusing one pre-built voice-clone prompt) and streamed
    to the browser over SSE, so prosody stays natural and playback starts fast.

Deploy:
    pip install modal
    modal setup
    modal run qwen3_tts_modal.py      # one-time: downloads ~4GB weights to the volume
    modal deploy qwen3_tts_modal.py   # deploys the web app, prints the URL
"""

import base64
import io
import re
import threading

import modal

app = modal.App("qwen3-tts-voice-cloner")

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
MODEL_DIR = "/models/qwen3-tts-12hz-1.7b-base"   # inside the volume

weights_volume = modal.Volume.from_name("qwen3-tts-weights", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "qwen-tts",              # pulls torch, transformers, soundfile, etc.
        "accelerate",            # needed for device_map="cuda:0"
        "fastapi[standard]",
        "python-multipart",
        "soundfile",
        "numpy",
    )
)


# --------------------------------------------------------------------------- #
# One-time weight download into the Modal volume
# --------------------------------------------------------------------------- #
@app.function(image=image, volumes={"/models": weights_volume}, timeout=60 * 60)
def download_model():
    from huggingface_hub import snapshot_download

    print(f"Downloading {MODEL_ID} into the volume (one-time)...")
    snapshot_download(MODEL_ID, local_dir=MODEL_DIR)
    weights_volume.commit()
    print("Done. Weights are cached in the 'qwen3-tts-weights' volume.")


@app.local_entrypoint()
def main():
    download_model.remote()


# --------------------------------------------------------------------------- #
# Sentence-level chunking — the key to natural-sounding long-form speech.
# Sending a whole script at once makes the model produce flat, run-on speech.
# --------------------------------------------------------------------------- #
_MAX_CHUNK_CHARS = 180  # ~1-2 sentences per synthesis call


def _split_long_sentence(sentence: str, max_chars: int) -> list[str]:
    """Split an over-long sentence at commas/colons without losing punctuation."""
    parts = re.split(r"(?<=[,，、:：;；—–-])\s+", sentence)
    out, cur = [], ""
    for p in parts:
        if len(cur) + len(p) + 1 <= max_chars:
            cur = f"{cur} {p}".strip()
        else:
            if cur:
                out.append(cur)
            cur = p
    if cur:
        out.append(cur)
    return out or [sentence]


def chunk_text(text: str) -> list[str]:
    """Split a script into natural chunks at sentence boundaries."""
    sentence_re = re.compile(r"[^.!?。！？\n]+[.!?。！？]*")
    sentences = [m.group(0).strip() for m in sentence_re.finditer(text)]
    sentences = [s for s in sentences if s]

    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) + 1 <= _MAX_CHUNK_CHARS:
            current = f"{current} {s}".strip()
        else:
            if current:
                chunks.append(current)
            if len(s) <= _MAX_CHUNK_CHARS:
                current = s
            else:
                chunks.extend(_split_long_sentence(s, _MAX_CHUNK_CHARS))
                current = ""
    if current:
        chunks.append(current)
    return chunks


# --------------------------------------------------------------------------- #
# The GPU-backed TTS service
# --------------------------------------------------------------------------- #
@app.cls(
    gpu="L4",
    image=image,
    volumes={"/models": weights_volume},
    scaledown_window=15 * 60,   # keep the container (and warm model) alive 15 min
    timeout=60 * 30,
)
class TTS:
    @modal.enter()
    def startup(self):
        """Runs when the container starts. IMPORTANT: this must return fast,
        because Modal starts routing HTTP traffic only after @modal.enter
        returns. So we kick the (slow) model load off in a daemon thread and
        serve the frontend immediately while weights load in the background."""
        self.ready_event = threading.Event()
        self.load_error = None
        self.gen_lock = threading.Lock()          # serialize GPU generation
        threading.Thread(target=self._load_model, daemon=True).start()

    def _load_model(self):
        try:
            import torch
            from qwen_tts import Qwen3TTSModel

            self.model = Qwen3TTSModel.from_pretrained(
                MODEL_DIR,                        # served from the cached volume
                device_map="cuda:0",
                dtype=torch.bfloat16,
                attn_implementation="sdpa",       # use "flash_attention_2" if you add flash-attn to the image
            )
        except Exception as e:                    # surfaced to the frontend
            self.load_error = f"{type(e).__name__}: {e}"
        finally:
            self.ready_event.set()

    # ------------------------------ endpoints ------------------------------ #
    @modal.fastapi_endpoint(method="GET")
    def status(self):
        """Frontend polls this to show a 'warming up' / 'ready' indicator."""
        return {
            "ready": self.ready_event.is_set() and self.load_error is None,
            "error": self.load_error,
        }

    @modal.fastapi_endpoint(method="GET")
    def home(self):
        """Serves the UI instantly — no model dependency."""
        from fastapi.responses import HTMLResponse

        return HTMLResponse(FRONTEND_HTML)

    @modal.fastapi_endpoint(method="POST")
    async def synthesize(
        self,
        ref_audio: "object",   # UploadFile — annotated below via fastapi
        ref_text: str = "",
        text: str = "",
        language: str = "auto",
    ):
        """Voice-clone endpoint: accepts your reference audio + transcript and
        the script to speak, then streams synthesized audio chunk-by-chunk
        (server-sent events) so playback starts after the first sentence."""
        from fastapi import UploadFile
        from fastapi.responses import StreamingResponse

        ref_audio: UploadFile = ref_audio  # type: ignore[no-redef]

        if not self.ready_event.is_set():
            return StreamingResponse(
                _sse({"error": "Model is still warming up — try again in a moment."}),
                media_type="text/event-stream",
            )
        if self.load_error:
            return StreamingResponse(
                _sse({"error": f"Model failed to load: {self.load_error}"}),
                media_type="text/event-stream",
            )
        if not text.strip():
            return StreamingResponse(_sse({"error": "No script provided."}),
                                     media_type="text/event-stream")

        ref_bytes = await ref_audio.read()
        return StreamingResponse(
            self._generate_stream(ref_bytes, ref_text, text, language),
            media_type="text/event-stream",
        )

    # --------------------------- generation core --------------------------- #
    def _generate_stream(self, ref_bytes: bytes, ref_text: str,
                         text: str, language: str):
        import soundfile as sf

        with self.gen_lock:
            try:
                # Decode the uploaded reference clip once
                ref_wav, ref_sr = sf.read(io.BytesIO(ref_bytes))

                # Build the voice-clone prompt ONCE and reuse it for every
                # chunk — this is what keeps the voice consistent throughout
                prompt = self.model.create_voice_clone_prompt(
                    ref_audio=(ref_wav, ref_sr),
                    ref_text=ref_text.strip(),
                    x_vector_only_mode=False,
                )

                chunks = chunk_text(text)
                yield _sse({"total": len(chunks)})

                for i, chunk in enumerate(chunks):
                    wavs, sr = self.model.generate_voice_clone(
                        text=chunk,
                        language=language,
                        voice_clone_prompt=prompt,
                        max_new_tokens=2048,
                    )
                    buf = io.BytesIO()
                    sf.write(buf, wavs[0], sr, format="WAV")
                    payload = base64.b64encode(buf.getvalue()).decode()
                    yield _sse({"i": i, "audio": payload})
                yield _sse({"done": True})
            except Exception as e:
                yield _sse({"error": f"{type(e).__name__}: {e}"})


def _sse(obj: dict) -> str:
    import json

    return f"data: {json.dumps(obj)}\n\n"


# --------------------------------------------------------------------------- #
# Frontend — self-contained, served instantly from the same Modal URL
# --------------------------------------------------------------------------- #
FRONTEND_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Qwen3-TTS Voice Cloner</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, -apple-system, sans-serif; background: #0d1117;
         color: #e6edf3; min-height: 100vh; display: flex; justify-content: center;
         padding: 40px 16px; }
  .card { width: 100%; max-width: 640px; background: #161b22; border: 1px solid #30363d;
          border-radius: 14px; padding: 28px; }
  h1 { font-size: 22px; margin-bottom: 4px; }
  .sub { color: #8b949e; font-size: 13px; margin-bottom: 20px; }
  .pill { display: inline-block; font-size: 12px; padding: 4px 12px; border-radius: 999px;
          margin-bottom: 20px; font-weight: 600; }
  .pill.loading { background: #2d2410; color: #e3b341; }
  .pill.ready { background: #12261e; color: #3fb950; }
  .pill.error { background: #2d1518; color: #f85149; }
  label { display: block; font-size: 13px; color: #8b949e; margin: 14px 0 6px; }
  input[type=file], input[type=text], select, textarea {
    width: 100%; background: #0d1117; border: 1px solid #30363d; color: #e6edf3;
    border-radius: 8px; padding: 10px 12px; font-size: 14px; }
  textarea { min-height: 140px; resize: vertical; line-height: 1.5; }
  input:focus, textarea:focus, select:focus { outline: 2px solid #1f6feb; }
  button { width: 100%; margin-top: 20px; padding: 12px; font-size: 15px; font-weight: 700;
           border: none; border-radius: 8px; cursor: pointer;
           background: #1f6feb; color: #fff; transition: background .15s; }
  button:hover:not(:disabled) { background: #388bfd; }
  button:disabled { background: #21262d; color: #484f58; cursor: not-allowed; }
  #status { margin-top: 14px; font-size: 13px; color: #8b949e; min-height: 20px; }
  #progressWrap { height: 6px; background: #21262d; border-radius: 3px;
                  margin-top: 10px; overflow: hidden; display: none; }
  #progress { height: 100%; width: 0%; background: #1f6feb; transition: width .2s; }
  #dl { display: none; margin-top: 12px; text-align: center; color: #58a6ff;
        font-size: 13px; cursor: pointer; }
</style>
</head>
<body>
<div class="card">
  <h1>🎙️ Qwen3-TTS Voice Cloner</h1>
  <div class="sub">Qwen3-TTS-12Hz-1.7B-Base · Modal L4 · long scripts are spoken sentence-by-sentence for natural prosody</div>
  <span id="pill" class="pill loading">⏳ Model warming up…</span>

  <form id="f">
    <label>1. Reference audio (3+ seconds of the voice to clone — wav/mp3/m4a)</label>
    <input type="file" id="refAudio" accept="audio/*" required>

    <label>2. Transcript of the reference audio (what is said in it, word for word)</label>
    <input type="text" id="refText" placeholder="e.g. Okay, yeah, I resent you. I love you…">

    <label>3. Language</label>
    <select id="lang">
      <option value="auto">Auto-detect</option>
      <option value="English">English</option><option value="Chinese">Chinese</option>
      <option value="Japanese">Japanese</option><option value="Korean">Korean</option>
      <option value="German">German</option><option value="French">French</option>
      <option value="Russian">Russian</option><option value="Portuguese">Portuguese</option>
      <option value="Spanish">Spanish</option><option value="Italian">Italian</option>
    </select>

    <label>4. Your script</label>
    <textarea id="text" placeholder="Paste your full script here. It will be split into natural sentences and spoken chunk by chunk — it won't be dumped into the model all at once."></textarea>

    <button id="go" disabled>Clone voice & speak</button>
  </form>

  <div id="status"></div>
  <div id="progressWrap"><div id="progress"></div></div>
  <a id="dl">⬇ Download full recording</a>
</div>

<script>
const pill = document.getElementById('pill'), go = document.getElementById('go'),
      statusEl = document.getElementById('status'), pw = document.getElementById('progressWrap'),
      prog = document.getElementById('progress'), dl = document.getElementById('dl');

// Poll model status — page itself is already usable, we just gate the button
async function poll() {
  try {
    const r = await fetch('/status'); const j = await r.json();
    if (j.ready) { pill.className = 'pill ready'; pill.textContent = '✅ Model ready';
      go.disabled = false; return; }
    if (j.error) { pill.className = 'pill error'; pill.textContent = '❌ ' + j.error; return; }
  } catch (e) {}
  setTimeout(poll, 2000);
}
poll();

let audioCtx = null, nextPlayTime = 0, collected = [], sampleRate = 24000;

document.getElementById('f').addEventListener('submit', async (e) => {
  e.preventDefault();
  const file = document.getElementById('refAudio').files[0];
  if (!file) return;
  go.disabled = true; dl.style.display = 'none'; collected = [];
  statusEl.textContent = 'Uploading & cloning voice…'; pw.style.display = 'block';
  prog.style.width = '2%';

  audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 24000 });
  nextPlayTime = Math.max(nextPlayTime, audioCtx.currentTime + 0.1);

  const fd = new FormData();
  fd.append('ref_audio', file);
  fd.append('ref_text', document.getElementById('refText').value);
  fd.append('text', document.getElementById('text').value);
  fd.append('language', document.getElementById('lang').value);

  const res = await fetch('/synthesize', { method: 'POST', body: fd });
  const reader = res.body.getReader(), dec = new TextDecoder();
  let buf = '';

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf('\n\n')) >= 0) {
      const raw = buf.slice(0, idx); buf = buf.slice(idx + 2);
      if (!raw.startsWith('data: ')) continue;
      const msg = JSON.parse(raw.slice(6));
      if (msg.error) { statusEl.textContent = '⚠️ ' + msg.error; break; }
      if (msg.total) { statusEl.textContent = `Speaking… chunk 1 of ${msg.total}`; continue; }
      if (msg.audio) {
        const wav = Uint8Array.from(atob(msg.audio), c => c.charCodeAt(0));
        const ab = await audioCtx.decodeAudioData(wav.buffer);
        sampleRate = ab.sampleRate; collected.push(ab);
        const src = audioCtx.createBufferSource(); src.buffer = ab;
        src.connect(audioCtx.destination);
        src.start(nextPlayTime);                       // seamless queueing
        nextPlayTime += ab.duration + 0.12;            // tiny breath between sentences
        statusEl.textContent = `Speaking… chunk ${msg.i + 1}`;
        prog.style.width = Math.min(99, (msg.i + 1) * 8) + '%';
      }
      if (msg.done) {
        statusEl.textContent = '✅ Done — played ' + collected.length + ' chunks in your cloned voice.';
        prog.style.width = '100%'; dl.style.display = 'block';
      }
    }
  }
  go.disabled = false;
});

// Concatenate all chunks (with gaps) and export one WAV file
dl.addEventListener('click', () => {
  const gap = Math.round(sampleRate * 0.12);
  const total = collected.reduce((s, b) => s + b.length + gap, 0);
  const out = audioCtx.createBuffer(collected[0].numberOfChannels, total, sampleRate);
  let off = 0;
  for (const b of collected) { for (let c = 0; c < b.numberOfChannels; c++)
    out.getChannelData(c).set(b.getChannelData(c), off); off += b.length + gap; }
  const n = out.length, data = out.getChannelData(0), bytes = new DataView(new ArrayBuffer(44 + n * 2));
  const ws = (o, s) => { for (let i = 0; i < s.length; i++) bytes.setUint8(o + i, s.charCodeAt(i)); };
  ws(0, 'RIFF'); bytes.setUint32(4, 36 + n * 2, true); ws(8, 'WAVEfmt ');
  bytes.setUint32(16, 16, true); bytes.setUint16(20, 1, true); bytes.setUint16(22, 1, true);
  bytes.setUint32(24, sampleRate, true); bytes.setUint32(28, sampleRate * 2, true);
  bytes.setUint16(32, 2, true); bytes.setUint16(34, 16, true); ws(36, 'data');
  bytes.setUint32(40, n * 2, true);
  for (let i = 0; i < n; i++) { const s = Math.max(-1, Math.min(1, data[i]));
    bytes.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true); }
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([bytes.buffer], { type: 'audio/wav' }));
  a.download = 'clone.wav'; a.click();
});
</script>
</body>
</html>"""
