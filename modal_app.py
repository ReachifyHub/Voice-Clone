import base64
import io
import os
import re
import tempfile

import modal

app = modal.App("qwen3-tts")

# Persist model weights so cold starts don't re-download
hf_cache = modal.Volume.from_name("qwen3-tts-hf-cache", create_if_missing=True)

# Your Netlify site (if you still want CORS for it). The Modal frontend is same-origin.
ALLOWED_ORIGINS = ["https://privoice.netlify.app", "*"]

MAX_CHARS = 10000
CHUNK_SIZE = 350        # chars per chunk (matches Qwen3-TTS community practice)
CHUNK_PAUSE_MS = 350    # silence between chunks (matches validated 350ms boundary)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.5.1",
        "torchaudio==2.5.1",
        "qwen-tts",
        "soundfile",
        "numpy",
        "fastapi[standard]",
    )
    .env({"HF_HOME": "/cache"})
)


# ──────────────────────────────────────────────────────────────
# HTML Frontend (Tailwind CSS, vanilla JS, Web Audio playback)
# ──────────────────────────────────────────────────────────────
FRONTEND_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Qwen3 TTS Studio</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    .spinner { border: 3px solid rgba(255,255,255,.2); border-top-color:#fff;
               border-radius:50%; width:20px; height:20px; animation:spin .6s linear infinite; }
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body class="bg-slate-900 text-slate-100 min-h-screen">
  <div class="max-w-3xl mx-auto px-4 py-10">
    <h1 class="text-3xl font-bold mb-2">Qwen3 TTS Studio</h1>
    <p class="text-slate-400 mb-8 text-sm">Voice cloning · Voice design · Chunked synthesis</p>

    <!-- Mode Tabs -->
    <div class="flex gap-2 mb-6">
      <button data-mode="design" class="mode-btn px-4 py-2 rounded-lg bg-indigo-600 text-sm font-medium">Voice Design</button>
      <button data-mode="clone"  class="mode-btn px-4 py-2 rounded-lg bg-slate-700 text-sm font-medium">Voice Clone</button>
    </div>

    <!-- Voice Design panel -->
    <div id="panel-design" class="space-y-4">
      <div>
        <label class="block text-sm text-slate-400 mb-1">Voice description</label>
        <input id="instruct" type="text" placeholder="e.g. female, british accent, warm tone"
               class="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm
                      focus:outline-none focus:ring-2 focus:ring-indigo-500" />
      </div>
    </div>

    <!-- Voice Clone panel -->
    <div id="panel-clone" class="space-y-4 hidden">
      <div>
        <label class="block text-sm text-slate-400 mb-1">Reference audio (3–10 s)</label>
        <input id="refAudio" type="file" accept="audio/*"
               class="block w-full text-sm text-slate-400 file:mr-3 file:py-2 file:px-4
                      file:rounded-lg file:border-0 file:bg-slate-700 file:text-slate-200
                      file:text-sm file:font-medium hover:file:bg-slate-600" />
      </div>
      <div>
        <label class="block text-sm text-slate-400 mb-1">Reference transcript</label>
        <input id="refText" type="text" placeholder="Exact words spoken in the clip"
               class="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm
                      focus:outline-none focus:ring-2 focus:ring-indigo-500" />
      </div>
    </div>

    <!-- Shared controls -->
    <div class="grid grid-cols-2 gap-4 mt-6">
      <div>
        <label class="block text-sm text-slate-400 mb-1">Language</label>
        <select id="language"
                class="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm">
          <option value="Auto">Auto</option>
          <option value="English" selected>English</option>
          <option value="Chinese">Chinese</option>
          <option value="Japanese">Japanese</option>
          <option value="Korean">Korean</option>
          <option value="German">German</option>
          <option value="French">French</option>
          <option value="Russian">Russian</option>
          <option value="Portuguese">Portuguese</option>
          <option value="Spanish">Spanish</option>
          <option value="Italian">Italian</option>
        </select>
      </div>
      <div>
        <label class="block text-sm text-slate-400 mb-1">Chunk size (chars)</label>
        <input id="chunkSize" type="number" value="350" min="100" max="800"
               class="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm" />
      </div>
    </div>

    <!-- Script input -->
    <div class="mt-6">
      <label class="block text-sm text-slate-400 mb-1">Script</label>
      <textarea id="text" rows="8" placeholder="Paste your full script here…"
                class="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm
                       focus:outline-none focus:ring-2 focus:ring-indigo-500 resize-y"></textarea>
      <div class="text-right text-xs text-slate-500 mt-1"><span id="charCount">0</span> chars</div>
    </div>

    <!-- Generate button -->
    <button id="generateBtn"
            class="mt-6 w-full bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50
                   disabled:cursor-not-allowed font-semibold py-3 rounded-lg transition
                   flex items-center justify-center gap-2">
      <span id="btnText">Generate Speech</span>
    </button>

    <!-- Progress -->
    <div id="progressWrap" class="mt-4 hidden">
      <div class="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
        <div id="progressBar" class="bg-indigo-500 h-2 transition-all duration-300" style="width:0%"></div>
      </div>
      <p id="progressLabel" class="text-xs text-slate-400 mt-2"></p>
    </div>

    <!-- Audio player -->
    <div id="audioWrap" class="mt-6 hidden">
      <audio id="player" controls class="w-full"></audio>
      <a id="downloadLink" download="qwen3-output.wav"
         class="inline-block mt-3 text-sm text-indigo-400 hover:text-indigo-300">↓ Download WAV</a>
    </div>

    <!-- Error -->
    <div id="errorBox" class="mt-4 hidden bg-red-900/40 border border-red-700 text-red-300
                              rounded-lg px-4 py-3 text-sm"></div>
  </div>

<script>
const $ = (s) => document.querySelector(s);
const API = "/tts";   // same-origin — Modal serves both frontend and API

let mode = "design";

/* ── Tab switching ─────────────────────────────── */
document.querySelectorAll(".mode-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    mode = btn.dataset.mode;
    document.querySelectorAll(".mode-btn").forEach(b => {
      b.classList.toggle("bg-indigo-600", b.dataset.mode === mode);
      b.classList.toggle("bg-slate-700",  b.dataset.mode !== mode);
    });
    $("#panel-design").classList.toggle("hidden", mode !== "design");
    $("#panel-clone").classList.toggle("hidden",  mode !== "clone");
  });
});

/* ── Char counter ──────────────────────────────── */
$("#text").addEventListener("input", e => {
  $("#charCount").textContent = e.target.value.length;
});

/* ── Helpers ───────────────────────────────────── */
function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result.split(",")[1]);
    r.onerror = reject;
    r.readAsDataURL(file);
  });
}

function showError(msg) {
  const el = $("#errorBox");
  el.textContent = msg;
  el.classList.remove("hidden");
}

function hideError() { $("#errorBox").classList.add("hidden"); }

function setBusy(busy) {
  $("#generateBtn").disabled = busy;
  $("#btnText").textContent = busy ? "Generating…" : "Generate Speech";
  if (busy) {
    $("#progressWrap").classList.remove("hidden");
  }
}

function setProgress(current, total, label) {
  const pct = total > 0 ? Math.round((current / total) * 100) : 0;
  $("#progressBar").style.width = pct + "%";
  $("#progressLabel").textContent = label || `Chunk ${current} of ${total}`;
}

/* ── Main generate ─────────────────────────────── */
$("#generateBtn").addEventListener("click", async () => {
  hideError();
  const text = $("#text").value.trim();
  if (!text) { showError("Please enter some text."); return; }

  const chunkSize = parseInt($("#chunkSize").value) || 350;
  const language  = $("#language").value;

  let refAudioB64 = null;
  let refText = null;
  if (mode === "clone") {
    const file = $("#refAudio").files[0];
    if (!file) { showError("Please upload a reference audio clip."); return; }
    refAudioB64 = await fileToBase64(file);
    refText = $("#refText").value.trim() || null;
  }

  // Send full text + chunk size — server handles splitting
  const body = {
    text,
    language,
    chunk_size: chunkSize,
    instruct: mode === "design" ? ($("#instruct").value.trim() || null) : null,
    ref_audio_b64: refAudioB64,
    ref_text: refText,
  };

  setBusy(true);
  setProgress(0, 1, "Starting…");

  try {
    const res = await fetch(API, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || "Server error");
    }

    // Server returns a single WAV (all chunks concatenated server-side)
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);

    $("#player").src = url;
    $("#downloadLink").href = url;
    $("#audioWrap").classList.remove("hidden");
    setProgress(1, 1, "Done ✓");
  } catch (e) {
    showError(e.message);
    $("#progressWrap").classList.add("hidden");
  } finally {
    setBusy(false);
  }
});
</script>
</body>
</html>
"""


@app.cls(
    image=image,
    gpu="L4",
    volumes={"/cache": hf_cache},
    scaledown_window=80,
    max_containers=1,
    timeout=1200,
)
class TTS:
    @modal.enter()
    def load(self):
        import torch
        from qwen_tts import Qwen3TTSModel

        # Base model supports voice cloning AND voice design
        self.model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        hf_cache.commit()

    # ── Chunking helper ──────────────────────────────────────
    @staticmethod
    def chunk_text(text: str, max_chars: int = 350) -> list[str]:
        """
        Split text into sentence-aligned chunks.
        Packs complete sentences into chunks of at most `max_chars`,
        then hard-splits any sentence that exceeds the limit.
        """
        # Normalise whitespace
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []

        # Split on sentence-ending punctuation (keeps the punctuation)
        sentences = re.split(r"(?<=[.!?…])\s+", text)

        chunks: list[str] = []
        current = ""

        for sentence in sentences:
            # If a single sentence is longer than max_chars, hard-split it
            while len(sentence) > max_chars:
                # Try to break at a comma or space near the limit
                split_at = sentence.rfind(", ", 0, max_chars)
                if split_at == -1:
                    split_at = sentence.rfind(" ", 0, max_chars)
                if split_at == -1:
                    split_at = max_chars
                piece = sentence[:split_at + 1].strip()
                sentence = sentence[split_at + 1:].strip()

                if current:
                    chunks.append(current.strip())
                    current = ""
                if piece:
                    chunks.append(piece)

            # Try to add this sentence to the current chunk
            if not current:
                current = sentence
            elif len(current) + 1 + len(sentence) <= max_chars:
                current += " " + sentence
            else:
                chunks.append(current.strip())
                current = sentence

        if current.strip():
            chunks.append(current.strip())

        return [c for c in chunks if c]

    # ── ASGI app ─────────────────────────────────────────────
    @modal.asgi_app()
    def web(self):
        import io as _io

        import numpy as np
        import soundfile as sf
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import HTMLResponse, Response
        from pydantic import BaseModel

        class Req(BaseModel):
            text: str
            language: str = "Auto"
            instruct: str | None = None
            ref_audio_b64: str | None = None
            ref_text: str | None = None
            chunk_size: int = 350

        api = FastAPI()
        api.add_middleware(
            CORSMiddleware,
            allow_origins=ALLOWED_ORIGINS,
            allow_methods=["POST", "OPTIONS"],
            allow_headers=["*"],
        )

        # ── Serve frontend at root ───────────────────────
        @api.get("/", response_class=HTMLResponse)
        def index():
            return FRONTEND_HTML

        # ── TTS endpoint ─────────────────────────────────
        @api.post("/tts")
        def tts(req: Req):
            if not req.text.strip():
                raise HTTPException(400, "Text is empty")
            if len(req.text) > MAX_CHARS:
                raise HTTPException(400, f"Text too long (max {MAX_CHARS} characters)")

            # 1. Chunk the script
            chunks = self.chunk_text(req.text, req.chunk_size)
            if not chunks:
                raise HTTPException(400, "No speakable text after chunking")

            # 2. Prepare reference audio temp file (if cloning)
            tmp_path = None
            if req.ref_audio_b64:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    f.write(base64.b64decode(req.ref_audio_b64))
                    tmp_path = f.name

            try:
                all_audio: list[np.ndarray] = []
                sr = 24000

                # 3. Synthesise each chunk independently
                for chunk in chunks:
                    if req.ref_audio_b64:
                        # Voice cloning path
                        wavs, sr = self.model.generate_voice_clone(
                            text=chunk,
                            language=req.language,
                            ref_audio=tmp_path,
                            ref_text=req.ref_text or "",
                        )
                    else:
                        # Voice design path
                        wavs, sr = self.model.generate_voice_design(
                            text=chunk,
                            language=req.language,
                            instruct=req.instruct or "",
                        )
                    all_audio.append(wavs[0])

                    # 4. Insert a short pause between chunks
                    pause = np.zeros(int(sr * CHUNK_PAUSE_MS / 1000), dtype=np.float32)
                    all_audio.append(pause)

                # 5. Concatenate, trimming the final pause
                if len(all_audio) > 1:
                    combined = np.concatenate(all_audio[:-1])
                else:
                    combined = np.concatenate(all_audio)

                # 6. Encode to WAV
                buf = _io.BytesIO()
                sf.write(buf, combined, sr, format="WAV")
                return Response(buf.getvalue(), media_type="audio/wav")

            finally:
                if tmp_path:
                    os.unlink(tmp_path)

        return api
