# app/main.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response, Request
from fastapi.responses import PlainTextResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import io, os, re, json, base64

from PIL import Image

# Import core library (đặt sẵn trong app/)
from . import image_to_text_full_v3 as lib

# ===== App & CORS =====
app = FastAPI(title="Image<->Text Encoder (v3)", version="1.1.0", docs_url="/docs", redoc_url="/redoc")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# ===== Config nhỏ =====
IMG_MIMES = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/webp": "webp",
    "image/bmp": "bmp",
    "image/gif": "gif",
    "image/tiff": "tiff",
    "image/x-tiff": "tiff",
}
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "15"))

INDEX_HTML = """
<!doctype html><html><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Image ↔ Text (v3)</title>
<style>
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial;margin:2rem;max-width:900px}
fieldset{margin-bottom:1.5rem;border:1px solid #ddd;padding:1rem;border-radius:8px}
.row{display:flex;gap:1rem;align-items:center;flex-wrap:wrap}
button{padding:.6rem 1rem;border-radius:8px;border:1px solid #ccc;background:#fafafa;cursor:pointer}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.box{padding:1rem;background:#f8f8ff;border:1px dashed #bbb;border-radius:8px}
</style></head><body>
<h1>Image ↔ Text (v3)</h1>
<p>Encode image → text (lossless / lossy-algo / lossy-nlp) & decode text → image. Or use <a href="/docs" target="_blank">OpenAPI</a>.</p>

<fieldset>
  <legend><b>Encode Image → Text</b></legend>
  <form id="encForm" class="row">
    <input type="file" name="file" accept="image/*" required />
    <label>Mode
      <select name="mode">
        <option>lossless</option>
        <option selected>lossy-algo</option>
        <option>lossy-nlp</option>
      </select>
    </label>
    <label>max_side <input name="max_side" type="number" value="512" class="mono"/></label>
    <label>palette_size <input name="palette_size" type="number" value="32" class="mono"/></label>
    <label>resample
      <select name="resample"><option>nearest</option><option>bilinear</option><option selected>bicubic</option><option>lanczos</option></select>
    </label>
    <label><input name="dither" type="checkbox"/> dither</label>
    <label><input name="lock_dims" type="checkbox"/> lock_dims</label>
    <label><input name="preserve_dims" type="checkbox"/> preserve_dims</label>
    <label>target_short_side <input name="target_short_side" type="number" value="512" class="mono"/></label>
    <label>palette_probe <input name="palette_probe" type="number" value="8" class="mono"/></label>
    <button type="submit">Encode</button>
  </form>
  <div class="box"><pre id="encOut" class="mono" style="white-space:pre-wrap;max-height:300px;overflow:auto"></pre></div>
</fieldset>

<fieldset>
  <legend><b>Decode Text → Image</b></legend>
  <form id="decForm" class="row">
    <input type="file" name="file" accept=".txt" required />
    <button type="submit">Decode</button>
  </form>
  <div class="box" id="decOut"></div>
</fieldset>

<script>
const encForm = document.getElementById('encForm');
const encOut = document.getElementById('encOut');
const decForm = document.getElementById('decForm');
const decOut = document.getElementById('decOut');

encForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  encOut.textContent = 'Encoding...';
  const fd = new FormData(encForm);
  const resp = await fetch('/encode', { method: 'POST', body: fd });
  if (!resp.ok) { encOut.textContent = 'Error: '+await resp.text(); return; }
  const txt = await resp.text();
  encOut.textContent = txt.slice(0, 100000);
});

decForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  decOut.textContent = 'Decoding...';
  const fd = new FormData(decForm);
  const resp = await fetch('/decode', { method: 'POST', body: fd });
  if (!resp.ok) { decOut.textContent = 'Error: '+await resp.text(); return; }
  const blob = await resp.blob();
  const url = URL.createObjectURL(blob);
  decOut.innerHTML = '<img style="max-width:100%" src="'+url+'"/><br/><a download="decoded.png" href="'+url+'">Download decoded.png</a>';
});
</script>
</body></html>
"""

# ===== Basic routes =====
@app.head("/")
async def index_head():
    return Response(status_code=200)

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(INDEX_HTML)

@app.get("/health")
async def health():
    return {"status": "ok"}

def _sniff_text_mode(txt: str) -> str:
    head = "\n".join(txt.strip().splitlines()[:4])
    if "LOSSLESS MANIFEST v2" in head:
        return "lossless"
    if "LOSSY-ALGO MANIFEST v2" in head:
        return "lossy-algo"
    if "LOSSY-NLP DESCRIPTION v2" in head:
        return "lossy-nlp"
    if re.search(r'"schema"\s*:\s*"LOSSY-IMAGE-DESCRIPTION v2"', head):
        return "lossy-nlp"
    raise HTTPException(status_code=400, detail="Unrecognized text schema. Expected v2 headers.")

# ===== Multipart endpoints (giữ nguyên) =====
@app.post("/encode", response_class=PlainTextResponse)
async def encode_image(
    file: UploadFile = File(...),
    mode: str = Form("lossy-algo"),
    lock_dims: bool = Form(False),
    max_side: int = Form(512),
    palette_size: int = Form(32),
    resample: str = Form("bicubic"),
    dither: bool = Form(False),
    preserve_dims: bool = Form(False),
    target_short_side: int = Form(512),
    palette_probe: int = Form(8),
):
    # size limit
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large")

    if file.content_type not in IMG_MIMES:
        raise HTTPException(status_code=400, detail=f"Unsupported image mime: {file.content_type}")
    try:
        Image.open(io.BytesIO(data)).load()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image data")

    tmp_path = f"/tmp/upload.{IMG_MIMES[file.content_type]}"
    with open(tmp_path, "wb") as f:
        f.write(data)

    if mode == "lossless":
        txt, _ = lib.encode_lossless_to_manifest(tmp_path)
    elif mode == "lossy-algo":
        txt, _ = lib.encode_lossy_algo_to_text(tmp_path, lock_dims=lock_dims, max_side=max_side,
                                               palette_size=palette_size, resample=resample, dither=dither)
    elif mode == "lossy-nlp":
        txt, _ = lib.encode_lossy_nlp_to_text(tmp_path, preserve_dims=preserve_dims,
                                              target_short_side=target_short_side, palette_probe=palette_probe)
    else:
        raise HTTPException(status_code=400, detail="Unknown mode")

    return PlainTextResponse(content=txt, headers={"Content-Disposition": 'attachment; filename="manifest_v3.txt"'})

@app.post("/decode")
async def decode_text_multipart(file: UploadFile = File(...)):
    txt_bytes = await file.read()
    if len(txt_bytes) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large")
    try:
        txt = txt_bytes.decode("utf-8", errors="replace")
    except Exception:
        raise HTTPException(status_code=400, detail="Body must be UTF-8 text")

    mode = _sniff_text_mode(txt)
    if mode == "lossless":
        out_path, _ = lib.decode_lossless_manifest_to_image(txt, output_dir="/tmp")
    elif mode == "lossy-algo":
        out_path, _ = lib.decode_lossy_algo_text_to_image(txt, output_dir="/tmp")
    elif mode == "lossy-nlp":
        out_path, _ = lib.decode_lossy_nlp_text_to_proxy_image(txt, output_dir="/tmp")

    img_bytes = open(out_path, "rb").read()
    return Response(content=img_bytes, media_type="image/png", headers={"Content-Disposition": 'inline; filename="decoded.png"'})

# ===== Binary & Text body endpoints (dành cho GPT Actions) =====
@app.post("/encode_octet", response_class=PlainTextResponse)
async def encode_octet(
    request: Request,
    mode: str = "lossy-algo",
    lock_dims: bool = False,
    max_side: int = 512,
    palette_size: int = 32,
    resample: str = "bicubic",
    dither: bool = False,
    preserve_dims: bool = False,
    target_short_side: int = 512,
    palette_probe: int = 8,
):
    data = await request.body()
    if not data:
        raise HTTPException(status_code=400, detail="Empty body")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large")

    try:
        Image.open(io.BytesIO(data)).load()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image data")

    tmp_path = "/tmp/upload.png"
    with open(tmp_path, "wb") as f:
        f.write(data)

    if mode == "lossless":
        txt, _ = lib.encode_lossless_to_manifest(tmp_path)
    elif mode == "lossy-algo":
        txt, _ = lib.encode_lossy_algo_to_text(tmp_path, lock_dims=lock_dims, max_side=max_side,
                                               palette_size=palette_size, resample=resample, dither=dither)
    elif mode == "lossy-nlp":
        txt, _ = lib.encode_lossy_nlp_to_text(tmp_path, preserve_dims=preserve_dims,
                                              target_short_side=target_short_side, palette_probe=palette_probe)
    else:
        raise HTTPException(status_code=400, detail="Unknown mode")

    return PlainTextResponse(content=txt, headers={"Content-Disposition": 'attachment; filename="manifest_v3.txt"'})

@app.post("/decode_text")
async def decode_text_raw(request: Request):
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Empty body")
    if len(body) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large")

    try:
        txt = body.decode("utf-8")
    except Exception:
        raise HTTPException(status_code=400, detail="Body must be UTF-8 text")

    mode = _sniff_text_mode(txt)
    if mode == "lossless":
        out_path, _ = lib.decode_lossless_manifest_to_image(txt, output_dir="/tmp")
    elif mode == "lossy-algo":
        out_path, _ = lib.decode_lossy_algo_text_to_image(txt, output_dir="/tmp")
    elif mode == "lossy-nlp":
        out_path, _ = lib.decode_lossy_nlp_text_to_proxy_image(txt, output_dir="/tmp")

    img_bytes = open(out_path, "rb").read()
    return Response(content=img_bytes, media_type="image/png", headers={"Content-Disposition": 'inline; filename="decoded.png"'})
