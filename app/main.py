
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response
from fastapi.responses import PlainTextResponse, StreamingResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import io, os, re, json
from PIL import Image
import base64

# Import internal library (placed alongside this app)
from . import image_to_text_full_v3 as lib

# ==== ADD THIS NEAR THE TOP ====
from fastapi import Request
import io

# ==== ADD THESE NEW ENDPOINTS ====

@app.post("/encode_octet", response_class=PlainTextResponse)
async def encode_octet(
    request: Request,
    mode: str = "lossy-algo",          # query: lossless | lossy-algo | lossy-nlp
    lock_dims: bool = False,
    max_side: int = 512,
    palette_size: int = 32,
    resample: str = "bicubic",
    dither: bool = False,
    preserve_dims: bool = False,
    target_short_side: int = 512,
    palette_probe: int = 8,
):
    """
    Accept raw image bytes in the request body (application/octet-stream).
    This avoids multipart/form-data so it's friendly to various connectors.
    """
    data = await request.body()
    if not data:
        raise HTTPException(status_code=400, detail="Empty body")

    # Validate image
    try:
        from PIL import Image
        Image.open(io.BytesIO(data)).load()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image data")

    # Save temp PNG for the existing library (expects a path)
    tmp_path = "/tmp/upload.png"
    with open(tmp_path, "wb") as f:
        f.write(data)

    if mode == "lossless":
        txt, _ = lib.encode_lossless_to_manifest(tmp_path)
    elif mode == "lossy-algo":
        txt, _ = lib.encode_lossy_algo_to_text(
            tmp_path,
            lock_dims=lock_dims,
            max_side=max_side,
            palette_size=palette_size,
            resample=resample,
            dither=dither,
        )
    elif mode == "lossy-nlp":
        txt, _ = lib.encode_lossy_nlp_to_text(
            tmp_path,
            preserve_dims=preserve_dims,
            target_short_side=target_short_side,
            palette_probe=palette_probe,
        )
    else:
        raise HTTPException(status_code=400, detail="Unknown mode")

    return PlainTextResponse(content=txt, headers={"Content-Disposition": 'attachment; filename="manifest_v3.txt"'})


@app.post("/decode_text")
async def decode_text(request: Request):
    """
    Accept plain text manifest in the request body (text/plain).
    Returns image/png.
    """
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Empty body")

    # Try UTF-8
    try:
        txt = body.decode("utf-8")
    except Exception:
        raise HTTPException(status_code=400, detail="Body must be UTF-8 text")

    # Sniff schema (reuse your helper)
    mode = _sniff_text_mode(txt)
    if mode == "lossless":
        out_path, _ = lib.decode_lossless_manifest_to_image(txt, output_dir="/tmp")
    elif mode == "lossy-algo":
        out_path, _ = lib.decode_lossy_algo_text_to_image(txt, output_dir="/tmp")
    elif mode == "lossy-nlp":
        out_path, _ = lib.decode_lossy_nlp_text_to_proxy_image(txt, output_dir="/tmp")
    else:
        raise HTTPException(status_code=400, detail="Unknown mode")

    img_bytes = open(out_path, "rb").read()
    return Response(content=img_bytes, media_type="image/png", headers={"Content-Disposition": 'inline; filename="decoded.png"'})

app = FastAPI(title="Image<->Text Encoder (v3)", version="1.0.0", docs_url="/docs", redoc_url="/redoc")

# CORS (allow all by default; tighten in production if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

INDEX_HTML = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Image ↔ Text (v3)</title>
    <style>
      body{font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 2rem; max-width: 900px}
      h1{margin-top:0}
      fieldset{margin-bottom:1.5rem; border:1px solid #ddd; padding:1rem; border-radius:8px}
      .row{display:flex; gap:1rem; align-items:center; flex-wrap:wrap}
      input[type="file"]{padding:.3rem}
      button{padding:.6rem 1rem; border-radius:8px; border:1px solid #ccc; background:#fafafa; cursor:pointer}
      code{background:#f3f3f3; padding:0 .4rem; border-radius:4px}
      .hint{color:#666}
      .box{padding:1rem; background:#f8f8ff; border:1px dashed #bbb; border-radius:8px}
      .mono{font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace}
    </style>
  </head>
  <body>
    <h1>Image ↔ Text (v3)</h1>
    <p class="hint">Encode an image to text (manifest) or decode a text manifest back to an image. Try the forms below or the <a href="/docs" target="_blank">OpenAPI</a>.</p>

    <fieldset>
      <legend><strong>Encode Image → Text</strong></legend>
      <form id="encForm" class="row">
        <div>
          <input type="file" name="file" accept="image/*" required />
        </div>
        <div>
          <label>Mode:
            <select name="mode">
              <option value="lossless">lossless</option>
              <option value="lossy-algo" selected>lossy-algo</option>
              <option value="lossy-nlp">lossy-nlp</option>
            </select>
          </label>
        </div>
        <div id="lossyAlgoParams" class="row">
          <label>max_side <input name="max_side" type="number" value="512" min="16" class="mono" /></label>
          <label>palette_size <input name="palette_size" type="number" value="32" min="2" class="mono" /></label>
          <label>resample 
            <select name="resample">
              <option>nearest</option><option>bilinear</option><option selected>bicubic</option><option>lanczos</option>
            </select>
          </label>
          <label><input name="dither" type="checkbox" /> dither</label>
        </div>
        <div id="lossyNlpParams" class="row" style="display:none">
          <label>target_short_side <input name="target_short_side" type="number" value="512" min="16" class="mono" /></label>
          <label>palette_probe <input name="palette_probe" type="number" value="8" min="2" class="mono" /></label>
          <label><input name="preserve_dims" type="checkbox" /> preserve_dims</label>
        </div>
        <div><button type="submit">Encode</button></div>
      </form>
      <div class="box">
        <div><strong>Result:</strong></div>
        <pre id="encOut" class="mono" style="white-space:pre-wrap; max-height:300px; overflow:auto"></pre>
        <div id="encDownload"></div>
      </div>
    </fieldset>

    <fieldset>
      <legend><strong>Decode Text → Image</strong></legend>
      <form id="decForm" class="row">
        <div>
          <input type="file" name="file" accept=".txt" required />
        </div>
        <div><button type="submit">Decode</button></div>
      </form>
      <div class="box">
        <div><strong>Result:</strong></div>
        <div id="decOut"></div>
      </div>
    </fieldset>

    <script>
      const encForm = document.getElementById('encForm');
      const encOut = document.getElementById('encOut');
      const encDl = document.getElementById('encDownload');
      const decForm = document.getElementById('decForm');
      const decOut = document.getElementById('decOut');
      const lossyAlgoParams = document.getElementById('lossyAlgoParams');
      const lossyNlpParams = document.getElementById('lossyNlpParams');

      encForm.mode.addEventListener('change', () => {
        const m = encForm.mode.value;
        lossyAlgoParams.style.display = (m === 'lossy-algo') ? 'flex' : 'none';
        lossyNlpParams.style.display = (m === 'lossy-nlp') ? 'flex' : 'none';
      });

      encForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        encOut.textContent = 'Encoding...';
        encDl.innerHTML = '';
        const fd = new FormData(encForm);
        const resp = await fetch('/encode', { method: 'POST', body: fd });
        if (!resp.ok) { encOut.textContent = 'Error: ' + await resp.text(); return; }
        const txt = await resp.text();
        encOut.textContent = txt.slice(0, 100000); // limit preview
        const blob = new Blob([txt], {type:'text/plain'});
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = 'manifest_v3.txt';
        a.textContent = 'Download manifest_v3.txt';
        encDl.appendChild(a);
      });

      decForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        decOut.textContent = 'Decoding...';
        const fd = new FormData(decForm);
        const resp = await fetch('/decode', { method: 'POST', body: fd });
        if (!resp.ok) { decOut.textContent = 'Error: ' + await resp.text(); return; }
        const blob = await resp.blob();
        const url = URL.createObjectURL(blob);
        const img = document.createElement('img');
        img.src = url;
        img.style.maxWidth = '100%';
        decOut.innerHTML = '';
        decOut.appendChild(img);
        const a = document.createElement('a');
        a.href = url; a.download = 'decoded.png'; a.textContent = 'Download decoded.png';
        decOut.appendChild(document.createElement('br'));
        decOut.appendChild(a);
      });
    </script>
  </body>
</html>
"""

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
    if re.search(r'\"schema\"\s*:\s*\"LOSSY-IMAGE-DESCRIPTION v2\"', json.dumps(head)):
        return "lossy-nlp"
    raise HTTPException(status_code=400, detail="Unrecognized text schema. Expected v2 headers.")

@app.post("/encode", response_class=PlainTextResponse)
async def encode_image(
    file: UploadFile = File(...),
    mode: str = Form("lossy-algo"),  # lossless | lossy-algo | lossy-nlp
    # lossy-algo params
    lock_dims: bool = Form(False),
    max_side: int = Form(512),
    palette_size: int = Form(32),
    resample: str = Form("bicubic"),
    dither: bool = Form(False),
    # lossy-nlp params
    preserve_dims: bool = Form(False),
    target_short_side: int = Form(512),
    palette_probe: int = Form(8),
):
    if file.content_type not in IMG_MIMES:
        raise HTTPException(status_code=400, detail=f"Unsupported image mime: {file.content_type}")
    data = await file.read()
    try:
        # verify loadable
        Image.open(io.BytesIO(data)).load()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image data")

    # Save temp (lib expects a file path interface)
    tmp_path = f"/tmp/upload.{IMG_MIMES[file.content_type]}"
    with open(tmp_path, "wb") as f:
        f.write(data)

    if mode == "lossless":
        txt, outname = lib.encode_lossless_to_manifest(tmp_path)
    elif mode == "lossy-algo":
        txt, outname = lib.encode_lossy_algo_to_text(
            tmp_path, lock_dims=lock_dims, max_side=max_side, palette_size=palette_size, resample=resample, dither=dither
        )
    elif mode == "lossy-nlp":
        txt, outname = lib.encode_lossy_nlp_to_text(
            tmp_path, preserve_dims=preserve_dims, target_short_side=target_short_side, palette_probe=palette_probe
        )
    else:
        raise HTTPException(status_code=400, detail="Unknown mode")

    headers = {"Content-Disposition": 'attachment; filename="manifest_v3.txt"'}
    return PlainTextResponse(content=txt, headers=headers)

@app.post("/decode")
async def decode_text(file: UploadFile = File(...)):
    if file.content_type not in ("text/plain", "application/json", "application/octet-stream"):
        raise HTTPException(status_code=400, detail=f"Unsupported text mime: {file.content_type}")
    txt = (await file.read()).decode("utf-8", errors="replace")
    mode = _sniff_text_mode(txt)
    if mode == "lossless":
        out_path, meta = lib.decode_lossless_manifest_to_image(txt, output_dir="/tmp")
    elif mode == "lossy-algo":
        out_path, meta = lib.decode_lossy_algo_text_to_image(txt, output_dir="/tmp")
    elif mode == "lossy-nlp":
        out_path, meta = lib.decode_lossy_nlp_text_to_proxy_image(txt, output_dir="/tmp")
    else:
        raise HTTPException(status_code=400, detail="Unknown mode")

    # Return as PNG regardless (library outputs PNG paths for lossy modes; lossless returns original bytes but we saved tmp)
    img_bytes = open(out_path, "rb").read()
    return Response(content=img_bytes, media_type="image/png", headers={"Content-Disposition": 'inline; filename="decoded.png"'})
