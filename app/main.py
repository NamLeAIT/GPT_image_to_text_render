# app/main.py
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io, os, re

# === import core lib (đã có trong app/) ===
from . import image_to_text_full_v3 as lib

# === create app FIRST ===
app = FastAPI(title="Image<->Text Encoder (v3-min)", version="1.1.0", docs_url="/docs", redoc_url="/redoc")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "15"))

@app.get("/health")
async def health():
    return {"status": "ok"}

def _sniff_text_mode(txt: str) -> str:
    head = "\n".join(txt.strip().splitlines()[:4])
    if "LOSSLESS MANIFEST v2" in head: return "lossless"
    if "LOSSY-ALGO MANIFEST v2" in head: return "lossy-algo"
    if "LOSSY-NLP DESCRIPTION v2" in head: return "lossy-nlp"
    if re.search(r'"schema"\s*:\s*"LOSSY-IMAGE-DESCRIPTION v2"', head): return "lossy-nlp"
    raise HTTPException(status_code=400, detail="Unrecognized text schema. Expected v2 headers.")

# ====== endpoint cho GPT Actions: body = ảnh (octet-stream) ======
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

    # validate image
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

# ====== endpoint cho GPT Actions: body = manifest text ======
@app.post("/decode_text")
async def decode_text(request: Request):
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
    else:
        raise HTTPException(status_code=400, detail="Unknown mode")

    img_bytes = open(out_path, "rb").read()
    return Response(content=img_bytes, media_type="image/png", headers={"Content-Disposition": 'inline; filename="decoded.png"'})
