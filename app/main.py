# app/main.py
from fastapi import FastAPI, HTTPException, Request, Response, UploadFile, File, Form, Body
from fastapi.responses import PlainTextResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageFile
import io, os, re, base64
import httpx

# chấp nhận file ảnh hơi lỗi
ImageFile.LOAD_TRUNCATED_IMAGES = True

# import core library của bạn (đặt trong app/)
from . import image_to_text_full_v3 as lib

app = FastAPI(
    title="Image<->Text Encoder (v4)",
    version="1.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS mở—siết lại trong production nếu cần
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "15"))
API_KEY = os.getenv("API_KEY")  # nếu đặt, sẽ yêu cầu header x-api-key

def _enforce_api_key(request: Request):
    if API_KEY and request.headers.get("x-api-key") != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.head("/")
async def head_root():
    return Response(status_code=200)

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse("<h1>Image ↔ Text Encoder v4</h1><p>Use <a href='/docs'>/docs</a>.</p>")

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

def _validate_and_save_image_bytes(img_bytes: bytes, tmp_path: str = "/tmp/upload.png") -> str:
    if not img_bytes:
        raise HTTPException(status_code=400, detail="Empty body")
    if len(img_bytes) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large")
    try:
        Image.open(io.BytesIO(img_bytes)).load()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image data")
    with open(tmp_path, "wb") as f:
        f.write(img_bytes)
    return tmp_path

# -------------------- JSON-first (khuyên dùng cho GPT Actions) --------------------

@app.post("/encode_json", response_class=PlainTextResponse)
async def encode_json(
    request: Request,
    payload: dict = Body(..., description="Provide either image_b64 or image_url"),
):
    _enforce_api_key(request)

    b64 = payload.get("image_b64")
    url = payload.get("image_url")
    mode = payload.get("mode", "lossy-algo")
    lock_dims = bool(payload.get("lock_dims", False))
    max_side = int(payload.get("max_side", 512))
    palette_size = int(payload.get("palette_size", 32))
    resample = payload.get("resample", "bicubic")
    dither = bool(payload.get("dither", False))
    preserve_dims = bool(payload.get("preserve_dims", False))
    target_short_side = int(payload.get("target_short_side", 512))
    palette_probe = int(payload.get("palette_probe", 8))

    img_bytes = None
    if b64:
        if "," in b64:  # hỗ trợ data:uri
            b64 = b64.split(",", 1)[1]
        try:
            img_bytes = base64.b64decode(b64, validate=True)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image")
    elif url:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(url)
                r.raise_for_status()
                img_bytes = r.content
        except Exception:
            raise HTTPException(status_code=400, detail="Cannot fetch image_url")
    else:
        raise HTTPException(status_code=400, detail="Provide image_b64 or image_url")

    tmp_path = _validate_and_save_image_bytes(img_bytes)

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

    return PlainTextResponse(
        content=txt,
        headers={"Content-Disposition": "attachment; filename=manifest_v3.txt"},
    )

@app.post("/decode_json")
async def decode_json(request: Request, payload: dict = Body(...)):
    _enforce_api_key(request)

    txt = payload.get("manifest_text")
    url = payload.get("manifest_url")

    if not txt and url:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(url)
                r.raise_for_status()
                txt = r.text
        except Exception:
            raise HTTPException(status_code=400, detail="Cannot fetch manifest_url")

    if not txt:
        raise HTTPException(status_code=400, detail="Provide manifest_text or manifest_url")

    mode = _sniff_text_mode(txt)
    if mode == "lossless":
        out_path, _ = lib.decode_lossless_manifest_to_image(txt, output_dir="/tmp")
    elif mode == "lossy-algo":
        out_path, _ = lib.decode_lossy_algo_text_to_image(txt, output_dir="/tmp")
    elif mode == "lossy-nlp":
        out_path, _ = lib.decode_lossy_nlp_text_to_proxy_image(txt, output_dir="/tmp")

    with open(out_path, "rb") as f:
        png_b64 = base64.b64encode(f.read()).decode("ascii")

    return JSONResponse({"image_png_b64": png_b64})

# -------------------- Fallbacks (multipart / octet / text) --------------------

@app.post("/encode", response_class=PlainTextResponse)
async def encode_multipart(
    request: Request,
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
    _enforce_api_key(request)

    data = await file.read()
    tmp_path = _validate_and_save_image_bytes(data, tmp_path="/tmp/upload")

    # cố gắng thêm phần mở rộng theo content-type (nếu có)
    if file.content_type and "/" in file.content_type:
        ext = file.content_type.split("/")[-1]
        if ext in ("png", "jpeg", "jpg", "webp", "bmp", "gif", "tiff"):
            tmp2 = f"/tmp/upload.{ext if ext != 'jpeg' else 'jpg'}"
            os.replace(tmp_path, tmp2)
            tmp_path = tmp2

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

    return PlainTextResponse(
        content=txt,
        headers={"Content-Disposition": "attachment; filename=manifest_v3.txt"},
    )

@app.post("/decode")
async def decode_multipart(request: Request, file: UploadFile = File(...)):
    _enforce_api_key(request)

    txt = (await file.read()).decode("utf-8", errors="replace")
    mode = _sniff_text_mode(txt)

    if mode == "lossless":
        out_path, _ = lib.decode_lossless_manifest_to_image(txt, output_dir="/tmp")
    elif mode == "lossy-algo":
        out_path, _ = lib.decode_lossy_algo_text_to_image(txt, output_dir="/tmp")
    elif mode == "lossy-nlp":
        out_path, _ = lib.decode_lossy_nlp_text_to_proxy_image(txt, output_dir="/tmp")

    with open(out_path, "rb") as f:
        return Response(
            content=f.read(),
            media_type="image/png",
            headers={"Content-Disposition": "inline; filename=decoded.png"},
        )

@app.post("/encode_octet", response_class=PlainTextResponse)
async def encode_octet(request: Request):
    _enforce_api_key(request)

    params = dict(request.query_params)
    mode = params.get("mode", "lossy-algo")
    lock_dims = params.get("lock_dims", "false").lower() == "true"
    max_side = int(params.get("max_side", 512))
    palette_size = int(params.get("palette_size", 32))
    resample = params.get("resample", "bicubic")
    dither = params.get("dither", "false").lower() == "true"
    preserve_dims = params.get("preserve_dims", "false").lower() == "true"
    target_short_side = int(params.get("target_short_side", 512))
    palette_probe = int(params.get("palette_probe", 8))

    data = await request.body()
    tmp_path = _validate_and_save_image_bytes(data)

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

    return PlainTextResponse(
        content=txt,
        headers={"Content-Disposition": "attachment; filename=manifest_v3.txt"},
    )

@app.post("/decode_text")
async def decode_text(request: Request):
    _enforce_api_key(request)

    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Empty body")
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

    with open(out_path, "rb") as f:
        return Response(
            content=f.read(),
            media_type="image/png",
            headers={"Content-Disposition": "inline; filename=decoded.png"},
        )
