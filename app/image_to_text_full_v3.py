# app/image_to_text_full_v3.py  (FALLBACK)
import base64, io, json, hashlib
from PIL import Image

def encode_lossless_to_manifest(path):
    with open(path, "rb") as f: b = f.read()
    b64 = base64.b64encode(b).decode("ascii")
    h = hashlib.sha256(b).hexdigest()
    txt = "LOSSLESS MANIFEST v2\nsha256:"+h+"\n\n"+b64
    return txt, "lossless_v3.txt"

def decode_lossless_manifest_to_image(txt, output_dir="."):
    lines = txt.splitlines()
    try:
        idx = lines.index("")
    except ValueError:
        idx = 2
    b64 = "\n".join(lines[idx+1:])
    raw = base64.b64decode(b64)
    out = output_dir.rstrip("/") + "/decoded_lossless.bin"
    with open(out, "wb") as f: f.write(raw)
    return out, {}

def encode_lossy_algo_to_text(path, **kw):
    im = Image.open(path).convert("RGBA")
    bio = io.BytesIO()
    im.save(bio, format="PNG")
    b64 = base64.b64encode(bio.getvalue()).decode("ascii")
    payload = {"schema":"FAKE-LOSSY-ALGO v2","png_b64":b64,"size":im.size}
    txt = "LOSSY-ALGO MANIFEST v2\n\n"+json.dumps(payload)
    return txt, "lossy_algo_v3.txt"

def decode_lossy_algo_text_to_image(txt, output_dir=".", out_name=None):
    j = json.loads("\n".join(txt.splitlines()[2:]))
    raw = base64.b64decode(j["png_b64"])
    out = output_dir.rstrip("/") + "/" + (out_name or "decoded_algo.png")
    with open(out, "wb") as f: f.write(raw)
    return out, {}

def encode_lossy_nlp_to_text(path, **kw):
    im = Image.open(path).convert("RGB")
    payload = {"schema":"LOSSY-IMAGE-DESCRIPTION v2","hint":"fallback","size":im.size}
    txt = "LOSSY-NLP DESCRIPTION v2\n\n"+json.dumps(payload)
    return txt, "lossy_nlp_v3.txt"

def decode_lossy_nlp_text_to_proxy_image(txt, output_dir=".", out_name=None):
    im = Image.new("RGB",(256,256),(200,200,200))
    out = output_dir.rstrip("/") + "/" + (out_name or "decoded_proxy.png")
    im.save(out, "PNG")
    return out, {}
