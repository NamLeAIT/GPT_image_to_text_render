
# image_to_text_full_v3.py
# Same functionality as v2, but with naming clarified:
# - Lossless -> "LOSSLESS MANIFEST v2"
# - Lossy-algo -> "LOSSY-ALGO MANIFEST v2"
# - Lossy-NLP -> "LOSSY-NLP DESCRIPTION v2"
# Output filenames end with *_v3.txt

import base64, hashlib, io, json, mimetypes, os, re, textwrap
from typing import List, Tuple, Dict, Any
from PIL import Image, ImageFilter, ImageDraw, ImageStat, ExifTags

# ========================
# Common helpers
# ========================

def _sha256(b: bytes) -> str: 
    return hashlib.sha256(b).hexdigest()

def _md5(b: bytes) -> str: 
    return hashlib.md5(b).hexdigest()

def _wrap(s: str, w: int = 76) -> str:
    import textwrap as _tw
    return "\n".join(_tw.wrap(s, w))

def _mime(path: str) -> str:
    m, _ = mimetypes.guess_type(path)
    return m or "application/octet-stream"

def _exif_dict(img: Image.Image) -> Dict[str, Any]:
    try:
        exif = img.getexif()
        if not exif:
            return {}
        tagmap = {ExifTags.TAGS.get(k, str(k)): v for k, v in exif.items()}
        return tagmap
    except Exception:
        return {}

def _exif_orientation(tagmap: Dict[str, Any]) -> int:
    val = tagmap.get("Orientation")
    try:
        return int(val)
    except Exception:
        return None

def _icc_b64(img: Image.Image) -> str:
    prof = img.info.get("icc_profile")
    if not prof:
        return None
    if isinstance(prof, bytes):
        return base64.b64encode(prof).decode("ascii")
    try:
        return base64.b64encode(prof.tobytes()).decode("ascii")
    except Exception:
        return None

def _image_meta(img: Image.Image, filename: str, mime: str) -> Dict[str, Any]:
    mode_to_bitdepth = {
        "1": 1, "L": 8, "P": 8, "RGB": 8, "RGBA": 8, "CMYK": 8,
        "I;16": 16, "I;16B": 16, "I": 32, "F": 32
    }
    bit_depth = mode_to_bitdepth.get(img.mode, 8)
    dpi = img.info.get("dpi")
    xres = yres = None
    if isinstance(dpi, tuple) and len(dpi) >= 2:
        xres, yres = float(dpi[0]), float(dpi[1])
    has_alpha = "A" in img.getbands()
    exif_map = _exif_dict(img)
    return {
        "filename": os.path.basename(filename),
        "mime_type": mime,
        "width": img.width,
        "height": img.height,
        "color_mode": img.mode,
        "bit_depth": bit_depth,
        "dpi_x": xres,
        "dpi_y": yres,
        "has_alpha": has_alpha,
        "exif_orientation": _exif_orientation(exif_map),
        "software": img.info.get("software"),
    }

# ============================================================
# 1) LOSSLESS (bit-perfect) — v2 header, v3 package
# ============================================================

def encode_lossless_to_manifest(
    image_path: str,
    source: str = "user_upload",
    chunk_size: int = 262144,  # 256 KiB
    line_width: int = 76
):
    with open(image_path, "rb") as f:
        blob = f.read()
    with Image.open(io.BytesIO(blob)) as img:
        mime = _mime(image_path)
        meta = _image_meta(img, image_path, mime)
        exif_bytes = img.info.get("exif")
        icc_b64 = _icc_b64(img)

    sha = _sha256(blob)
    md5 = _md5(blob)
    size = len(blob)

    b64 = base64.b64encode(blob).decode("ascii")
    chunks = [b64[i:i + chunk_size] for i in range(0, len(b64), chunk_size)]

    header = [
        "LOSSLESS MANIFEST v2",
        f"source: {source}",
        f"filename: {meta['filename']}",
        f"mime_type: {meta['mime_type']}",
        f"filesize_bytes: {size}",
        f"width: {meta['width']}",
        f"height: {meta['height']}",
        f"color_mode: {meta['color_mode']}",
        f"bit_depth: {meta['bit_depth']}",
        f"dpi_x: {meta['dpi_x'] if meta['dpi_x'] is not None else 'null'}",
        f"dpi_y: {meta['dpi_y'] if meta['dpi_y'] is not None else 'null'}",
        f"has_alpha: {str(meta['has_alpha']).lower()}",
        f"exif_orientation: {meta['exif_orientation'] if meta['exif_orientation'] is not None else 'null'}",
        f"software: {meta['software'] if meta['software'] else 'null'}",
        f"sha256: {sha}",
        f"md5: {md5}",
        f"icc_profile_b64_present: {str(bool(icc_b64)).lower()}",
        f"exif_b64_present: {str(bool(exif_bytes)).lower()}",
        "",
        f"chunk_count: {len(chunks)}",
        "chunk_encoding: base64",
        f"chunk_line_width: {line_width}",
        ""
    ]

    body = []
    for i, c in enumerate(chunks, 1):
        body += [f"CHUNK {i}/{len(chunks)}", _wrap(c, line_width), "END CHUNK", ""]

    extras = []
    if icc_b64:
        extras += ["ICC_PROFILE_START", _wrap(icc_b64, line_width), "ICC_PROFILE_END", ""]
    if exif_bytes:
        exif_b64 = base64.b64encode(exif_bytes).decode("ascii")
        extras += ["EXIF_START", _wrap(exif_b64, line_width), "EXIF_END", ""]

    instr = [
        "RECONSTRUCT_INSTRUCTIONS:",
        "1) Concatenate CHUNKs in order.",
        "2) Base64-decode to bytes.",
        "3) Verify SHA-256 matches \"sha256\" (md5 provided for convenience).",
        "4) Save bytes as \"filename\" with \"mime_type\". (All metadata is preserved because bytes are original.)",
        "EOF"
    ]

    manifest_text = "\n".join(header + body + extras + instr)
    name = os.path.splitext(meta['filename'])[0] + ".lossless_manifest_v3.txt"
    return manifest_text, name

def decode_lossless_manifest_to_image(manifest_text: str, output_dir: str = "."):
    def grab(key: str):
        m = re.search(rf"^{re.escape(key)}:\s*(.+)$", manifest_text, flags=re.M)
        return m.group(1).strip() if m else None

    filename = grab("filename")
    mime = grab("mime_type")
    sha_expected = grab("sha256")
    chunk_count = int(grab("chunk_count") or "0")

    blocks = re.findall(r"CHUNK\s+\d+/\d+\s*\n(.*?)\nEND CHUNK", manifest_text, flags=re.S)
    if len(blocks) != chunk_count:
        raise ValueError(f"Found {len(blocks)} CHUNK blocks but header says {chunk_count}")

    b64 = "".join(x.replace("\n", "").strip() for x in blocks)
    blob = base64.b64decode(b64)
    if _sha256(blob) != sha_expected:
        raise ValueError("SHA-256 mismatch!")

    out_path = os.path.join(output_dir, filename)
    with open(out_path, "wb") as f:
        f.write(blob)
    return out_path, {"filename": filename, "mime_type": mime, "bytes": len(blob), "sha256": _sha256(blob)}

# ============================================================
# 2) LOSSY-ALGO (deterministic, non-NLP) — v2 header
# ============================================================

def _resample_lookup(name: str):
    name = name.lower()
    from PIL import Image as _I
    if name == "nearest": return _I.NEAREST
    if name == "bilinear": return _I.BILINEAR
    if name == "bicubic": return _I.BICUBIC
    if name == "lanczos": return _I.LANCZOS
    raise ValueError("Unknown resample method. Use: nearest|bilinear|bicubic|lanczos")

def _rle_encode(seq: List[int]) -> List[List[int]]:
    if not seq: return []
    out = []; prev = seq[0]; cnt = 1
    for v in seq[1:]:
        if v == prev and cnt < 2**31 - 1:
            cnt += 1
        else:
            out.append([cnt, prev]); prev = v; cnt = 1
    out.append([cnt, prev])
    return out

def _rle_decode(pairs: List[List[int]]) -> List[int]:
    out = []
    for cnt, val in pairs: 
        out.extend([val] * cnt)
    return out

def _palette_from_P(imgP, palette_size):
    pal = imgP.getpalette() or []
    trip = [tuple(pal[i:i+3]) for i in range(0, len(pal), 3)]
    return trip[:palette_size]

def encode_lossy_algo_to_text(
    image_path: str,
    source: str = "user_upload",
    lock_dims: bool = True,
    max_side: int = 128,
    palette_size: int = 32,
    resample: str = "bicubic",
    dither: bool = False
):
    with Image.open(image_path) as im0:
        has_alpha = "A" in im0.getbands()
        if has_alpha:
            alpha = list(im0.getchannel("A").getdata())
        img = im0.convert("RGB")
        ow, oh = img.width, img.height

        if lock_dims:
            nw, nh = ow, oh
            img_resampled = img.copy()
        else:
            scale = min(1.0, max_side / max(ow, oh)) if max(ow, oh) > max_side else 1.0
            nw, nh = max(1, int(round(ow * scale))), max(1, int(round(oh * scale)))
            img_resampled = img.resize((nw, nh), resample=_resample_lookup(resample))

        dither_flag = Image.FLOYDSTEINBERG if dither else Image.NONE
        imgP = img_resampled.quantize(colors=palette_size, method=Image.MEDIANCUT, dither=dither_flag)
        pal = _palette_from_P(imgP, palette_size)
        idxs = list(imgP.getdata())
        rle_idx = _rle_encode(idxs)

        alpha_payload = None
        if has_alpha:
            if not lock_dims:
                alpha_img = im0.getchannel("A").resize((nw, nh), resample=_resample_lookup(resample))
                alpha = list(alpha_img.getdata())
            alpha_payload = _rle_encode(alpha)

        act_palette = len([c for c in pal if c is not None])
        index_bits = max(1, (act_palette - 1).bit_length())

        payload = {
            "schema": "IMG-ALGO-LOSSY v2",
            "source": source,
            "original": {"width": ow, "height": oh, "mode": im0.mode, "has_alpha": has_alpha},
            "params": {
                "lock_dims": bool(lock_dims),
                "max_side": max_side,
                "palette_size": palette_size,
                "resample": resample,
                "dither": bool(dither)
            },
            "result_size": {"width": nw, "height": nh},
            "palette_rgb_hex": ["#{:02X}{:02X}{:02X}".format(r, g, b) for (r, g, b) in pal],
            "palette_size_actual": act_palette,
            "index_bit_depth": index_bits,
            "pixels_rle": rle_idx,
            "alpha_rle": alpha_payload
        }

        txt = "\n".join([
            "LOSSY-ALGO MANIFEST v2",
            "note: deterministic; rebuild matches this lossy image exactly (size + alpha).",
            f"filename: {os.path.basename(image_path)}",
            "",
            "JSON_START",
            json.dumps(payload, ensure_ascii=False, indent=2),
            "JSON_END"
        ])
        name = os.path.splitext(os.path.basename(image_path))[0] + ".lossy_algo_manifest_v3.txt"
        return txt, name

def decode_lossy_algo_text_to_image(
    lossy_text: str,
    output_dir: str = ".",
    out_name: str = None
):
    m = re.search(r"JSON_START\s*(\{.*?\})\s*JSON_END", lossy_text, flags=re.S)
    if not m:
        raise ValueError("JSON payload not found.")
    payload = json.loads(m.group(1))
    if payload.get("schema") not in ("IMG-ALGO-LOSSY v2", "IMG-ALGO-LOSSY v1"):
        raise ValueError("Unsupported schema.")

    W = payload["result_size"]["width"]
    H = payload["result_size"]["height"]
    pal_hex = payload["palette_rgb_hex"]
    idxs = _rle_decode(payload["pixels_rle"])
    if len(idxs) != W * H:
        raise ValueError("Pixel count mismatch.")

    imgP = Image.new("P", (W, H))
    pal = []
    for hexcol in pal_hex:
        hexcol = hexcol.lstrip("#")
        pal += [int(hexcol[0:2], 16), int(hexcol[2:4], 16), int(hexcol[4:6], 16)]
    pal += [0, 0, 0] * (256 - len(pal_hex))
    imgP.putpalette(pal)
    imgP.putdata(idxs)
    imgRGB = imgP.convert("RGB")

    alpha_rle = payload.get("alpha_rle")
    if alpha_rle:
        a = _rle_decode(alpha_rle)
        if len(a) != W * H:
            raise ValueError("Alpha length mismatch.")
        alpha_img = Image.new("L", (W, H))
        alpha_img.putdata(a)
        imgRGBA = Image.merge("RGBA", (imgRGB.split()[0], imgRGB.split()[1], imgRGB.split()[2], alpha_img))
        out = imgRGBA
    else:
        out = imgRGB

    if out_name is None:
        out_name = "rebuilt_lossy_algo_v3.png"
    out_path = os.path.join(output_dir, out_name)
    out.save(out_path, format="PNG")
    return out_path, {"size": (W, H), "has_alpha": bool(alpha_rle), "palette_entries": len(pal_hex)}

# ============================================================
# 3) LOSSY-NLP (deterministic textual description + deterministic proxy) — v2 header
# ============================================================

def _dominant_palette_via_quantize(img: Image.Image, k: int = 8):
    imgP = img.convert("RGB").quantize(colors=k, method=Image.MEDIANCUT, dither=Image.NONE)
    pal = imgP.getpalette() or []
    trip = [tuple(pal[i:i+3]) for i in range(0, len(pal), 3)]
    idxs = list(imgP.getdata()); counts = {}
    for t in idxs:
        counts[t] = counts.get(t, 0) + 1
    order = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return [trip[idx] for idx, _ in order[:k]]

def _rgb_to_hex(c): 
    return "#{:02X}{:02X}{:02X}".format(*c)

def encode_lossy_nlp_to_text(
    image_path: str,
    source: str = "user_upload",
    preserve_dims: bool = True,
    target_short_side: int = 512,
    palette_probe: int = 8
):
    with Image.open(image_path) as im0:
        img = im0.convert("RGB")
        w, h = img.width, img.height
        dom = _dominant_palette_via_quantize(img, k=palette_probe)

        # choose render dims
        if preserve_dims:
            rw, rh = w, h
        else:
            if w >= h:
                rh = target_short_side
                rw = max(1, int(round(rh * (w / h))))
            else:
                rw = target_short_side
                rh = max(1, int(round(rw * (h / w))))

        # texture/contrast stats
        g = img.convert("L"); stat = ImageStat.Stat(g)
        mean = stat.mean[0]; std = stat.stddev[0] if stat.stddev else 0.0
        contrast = "low" if std < 30 else ("medium" if std < 60 else "high")

        # quadrant colors
        quads = {
            "top_left":      (0, 0, w // 2, h // 2),
            "top_right":     (w // 2, 0, w, h // 2),
            "bottom_left":   (0, h // 2, w // 2, h),
            "bottom_right":  (w // 2, h // 2, w, h),
        }
        qhex = {}
        for name, (x0, y0, x1, y1) in quads.items():
            crop = img.crop((x0, y0, x1, y1))
            pal = _dominant_palette_via_quantize(crop, k=1)
            qhex[name] = _rgb_to_hex(pal[0]) if pal else "#CCCCCC"

        desc = {
            "schema": "LOSSY-IMAGE-DESCRIPTION v2",
            "source": source,
            "dimensions": {"width_px": w, "height_px": h, "aspect_ratio": round(w / h, 6)},
            "render_target": {"width_px": rw, "height_px": rh, "preserve_original_dims": bool(preserve_dims)},
            "colors": {
                "dominant_hex": [_rgb_to_hex(c) for c in dom],
                "quadrants_hex": qhex
            },
            "appearance": {
                "brightness_mean_0_255": round(mean, 2),
                "contrast_level": contrast
            },
            "composition_hints": [
                "Maintain aspect ratio and orientation.",
                "Preserve rough quadrant color placement.",
                "Use the dominant palette listed above."
            ]
        }
        text = "\n".join([
            "LOSSY-NLP DESCRIPTION v2",
            "note: deterministic textual description; proxy renderer uses render_target dims.",
            f"filename: {os.path.basename(image_path)}",
            "",
            "JSON_START",
            json.dumps(desc, ensure_ascii=False, indent=2),
            "JSON_END"
        ])
        name = os.path.splitext(os.path.basename(image_path))[0] + ".lossy_nlp_description_v3.txt"
        return text, name

def decode_lossy_nlp_text_to_proxy_image(
    nlp_text: str,
    output_dir: str = ".",
    out_name: str = "rebuilt_lossy_nlp_proxy_v3.png"
):
    m = re.search(r"JSON_START\s*(\{.*?\})\s*JSON_END", nlp_text, flags=re.S)
    if not m:
        raise ValueError("JSON payload not found.")
    payload = json.loads(m.group(1))
    if payload.get("schema") not in ("LOSSY-IMAGE-DESCRIPTION v2", "LOSSY-IMAGE-DESCRIPTION v1"):
        raise ValueError("Unsupported NLP schema.")

    dims = payload["render_target"]
    W, H = dims["width_px"], dims["height_px"]
    quads_hex = payload["colors"]["quadrants_hex"]
    dom = payload["colors"]["dominant_hex"][:8]

    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    quad_boxes = {
        "top_left":      (0, 0, W // 2, H // 2),
        "top_right":     (W // 2, 0, W, H // 2),
        "bottom_left":   (0, H // 2, W // 2, H),
        "bottom_right":  (W // 2, H // 2, W, H),
    }
    for name, box in quad_boxes.items():
        c = quads_hex.get(name, "#CCCCCC").lstrip("#")
        col = (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16))
        draw.rectangle(box, fill=col)

    if dom:
        sw = max(10, W // len(dom))
        y0 = max(0, H - sw); x = 0
        for hexcol in dom:
            c = hexcol.lstrip("#")
            col = (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16))
            draw.rectangle((x, y0, x + sw, H), fill=col)
            x += sw

    out_path = os.path.join(output_dir, out_name)
    img.save(out_path, format="PNG")
    return out_path, {"size": (W, H), "dominant_swatches": len(dom)}
