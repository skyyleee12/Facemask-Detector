import zipfile, json

src = "face_mask_detector.keras"
dst = "face_mask_detector_fixed.keras"

def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()
                if k not in ("input_axes", "output_axes", "quantization_config")}
    if isinstance(o, list):
        return [clean(i) for i in o]
    return o

with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
    for item in zin.infolist():
        data = zin.read(item.filename)
        if item.filename == "config.json":
            data = json.dumps(clean(json.loads(data))).encode()
        zout.writestr(item, data)

print("Done:", dst)