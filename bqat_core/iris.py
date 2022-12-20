import os
import csv
from PIL import Image, ImageOps
import subprocess
from io import StringIO


def scan_iris(
    img_path: str,
) -> dict:
    """_summary_

    Args:
        img_path (str): _description_

    Returns:
        dict: _description_
    """
    output = {}
    target_size = (640, 480)
    target_format = "png"
    processed = False

    try:
        img = Image.open(img_path)
        w, h = img.size
        output.update({
            "width": w,
            "height": h,
        })
    except Exception as e:
        output["log"].update({"load image": str(e)})
    
    try:
        if w > target_size[0] or h > target_size[1]:
            img = ImageOps.fit(img, target_size)
            img = ImageOps.grayscale(img)
            output["log"].update({"resize": f"input resized to {target_size} as {target_format.upper()}"})
            processed = True
        if processed:
            img_path = os.path.splitext(img_path)[0] + f".processed.{target_format}"
            img.save(img_path)
    except Exception as e:
        output["log"].update({"preprocess": str(e)})
        if processed: os.remove(img_path)

    output.update(meta) if not (meta:=get_attributes(img_path)).get("error") else output["log"].update({"iris attributes": meta["error"]})

    if processed: os.remove(img_path)

    return output


def get_attributes(img_path: str) -> dict:
    try:
        output = {}
        raw = subprocess.check_output(["biqt", "-m", "iris", img_path])
        content = StringIO(raw.decode())
        attributes = csv.DictReader(content)
        for attribute in attributes:
            output.update({attribute.get("Key"): attribute.get("Value")})
        quality_score = {"quality": output.get("quality")}
        output.pop("quality")
        output.pop("fast_quality")
        quality_score.update(output)
        output = quality_score
    except Exception as e:
        return {"error": str(e)}
    return output
