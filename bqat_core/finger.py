import csv
import subprocess
from io import StringIO

import numpy as np
from PIL import Image, ImageOps

from .utils import camel_to_snake

# from scipy.stats import normaltest


def scan_finger(
    img_path: str,
) -> dict:
    """_summary_

    Args:
        img_path (str): _description_

    Returns:
        dict: _description_
    """
    output = {"log": []}

    try:
        img = Image.open(img_path)
        w, h = img.size
        output.update(
            {
                "image_width": w,
                "image_height": h,
            }
        )
    except Exception as e:
        output["log"].append({"load image": str(e)})
        return output

    # NFIQ2 Analysis
    meta = get_nfiq2(img_path)
    if meta.get("error"):
        output["log"].append({"nfiq2": meta.pop("error")})
    output.update(meta)

    # Faulty Image Detection
    meta = detect_fault(img)
    if meta.get("error"):
        output["log"].append({"fault detection": meta.pop("error")})
    output.update(meta)

    if not output.get("log"):
        output.pop("log")

    return output


def get_nfiq2(img_path: str) -> dict:
    try:
        raw = subprocess.check_output(["nfiq2", "-F", "-a", "-i", img_path])
        content = StringIO(raw.decode())
        output = next(csv.DictReader(content))
        output.pop("FingerCode")
        output.pop("Filename")
        output = {
            "nfiq2_score" if k == "QualityScore" else camel_to_snake(k): v
            for k, v in output.items()
        }
    except Exception as e:
        output = {
            "nfiq2_score": None,
            "uniform_image": None,
            "empty_image_or_contrast_too_low": None,
            "fingerprint_image_with_minutiae": None,
            "sufficient_fingerprint_foreground": None,
            "edge_std": None,
            "error": str(e),
        }
    if (error := output.get(camel_to_snake("OptionalError"))) != "NA":
        output.update({"error": error})
    output.pop(camel_to_snake("OptionalError"))
    return output


def detect_fault(img) -> dict:
    """
    Checks to see if there is an issue with the image by taking a strip one pixel wide
    From the edges of the fingerprint image. If the finger print is corrupt this will have high varience
    """
    try:
        img = ImageOps.grayscale(img)
        size = img.size
        lres = np.array(img.copy().crop((0, 0, 1, size[1])))
        rres = np.array(img.copy().crop((size[0] - 1, 0, size[0], size[1])))
        left_std = np.std(lres)
        right_std = np.std(rres)
        # lstat, p = normaltest(rres)
        # rstat, p = normaltest(rres)
        # return {"edge_skew": lstat[0] + rstat[0] }
        edge_std = (left_std + right_std) / 2.0
    except Exception as e:
        return {
            "edge_std": None,
            "error": str(e),
        }
    return {"edge_std": edge_std}
