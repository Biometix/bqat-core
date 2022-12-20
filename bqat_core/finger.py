import csv
from PIL import Image, ImageOps
import numpy as np
import subprocess
from io import StringIO

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
    output = {}

    try:
        img = Image.open(img_path)
        output.update({
            "Width": img.size[0],
            "Height": img.size[1],
        })
    except Exception as e:
        output["log"].update({"load image": str(e)})

    output.update(meta) if not (meta:=get_nfiq2(img_path)).get("error") else output["log"].update({"nfiq2": meta["error"]})
    output.update(meta) if not (meta:=detect_fault(img)).get("error") else output["log"].update({"fault detection": meta["error"]})

    return output


def get_nfiq2(img_path: str) -> dict:
    try:
        raw = subprocess.check_output(["nfiq2", "-F", "-a", "-i", img_path])
        content = StringIO(raw.decode())
        output = next(csv.DictReader(content))
        output = {
            "NFIQ2" if k == "QualityScore" else k: v for k, v in output.items()
        }
        output.pop("FingerCode")
        output.pop("Filename")
    except Exception as e:
        return {"error": str(e)}
    if (error := output.get("OptionalError")) != "NA":
        return {"error": error}
    output.pop("OptionalError")
    return output


def detect_fault(img: object) -> dict:
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
    except Exception as e:
        return {"error": str(e)}
    return {"EdgeStd": (left_std + right_std) / 2.0}
