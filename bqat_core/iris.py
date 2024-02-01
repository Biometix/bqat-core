import csv
import os
import subprocess
from io import StringIO

import cv2 as cv
import imutils


def scan_iris(
    img_path: str,
) -> dict:
    """_summary_

    Args:
        img_path (str): _description_

    Returns:
        dict: _description_
    """
    output = {"log": {}}

    try:
        img = cv.imread(img_path)
        h, w, _ = img.shape
    except Exception as e:
        output["log"].update({"load image": str(e)})
        return output

    try:
        result = resize_input(
            img_path,
            upper=(640, 480),
        )  # lower resolution to improve robustness #17
        if result["resize"]:
            output["log"].update(
                {"resize": f"input resized to ({result['width']}, {result['height']})"}
            )
            img_path = result["path"]
    except Exception as e:
        output["log"].update({"preprocess": str(e)})
        return output

    output.update(meta) if not (meta := get_attributes(img_path)).get(
        "error"
    ) else output["log"].update({"iris attributes": meta["error"]})

    if result["resize"]:
        os.remove(img_path)
        output.update(
            {
                "image_height": h,
                "image_width": w,
            }
        )

    if not output["log"]:
        output.pop("log")
    return output


def get_attributes(img_path: str) -> dict:
    try:
        output = {}
        try:
            raw = subprocess.check_output(["biqt", "-m", "iris", img_path])
        except Exception:
            raise RuntimeError("Engine failed")
        content = StringIO(raw.decode())
        attributes = csv.DictReader(content)
        for attribute in attributes:
            output.update({attribute.get("Key"): float(attribute.get("Value"))})
        if not output:
            raise RuntimeError("Engine failed")
        quality_score = {"quality": output.get("quality")}
        output.pop("quality")
        # output.pop("fast_quality")
        quality_score.update(output)
        output = quality_score
    except Exception as e:
        return {"error": str(e)}
    return output


def resize_input(
    input,
    upper=(1000, 680),
    lower=(256, 256),
    format="png",
    grayscale=False,
):
    try:
        raw = cv.imread(input)
        img = raw.copy()
        h, w, _ = img.shape
        result = {"resize": False, "width": w, "height": h, "path": input}
    except Exception as e:
        raise RuntimeError(f"failed to load: {str(e)}")

    if w > upper[0] or h > upper[1] or w < lower[0] or h < lower[1]:
        result["resize"] = True

        while w > upper[0] or h > upper[1] or w < lower[0] or h < lower[1]:
            if w > upper[0]:
                img = imutils.resize(img, width=upper[0])
                h, w, _ = img.shape
            if h > upper[1]:
                img = imutils.resize(img, height=upper[1])
                h, w, _ = img.shape
            if w < lower[0]:
                img = imutils.resize(img, width=lower[0])
                h, w, _ = img.shape
            if h < lower[1]:
                img = imutils.resize(img, height=lower[1])
                h, w, _ = img.shape

    if result["resize"]:
        if grayscale:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            # img = cv.GaussianBlur(img, (11, 11), 0)
        if format:
            img_path = os.path.splitext(input)[0] + f".resized.{format}"
        else:
            filename, format = os.path.splitext(input)[0], os.path.splitext(input)[1]
            img_path = filename + f".resized.{format}"
        cv.imwrite(img_path, img)

        result["path"] = img_path
        result["width"] = w
        result["height"] = h

    return result
