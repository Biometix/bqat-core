import csv
import os
import subprocess
from io import StringIO

import cv2 as cv
import imutils
from PIL import Image, ImageOps


def scan_iris(
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
        img = cv.imread(img_path)
        if img is None:
            raise RuntimeError("failed to open file.")
        h, w, _ = img.shape
        output.update(
            {
                "image_height": h,
                "image_width": w,
            }
        )
    except Exception as e:
        output["log"].append({"load image": str(e)})
        return output

    try:
        # result = crop_input(img_path) # won't work better
        result = resize_input(
            img_path,
            upper=(630, 470),
            lower=(400, 400),
            file_format="png",
        )  # reduce range resolution to improve robustness #17
        if result["resize"]:
            output["log"].append(
                {"resize": f"input resized to ({result['width']}, {result['height']})"}
            )
            img_path = result["path"]
        if result["format"]:
            output["log"].append(
                {"convert": f"input converted to {result['format'].upper()} format"}
            )
            img_path = result["path"]
    except Exception as e:
        output["log"].append({"preprocess": str(e)})
        return output

    # Iris Analysis
    meta = get_attributes(img_path)
    if meta.get("error"):
        output["log"].append(
            {"iris image analysis": meta.pop("error")}
        )
    output.update(meta)

    if result["resize"] or result["convert"]:
        os.remove(img_path)

    if not output.get("log"):
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
        return {
            "quality": None,
            "contrast": None,
            "sharpness": None,
            "iris_diameter": None,
            "percent_visible_iris": None,
            "iris_pupil_gs": None,
            "iris_sclera_gs": None,
            "iso_overall_quality": None,
            "iso_greyscale_utilization": None,
            "iso_iris_pupil_concentricity": None,
            "iso_iris_pupil_contrast": None,
            "iso_iris_pupil_ratio": None,
            "iso_iris_sclera_contrast": None,
            "iso_margin_adequacy": None,
            "iso_pupil_boundary_circularity": None,
            "iso_sharpness": None,
            "iso_usable_iris_area": None,
            "error": str(e)
        }
    return output


def resize_input(
    input_path,
    upper=(1000, 680),
    lower=(256, 256),
    file_format="",
    grayscale=False,
):
    try:
        raw = cv.imread(input_path)
        img = raw.copy()
        h, w, _ = img.shape
        result = {
            "resize": False,
            "convert": False,
            "format": file_format,
            "width": w,
            "height": h,
            "path": input_path,
        }
    except Exception as e:
        raise RuntimeError(f"failed to load: {str(e)}")

    if w > upper[0] or h > upper[1] or w < lower[0] or h < lower[1]:
        result["resize"] = True

        inf_count = 5  # break inf loop
        while w > upper[0] or h > upper[1] or w < lower[0] or h < lower[1]:
            inf_count -= 1
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
            if inf_count < 0:
                break

    if grayscale:
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # img = cv.GaussianBlur(img, (11, 11), 0)

    if result["resize"]:
        if file_format:
            result["convert"] = True
            img_path = os.path.splitext(input_path)[0] + f".resized.{file_format}"
        else:
            filename, file_format = os.path.splitext(input_path)[0], os.path.splitext(input_path)[1]
            img_path = filename + f".resized.{file_format}"
            result["format"] = file_format
        cv.imwrite(img_path, img)
    else:
        if file_format:
            result["convert"] = True
            img_path = os.path.splitext(input_path)[0] + f".converted.{file_format}"
            cv.imwrite(img_path, img)
        else:
            img_path = input_path

    result["path"] = img_path
    result["width"] = w
    result["height"] = h

    return result


def crop_input(
    input_path,
):
    """Older version of BIQT claims it supports only 480 by 480 or 640 by 480 input

    Args:
        input_path (_type_): _description_
        target (tuple, optional): _description_. Defaults to (640,480).
    """
    try:
        img = Image.open(input_path)
        w, h = img.size
    except Exception as e:
        raise RuntimeError(f"failed to load: {str(e)}")

    result = {"resize": False, "width": w, "height": h, "path": input_path}

    if h == 480 and (w == 480 or 640):
        return result

    if (w / h) > 640 / 480 or (w / h) > 1.1:
        img = ImageOps.fit(img, size=(640, 480))

    else:
        img = ImageOps.fit(img, size=(480, 480))

    filename, file_format = os.path.splitext(input_path)[0], os.path.splitext(input_path)[1]
    img_path = filename + f".resized{file_format}"
    img.save(img_path)

    w, h = img.size
    result["resize"] = True
    result["path"] = img_path
    result["width"] = w
    result["height"] = h

    return result