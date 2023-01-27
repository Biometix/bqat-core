import os

from .face import scan_face
from .finger import scan_finger
from .iris import scan_iris
from .utils import convert

SOURCE_TYPE = ["jpg", "jpeg", "bmp", "jp2", "wsq"]
TARGET_TYPE = "png"


def scan(file: str, **params) -> dict:
    """_summary_

    Args:
        file (str): _description_

    Returns:
        dict: _description_
    """
    meta = {"file": file}
    error = []

    if params.get("mode") == "iris":
        try:
            output = scan_iris(
                img_path=file,
            )
            meta.update(output)
        except Exception as e:
            error.append(str(e))

    if params.get("mode") == "face":
        try:
            output = scan_face(
                img_path=file,
                engine=params.get("engine", "default"),
                confidence=params.get("confidence", 0.7)
            )
            meta.update(output)
        except Exception as e:
            error.append(str(e))
    
    if params.get("mode") in ("finger", "fingerprint"):
        try:
            converted = False
            source = params["source"] if params.get("source") else SOURCE_TYPE
            target = params["target"] if params.get("target") else TARGET_TYPE
            file, input_type, output_type = convert(file, source, target)
            converted = True if output_type != input_type else False
            output = scan_finger(
                img_path=file,
            )
            meta.update(output)
            if converted:
                os.remove(file)
                meta.update({"converted": f"{input_type} -> {output_type}"})

        except Exception as e:
            error.append(str(e))
            if converted: os.remove(file)

    if error:
        meta.update({"error": error})

    return meta
