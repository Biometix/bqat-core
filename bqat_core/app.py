import os

from .face import scan_face
from .finger import scan_finger
from .iris import scan_iris
from .utils import convert


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
                confidence=params.get("confidence", 0.7),
            )
            meta.update(output)
        except Exception as e:
            error.append(str(e))
    
    if params.get("mode") in ("finger", "fingerprint"):
        try:
            converted = False
            if (source:=params.get("source")) and (target:=params.get("target")):
                file = convert(file, source, target)
                converted = True if file else False
                meta.update({
                    "source": list(source),
                    "target": target
                })
            if file:
                output = scan_finger(
                    img_path=file,
                )
                meta.update(output)
            if converted: os.remove(file)
        except Exception as e:
            error.append(str(e))
            if converted: os.remove(file)

    if error:
        meta.update({"error": error})

    return meta
