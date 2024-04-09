import os

from .face import scan_face
from .finger import scan_finger
from .iris import scan_iris
from .speech import process_speech
from .utils import convert

SOURCE_TYPE = ["jpg", "jpeg", "bmp", "jp2", "wsq"]
TARGET_TYPE = "png"


def scan(file: str, mode: str, **params):
    """_summary_

    Args:
        file (str): _description_

    Returns:
        dict: _description_
    """
    meta = {"file": file}
    error = []

    if mode == "iris":
        try:
            output = scan_iris(
                img_path=file,
            )
            meta.update(output)
        except Exception as e:
            error.append(str(e))

    if mode == "face":
        try:
            if (engine:=params.get("engine")) == "ofiq":
                return scan_face(
                    img_path=file,
                    engine=engine,
                )
            else:
                output = scan_face(
                    img_path=file,
                    engine=params.get("engine", "bqat"),
                    confidence=params.get("confidence", 0.7),
                )
                meta.update(output)
        except Exception as e:
            error.append(str(e))

    if mode in ("finger", "fingerprint"):
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
            if converted:
                os.remove(file)

    if mode == "speech":
        if params["type"] == "file":
            try:
                output = process_speech(input_path=file, input_type=params.get("type"))
                meta.update(output)
            except Exception as e:
                error.append(str(e))
        elif params["type"] == "folder":
            output = process_speech(input_path=file, input_type=params.get("type"))
            meta.update(output)
        else:
            raise RuntimeError("task type not provided")

    if error:
        meta.update({"error": error})

    return meta
