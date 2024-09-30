import os
import tempfile

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

    elif mode == "face":
        try:
            if (params.get("engine")) == "ofiq" and params.get("type") == "folder":
                return scan_face(
                    path=file,
                    **params,
                )
            else:
                output = scan_face(
                    path=file,
                    **params,
                )
                meta.update(output)
        except Exception as e:
            error.append(str(e))

    elif mode in ("finger", "fingerprint"):
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                converted = False
                source = params["source"] if params.get("source") else SOURCE_TYPE
                target = params["target"] if params.get("target") else TARGET_TYPE
                file, input_type, output_type = convert(
                    file,
                    source,
                    target,
                    directory=tmpdir,
                )
                converted = True if output_type != input_type else False
                output = scan_finger(
                    img_path=file,
                )
                meta.update(output)
                if converted:
                    os.remove(file)
                    if not meta.get("log"):
                        meta["log"] = []
                    meta["log"].append({"converted": f"{input_type} -> {output_type}"})
        except Exception as e:
            error.append(str(e))

    elif mode == "speech":
        try:
            if params["type"] == "file":
                output = process_speech(input_path=file, input_type=params.get("type"))
                meta.update(output)
            elif params["type"] == "folder":
                output = process_speech(input_path=file, input_type=params.get("type"))
                meta.update(output)
                if log := output.get("log"):
                    error.append(log)
            else:
                raise RuntimeError("task type not provided")
        except Exception as e:
            error.append(str(e))

    if error:
        meta.update({"error": error})

    return meta