import os
import tempfile

from .face import scan_face
from .finger import scan_finger
from .iris import scan_iris
from .speech import process_speech
from .utils import convert

SOURCE_TYPE = ["jpg", "jpeg", "bmp", "jp2", "wsq"]
TARGET_TYPE = "png"


def scan(input: str, mode: str, **params):
    """Distribute biometric data to the interface of specified modality.

    Args:
        input: path to the input, file path or directory.

    Returns:
        A dictionary containing the results as well as the error message.
    """
    meta = {"file": input}
    error = []

    match mode:
        case "iris":
            try:
                output = scan_iris(
                    img_path=input,
                )
                meta.update(output)
            except Exception as e:
                error.append(str(e))

        case "face":
            try:
                if (params.get("engine")) == "ofiq" and params.get("type") == "folder":
                    return scan_face(
                        path=input,
                        **params,
                    )
                else:
                    output = scan_face(
                        path=input,
                        **params,
                    )
                    meta.update(output)
            except Exception as e:
                error.append(str(e))

        case "finger" | "fingerprint":
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    converted = False
                    source = params["source"] if params.get("source") else SOURCE_TYPE
                    target = params["target"] if params.get("target") else TARGET_TYPE
                    input, input_type, output_type = convert(
                        input,
                        source,
                        target,
                        directory=tmpdir,
                    )
                    converted = True if output_type != input_type else False
                    output = scan_finger(
                        img_path=input,
                    )
                    meta.update(output)
                    if converted:
                        os.remove(input)
                        if not meta.get("log"):
                            meta["log"] = []
                        meta["log"].append(
                            {"converted": f"{input_type} -> {output_type}"}
                        )
            except Exception as e:
                error.append(str(e))

        case "speech":
            try:
                if params["type"] == "file":
                    output = process_speech(
                        input_path=input, input_type=params.get("type")
                    )
                    meta.update(output)
                elif params["type"] == "folder":
                    output = process_speech(
                        input_path=input, input_type=params.get("type")
                    )
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