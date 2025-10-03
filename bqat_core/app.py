import os
import tempfile

from .face import scan_face
from .finger import scan_finger
from .iris import scan_iris
from .speech import process_speech
from .utils import convert

SOURCE_TYPE = ["jpg", "jpeg", "bmp", "jp2", "wsq"]
TARGET_TYPE = "png"


def scan(data: str, mode: str, **params):
    """Distribute biometric data to the interface of specified modality.

    Args:
        data: path to the input, file path or directory.
        mode: modality of the input data

    Returns:
        A dictionary containing the results as well as the error message.
    """
    meta = {"file": data}
    error = []

    match mode:
        case "iris":
            try:
                output = scan_iris(
                    img_path=data,
                )
                meta.update(output)
            except Exception as e:
                error.append(str(e))

        case "face":
            try:
                if (params.get("engine")) == "ofiq" and params.get("type") == "folder":
                    return scan_face(
                        path=data,
                        **params,
                    )
                else:
                    output = scan_face(
                        path=data,
                        **params,
                    )
                    meta.update(output)
            except Exception as e:
                error.append(str(e))

        case "finger" | "fingerprint":
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    source = params["source"] if params.get("source") else SOURCE_TYPE
                    target = params["target"] if params.get("target") else TARGET_TYPE
                    data, input_type, output_type = convert(
                        data,
                        source,
                        target,
                        directory=tmpdir,
                    )
                    converted = True if output_type != input_type else False
                    output = scan_finger(
                        img_path=data,
                    )
                    meta.update(output)
                    if converted:
                        os.remove(data)
                        if not meta.get("log"):
                            meta["log"] = []
                        meta["log"].append(
                            {"converted": f"{input_type} -> {output_type}"}
                        )
            except Exception as e:
                error.append(str(e))

        case "speech":
            try:
                # if params["type"] == "file":
                #     output = process_speech(
                #         input_path=data, input_type=params.get("type")
                #     )
                #     meta.update(output)
                # elif params["type"] == "folder":
                #     output = process_speech(
                #         input_path=data, input_type=params.get("type")
                #     )
                #     meta.update(output)
                #     if logs := output.get("log"):
                #         error.extend(logs)
                # else:
                #     raise RuntimeError("task type not provided")
                output = process_speech(input_path=data, input_type="folder")
                meta.update(output)
                if logs := output.get("log"):
                    error.extend(logs)
            except Exception as e:
                error.append(str(e))

    if error:
        meta.update({"error": error})

    return meta
