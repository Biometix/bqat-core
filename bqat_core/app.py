import os
from .face import scan_face
from .finger import scan_finger
from .utils import convert


def run(file: str, **params) -> dict:
    """_summary_

    Args:
        file (str): _description_

    Returns:
        dict: _description_
    """
    meta = {"file": file}
    error = []

    if params.get("mode") == "face":
        try:
            output = scan_face(
                img_path=file,
                confidence=params.get("confidence", 0.7),
            )
            meta.update(output)
        except Exception as e:
            error.append(str(e))
    
    if params.get("mode") == "finger":
        try:
            converted = False
            if (source:=params.get("source")) and (target:=params.get("target")):
                file = convert(file, source, target)
                converted = True
                meta.update({
                    "source": source,
                    "target": target
                })
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
    
    print(meta)

    return meta
