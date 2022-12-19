from .face import scan_face
import cv2 as cv


def main(path: str, **params) -> dict:
    """_summary_

    Args:
        path (str): _description_

    Returns:
        dict: _description_
    """
    meta = {"file": path}
    error = []

    try:
        img = cv.imread(path)
        h, w, _ = img.shape
    except Exception as e:
        error.append(str(e))

    meta.update({"img_h": h, "img_w": w})

    if params.get("mode") == "face":
        try:
            output = scan_face(
                img=img,
                confidence=params.get("confidence", 0.7),
            )
            meta.update(output)
        except Exception as e:
            error.append(str(e))

    if error:
        meta.update({"error": error})

    return meta
