import csv
import subprocess
from io import StringIO

import cv2 as cv
import numpy as np

# import imquality.brisque as bk
# from deepface import DeepFace as df
from mediapipe.python.solutions.face_detection import FaceDetection
from mediapipe.python.solutions.face_mesh import FaceMesh
from scipy.spatial.distance import euclidean


def scan_face(img_path: str, engine: str = "default", **params) -> dict:
    if engine == "default":
        output = default_engine(img_path, params.get("confidence", 0.7))
    elif engine == "biqt":
        output = biqt_engine(img_path)
    return output


def default_engine(
    img_path: str,
    confidence: float = 0.7,
) -> dict:
    """_summary_

    Args:
        img_path (str): _description_
        confidence (float, optional): _description_. Defaults to 0.7.

    Returns:
        dict: _description_
    """
    output = {"log": {}}

    try:
        img = cv.imread(img_path)
        h, w, _ = img.shape
        output.update({"image_height": h, "image_width": w})
    except Exception as e:
        output["log"].update({"load image": str(e)})
        return output

    try:
        with FaceDetection(
            model_selection=1,  # full-range detection model
            min_detection_confidence=confidence,
        ) as face_detection:
            detections = face_detection.process(cv.cvtColor(img, cv.COLOR_BGR2RGB))
            if not getattr(detections, "detections"):
                # print(">> fallback to short-range model.")
                with FaceDetection(
                    model_selection=0,  # short-range detection model
                    min_detection_confidence=confidence,
                ) as face_detection:
                    detections = face_detection.process(
                        cv.cvtColor(img, cv.COLOR_BGR2RGB)
                    )
                if not getattr(detections, "detections"):
                    raise RuntimeError("no face found")

            score = 0
            index = 0
            for detection in getattr(detections, "detections"):
                detection_score = detection.score[0]
                detection_label_id = detection.label_id[0]
                if detection_score > score:
                    score = detection_score
                    index = detection_label_id
            detection = getattr(detections, "detections")[index]
            x = detection.location_data.relative_bounding_box.xmin * img.shape[1]
            y = detection.location_data.relative_bounding_box.ymin * img.shape[0]
            w = detection.location_data.relative_bounding_box.width * img.shape[1]
            h = detection.location_data.relative_bounding_box.height * img.shape[0]
            bbox = {
                "left": int(x),
                "upper": int(y),
                "right": int(w + x),
                "lower": int(h + y),
            }
            output.update({"confidence": detection.score[0]})
            output.update(
                {
                    "bbox_left": bbox["left"],
                    "bbox_upper": bbox["upper"],
                    "bbox_right": bbox["right"],
                    "bbox_lower": bbox["lower"],
                }
            )

            # Crop face region to ensure all components work on same target area.
            upper = bbox["upper"] if bbox["upper"] > 0 else 0
            lower = bbox["lower"] if bbox["lower"] < img.shape[0] else img.shape[0]
            left = bbox["left"] if bbox["left"] > 0 else 0
            right = bbox["right"] if bbox["right"] < img.shape[1] else img.shape[1]
            target_region = img[upper:lower, left:right]
    except Exception as e:
        output["log"].update({"face detection": str(e)})
        return output

    try:
        with FaceMesh(
            static_image_mode=True,
            min_detection_confidence=confidence,
            max_num_faces=1,
            refine_landmarks=True,
        ) as model:
            mesh = model.process(cv.cvtColor(target_region, cv.COLOR_BGR2RGB))

        if mesh.multi_face_landmarks:
            face_mesh = True
            mesh = mesh.multi_face_landmarks[0]
        else:
            face_mesh = False
            raise RuntimeError("fail to get face mesh")

    except Exception as e:
        output["log"].update({"face mesh": str(e)})
        # return output

    # output.update(meta) if not (meta:=get_img_quality(target_region)).get("error") else output["log"].update({"image quality": meta["error"]})
    # output.update(meta) if not (meta:=get_attributes(target_region)).get("error") else output["log"].update({"face attributes": meta["error"]})
    output.update(meta) if not (meta := is_smile(target_region)).get(
        "error"
    ) else output["log"].update({"smile detection": meta["error"]})

    if face_mesh:
        output.update(meta) if not (meta := is_eye_closed(mesh, target_region)).get(
            "error"
        ) else output["log"].update({"closed eye detection": meta["error"]})
        output.update(meta) if not (meta := get_ipd(mesh, target_region)).get(
            "error"
        ) else output["log"].update({"ipd": meta["error"]})
        output.update(meta) if not (meta := get_orientation(mesh, target_region)).get(
            "error"
        ) else output["log"].update({"head pose": meta["error"]})
        output.update(meta) if not (meta := is_glasses(mesh, target_region)).get(
            "error"
        ) else output["log"].update({"glasses detection": meta["error"]})

    if not output["log"]:
        output.pop("log")
    return output


def biqt_engine(
    img_path: str,
) -> dict:
    """_summary_

    Args:
        img_path (str): _description_

    Returns:
        dict: _description_
    """
    output = {"log": {}}

    output.update(meta) if not (meta := get_biqt_attr(img_path)).get(
        "error"
    ) else output["log"].update({"face attributes": meta["error"]})

    if not output["log"]:
        output.pop("log")
    return output


def get_biqt_attr(img_path: str) -> dict:
    try:
        output = {}
        try:
            raw = subprocess.check_output(["biqt", "-m", "face", img_path])
        except Exception:
            raise RuntimeError("Engine failed")
        content = StringIO(raw.decode())
        attributes = csv.DictReader(content)
        for attribute in attributes:
            output.update({attribute.get("Key"): float(attribute.get("Value"))})
        if not output:
            raise RuntimeError("Engine failed")
        output["quality"] *= 10  # Observe ISO/IEC 29794-1
    except Exception as e:
        return {"error": str(e)}
    return output


# def get_img_quality(img: np.array) -> dict:
#     try:
#         # if img.shape[-1] == 4:
#         #     img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
#         image_quality = bk.score(img)
#     except Exception as e:
#         return {"error": str(e)}
#     return {"quality": image_quality}


# def get_attributes(img: np.array) -> dict:
#     try:
#         # backends = ["opencv", "ssd", "dlib", "mtcnn", "retinaface", "mediapipe"]
#         face_attributes = df.analyze(
#             img_path=img,  # numpy array (BGR)
#             # detector_backend=backends[1],
#             actions=["age", "gender", "race", "emotion"],
#             models={
#                 "age": df.build_model("Age"),
#                 "gender": df.build_model("Gender"),
#                 "emotion": df.build_model("Emotion"),
#                 "race": df.build_model("Race"),
#             },
#             enforce_detection=False,
#             # prog_bar=False,
#         )
#     except Exception as e:
#         return {"error": str(e)}
#     return {
#         "age": face_attributes["age"],
#         "gender": face_attributes["gender"],
#         "ethnicity": face_attributes["dominant_race"],
#         "emotion": face_attributes["dominant_emotion"],
#     }


def is_smile(img: np.array) -> dict:
    try:
        img_h, img_w, _ = img.shape
        smileCascade = cv.CascadeClassifier("bqat_core/misc/haarcascade_smile.xml")
        smile = smileCascade.detectMultiScale(
            cv.cvtColor(img, cv.COLOR_BGR2GRAY),
            scaleFactor=1.15,
            minNeighbors=20,
            minSize=(int(img_h / 6), int(img_w / 3)),
            maxSize=(int(img_h / 4), int(img_w / 2)),
            flags=cv.CASCADE_DO_CANNY_PRUNING,
        )
        smile = True if len(smile) >= 1 else False
    except Exception as e:
        return {"error": str(e)}
    return {"smile": smile}


def is_eye_closed(face_mesh: object, img: np.array) -> dict:
    try:
        img_h, img_w, _ = img.shape
        right_upper = [384, 385, 386, 387]
        right_lower = [381, 380, 374, 373]
        right_corner = [362, 263]
        left_upper = [160, 159, 158, 157]
        left_lower = [144, 145, 153, 154]
        left_corner = [33, 133]

        r_u, r_l, r_c, l_u, l_l, l_c = [], [], [], [], [], []
        for i, mark in enumerate(face_mesh.landmark):
            if i in right_upper:
                r_u.append((mark.x * img_w, mark.y * img_h))
            if i in right_lower:
                r_l.append((mark.x * img_w, mark.y * img_h))
            if i in right_corner:
                r_c.append((mark.x * img_w, mark.y * img_h))
            if i in left_lower:
                l_l.append((mark.x * img_w, mark.y * img_h))
            if i in left_upper:
                l_u.append((mark.x * img_w, mark.y * img_h))
            if i in left_corner:
                l_c.append((mark.x * img_w, mark.y * img_h))

        r_l.reverse()
        l_u.reverse()

        right_vertical = np.mean([euclidean(u, l) for u, l in zip(r_u, r_l)])
        right_horizontal = euclidean(r_c[0], r_c[1])
        left_vertical = np.mean([euclidean(u, l) for u, l in zip(l_u, l_l)])
        left_horizontal = euclidean(l_c[0], l_c[1])

        right_ratio = right_vertical / right_horizontal
        left_ratio = left_vertical / left_horizontal

        threshold = 0.15
        right = True if right_ratio < threshold else False
        left = True if left_ratio < threshold else False
    except Exception as e:
        return {"error": str(e)}
    return {
        "eye_closed_left": left,
        "eye_closed_right": right,
    }


def get_ipd(face_mesh: object, img: np.array) -> dict:
    try:
        img_h, img_w, _ = img.shape
        right_iris = [469, 470, 471, 472]
        left_iris = [474, 475, 476, 477]

        r_x, r_y, l_x, l_y = [0] * 4
        r_x = np.mean(
            [lm.x for i, lm in enumerate(face_mesh.landmark) if i in right_iris]
        )
        r_y = np.mean(
            [lm.y for i, lm in enumerate(face_mesh.landmark) if i in right_iris]
        )

        l_x = np.mean(
            [lm.x for i, lm in enumerate(face_mesh.landmark) if i in left_iris]
        )
        l_y = np.mean(
            [lm.y for i, lm in enumerate(face_mesh.landmark) if i in left_iris]
        )

        r = (int(r_x * img_w), int(r_y * img_h))
        l = (int(l_x * img_w), int(l_y * img_h))
        dist = 0
        for u, v in zip(r, l):
            dist += (u - v) ** 2
    except Exception as e:
        return {"error": str(e)}
    return {
        "ipd": int(dist**0.5),
        "pupil_right_x": r[0],
        "pupil_right_y": r[1],
        "pupil_left_x": l[0],
        "pupil_left_y": l[1],
    }


def get_orientation(face_mesh: object, img: np.array) -> dict:
    try:
        img_h, img_w, _ = img.shape
        poi = [
            1,  # nose tip
            152,  # chin
            33,  # left corner of left eye
            263,  # right corner of right eye
            61,  # mouth left
            291,  # mouth right
            129,  # nose left
            358,  # nose right
        ]

        face_3d = []
        face_2d = []
        for i, lm in enumerate(face_mesh.landmark):
            if i in poi:
                x, y = int(lm.x * img_w), int(lm.y * img_h)
                face_2d.append((x, y))
                face_3d.append((x, y, lm.z))

        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)

        focal_length = img_w
        center = (img_w / 2, img_h / 2)
        camera_matrix = np.array(
            [
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1.0],
            ],
            dtype=np.float64,
        )
        distortion_coefficients = np.zeros((4, 1), dtype=np.float64)

        _, rotation_vec, _ = cv.solvePnP(
            face_3d,
            face_2d,
            camera_matrix,
            distortion_coefficients,
            flags=cv.SOLVEPNP_ITERATIVE,
        )

        rot_mat, _ = cv.Rodrigues(rotation_vec)
        try:
            angles, _, _, _, _, _ = cv.RQDecomp3x3(rot_mat)
            degree_yaw = -angles[1] * 360
            degree_pitch = angles[0] * 360
            degree_roll = angles[2] * 360
        except:
            raise RuntimeError("unable to get head pose angles.")

        if abs(degree_yaw) < 3:
            pose_yaw = "Forward"
        else:
            pose_yaw = "Left" if degree_yaw > 0 else "Right"
        if abs(degree_pitch) < 3:
            pose_pitch = "Level"
        else:
            pose_pitch = "Up" if degree_pitch > 0 else "Down"
        if abs(degree_roll) < 3:
            pose_roll = "Level"
        else:
            pose_roll = "Anti-clockwise" if degree_roll > 0 else "Clockwise"
    except Exception as e:
        return {"error": str(e)}
    return {
        "yaw_pose": pose_yaw,
        "yaw_degree": degree_yaw,
        "pitch_pose": pose_pitch,
        "pitch_degree": degree_pitch,
        "roll_pose": pose_roll,
        "roll_degree": degree_roll,
    }


def is_glasses(face_mesh: object, img: np.array) -> dict:
    try:
        img_h, img_w, _ = img.shape

        nose_bridge = [6, 197, 195, 5, 4, 1, 237, 44, 274, 457]

        nose_bridge_x = []
        nose_bridge_y = []
        for i, lm in enumerate(face_mesh.landmark):
            if i in nose_bridge:
                nose_bridge_x.append(lm.x)
                nose_bridge_y.append(lm.y)
            if i == 8:
                y_min = int(lm.y * img_h)
            if i == 5:
                y_max = int(lm.y * img_h)

        x_min = int(min(nose_bridge_x) * img_w)
        x_max = int(max(nose_bridge_x) * img_w)

        target_area = img[x_min:x_max, y_min:y_max]

        bl = cv.GaussianBlur(target_area, (3, 3), sigmaX=0, sigmaY=0)
        eg = cv.Canny(bl, 100, 200)

        center = eg.T[(int(len(eg.T) / 2))]

    except Exception as e:
        return {"error": str(e)}
    return {"glasses": True if 255 in center else False}
