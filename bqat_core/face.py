import csv
import multiprocessing
import subprocess
import tempfile
import traceback
from io import StringIO
from pathlib import Path
from uuid import uuid4

import cv2 as cv
import numpy as np

# from mediapipe import Image, ImageFormat
# import imquality.brisque as bk
# from deepface import DeepFace as df
from mediapipe.python.solutions.face_detection import FaceDetection
from mediapipe.python.solutions.face_mesh import FaceMesh

# from mediapipe.tasks.python import BaseOptions
# from mediapipe.tasks.python.vision import (
#     ImageSegmenter,
#     ImageSegmenterOptions,
#     RunningMode,
# )
from scipy.spatial.distance import euclidean

# from skimage import measure
# from sklearn.cluster import KMeans
from .utils import (
    camel_to_snake,
    convert_values_to_number,
    # get_color_name,
)


def scan_face(path: str, engine: str = "bqat", **params) -> dict:
    match engine.casefold():
        case "bqat":
            output = default_engine(path, params.get("confidence", 0.7))
        case "biqt":
            output = biqt_engine(path)
        case "ofiq":
            # dir = True if params.get("type") == "folder" else False
            # output = ofiq_engine(path, dir=dir)
            output = ofiq_engine(path, dir=True)
        case "fusion":
            output = fusion_engine(path, params.get("fusion", 6))
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
    output = {"log": []}

    try:
        img = cv.imread(img_path)
        h, w, _ = img.shape
        output.update(
            {
                "image_height": h,
                "image_width": w,
            }
        )
    except Exception as e:
        traceback.print_exception(e)
        output["log"].append({"load image": str(e)})
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
            output.update({"face_detection": detection.score[0]})
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
        traceback.print_exception(e)
        output["log"].append({"face detection": str(e)})
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

    except RuntimeError as e:
        output["log"].append({"face mesh": str(e)})
    except Exception as e:
        traceback.print_exception(e)
        output["log"].append({"face mesh": str(e)})
        # return output

    # output.update(meta) if not (meta:=get_img_quality(target_region)).get("error") else output["log"].append({"image quality": meta["error"]})
    # output.update(meta) if not (meta:=get_attributes(target_region)).get("error") else output["log"].append({"face attributes": meta["error"]})
    output.update(meta) if not (meta := is_smile(target_region)).get(
        "error"
    ) else output["log"].append({"smile detection": meta["error"]})
    # output.update(meta) if not (meta := get_offset(output)).get("error") else output[
    #     "log"
    # ].append({"offset detection": meta["error"]})
    output.update(meta) if not (meta := get_face_ratio(output)).get(
        "error"
    ) else output["log"].append({"face ratio": meta["error"]})
    output.update(meta) if not (meta := get_image_meta(img)).get("error") else output[
        "log"
    ].append({"image metadata": meta["error"]})
    # output.update(meta) if not (meta := get_background_color(img)).get(
    #     "error"
    # ) else output["log"].append({"image background": meta["error"]})
    # output.update(meta) if not (meta := get_hair_cover(img, output)).get(
    #     "error"
    # ) else output["log"].append({"hair coverage": meta["error"]})
    # output.update(meta) if not (meta := is_blurry(img)).get("error") else output[
    #     "log"
    # ].append({"image blurriness": meta["error"]})

    if face_mesh:
        output.update(meta) if not (meta := is_eye_closed(mesh, target_region)).get(
            "error"
        ) else output["log"].append({"closed eye detection": meta["error"]})
        output.update(meta) if not (meta := get_ipd(mesh, output)).get(
            "error"
        ) else output["log"].append({"ipd": meta["error"]})
        output.update(meta) if not (meta := get_orientation(mesh, target_region)).get(
            "error"
        ) else output["log"].append({"head pose": meta["error"]})
        output.update(meta) if not (meta := is_glasses(mesh, target_region)).get(
            "error"
        ) else output["log"].append({"glasses detection": meta["error"]})
        output.update(meta) if not (meta := get_offset(mesh, output)).get(
            "error"
        ) else output["log"].append({"offset detection": meta["error"]})
        # output.update(meta) if not (meta := get_gaze_degree(mesh)).get(
        #     "error"
        # ) else output["log"].append({"gaze direction": meta["error"]})
        # output.update(meta) if not (
        #     meta := get_pupil_color(mesh, target_region, output)
        # ).get("error") else output["log"].append(
        #     {"pupil color detection": meta["error"]}
        # )

    if output.get("log"):
        output["log"] = output.pop("log")
    else:
        output.pop("log")

    output["file"] = img_path

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
    ) else output["log"].append({"face attributes": meta["error"]})

    if not output["log"]:
        output.pop("log")

    output["file"] = img_path

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
        traceback.print_exception(e)
        return {"error": str(e)}
    return output


# def get_img_quality(img: np.array) -> dict:
#     try:
#         # if img.shape[-1] == 4:
#         #     img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
#         image_quality = bk.score(img)
#     except Exception as e:
#         traceback.print_exception(e)
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
#         traceback.print_exception(e)
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
        traceback.print_exception(e)
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
        traceback.print_exception(e)
        return {"error": str(e)}
    return {
        "eye_closed_left": left,
        "eye_closed_right": right,
    }


def get_ipd(face_mesh: object, output: dict) -> dict:
    try:
        right_iris = [474, 475, 476, 477]
        left_iris = [469, 470, 471, 472]

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

        pupil_r = (
            int(
                (r_x * (output["bbox_right"] - output["bbox_left"]))
                + output["bbox_left"]
            ),
            int(
                (r_y * (output["bbox_lower"] - output["bbox_upper"]))
                + output["bbox_upper"]
            ),
        )
        pupil_l = (
            int(
                (l_x * (output["bbox_right"] - output["bbox_left"]))
                + output["bbox_left"]
            ),
            int(
                (l_y * (output["bbox_lower"] - output["bbox_upper"]))
                + output["bbox_upper"]
            ),
        )
        dist = 0
        for u, v in zip(pupil_r, pupil_l):
            dist += (u - v) ** 2
    except Exception as e:
        traceback.print_exception(e)
        return {"error": str(e)}
    return {
        "ipd": int(dist**0.5),
        "pupil_right_x": pupil_r[0],
        "pupil_right_y": pupil_r[1],
        "pupil_left_x": pupil_l[0],
        "pupil_left_y": pupil_l[1],
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
            pose_yaw = "forward"
        else:
            pose_yaw = "left" if degree_yaw > 0 else "right"
        if abs(degree_pitch) < 3:
            pose_pitch = "level"
        else:
            pose_pitch = "up" if degree_pitch > 0 else "down"
        if abs(degree_roll) < 3:
            pose_roll = "level"
        else:
            pose_roll = "anti-clockwise" if degree_roll > 0 else "clockwise"
    except Exception as e:
        traceback.print_exception(e)
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
        traceback.print_exception(e)
        return {"error": str(e)}
    return {"glasses": True if 255 in center else False}


def ofiq_engine(
    path: str,
    dir: bool = False,
) -> dict:
    """_summary_

    Args:
        dir_path (str): _description_

    Returns:
        dict: _description_
    """
    output = {"log": {}}

    if dir:
        output.update({"results": meta.get("results")}) if not (
            meta := get_ofiq_attr(path, dir=dir)
        ).get("error") else output["log"].append({"face attributes": meta["error"]})
    else:
        output.update(meta) if not (meta := get_ofiq_attr(path, dir=dir)).get(
            "error"
        ) else output["log"].append({"face attributes": meta["error"]})

    if not output["log"]:
        output.pop("log")
    return output


def get_ofiq_attr(path: str, dir: bool = False) -> list:
    try:
        if dir:
            with tempfile.TemporaryDirectory() as tmpdir:
                temp_output = Path(tmpdir) / f"{uuid4()}.csv"
                temp_log = Path(tmpdir) / f"{uuid4()}.log"
                output = {"results": []}
                try:
                    with open(temp_log, "w") as f:
                        subprocess.run(
                            [
                                "./OFIQ/bin/OFIQSampleApp",
                                "-c",
                                "./OFIQ/ofiq_config.jaxn",
                                "-i",
                                path,
                                "-o",
                                temp_output,
                            ],
                            stdout=f,
                            text=True,
                        )
                except Exception as e:
                    traceback.print_exception(e)
                    raise RuntimeError(f"Engine failed: {str(e)}")
                with open(temp_output) as content:
                    lines = csv.DictReader(content, delimiter=";")
                    for line in lines:
                        line = {
                            (
                                camel_to_snake(key) if key != "Filename" else "file"
                            ): value
                            for key, value in line.items()
                        }
                        line.pop("")
                        rectified = {
                            "file": line.pop("file"),
                            # "quality": line.get("unified_quality_score_scalar"),
                        }
                        rectified.update(line)
                        output["results"].append(convert_values_to_number(rectified))
                    if not output["results"]:
                        raise RuntimeError("No output")
        else:
            output = {}
            try:
                raw = subprocess.check_output(
                    [
                        "./OFIQ/bin/OFIQSampleApp",
                        "-c",
                        "./OFIQ/ofiq_config.jaxn",
                        "-i",
                        path,
                    ]
                )
            except Exception:
                raise RuntimeError("Engine failed")
            content = StringIO(raw.decode())

            # OFIQ v1.0.0-rc.1
            # header = [
            #     "unified_quality_score",
            #     "background_uniformity",
            #     "illumination_uniformity",
            #     "luminance_mean",
            #     "luminance_variance",
            #     "under_exposure_prevention",
            #     "over_exposure_prevention",
            #     "dynamic_range",
            #     "sharpness",
            #     "compression_artifacts",
            #     "natural_colour",
            #     "single_face_present",
            #     "eyes_open",
            #     "mouth_closed",
            #     "eyes_visible",
            #     "mouth_occlusion_prevention",
            #     "face_occlusion_prevention",
            #     "inter_eye_distance",
            #     "head_size",
            #     "leftward_crop_of_the_face_image",
            #     "rightward_crop_of_the_face_image",
            #     "downward_crop_of_the_face_image",
            #     "upward_crop_of_the_face_image",
            #     "head_pose_yaw",
            #     "head_pose_pitch",
            #     "head_pose_roll",
            #     "expression_neutrality",
            #     "no_head_coverings",
            #     "unified_quality_score_scalar",
            #     "background_uniformity_scalar",
            #     "illumination_uniformity_scalar",
            #     "luminance_mean_scalar",
            #     "luminance_variance_scalar",
            #     "under_exposure_prevention_scalar",
            #     "over_exposure_prevention_scalar",
            #     "dynamic_range_scalar",
            #     "sharpness_scalar",
            #     "compression_artifacts_scalar",
            #     "natural_colour_scalar",
            #     "single_face_present_scalar",
            #     "eyes_open_scalar",
            #     "mouth_closed_scalar",
            #     "eyes_visi# OFIQ v1.0.0-rc.1ble_scalar",
            #     "mouth_occlusion_prevention_scalar",
            #     "face_occlusion_prevention_scalar",
            #     "inter_eye_distance_scalar",
            #     "head_size_scalar",
            #     "leftward_crop_of_the_face_image_scalar",
            #     "rightward_crop_of_the_face_image_scalar",
            #     "downward_crop_of_the_face_image_scalar",
            #     "upward_crop_of_the_face_image_scalar",
            #     "head_pose_yaw_scalar",
            #     "head_pose_pitch_scalar",
            #     "head_pose_roll_scalar",
            #     "expression_neutrality_scalar",
            #     "no_head_coverings_scalar",
            # ]
            # output = next(
            #     csv.DictReader(
            #         StringIO(
            #             list(next(csv.DictReader(content)).values())[0].split(":")[1]
            #         ),
            #         delimiter=";",
            #         fieldnames=header[0:28],
            #     )
            # )

            raw = list(csv.DictReader(content))
            header = [
                camel_to_snake(item)
                for item in next(
                    csv.reader(
                        StringIO(list(raw[3].values())[0]),
                        delimiter=";",
                    )
                )
            ][1:]
            header.pop()
            raw_scores = list(raw[1].values())[0].split(":")[1]
            scalar_scores = list(raw[2].values())[0].split(":")[1]
            output = next(
                csv.DictReader(
                    StringIO(raw_scores + ";" + scalar_scores),
                    delimiter=";",
                    fieldnames=header,
                )
            )

            output = {key: float(value) for key, value in output.items()}

            if not output:
                raise RuntimeError("Engine failed")
    except Exception as e:
        traceback.print_exception(e)
        return {"error": str(e)}
    return output


# def get_offset(output: dict) -> dict:
#     # Use face area bouding box to calculate face offset
#     try:
#         return {
#             "face_offset_x": (
#                 (output["image_width"] - (output["bbox_right"] - output["bbox_left"]))
#                 / 2
#                 - (output["image_width"] - output["bbox_right"])
#             )
#             / output["image_width"],
#             "face_offset_y": (
#                 (output["image_height"] - (output["bbox_lower"] - output["bbox_upper"]))
#                 / 2
#                 - (output["image_height"] - output["bbox_lower"])
#             )
#             / output["image_height"],
#         }
#     except Exception as e:
#         traceback.print_exception(e)
#         return {"error": str(e)}


def get_offset(face_mesh: object, output: dict) -> dict:
    # Use face mask geometrical centre to calculate face offset
    try:
        img_h, img_w = output["image_height"], output["image_width"]
        centre = [6]

        centre_x = img_w / 2
        centre_y = img_h / 2

        for i, lm in enumerate(face_mesh.landmark):
            if i in centre:
                centre_x = (
                    lm.x * (output["bbox_right"] - output["bbox_left"])
                    + output["bbox_left"]
                )
                centre_y = (
                    lm.y * (output["bbox_lower"] - output["bbox_upper"])
                ) + output["bbox_upper"]
                break

        offset_x = (centre_x - img_w / 2) / img_w
        offset_y = (centre_y - img_h / 2) / img_h
    except Exception as e:
        traceback.print_exception(e)
        return {"error": str(e)}
    return {
        "face_offset_x": offset_x,
        "face_offset_y": offset_y,
    }


def get_face_ratio(output: dict) -> dict:
    try:
        return {
            "face_ratio": (output["bbox_right"] - output["bbox_left"])
            * (output["bbox_lower"] - output["bbox_upper"])
            / (output["image_width"] * output["image_height"])
        }
    except Exception as e:
        traceback.print_exception(e)
        return {"error": str(e)}


# def get_gaze_degree(face_mesh: object) -> dict:
#     try:
#         right_pupil_centre = 473
#         right_eye_right = 263
#         right_eye_left = 362
#         right_eye_top = 386
#         right_eye_bottom = 374
#         left_pupil_centre = 468
#         left_eye_right = 133
#         left_eye_left = 33
#         left_eye_top = 159
#         left_eye_bottom = 145

#         landmarks = {index: lm for index, lm in enumerate(face_mesh.landmark)}

#         rpc = (landmarks[right_pupil_centre].x, landmarks[right_pupil_centre].y)
#         rer = (landmarks[right_eye_right].x, landmarks[right_eye_right].y)
#         rel = (landmarks[right_eye_left].x, landmarks[right_eye_left].y)
#         ret = (landmarks[right_eye_top].x, landmarks[right_eye_top].y)
#         reb = (landmarks[right_eye_bottom].x, landmarks[right_eye_bottom].y)
#         lpc = (landmarks[left_pupil_centre].x, landmarks[left_pupil_centre].y)
#         ler = (landmarks[left_eye_right].x, landmarks[left_eye_right].y)
#         lel = (landmarks[left_eye_left].x, landmarks[left_eye_left].y)
#         let = (landmarks[left_eye_top].x, landmarks[left_eye_top].y)
#         leb = (landmarks[left_eye_bottom].x, landmarks[left_eye_bottom].y)

#         gaze_right_x = (rpc[0] - ((rer[0] + rel[0]) / 2)) / (rer[0] - rel[0])
#         gaze_right_y = (rpc[1] - ((ret[1] + reb[1]) / 2)) / (reb[1] - ret[1])
#         gaze_left_x = (lpc[0] - ((ler[0] + lel[0]) / 2)) / (ler[0] - lel[0])
#         gaze_left_y = (lpc[1] - ((let[1] + leb[1]) / 2)) / (leb[1] - let[1])

#     except Exception as e:
#         traceback.print_exception(e)
#         return {"error": str(e)}
#     return {
#         "gaze_right_x": gaze_right_x,
#         "gaze_right_y": gaze_right_y,
#         "gaze_left_x": gaze_left_x,
#         "gaze_left_y": gaze_left_y,
#     }


# def get_pupil_color(face_mesh: object, img: np.array, output: dict) -> dict:
#     try:
#         right_pupil_centre = 473
#         right_pupil_right = 474
#         right_pupil_top = 475
#         right_pupil_left = 476
#         right_pupil_bottom = 477
#         left_pupil_centre = 468
#         left_pupil_right = 469
#         left_pupil_top = 470
#         left_pupil_left = 471
#         left_pupil_bottom = 472

#         landmarks = {index: lm for index, lm in enumerate(face_mesh.landmark)}

#         rpc = (landmarks[right_pupil_centre].x, landmarks[right_pupil_centre].y)
#         rpr = (landmarks[right_pupil_right].x, landmarks[right_pupil_right].y)
#         rpt = (landmarks[right_pupil_top].x, landmarks[right_pupil_top].y)
#         rpl = (landmarks[right_pupil_left].x, landmarks[right_pupil_left].y)
#         rpb = (landmarks[right_pupil_bottom].x, landmarks[right_pupil_bottom].y)
#         lpc = (landmarks[left_pupil_centre].x, landmarks[left_pupil_centre].y)
#         lpr = (landmarks[left_pupil_right].x, landmarks[left_pupil_right].y)
#         lpt = (landmarks[left_pupil_top].x, landmarks[left_pupil_top].y)
#         lpl = (landmarks[left_pupil_left].x, landmarks[left_pupil_left].y)
#         lpb = (landmarks[left_pupil_bottom].x, landmarks[left_pupil_bottom].y)

#         def get_color(pc, pr, pt, pl, pb) -> str:
#             top = int(
#                 (pc[1] + pt[1]) / 2 * (output["bbox_lower"] - output["bbox_upper"])
#             )
#             bottom = int(
#                 (pc[1] + pb[1]) / 2 * (output["bbox_lower"] - output["bbox_upper"])
#             )
#             right = int(
#                 (pc[0] + pr[0]) / 2 * (output["bbox_right"] - output["bbox_left"])
#             )
#             left = int(
#                 (pc[0] + pl[0]) / 2 * (output["bbox_right"] - output["bbox_left"])
#             )

#             if (pupil := img[top:bottom, left:right]).any():
#                 pupil = cv.cvtColor(pupil, cv.COLOR_BGR2RGB)
#             else:
#                 raise RuntimeError("fail to crop pupil area")

#             # Reshape the image to be a list of pixels
#             pixels = pupil.reshape(-1, 3)

#             # Handle low resolution images (will intriduce warning for duplicate pixels)
#             if len(pixels) < 10:
#                 pixels = np.repeat(pixels, 3, axis=0)
#                 n_clusters = 1
#             else:
#                 n_clusters = 3

#             # Use KMeans to find the most common colors
#             kmeans = KMeans(n_clusters=n_clusters)
#             kmeans.fit(pixels)

#             # Get the cluster centers
#             centers = kmeans.cluster_centers_

#             # Calculate the size of each cluster
#             _, counts = np.unique(kmeans.labels_, return_counts=True)

#             # Find the largest cluster
#             largest_cluster_index = np.argmax(counts)

#             # Get the largest cluster center
#             largest_center = centers[largest_cluster_index]
#             dominant_color = tuple(map(int, largest_center))

#             return get_color_name(dominant_color), dominant_color

#         pupil_color_right, pupil_right_rgb = get_color(rpc, rpr, rpt, rpl, rpb)
#         pupil_color_left, pupil_left_rgb = get_color(lpc, lpr, lpt, lpl, lpb)

#     except RuntimeError as e:
#         return {"error": str(e)}
#     except Exception as e:
#         traceback.print_exception(e)
#         return {"error": str(e)}
#     return {
#         "pupil_colour_right_name": pupil_color_right,
#         "pupil_colour_right_rgb": pupil_right_rgb,
#         "pupil_colour_left_name": pupil_color_left,
#         "pupil_colour_left_rgb": pupil_left_rgb,
#     }


def get_image_meta(img: np.array) -> dict:
    try:
        gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        brightness = np.mean(gray_image)
        laplacian = cv.Laplacian(gray_image, cv.CV_64F)
        variance = np.var(laplacian)
        # luminance = np.mean(
        #     0.299 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0]
        # )
        sorted_array = np.sort(gray_image.flatten())
        num_elements = int(0.01 * len(sorted_array))
        min_val = np.mean(sorted_array[:num_elements])
        max_val = np.mean(sorted_array[-num_elements:])
        dynamic_range = max_val - min_val
        contrast = np.std(img)
    except Exception as e:
        traceback.print_exception(e)
        return {"error": str(e)}
    return {
        "brightness": float(brightness),
        "dynamic_range": float(dynamic_range),
        "sharpness": float(variance),
        # "luminance": float(luminance),
        "contrast": contrast,
    }


# def get_background_color(img: np.array) -> dict:
#     try:
#         # Create a image segmenter instance with the image mode:
#         options = ImageSegmenterOptions(
#             base_options=BaseOptions(
#                 model_asset_path="bqat_core/misc/BQAT/selfie_segmenter.tflite"
#             ),
#             running_mode=RunningMode.IMAGE,
#             output_category_mask=True,
#         )

#         # Load the input image from a numpy array.
#         mp_image = Image(image_format=ImageFormat.SRGB, data=img)

#         with ImageSegmenter.create_from_options(options) as segmenter:
#             segmented_masks = segmenter.segment(mp_image)

#         image_data = cv.cvtColor(mp_image.numpy_view(), cv.COLOR_BGR2RGB)

#         condition = (
#             np.stack((segmented_masks.category_mask.numpy_view(),) * 3, axis=-1) > 0.2
#         )

#         # # Generate output with transparent foreground
#         # fg_image = np.zeros(
#         #     (
#         #         image_data.shape[0],
#         #         image_data.shape[1],
#         #         4,
#         #     ),
#         #     dtype=np.uint8,
#         # )
#         # rgba_image = np.concatenate(
#         #     (
#         #         image_data,
#         #         np.ones(
#         #             (
#         #                 image_data.shape[0],
#         #                 image_data.shape[1],
#         #                 1,
#         #             ),
#         #             dtype=np.uint8,
#         #         )
#         #         * 255,
#         #     ),
#         #     axis=2,
#         # )
#         # output_image = np.where(condition, rgba_image, fg_image)

#         # Generate output without foreground
#         fg_image = np.full(
#             (
#                 image_data.shape[0],
#                 image_data.shape[1],
#                 3,
#             ),
#             111,
#             dtype=np.uint8,
#         )
#         output_image = np.where(condition, image_data, fg_image)

#         # Reshape the image to be a list of pixels
#         pixels = output_image.reshape(-1, 3)

#         # Remove foreground pixels
#         if not (pixels := pixels[~(np.all(pixels == [111, 111, 111], axis=1))]).any():
#             raise RuntimeError("fail to get background image.")

#         # Use KMeans to find the most common colors
#         kmeans = KMeans(n_clusters=3)
#         kmeans.fit(pixels)

#         # Get the cluster centers
#         centers = kmeans.cluster_centers_

#         # Calculate the size of each cluster
#         _, counts = np.unique(kmeans.labels_, return_counts=True)

#         # Find the largest cluster
#         largest_cluster_index = np.argmax(counts)

#         # Get the largest cluster center
#         largest_center = centers[largest_cluster_index]
#         dominant_color = tuple(map(int, largest_center))

#         background_color = get_color_name(dominant_color)

#         # Calculate the standard deviation of the R, G, and B channels separately
#         output_image = cv.GaussianBlur(output_image, (5, 5), 0)

#         r_std = np.std(output_image[:, :, 0])
#         g_std = np.std(output_image[:, :, 1])
#         b_std = np.std(output_image[:, :, 2])

#         # Calculate overall average color standard deviation
#         background_uniformity = (r_std + g_std + b_std) / 3

#     except RuntimeError as e:
#         return {"error": str(e)}
#     except Exception as e:
#         traceback.print_exception(e)
#         return {"error": str(e)}
#     return {
#         "background_color_name": background_color,
#         "background_color_rgb": dominant_color,
#         "background_uniformity": background_uniformity,
#     }


# def get_hair_cover(img: np.array, output: dict) -> dict:
#     try:
#         # Create a image segmenter instance with the image mode:
#         options = ImageSegmenterOptions(
#             base_options=BaseOptions(
#                 model_asset_path="bqat_core/misc/BQAT/hair_segmenter.tflite",
#             ),
#             running_mode=RunningMode.IMAGE,
#             output_category_mask=True,
#         )

#         # Load the input image from a numpy array.
#         mp_image = Image(image_format=ImageFormat.SRGB, data=img)

#         with ImageSegmenter.create_from_options(options) as segmenter:
#             segmented_masks = segmenter.segment(mp_image)

#         image_data = cv.cvtColor(mp_image.numpy_view(), cv.COLOR_BGR2RGB)

#         condition = (
#             np.stack(
#                 (segmented_masks.category_mask.numpy_view(),) * 3,
#                 axis=-1,
#             )
#             > 0.2
#         )

#         # Pure black background
#         bg_image = np.full(
#             (
#                 image_data.shape[0],
#                 image_data.shape[1],
#                 3,
#             ),
#             0,
#             dtype=np.uint8,
#         )
#         # Pure white foreground
#         fg_image = np.full(
#             (
#                 image_data.shape[0],
#                 image_data.shape[1],
#                 3,
#             ),
#             255,
#             dtype=np.uint8,
#         )
#         # Mask hair area
#         hair_area_image = np.where(condition, fg_image, bg_image)

#         # Mask face area
#         cv.rectangle(
#             bg_image,
#             (
#                 output.get("bbox_left"),
#                 output.get("bbox_upper"),
#             ),
#             (
#                 output.get("bbox_right"),
#                 output.get("bbox_lower"),
#             ),
#             (255, 255, 255),
#             -1,
#         )

#         face_area_hsv = cv.cvtColor(bg_image, cv.COLOR_BGR2HSV)
#         hair_area_image_hsv = cv.cvtColor(hair_area_image, cv.COLOR_BGR2HSV)

#         mask_face = cv.inRange(
#             face_area_hsv, np.array([0, 0, 100]), np.array([0, 0, 255])
#         )
#         mask_hair = cv.inRange(
#             hair_area_image_hsv, np.array([0, 0, 100]), np.array([0, 0, 255])
#         )

#         # Find intersection of the two masks
#         intersection = cv.bitwise_and(mask_face, mask_hair)
#         covered_area = cv.countNonZero(intersection)

#         # Face area
#         face_area = (output.get("bbox_right") - output.get("bbox_left")) * (
#             output.get("bbox_lower") - output.get("bbox_upper")
#         )

#         overlap = covered_area / face_area if face_area > 0 else 0
#     except Exception as e:
#         traceback.print_exception(e)
#         return {"error": str(e)}
#     return {
#         "hair_coverage": overlap,
#     }


# def is_blurry(img: np.array) -> dict:
#     try:
#         gs = cv.cvtColor(img.copy(), cv.COLOR_BGR2GRAY)
#         blur_metric = measure.blur_effect(gs, h_size=15)
#         lap = cv.Laplacian(gs, cv.CV_64F)
#     except Exception as e:
#         return {"error": str(e)}
#     return {
#         "blur_lap_var": lap.var(),
#         "blurriness": blur_metric,
#     }


def fusion_engine(path: str, fusion_code: int = 6):
    def process_images(engine_func, path, pool):
        results = {}
        for result in pool.map(
            engine_func, [p.as_posix() for p in Path(path).rglob("*")]
        ):
            results[result["file"]] = result
        return results

    with multiprocessing.Pool(
        processes=(
            int(multiprocessing.cpu_count() * 0.7)
            if int(multiprocessing.cpu_count() * 0.7) > 1
            else 1
        )
    ) as pool:
        match fusion_code:
            case 6:
                ofiq = ofiq_engine(path, dir=True)
                bqat_results = process_images(default_engine, path, pool)
                output = {
                    "results": [i | bqat_results[i["file"]] for i in ofiq["results"]],
                    "logs": ofiq.get("logs"),
                }
            case 5:
                ofiq = ofiq_engine(path, dir=True)
                biqt_results = process_images(biqt_engine, path, pool)
                output = {
                    "results": [i | biqt_results[i["file"]] for i in ofiq["results"]],
                    "logs": ofiq.get("logs"),
                }
            case 3:
                bqat_results = process_images(default_engine, path, pool)
                biqt_results = process_images(biqt_engine, path, pool)
                output = {
                    "results": [
                        biqt_results[i["file"]] | i for i in bqat_results.values()
                    ]
                }
            case 7:
                ofiq = ofiq_engine(path, dir=True)
                bqat_results = process_images(default_engine, path, pool)
                biqt_results = process_images(biqt_engine, path, pool)
                output = {
                    "results": [
                        i | bqat_results[i["file"]] | biqt_results[i["file"]]
                        for i in ofiq["results"]
                    ],
                    "logs": ofiq.get("logs"),
                }
            case _:
                raise ValueError(f"Unsupported fusion code: {fusion_code}")

        return output
