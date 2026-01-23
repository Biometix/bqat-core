import csv
import multiprocessing
import subprocess
import tempfile
import traceback
from functools import reduce
from io import StringIO
from itertools import combinations
from pathlib import Path
from uuid import uuid4

import cv2 as cv
import numpy as np
import PIL

# import onnxruntime
from mediapipe import Image, ImageFormat

# import imquality.brisque as bk
# from deepface import DeepFace as df
from mediapipe.python.solutions.face_detection import FaceDetection
from mediapipe.python.solutions.face_mesh import FaceMesh
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    ImageSegmenter,
    ImageSegmenterOptions,
    RunningMode,
)
from scipy.spatial.distance import euclidean

# from skimage import measure
from sklearn.cluster import KMeans

from .utils import (
    BQAT_CWD,  # prepare_input,; rgb_to_color_temperature,
    OFIQ_CWD,
    SETTINGS,
    Engine,
    camel_to_snake,
    convert_values_to_number,
    cpu_usage_to_processes,
    get_color_name,
    merge_results,
)


def scan_face(path: str, engine: str = "bqat", **params) -> dict:
    """Process image with engine specified.

    Args:
        path: Path to the image.
        engine: Name of the engine. Defaults to "bqat".
        **params: Arbitrary keyword arguments.

    Returns:
        A dictionary containing the results and logs from submodules.
    """
    match engine.casefold():
        case "bqat":
            output = default_engine(
                path,
                params.get("confidence", 0.7),
                params.get("settings", {}),
                # params.get("pro", False),
            )
        case "biqt":
            output = biqt_engine(path)
        case "ofiq":
            # dir = True if params.get("type") == "folder" else False
            # output = ofiq_engine(path, dir=dir)
            output = ofiq_engine(path, dir=True)
        case "fusion":
            output = fusion_engine(
                path,
                params.get("fusion", 6),
                params.get("cpu", 0.7),
            )
        case _:
            raise ValueError(f"Unknown engine: {engine}")
    return output


def default_engine(
    img_path: str,
    confidence: float = 0.7,
    settings: dict = {},
    verbose: bool = False,
    # pro: bool = False,
) -> dict:
    """Process image with native BQAT engine.

    Sends image path into BQAT engine and returns the results.

    Args:
        img_path: Path to the image.
        confidence: Face detection confidence level threshold. Defaults to '0.7'.
        pro: togle to enable pro version of native BQAT engine. Defaults to False.

    Returns:
        A dictionary containing the results and logs from submodules.
    """
    if not settings:
        settings = SETTINGS
    if verbose:
        settings = {k: True for k, _ in settings.items()}

    output = {"log": [], "file": img_path}

    try:
        img = cv.imread(img_path)
        if img is None:
            raise RuntimeError("failed to open file.")
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

    target_region, output = get_face(img, output, confidence)
    mesh, output = get_mesh(target_region, output, confidence)

    # # BRISQUE Analysis
    # meta = (
    #     get_img_quality(target_region) if settings.get("image_quality", False) else {}
    # )
    # if meta.get("error"):
    #     output["log"].append({"Image quality": meta.pop("error")})
    # output.update(meta)

    # # Face Attributes Analysis
    # meta = (
    #     get_attributes(target_region) if settings.get("face_attributes", False) else {}
    # )
    # if meta.get("error"):
    #     output["log"].append({"Face attributes": meta.pop("error")})
    # output.update(meta)

    # Face Smile Detection
    meta = is_smile(target_region) if settings.get("smile_detection", False) else {}
    if meta.get("error"):
        output["log"].append({"Smile detection": meta.pop("error")})
    output.update(meta)

    # Image Metadata
    meta = (
        get_image_meta(img, img_path) if settings.get("image_metadata", False) else {}
    )
    if meta.get("error"):
        output["log"].append({"Image metadata": meta.pop("error")})
    output.update(meta)

    # Background Colour
    meta = get_background_color(img) if settings.get("background_colour", False) else {}
    if meta.get("error"):
        output["log"].append({"Background colour": meta.pop("error")})
    output.update(meta)

    # Hair Coverage Estimation
    meta = get_hair_cover(img, output) if settings.get("hair_coverage", False) else {}
    if meta.get("error"):
        output["log"].append({"Hair coverage": meta.pop("error")})
    output.update(meta)

    # # Image Blurriness
    # meta = is_blurry(img) if settings.get("image_blurriness", False) else {}
    # if meta.get("error"):
    #     output["log"].append({"Image blurriness": meta.pop("error")})
    # output.update(meta)

    # # Colour Temperature
    # meta = (
    #     get_colour_temperature(img) if settings.get("colour_temperature", False) else {}
    # )
    # if meta.get("error"):
    #     output["log"].append({"Colour temperature": meta.pop("error")})
    # output.update(meta)

    # Image Brightness Variance
    meta = (
        get_brightness_variance(img)
        if settings.get("brightness_variance", False)
        else {}
    )
    if meta.get("error"):
        output["log"].append({"Brightness variance": meta.pop("error")})
    output.update(meta)

    # Face Ratio Estimation
    meta = get_face_ratio(output) if settings.get("face_ratio", False) else {}
    if meta.get("error"):
        output["log"].append({"Face ratio": meta.pop("error")})
    output.update(meta)

    # # Headwear Detection
    # meta = get_hijab(img, output) if settings.get("headwear_detection", False) else {}
    # if meta.get("error"):
    #     output["log"].append({"Headwear detection": meta.pop("error")})
    # output.update(meta)

    # Eyes detection
    meta = (
        is_eye_closed(mesh, target_region)
        if settings.get("eyes_detection", False)
        else {}
    )
    if meta.get("error"):
        output["log"].append({"Eyes detection": meta.pop("error")})
    output.update(meta)

    # Inter-pupillary Distance Estimation
    meta = get_ipd(mesh, output) if settings.get("ipd_estimation", False) else {}
    if meta.get("error"):
        output["log"].append({"IPD estimation": meta.pop("error")})
    output.update(meta)

    # Head Pose Estimation
    meta = (
        get_orientation(mesh, target_region)
        if settings.get("pose_estimation", False)
        else {}
    )
    if meta.get("error"):
        output["log"].append({"Head pose estimation": meta.pop("error")})
    output.update(meta)

    # Glasses Detection
    meta = (
        is_glasses(mesh, target_region)
        if settings.get("glasses_detection", False)
        else {}
    )
    if meta.get("error"):
        output["log"].append({"Glasses detection": meta.pop("error")})
    output.update(meta)

    # Face Offset Estimation
    meta = get_offset(mesh, output) if settings.get("face_offset", False) else {}
    if meta.get("error"):
        output["log"].append({"Face offset": meta.pop("error")})
    output.update(meta)

    # Gaze Direction Estimation
    meta = get_gaze_degree(mesh) if settings.get("gaze_estimation", False) else {}
    if meta.get("error"):
        output["log"].append({"Gaze estimation": meta.pop("error")})
    output.update(meta)

    # Pupil Color Estimation
    meta = (
        get_pupil_color(mesh, target_region, output)
        if settings.get("pupil_estimation", False)
        else {}
    )
    if meta.get("error"):
        output["log"].append({"Pupil color estimation": meta.pop("error")})
    output.update(meta)

    # Head Location Estimation
    meta = (
        get_head_location(mesh, img, target_region, output)
        if settings.get("head_location", False)
        else {}
    )
    if meta.get("error"):
        output["log"].append({"Head location": meta.pop("error")})
    output.update(meta)

    if not output.get("log"):
        output.pop("log")

    return output


def biqt_engine(
    img_path: str,
) -> dict:
    """Process image with native BIQT engine.

    Args:
        img_path: Path to the image.

    Returns:
        A dictionary containing the results from BIQT.
    """
    output = {"log": [], "file": img_path}

    # BIQT Face Analysis
    meta = get_biqt_attr(img_path)
    if meta.get("error"):
        output["log"].append({"biqt": meta.pop("error")})
    output.update(meta)

    if not output.get("log"):
        output.pop("log")

    return output


def get_biqt_attr(img_path: str) -> dict:
    try:
        output = {}
        try:
            raw = subprocess.check_output(["biqt", "-m", "face", img_path])
        except Exception:
            raise RuntimeError("biqt engine failed")
        content = StringIO(raw.decode())
        attributes = csv.DictReader(content)
        for attribute in attributes:
            output.update({attribute.get("Key"): float(attribute.get("Value"))})
        if not output:
            raise RuntimeError("biqt engine failed")
        output["quality"] *= 10  # Observe ISO/IEC 29794-1, requested from MOSIP
    except Exception as e:
        traceback.print_exception(e)
        return {
            "background_deviation": None,
            "background_grayness": None,
            "blur": None,
            "blur_face": None,
            "focus": None,
            "focus_face": None,
            "openbr_IPD": None,
            "openbr_confidence": None,
            "opencv_IPD": None,
            "opencv_eye_count": None,
            "opencv_face_found": None,
            "opencv_face_height": None,
            "opencv_face_width": None,
            "opencv_frontal_face_found": None,
            "opencv_landmarks_count": None,
            "opencv_mouth_count": None,
            "opencv_nose_count": None,
            "opencv_profile_face_found": None,
            "over_exposure": None,
            "over_exposure_face": None,
            "quality": None,
            "sap_code": None,
            "skin_ratio_face": None,
            "skin_ratio_full": None,
            "image_area": None,
            "image_channels": None,
            "image_height": None,
            "image_ratio": None,
            "image_width": None,
            "openbr_left_eye_x": None,
            "openbr_left_eye_y": None,
            "openbr_right_eye_x": None,
            "openbr_right_eye_y": None,
            "opencv_face_center_of_mass_x": None,
            "opencv_face_center_of_mass_y": None,
            "opencv_face_offset_x": None,
            "opencv_face_offset_y": None,
            "opencv_face_x": None,
            "opencv_face_y": None,
            "opencv_left_eye_x": None,
            "opencv_left_eye_y": None,
            "opencv_mouth_x": None,
            "opencv_mouth_y": None,
            "opencv_nose_x": None,
            "opencv_nose_y": None,
            "opencv_right_eye_x": None,
            "opencv_right_eye_y": None,
            "error": str(e),
        }
    return output


# def get_img_quality(img: np.array) -> dict:
#     try:
#         # if img.shape[-1] == 4:
#         #     img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
#         image_quality = bk.score(img)
#     except Exception as e:
#         traceback.print_exception(e)
#         return {
#             "brisque_quality": None,
#             "error": str(e),
#         }
#     return {"brisque_quality": image_quality}


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
#         return {
#             "age": None,
#             "gender": None,
#             "ethnicity": None,
#             "emotion": None,
#             "error": str(e),
#         }
#     return {
#         "age": face_attributes["age"],
#         "gender": face_attributes["gender"],
#         "ethnicity": face_attributes["dominant_race"],
#         "emotion": face_attributes["dominant_emotion"],
#     }


def is_smile(img: np.array) -> dict:
    try:
        # img_h, img_w, _ = img.shape
        smileCascade = cv.CascadeClassifier(f"{BQAT_CWD}haarcascade_smile.xml")
        if len(img.shape) == 3 and img.shape[2] == 3:
            gs = cv.resize(cv.cvtColor(img, cv.COLOR_BGR2GRAY), (128, 128))
        else:
            gs = cv.resize(img.copy(), (128, 128))
        smile = smileCascade.detectMultiScale(
            gs,
            scaleFactor=1.5,
            minNeighbors=30,
            # minSize=(int(img_h / 6), int(img_w / 3)),
            # maxSize=(int(img_h / 4), int(img_w / 2)),
            flags=cv.CASCADE_DO_CANNY_PRUNING,
        )
        smile = 1 if len(smile) >= 1 else 0
    except Exception as e:
        traceback.print_exception(e)
        return {
            "smile": None,
            "error": str(e),
        }
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

        right_vertical = np.mean([euclidean(up, lo) for up, lo in zip(r_u, r_l)])
        right_horizontal = euclidean(r_c[0], r_c[1])
        left_vertical = np.mean([euclidean(up, lo) for up, lo in zip(l_u, l_l)])
        left_horizontal = euclidean(l_c[0], l_c[1])

        right_ratio = right_vertical / right_horizontal
        left_ratio = left_vertical / left_horizontal

        threshold = 0.15
        right = 1 if right_ratio < threshold else 0
        left = 1 if left_ratio < threshold else 0
    except Exception as e:
        traceback.print_exception(e)
        return {
            "eye_closed_left": None,
            "eye_closed_right": None,
            "error": str(e),
        }
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
                (r_x * (output["face_right"] - output["face_left"]))
                + output["face_left"]
            ),
            int(
                (r_y * (output["face_bottom"] - output["face_top"]))
                + output["face_top"]
            ),
        )
        pupil_l = (
            int(
                (l_x * (output["face_right"] - output["face_left"]))
                + output["face_left"]
            ),
            int(
                (l_y * (output["face_bottom"] - output["face_top"]))
                + output["face_top"]
            ),
        )
        dist = 0
        for u, v in zip(pupil_r, pupil_l):
            dist += (u - v) ** 2
    except Exception as e:
        traceback.print_exception(e)
        return {
            "ipd": None,
            "pupil_right_x": None,
            "pupil_right_y": None,
            "pupil_left_x": None,
            "pupil_left_y": None,
            "error": str(e),
        }
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
        except Exception:
            raise RuntimeError("unable to get head pose angles.")

        if abs(degree_yaw) < 3:
            pose_yaw = "FORWARD"
        else:
            pose_yaw = "LEFT" if degree_yaw > 0 else "RIGHT"
        if abs(degree_pitch) < 3:
            pose_pitch = "LEVEL"
        else:
            pose_pitch = "UP" if degree_pitch > 0 else "DOWN"
        if abs(degree_roll) < 3:
            pose_roll = "LEVEL"
        else:
            pose_roll = "ANTI-CLOCKWISE" if degree_roll > 0 else "CLOCKWISE"
    except Exception as e:
        traceback.print_exception(e)
        return {
            "yaw_pose": None,
            "yaw_degree": None,
            "pitch_pose": None,
            "pitch_degree": None,
            "roll_pose": None,
            "roll_degree": None,
            "error": str(e),
        }
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

        nose_bridge = [5, 9, 243, 463]

        nose_bridge_x = [
            lm.x for i, lm in enumerate(face_mesh.landmark) if i in nose_bridge
        ]
        nose_bridge_y = [
            lm.y for i, lm in enumerate(face_mesh.landmark) if i in nose_bridge
        ]

        y_min = int(min(nose_bridge_y) * img_h)
        y_max = int(max(nose_bridge_y) * img_h)
        x_min = int(min(nose_bridge_x) * img_w)
        x_max = int(max(nose_bridge_x) * img_w)

        target_area = img[y_min:y_max, x_min:x_max]

        if not target_area.any():
            return {"glasses": None}

        if len(img.shape) == 3 and img.shape[2] == 3:
            target_area = cv.cvtColor(target_area, cv.COLOR_BGR2GRAY)

        target_area = np.float32(target_area)

        contrast_factor = 0.9
        target_area = cv.convertScaleAbs(target_area, alpha=contrast_factor, beta=0)

        sharpen_kernel = np.array(
            [
                [1, 2, 1],
                [0, 0, 0],
                [-1, -2, -1],
            ]
        )
        out = cv.filter2D(target_area, -1, sharpen_kernel)

        sharpen_kernel = np.array(
            [
                [0, -1, 0],
                [-1, 4, -1],
                [0, -1, 0],
            ]
        )
        out = cv.filter2D(out, -1, sharpen_kernel)

        edge = cv.Canny(out, threshold1=300, threshold2=500)

        center = edge[:, edge.shape[1] // 2]

    except Exception as e:
        traceback.print_exception(e)
        return {
            "glasses": None,
            "error": str(e),
        }
    return {"glasses": 1 if np.any(center == 255) else 0}


def ofiq_engine(
    path: str,
    dir: bool = False,
) -> dict:
    """Process image with native BQAT engine.

    Args:
        path: Path to the image or directory.
        dir: Whether the input path is a directory.

    Returns:
        A dictionary containing the results from OFIQ.
    """
    output = {"log": []}

    if dir:
        if (meta := get_ofiq_attr(path, dir=dir)).get("error"):
            output["log"].append({"ofiq attributes": meta["error"]})
        output["results"] = meta.get("results", [])
    else:
        output["file"] = path
        output.update(meta) if not (meta := get_ofiq_attr(path, dir=dir)).get(
            "error"
        ) else output["log"].append({"ofiq attributes": meta["error"]})
    if not output["log"]:
        output.pop("log")
    return output


def get_ofiq_attr(path: str, dir: bool = False) -> list:
    try:
        if dir:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_output = Path(temp_dir) / f"{uuid4()}.csv"
                output = {"results": []}
                raw = subprocess.run(
                    [
                        f"{OFIQ_CWD}bin/OFIQSampleApp",
                        "-c",
                        f"{OFIQ_CWD}ofiq_config.jaxn",
                        "-i",
                        path,
                        "-o",
                        temp_output,
                    ],
                    capture_output=True,
                )
                if error := raw.stderr.decode():
                    output["error"] = error
                    print(f"ofiq engine error: {error}")
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
        else:
            output = {}
            raw = subprocess.check_output(
                [
                    f"{OFIQ_CWD}bin/OFIQSampleApp",
                    "-c",
                    f"{OFIQ_CWD}ofiq_config.jaxn",
                    "-i",
                    path,
                ]
            )
            content = StringIO(raw.decode())

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
    except Exception as e:
        traceback.print_exception(e)
        return {
            "unified_quality_score": None,
            "background_uniformity": None,
            "illumination_uniformity": None,
            "luminance_mean": None,
            "luminance_variance": None,
            "under_exposure_prevention": None,
            "over_exposure_prevention": None,
            "dynamic_range": None,
            "sharpness": None,
            "compression_artifacts": None,
            "natural_colour": None,
            "single_face_present": None,
            "eyes_open": None,
            "mouth_closed": None,
            "eyes_visible": None,
            "mouth_occlusion_prevention": None,
            "face_occlusion_prevention": None,
            "inter_eye_distance": None,
            "head_size": None,
            "leftward_crop_of_the_face_image": None,
            "rightward_crop_of_the_face_image": None,
            "margin_above_of_the_face_image": None,
            "margin_below_of_the_face_image": None,
            "head_pose_yaw": None,
            "head_pose_pitch": None,
            "head_pose_roll": None,
            "expression_neutrality": None,
            "no_head_coverings": None,
            "unified_quality_score_scalar": None,
            "background_uniformity_scalar": None,
            "illumination_uniformity_scalar": None,
            "luminance_mean_scalar": None,
            "luminance_variance_scalar": None,
            "under_exposure_prevention_scalar": None,
            "over_exposure_prevention_scalar": None,
            "dynamic_range_scalar": None,
            "sharpness_scalar": None,
            "compression_artifacts_scalar": None,
            "natural_colour_scalar": None,
            "single_face_present_scalar": None,
            "eyes_open_scalar": None,
            "mouth_closed_scalar": None,
            "eyes_visible_scalar": None,
            "mouth_occlusion_prevention_scalar": None,
            "face_occlusion_prevention_scalar": None,
            "inter_eye_distance_scalar": None,
            "head_size_scalar": None,
            "leftward_crop_of_the_face_image_scalar": None,
            "rightward_crop_of_the_face_image_scalar": None,
            "margin_above_of_the_face_image_scalar": None,
            "margin_below_of_the_face_image_scalar": None,
            "head_pose_yaw_scalar": None,
            "head_pose_pitch_scalar": None,
            "head_pose_roll_scalar": None,
            "expression_neutrality_scalar": None,
            "no_head_coverings_scalar": None,
            "error": str(e),
        }
    return output


# def get_offset(output: dict) -> dict:
#     # Use face area bouding box to calculate face offset
#     try:
#         return {
#             "face_offset_x": (
#                 (output["image_width"] - (output["face_right"] - output["face_left"]))
#                 / 2
#                 - (output["image_width"] - output["face_right"])
#             )
#             / output["image_width"],
#             "face_offset_y": (
#                 (output["image_height"] - (output["face_bottom"] - output["face_top"]))
#                 / 2
#                 - (output["image_height"] - output["face_bottom"])
#             )
#             / output["image_height"],
#         }
#     except Exception as e:
#         traceback.print_exception(e)
#         return {
#             "face_offset_x": None,
#             "face_offset_y": None,
#             "error": str(e),
#         }


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
                    lm.x * (output["face_right"] - output["face_left"])
                    + output["face_left"]
                )
                centre_y = (
                    lm.y * (output["face_bottom"] - output["face_top"])
                ) + output["face_top"]
                break

        offset_x = (centre_x - img_w / 2) / img_w
        offset_y = (centre_y - img_h / 2) / img_h
    except Exception as e:
        traceback.print_exception(e)
        return {
            "face_offset_x": None,
            "face_offset_y": None,
            "error": str(e),
        }
    return {
        "face_offset_x": offset_x,
        "face_offset_y": offset_y,
    }


def get_face_ratio(output: dict) -> dict:
    try:
        return {
            "face_ratio": (output["face_right"] - output["face_left"])
            * (output["face_bottom"] - output["face_top"])
            / (output["image_width"] * output["image_height"])
        }
    except Exception as e:
        traceback.print_exception(e)
        return {
            "face_ratio": None,
            "error": str(e),
        }


def get_gaze_degree(face_mesh: object) -> dict:
    try:
        right_pupil_centre = 473
        right_eye_right = 263
        right_eye_left = 362
        right_eye_top = 386
        right_eye_bottom = 374
        left_pupil_centre = 468
        left_eye_right = 133
        left_eye_left = 33
        left_eye_top = 159
        left_eye_bottom = 145

        landmarks = {index: lm for index, lm in enumerate(face_mesh.landmark)}

        rpc = (landmarks[right_pupil_centre].x, landmarks[right_pupil_centre].y)
        rer = (landmarks[right_eye_right].x, landmarks[right_eye_right].y)
        rel = (landmarks[right_eye_left].x, landmarks[right_eye_left].y)
        ret = (landmarks[right_eye_top].x, landmarks[right_eye_top].y)
        reb = (landmarks[right_eye_bottom].x, landmarks[right_eye_bottom].y)
        lpc = (landmarks[left_pupil_centre].x, landmarks[left_pupil_centre].y)
        ler = (landmarks[left_eye_right].x, landmarks[left_eye_right].y)
        lel = (landmarks[left_eye_left].x, landmarks[left_eye_left].y)
        let = (landmarks[left_eye_top].x, landmarks[left_eye_top].y)
        leb = (landmarks[left_eye_bottom].x, landmarks[left_eye_bottom].y)

        gaze_right_x = (rpc[0] - ((rer[0] + rel[0]) / 2)) / (rer[0] - rel[0])
        gaze_right_y = (rpc[1] - ((ret[1] + reb[1]) / 2)) / (reb[1] - ret[1])
        gaze_left_x = (lpc[0] - ((ler[0] + lel[0]) / 2)) / (ler[0] - lel[0])
        gaze_left_y = (lpc[1] - ((let[1] + leb[1]) / 2)) / (leb[1] - let[1])

    except Exception as e:
        traceback.print_exception(e)
        return {
            "gaze_right_x": None,
            "gaze_right_y": None,
            "gaze_left_x": None,
            "gaze_left_y": None,
            "error": str(e),
        }
    return {
        "gaze_right_x": gaze_right_x,
        "gaze_right_y": gaze_right_y,
        "gaze_left_x": gaze_left_x,
        "gaze_left_y": gaze_left_y,
    }


def get_pupil_color(face_mesh: object, img: np.array, output: dict) -> dict:
    try:
        right_pupil_centre = 473
        right_pupil_right = 474
        right_pupil_top = 475
        right_pupil_left = 476
        right_pupil_bottom = 477
        left_pupil_centre = 468
        left_pupil_right = 469
        left_pupil_top = 470
        left_pupil_left = 471
        left_pupil_bottom = 472

        landmarks = {index: lm for index, lm in enumerate(face_mesh.landmark)}

        rpc = (landmarks[right_pupil_centre].x, landmarks[right_pupil_centre].y)
        rpr = (landmarks[right_pupil_right].x, landmarks[right_pupil_right].y)
        rpt = (landmarks[right_pupil_top].x, landmarks[right_pupil_top].y)
        rpl = (landmarks[right_pupil_left].x, landmarks[right_pupil_left].y)
        rpb = (landmarks[right_pupil_bottom].x, landmarks[right_pupil_bottom].y)
        lpc = (landmarks[left_pupil_centre].x, landmarks[left_pupil_centre].y)
        lpr = (landmarks[left_pupil_right].x, landmarks[left_pupil_right].y)
        lpt = (landmarks[left_pupil_top].x, landmarks[left_pupil_top].y)
        lpl = (landmarks[left_pupil_left].x, landmarks[left_pupil_left].y)
        lpb = (landmarks[left_pupil_bottom].x, landmarks[left_pupil_bottom].y)

        def get_color(pc, pr, pt, pl, pb) -> str:
            top = int(
                (pc[1] + pt[1]) / 2 * (output["face_bottom"] - output["face_top"])
            )
            bottom = int(
                (pc[1] + pb[1]) / 2 * (output["face_bottom"] - output["face_top"])
            )
            right = int(
                (pc[0] + pr[0]) / 2 * (output["face_right"] - output["face_left"])
            )
            left = int(
                (pc[0] + pl[0]) / 2 * (output["face_right"] - output["face_left"])
            )

            if (pupil := img[top:bottom, left:right]).any():
                pupil = cv.cvtColor(pupil, cv.COLOR_BGR2RGB)
            else:
                raise RuntimeError("fail to crop pupil area")

            # Reshape the image to be a list of pixels
            pixels = pupil.reshape(-1, 3)

            # Handle low resolution images (will intriduce warning for duplicate pixels)
            if len(pixels) < 10:
                pixels = np.repeat(pixels, 3, axis=0)
                n_clusters = 1
            else:
                n_clusters = 3

            # Use KMeans to find the most common colors
            kmeans = KMeans(n_clusters=n_clusters)
            kmeans.fit(pixels)

            # Get the cluster centers
            centers = kmeans.cluster_centers_

            # Calculate the size of each cluster
            _, counts = np.unique(kmeans.labels_, return_counts=True)

            # Find the largest cluster
            largest_cluster_index = np.argmax(counts)

            # Get the largest cluster center
            largest_center = centers[largest_cluster_index]
            dominant_color = tuple(map(int, largest_center))

            return get_color_name(dominant_color), dominant_color

        pupil_color_right, pupil_right_rgb = get_color(rpc, rpr, rpt, rpl, rpb)
        pupil_color_left, pupil_left_rgb = get_color(lpc, lpr, lpt, lpl, lpb)

    except Exception as e:
        traceback.print_exception(e)
        return {
            "pupil_colour_right_name": None,
            "pupil_colour_right_rgb": None,
            "pupil_colour_left_name": None,
            "pupil_colour_left_rgb": None,
            "error": str(e),
        }
    return {
        "pupil_colour_right_name": pupil_color_right,
        "pupil_colour_right_rgb": pupil_right_rgb,
        "pupil_colour_left_name": pupil_color_left,
        "pupil_colour_left_rgb": pupil_left_rgb,
    }


def get_image_meta(img: np.array, file_path: str) -> dict:
    try:
        if len(img.shape) == 3 and img.shape[2] == 3:
            gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        else:
            gray_image = img.copy()
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

        file_size_bytes = Path(file_path).stat().st_size
        with PIL.Image.open(file_path) as img:
            image_format = img.format
            image_mode = img.mode
    except Exception as e:
        traceback.print_exception(e)
        return {
            "brightness": None,
            "dynamic_range": None,
            "sharpness": None,
            # "luminance": None,
            "contrast": None,
            "file_size": None,
            "image_format": None,
            "image_mode": None,
            "error": str(e),
        }
    return {
        "brightness": float(brightness),
        "dynamic_range": float(dynamic_range),
        "sharpness": float(variance),
        # "luminance": float(luminance),
        "contrast": contrast,
        "file_size_bytes": file_size_bytes,
        "image_format": image_format,
        "image_mode": image_mode,
    }


def get_background_color(img: np.array) -> dict:
    try:
        # Create a image segmenter instance with the image mode:
        options = ImageSegmenterOptions(
            base_options=BaseOptions(
                model_asset_path=f"{BQAT_CWD}selfie_segmenter.tflite"
            ),
            running_mode=RunningMode.IMAGE,
            output_category_mask=True,
        )

        # Load the input image from a numpy array.
        mp_image = Image(image_format=ImageFormat.SRGB, data=img)

        with ImageSegmenter.create_from_options(options) as segmenter:
            segmented_masks = segmenter.segment(mp_image)

        image_data = cv.cvtColor(mp_image.numpy_view(), cv.COLOR_BGR2RGB)

        condition = (
            np.stack((segmented_masks.category_mask.numpy_view(),) * 3, axis=-1) > 0.2
        )

        # # Generate output with transparent foreground
        # fg_image = np.zeros(
        #     (
        #         image_data.shape[0],
        #         image_data.shape[1],
        #         4,
        #     ),
        #     dtype=np.uint8,
        # )
        # rgba_image = np.concatenate(
        #     (
        #         image_data,
        #         np.ones(
        #             (
        #                 image_data.shape[0],
        #                 image_data.shape[1],
        #                 1,
        #             ),
        #             dtype=np.uint8,
        #         )
        #         * 255,
        #     ),
        #     axis=2,
        # )
        # output_image = np.where(condition, rgba_image, fg_image)

        # Generate output without foreground
        fg_image = np.full(
            (
                image_data.shape[0],
                image_data.shape[1],
                3,
            ),
            111,
            dtype=np.uint8,
        )
        output_image = np.where(condition, image_data, fg_image)

        # Reshape the image to be a list of pixels
        pixels = output_image.reshape(-1, 3)

        # Remove foreground pixels
        if not (pixels := pixels[~(np.all(pixels == [111, 111, 111], axis=1))]).any():
            raise RuntimeError("fail to get background image.")

        # Use KMeans to find the most common colors
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(pixels)

        # Get the cluster centers
        centers = kmeans.cluster_centers_

        # Calculate the size of each cluster
        _, counts = np.unique(kmeans.labels_, return_counts=True)

        # Find the largest cluster
        largest_cluster_index = np.argmax(counts)

        # Get the largest cluster center
        largest_center = centers[largest_cluster_index]
        dominant_color = tuple(map(int, largest_center))

        background_color = get_color_name(dominant_color)

        # Calculate the standard deviation of the R, G, and B channels separately
        output_image = cv.GaussianBlur(output_image, (5, 5), 0)

        # Remove foreground pixels
        r_pixels = output_image[:, :, 0].flatten()
        r_pixels = r_pixels[r_pixels != 111]

        g_pixels = output_image[:, :, 1].flatten()
        g_pixels = g_pixels[g_pixels != 111]

        b_pixels = output_image[:, :, 2].flatten()
        b_pixels = b_pixels[b_pixels != 111]

        r_std = np.std(r_pixels)
        g_std = np.std(g_pixels)
        b_std = np.std(b_pixels)

        # Calculate overall average color standard deviation
        background_variance = (r_std + g_std + b_std) / 3

    except Exception as e:
        traceback.print_exception(e)
        return {
            "background_colour_name": None,
            "background_colour_rgb": None,
            "background_colour_variance": None,
            "error": str(e),
        }
    return {
        "background_colour_name": background_color,
        "background_colour_rgb": dominant_color,
        "background_colour_variance": background_variance,
    }


def get_hair_cover(img: np.array, output: dict) -> dict:
    try:
        # Create a image segmenter instance with the image mode:
        options = ImageSegmenterOptions(
            base_options=BaseOptions(
                model_asset_path=f"{BQAT_CWD}hair_segmenter.tflite",
            ),
            running_mode=RunningMode.IMAGE,
            output_category_mask=True,
        )

        # Load the input image from a numpy array.
        mp_image = Image(image_format=ImageFormat.SRGB, data=img)

        with ImageSegmenter.create_from_options(options) as segmenter:
            segmented_masks = segmenter.segment(mp_image)

        image_data = cv.cvtColor(mp_image.numpy_view(), cv.COLOR_BGR2RGB)

        condition = (
            np.stack(
                (segmented_masks.category_mask.numpy_view(),) * 3,
                axis=-1,
            )
            > 0.2
        )

        # Pure black background
        bg_image = np.full(
            (
                image_data.shape[0],
                image_data.shape[1],
                3,
            ),
            0,
            dtype=np.uint8,
        )
        # Pure white foreground
        fg_image = np.full(
            (
                image_data.shape[0],
                image_data.shape[1],
                3,
            ),
            255,
            dtype=np.uint8,
        )
        # Mask hair area
        hair_area_image = np.where(condition, fg_image, bg_image)

        # Mask face area
        cv.rectangle(
            bg_image,
            (
                output.get("face_left"),
                output.get("face_top"),
            ),
            (
                output.get("face_right"),
                output.get("face_bottom"),
            ),
            (255, 255, 255),
            -1,
        )

        face_area_hsv = cv.cvtColor(bg_image, cv.COLOR_BGR2HSV)
        hair_area_image_hsv = cv.cvtColor(hair_area_image, cv.COLOR_BGR2HSV)

        mask_face = cv.inRange(
            face_area_hsv, np.array([0, 0, 100]), np.array([0, 0, 255])
        )
        mask_hair = cv.inRange(
            hair_area_image_hsv, np.array([0, 0, 100]), np.array([0, 0, 255])
        )

        # Find intersection of the two masks
        intersection = cv.bitwise_and(mask_face, mask_hair)
        covered_area = cv.countNonZero(intersection)

        # Face area
        face_area = (output.get("face_right") - output.get("face_left")) * (
            output.get("face_bottom") - output.get("face_top")
        )

        overlap = covered_area / face_area if face_area > 0 else 0
    except Exception as e:
        traceback.print_exception(e)
        return {
            "hair_coverage": None,
            "error": str(e),
        }
    return {
        "hair_coverage": overlap,
    }


# def is_blurry(img: np.array) -> dict:
#     try:
#         if len(img.shape) == 3 and img.shape[2] == 3:
#             gs = cv.cvtColor(img.copy(), cv.COLOR_BGR2GRAY)
#         else:
#             gs = img.copy()
#         blur_metric = measure.blur_effect(gs, h_size=15)
#         lap = cv.Laplacian(gs, cv.CV_64F)
#     except Exception as e:
#         return {
#             "blur_lap_var": None,
#             "blurriness": None,
#             "error": str(e),
#         }
#     return {
#         "blur_lap_var": lap.var(),
#         "blurriness": blur_metric,
#     }


def fusion_engine(path: str, fusion_code: int = 6, cpu: float = 0.8):
    """Process images using multiple engines and combine results.

    Args:
        path: Path to the image or directory containing images.
        fusion_code: Bitmask indicating which engines to use (e.g., BQAT, OFIQ, BIQT).
        cpu: Fraction of CPU resources to allocate for multiprocessing.

    Returns:
        A dictionary containing combined results from the selected engines.
    """

    def process_images(engine_func, path, pool, chunk_size=100):
        files = [p.as_posix() for p in Path(path).rglob("*")]
        results = []
        for chunk_start in range(0, len(files), chunk_size):
            chunk = files[chunk_start : chunk_start + chunk_size]
            results.extend(pool.imap_unordered(engine_func, chunk))
        return results

    output = {"log": []}
    results = []
    with multiprocessing.Pool(
        processes=cpu_usage_to_processes(cpu),
    ) as pool:
        if fusion_code & Engine.BQAT:
            results.append(("bqat", process_images(default_engine, path, pool)))
        if fusion_code & Engine.OFIQ:
            ofiq = ofiq_engine(path, dir=True)
            if ofiq_logs := ofiq.get("log"):
                output["log"].extend(ofiq_logs)
            results.append(("ofiq", ofiq.get("results", [])))
        if fusion_code & Engine.BIQT:
            results.append(("biqt", process_images(biqt_engine, path, pool)))

    pairs = combinations(results, 2)
    # Merge results from all combinations of engines
    results = [
        merge_results(
            res_a=item_a[1],
            res_b=item_b[1],
            name_a=item_a[0],
            name_b=item_b[0],
        )
        for item_a, item_b in pairs
    ]

    # Use reduce to combine all merged results into a single output
    # This step iteratively applies merge_results to the list of results
    combined_results = reduce(merge_results, results) if results else []

    # Store the final combined results in the output dictionary
    output["results"] = combined_results

    if not output["log"]:
        output.pop("log")

    return output


# def get_hijab(img: np.array, output: dict, margin_factor: int = 0.4) -> dict:
#     try:
#         img = prepare_input(
#             img=img,
#             meta=output,
#             width=320,
#             height=320,
#             margin_factor=margin_factor,
#             onnx=True,
#         )

#         # Detect any hijab
#         # # Use torch model
#         # model = YOLO(f"{CWD}hijab_classifier.pt")
#         # model.to("cpu")
#         # results = model(
#         #     img,
#         #     max_det=1,
#         #     verbose=False,
#         #     device="cpu",
#         # )
#         # hijab_detection = results[0].probs.data[2].item()

#         # Use onnx model
#         sess = ort.InferenceSession(f"{BQAT_CWD}hijab_classifier.onnx")
#         results = sess.run(None, {sess.get_inputs()[0].name: img})
#         hijab_detection = results[0][0][2].item()
#         if hijab_detection < 0.1:
#             hijab_detection = 0

#         # Detect dark hijab
#         if hijab_detection > 0.9:
#             sess = ort.InferenceSession(f"{BQAT_CWD}hijab_dark_classifier.onnx")
#             results = sess.run(None, {sess.get_inputs()[0].name: img})
#             hijab_detection_dark = results[0][0][2].item()
#             if hijab_detection_dark < 0.1:
#                 hijab_detection_dark = 0
#         else:
#             hijab_detection_dark = 0

#     except Exception as e:
#         traceback.print_exception(e)
#         return {
#             "hijab_detection": None,
#             "hijab_detection_dark": None,
#             "error": str(e),
#         }
#     return {
#         "headgear_detection": hijab_detection,
#         "headgear_detection_dark": hijab_detection_dark,
#     }


# def get_hijab_batch(path: str, margin_factor: int = 0.4) -> list:
#     try:
#         with tempfile.TemporaryDirectory() as tmpdir:
#             results = []
#             files = Path(path).rglob("*.*")
#             for f in files:
#                 result = {"file": f.as_posix(), "log": []}
#                 try:
#                     try:
#                         img = cv.imread(f)
#                         h, w, _ = img.shape
#                         result.update(
#                             {
#                                 "image_height": h,
#                                 "image_width": w,
#                             }
#                         )
#                     except Exception as e:
#                         raise RuntimeError(f"load image: {str(e)}")

#                     result.update(meta) if not (meta := get_face(img)).get(
#                         "error"
#                     ) else result["log"].append({"face detection": meta["error"]})

#                     img = prepare_input(
#                         img=img,
#                         meta=result,
#                         width=320,
#                         height=320,
#                         margin_factor=margin_factor,
#                     )
#                     cv.imwrite(Path(tmpdir) / f.name, img)
#                 except Exception as e:
#                     result["log"].append({"hijab detector preprocessing": str(e)})

#                 if not result.get("log"):
#                     result.pop("log")

#                 results.append(result)

#             model = YOLO(f"{BQAT_CWD}hijab_classifier.pt")
#             model.to("cpu")

#             predicts = model(
#                 source=tmpdir,
#                 max_det=1,
#                 verbose=False,
#                 device="cpu",
#                 stream=True,
#             )

#             temp = []
#             count = 0
#             for p in predicts:
#                 count += 1
#                 for r in results:
#                     if Path(r["file"]).name == Path(p.path).name:
#                         r["hijab_detection"] = p.probs.data[2].item()
#                     temp.append(r)

#             results = temp

#         output = {"results": results}
#     except Exception as e:
#         traceback.print_exception(e)
#         output = {"results": results, "error": str(e)}
#     return output


def get_face(
    img: cv.typing.MatLike,
    output: dict = {"log": []},
    confidence: float = 0.7,
) -> tuple[cv.typing.MatLike, dict]:
    """
    Detects a face in the given image and returns the cropped face region along with detection metadata.

    This function attempts to detect a face using a full-range detection model first. If no face is found,
    it falls back to a short-range detection model. The face with the highest detection score is selected.
    The function returns the cropped face region and updates the provided output dictionary with detection
    confidence and bounding box coordinates.

    Args:
        img (cv.typing.MatLike): The input image in which to detect a face.
        output (dict, optional): A dictionary to store detection metadata and logs. Defaults to {"log": []}.
        confidence (float, optional): Minimum confidence threshold for face detection. Defaults to 0.7.

    Returns:
        tuple[cv.typing.MatLike, dict]: A tuple containing the cropped face region as an image and the updated output dictionary.

    Raises:
        RuntimeError: If no face is found in the image.

    Notes:
        - If no face is detected, the original image is returned and the error is logged in the output dictionary.
        - The output dictionary contains the detection confidence and bounding box coordinates (face_left, face_top, face_right, face_bottom).
    """
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
                    "face_left": bbox["left"],
                    "face_top": bbox["upper"],
                    "face_right": bbox["right"],
                    "face_bottom": bbox["lower"],
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
        output.update(
            {
                "face_detection": output.get("face_detection", None),
                "face_left": output.get("face_left", None),
                "face_top": output.get("face_top", None),
                "face_right": output.get("face_right", None),
                "face_bottom": output.get("face_bottom", None),
            }
        )
        target_region = img.copy()
    return target_region, output


def get_mesh(
    img: cv.typing.MatLike,
    output: dict = {"log": []},
    confidence: float = 0.7,
) -> tuple[object, dict]:
    """
    Extracts a face mesh from an input image using the FaceMesh model.

    Args:
        img (cv.typing.MatLike): The input image in which to detect the face mesh.
        output (dict, optional): A dictionary for logging information. Defaults to {"log": []}.
        confidence (float, optional): Minimum confidence threshold for face detection. Defaults to 0.7.

    Returns:
        tuple[object, dict]: A tuple containing the extracted face mesh object (or None if extraction fails)
            and the output dictionary with logs and error messages.

    Notes:
        - Only the first detected face mesh is returned if multiple faces are present.
        - If face mesh extraction fails, an error message is appended to the output log.
        - Exceptions during processing are caught and logged in the output dictionary.
    """
    try:
        with FaceMesh(
            static_image_mode=True,
            min_detection_confidence=confidence,
            max_num_faces=1,
            refine_landmarks=True,
        ) as model:
            mesh = model.process(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        if mesh.multi_face_landmarks:
            mesh = mesh.multi_face_landmarks[0]
        else:
            output["log"].append({"face mesh": "failed to extract face mesh"})
            mesh = None
    except Exception as e:
        traceback.print_exception(e)
        mesh = None
        output["log"].append({"face mesh": str(e)})
    return mesh, output


# def get_colour_temperature(img: np.array) -> dict:
#     try:
#         img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

#         # Initialize a list to store temperatures
#         temperatures = []

#         # Iterate over all pixels and calculate the color temperature
#         for row in img:
#             for pixel in row:
#                 r, g, b = pixel
#                 try:
#                     temp = rgb_to_color_temperature(r, g, b)
#                 except Exception:
#                     continue
#                 temperatures.append(temp)

#         # Calculate the average color temperature
#         average_temp = np.mean(temperatures)

#     except RuntimeError as e:
#         return {"error": str(e)}
#     except Exception as e:
#         traceback.print_exception(e)
#         return {"error": str(e)}
#     return {
#         "colour_temperature": average_temp,
#     }


def get_brightness_variance(img: np.array, block_size=(50, 50)) -> dict:
    try:
        img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        img = img[:, :, 2]
        min_val = np.min(img)
        max_val = np.max(img)
        img = ((img - min_val) / (max_val - min_val)) * 255
        h, w = img.shape
        brightness_values = []

        # Divide the image into blocks and calculate mean brightness for each block
        for i in range(0, h, block_size[0]):
            for j in range(0, w, block_size[1]):
                block = img[i : i + block_size[0], j : j + block_size[1]]
                mean_brightness = np.mean(block)
                brightness_values.append(mean_brightness)

        brightness_variance = np.std(brightness_values)

    except Exception as e:
        traceback.print_exception(e)
        return {
            "brightness_variance": None,
            "error": str(e),
        }
    return {
        "brightness_variance": brightness_variance,
    }


def get_head_location(
    face_mesh: object,
    img: np.array,
    target_region: np.array,
    output: dict,
) -> dict:
    try:
        img_h, img_w, _ = target_region.shape

        left_side = [127, 234, 93]
        right_side = [356, 454, 323]
        bottom_chin = [148, 152, 377]

        head_x = [
            lm.x
            for i, lm in enumerate(face_mesh.landmark)
            if i in left_side + right_side
        ]
        head_y = [lm.y for i, lm in enumerate(face_mesh.landmark) if i in bottom_chin]

        y_max = int(int(max(head_y) * img_h) + output["face_top"])
        x_min = int(int(min(head_x) * img_w) + output["face_left"])
        x_max = int(int(max(head_x) * img_w) + output["face_left"])

        options = ImageSegmenterOptions(
            base_options=BaseOptions(
                model_asset_path=f"{BQAT_CWD}selfie_segmenter.tflite",
            ),
            running_mode=RunningMode.IMAGE,
            output_category_mask=True,
        )

        mp_image = Image(image_format=ImageFormat.SRGB, data=img)

        with ImageSegmenter.create_from_options(options) as segmenter:
            segmented_masks = segmenter.segment(mp_image)
            image_data = cv.cvtColor(mp_image.numpy_view(), cv.COLOR_BGR2RGB)

            condition = (
                np.stack(
                    (segmented_masks.category_mask.numpy_view(),) * 3,
                    axis=-1,
                )
                > 0.2
            )

            bg_image = np.full(
                (
                    image_data.shape[0],
                    image_data.shape[1],
                    3,
                ),
                0,
                dtype=np.uint8,
            )
            fg_image = np.full(
                (
                    image_data.shape[0],
                    image_data.shape[1],
                    3,
                ),
                255,
                dtype=np.uint8,
            )
            output_image = np.where(condition, fg_image, bg_image)

        black_pixels = np.column_stack(np.where(output_image == 0))
        if not black_pixels.size:
            y_min = -1
        else:
            y_min, _, _ = black_pixels[np.argmin(black_pixels[:, 0])]
            y_min = int(y_min)

        # x_max, _, _ = black_pixels[np.argmax(black_pixels[:, 1])]
        # x_min, _, _ = black_pixels[np.argmin(black_pixels[:, 1])]

    except Exception as e:
        traceback.print_exception(e)
        return {
            "head_top": None,
            "head_bottom": None,
            "head_right": None,
            "head_left": None,
            "error": str(e),
        }
    return {
        "head_top": y_min,
        "head_bottom": y_max,
        "head_right": x_max,
        "head_left": x_min,
    }
