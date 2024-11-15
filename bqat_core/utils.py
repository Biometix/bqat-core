import math
import os
import re
from pathlib import Path
from typing import List, Tuple

# import cv2 as cv
# import numpy as np
# from typing import List, Tuple
# import webcolors
import wsq
from PIL import Image, ImageOps

# from shapely import area, box, intersection, union


def convert(file, source, target, grayscale=False, directory=None):
    input_type = file.rsplit(".")[-1]
    if input_type == target:
        return file, input_type, target
    if target == "wsq":
        grayscale = True
    if input_type in extend(source):
        img = Image.open(file)
        if grayscale:
            img = ImageOps.grayscale(img)
            # img = img.convert("L")
        converted = Path(directory) / f"{os.path.splitext(file)[0]}.{target}"
        if not converted.parent.exists():
            converted.parent.mkdir(parents=True, exist_ok=True)
        img.save(converted)
        output = str(converted)
    else:
        output = file
    output_type = output.rsplit(".")[-1]
    return output, input_type, output_type


def extend(suffixes: list):
    suffixes = [s.casefold() for s in suffixes]
    extended = []
    for s in suffixes:
        extended.append(s.capitalize())
        extended.append(s.upper())
        extended.append(s)
    return extended


def camel_to_snake(name: str) -> str:
    name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()
    return re.sub(r"\.", "_", name)


def convert_values_to_number(d) -> dict[str, int | float | str | bool]:
    def try_convert(value) -> int | float | str | bool:
        try:
            return int(value)
        except ValueError:
            try:
                f = float(value)
                return f if not math.isnan(f) else None
            except Exception:
                return value

    return {k: try_convert(v) for k, v in d.items()}


# def closest_color(requested_color):
#     min_colors = {}
#     for name in webcolors.names(spec=webcolors.CSS3):
#         r_c, g_c, b_c = webcolors.name_to_rgb(name)
#         rd = (r_c - requested_color[0]) ** 2
#         gd = (g_c - requested_color[1]) ** 2
#         bd = (b_c - requested_color[2]) ** 2
#         min_colors[(rd + gd + bd)] = name
#     return min_colors[min(min_colors.keys())]


# def get_color_name(requested_color):
#     try:
#         closest_name = webcolors.rgb_to_name(requested_color)
#     except ValueError:
#         closest_name = closest_color(requested_color)
#     return closest_name


# def get_overlap_ratio(obstacles: List[Tuple[int]], target: Tuple[int]):
#     upper, lower, left, right = target[0], target[1], target[2], target[3]
#     face_poly = box(left, upper, right, lower)
#     obstacle_poly = box(0, 0, 0, 0)

#     for obj in obstacles:
#         x, y, w, h = obj[0], obj[1], obj[2], obj[3]

#         if x > right or y > lower or x + w < left or y + h < upper:
#             continue

#         else:
#             obstacle_poly = union(
#                 obstacle_poly,
#                 box(x, y, x + w, y + h),
#             )

#     inter = intersection(face_poly, obstacle_poly)

#     return area(inter) / area(face_poly)


# def resize_with_aspect_ratio(image, width, height, inter=cv.INTER_AREA):
#     # Grab the dimensions of the image
#     (h, w) = image.shape[:2]

#     # If the width is None, calculate the ratio of the height and construct the dimensions
#     if h >= w:
#         r = height / float(h)
#         dim = (int(w * r), height)

#     # Otherwise, the height is None, calculate the ratio of the width and construct the dimensions
#     else:
#         r = width / float(w)
#         dim = (width, int(h * r))

#     # Resize the image
#     resized = cv.resize(image, dim, interpolation=inter)

#     return resized


# def add_padding(image, width, height):
#     (h, w) = image.shape[:2]
#     top = (height - h) // 2
#     bottom = height - h - top
#     left = (width - w) // 2
#     right = width - w - left

#     color = [0, 0, 0]  # Black
#     padded_image = cv.copyMakeBorder(
#         image, top, bottom, left, right, cv.BORDER_CONSTANT, value=color
#     )
#     return padded_image


def merge_outputs(list_a: List[dict], list_b: List[dict], key: str) -> List[dict]:
    output = []
    dict_list_b = {item[key]: item for item in list_b}

    for item in list_a:
        logs = []
        target = item[key]
        if item.get("log"):
            logs.extend(item["log"])
        if target in dict_list_b:
            if dict_list_b[target].get("log"):
                logs.extend(dict_list_b[target]["log"])
            merged = item | dict_list_b[target]
            if logs:
                merged.update({"log": logs})
            output.append(merged)
        else:
            output.append(item)

    for item in list_b:
        if item[key] not in [i[key] for i in output]:
            output.append(item)

    return output


# def prepare_input(
#     img: np.array,
#     meta: dict,
#     width: int = 320,
#     height: int = 320,
#     margin_factor: int = 0.4,
# ) -> dict:
#     h, w, _ = img.shape

#     upper = meta.get("bbox_upper")
#     lower = meta.get("bbox_lower")
#     left = meta.get("bbox_left")
#     right = meta.get("bbox_right")

#     margin = int((right - left) * margin_factor)

#     right += margin
#     left -= margin
#     if right > w:
#         img = add_padding(img, width=w + (right - w) * 2, height=h)
#         left += right - w
#         h, w, _ = img.shape
#         right = w
#     if left < 0:
#         img = add_padding(img, width=w + (-left) * 2, height=h)
#         right += -left
#         h, w, _ = img.shape
#         left = 0

#     lower += int(margin * (1 - margin_factor))
#     upper -= int(margin * (1 + margin_factor))
#     if lower >= h:
#         img = add_padding(img, width=w, height=h + (lower - h) * 2)
#         upper += lower - h
#         h, w, _ = img.shape
#         lower = h
#     if upper <= 0:
#         img = add_padding(img, width=w, height=h + (-upper) * 2)
#         lower += -upper
#         h, w, _ = img.shape
#         upper = 0

#     img = img[upper:lower, left:right]
#     img = resize_with_aspect_ratio(img, width=width, height=height)
#     img = add_padding(img, width=width, height=height)

#     return img