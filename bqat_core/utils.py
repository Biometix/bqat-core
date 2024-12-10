import math
import os
import re
from pathlib import Path
from typing import List

# import cv2 as cv
# import numpy as np
# import webcolors
import wsq
from PIL import Image, ImageOps

# from scipy.interpolate import interp1d
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
#     onnx: bool = False,
# ) -> dict:
#     h, w, _ = img.shape

#     upper = meta.get("bbox_upper")
#     lower = meta.get("bbox_lower")
#     left = meta.get("bbox_left")
#     right = meta.get("bbox_right")

#     margin = int((right - left) * margin_factor)

#     # # Face crop with margin of black filling
#     # right += margin
#     # left -= margin
#     # if right > w:
#     #     img = add_padding(img, width=w + (right - w) * 2, height=h)
#     #     left += right - w
#     #     h, w, _ = img.shape
#     #     right = w
#     # if left < 0:
#     #     img = add_padding(img, width=w + (-left) * 2, height=h)
#     #     right += -left
#     #     h, w, _ = img.shape
#     #     left = 0

#     # lower += int(margin * (1 - margin_factor))
#     # upper -= int(margin * (1 + margin_factor))
#     # if lower >= h:
#     #     img = add_padding(img, width=w, height=h + (lower - h) * 2)
#     #     upper += lower - h
#     #     h, w, _ = img.shape
#     #     lower = h
#     # if upper <= 0:
#     #     img = add_padding(img, width=w, height=h + (-upper) * 2)
#     #     lower += -upper
#     #     h, w, _ = img.shape
#     #     upper = 0

#     # Calculate face crop with margin of original image
#     left = max(left - margin, 0)
#     upper = max(upper - margin, 0)
#     right = min(right + margin, w)
#     lower = min(lower + margin, h)

#     # Crop and resize
#     img = img[upper:lower, left:right]
#     img = resize_with_aspect_ratio(img, width=width, height=height)
#     img = add_padding(img, width=width, height=height)

#     # Additional preprocesses required by onnx model
#     if onnx:
#         img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # Convert BGR to RGB
#         img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
#         img = np.transpose(img, (2, 0, 1))  # Change shape to (C, H, W)
#         img = np.expand_dims(img, axis=0)  # Add batch dimension

#     return img


# def rgb_to_xyz(r, g, b):
#     # Normalize RGB to [0, 1]
#     r /= 255.0
#     g /= 255.0
#     b /= 255.0

#     # Apply gamma correction (sRGB to linear RGB)
#     r = r**2.2
#     g = g**2.2
#     b = b**2.2

#     # RGB to XYZ matrix (sRGB to XYZ D65)
#     x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
#     y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
#     z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b

#     return x, y, z


# # Convert XYZ to chromaticity coordinates (x, y)
# def xyz_to_chromaticity(x, y, z):
#     total = x + y + z
#     x_chromaticity = x / total
#     y_chromaticity = y / total
#     return x_chromaticity, y_chromaticity


# # Function to calculate color temperature based on RGB
# def rgb_to_color_temperature(r, g, b):
#     # Step 1: Convert RGB to XYZ
#     x, y, z = rgb_to_xyz(r, g, b)

#     # Step 2: Convert XYZ to chromaticity coordinates (x, y)
#     x_chromaticity, y_chromaticity = xyz_to_chromaticity(x, y, z)

#     # Planckian locus data (approximate x, y coordinates for different temperatures)
#     # This is a simplified version of the Planckian locus (typically we'd use a table with more points)
#     temperature_data = np.array(
#         [
#             (2000, 0.450, 0.370),  # 2000K
#             (2500, 0.433, 0.375),  # 2500K
#             (3000, 0.410, 0.380),  # 3000K
#             (3500, 0.389, 0.385),  # 3500K
#             (4000, 0.380, 0.390),  # 4000K
#             (4500, 0.371, 0.395),  # 4500K
#             (5000, 0.363, 0.400),  # 5000K
#             (5500, 0.356, 0.405),  # 5500K
#             (6000, 0.350, 0.410),  # 6000K
#             (6500, 0.345, 0.415),  # 6500K (daylight)
#         ]
#     )

#     # Extract temperature, x, y values for interpolation
#     temperatures = temperature_data[:, 0]
#     x_values = temperature_data[:, 1]
#     y_values = temperature_data[:, 2]

#     # Interpolation functions for Planckian locus (x, y) to Temperature
#     interpolate_x = interp1d(
#         x_values, temperatures, kind="linear", fill_value="extrapolate"
#     )
#     interpolate_y = interp1d(
#         y_values, temperatures, kind="linear", fill_value="extrapolate"
#     )

#     # Step 3: Interpolate the chromaticity values to find the corresponding color temperature
#     temperature_x = interpolate_x(x_chromaticity)
#     temperature_y = interpolate_y(y_chromaticity)

#     # Return the average temperature from both x and y interpolations
#     temperature = (temperature_x + temperature_y) / 2
#     return round(temperature)
