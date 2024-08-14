import os
import re

import wsq
from PIL import Image, ImageOps


def convert(file, source, target, grayscale=False):
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
        converted = os.path.splitext(file)[0] + f".converted.{target}"
        img.save(converted)
        output = converted
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
