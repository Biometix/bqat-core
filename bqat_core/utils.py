import os
import re
from pathlib import Path

import wsq
from PIL import Image, ImageOps


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
