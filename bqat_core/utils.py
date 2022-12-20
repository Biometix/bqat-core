from PIL import Image, ImageOps
import wsq
import os


def convert(file, source, target, grayscale=False):
    if file.rsplit(".")[-1] in extend(source):
        img = Image.open(file)
        if grayscale:
            img = ImageOps.grayscale(img)
            # img = img.convert("L")
        converted = os.path.splitext(file)[0] + f".converted.{target}"
        img.save(converted)
        return converted
    return False


def extend(suffixes: list):
    suffixes = [s.casefold() for s in suffixes]
    extended = []
    for s in suffixes:
        extended.append(s.capitalize())
        extended.append(s.upper())
        extended.append(s)
    return extended
