import csv
import re
import subprocess
from io import StringIO
from pathlib import Path

import pandas as pd

from .utils import NISQA_CWD


def process_speech(
    input_path: str,
    input_type: str,
):
    output = {"log": []}
    try:
        match input_type:
            case "file":
                # Transmitted Speech Analysis
                meta = get_metrics_transmitted(input_path)
                if meta.get("error"):
                    output["log"].append(
                        {"transmitted speech analysis": meta.pop("error")}
                    )
                output.update(meta)
                # Synthesized Speech Analysis
                meta = get_metrics_synthesized(input_path)
                if meta.get("error"):
                    output["log"].append(
                        {"synthesized speech analysis": meta.pop("error")}
                    )
                output.update(meta)
            case "folder":
                results, log = get_metrics_batch(input_path)
                output.update({"results": results})
                output["log"].extend(log)
            case _:
                raise TypeError("illegal task type")
        if not output.get("log"):
            output.pop("log")
    except Exception as e:
        raise RuntimeError(str(e))
    return output


def get_metrics_transmitted(file_path: str) -> dict:
    try:
        raw = subprocess.check_output(
            [
                # "conda",
                # "run",
                # "-n",
                # "nisqa",
                # "python",
                "python3",
                f"{NISQA_CWD}run.py",
                "--mode",
                "predict_file",
                "--pretrained_model",
                f"{NISQA_CWD}weights/nisqa.tar",
                "--deg",
                file_path,
                "--output_dir",
                "/",
            ]
        )
        content = StringIO(raw.decode())
        output = next(csv.DictReader(content))
        output = {"quality" if k == "mos_pred" else k: v for k, v in output.items()}
        output = {"noisiness" if k == "noi_pred" else k: v for k, v in output.items()}
        output = {"coloration" if k == "col_pred" else k: v for k, v in output.items()}
        output = {
            "discontinuity" if k == "dis_pred" else k: v for k, v in output.items()
        }
        output = {"loudness" if k == "loud_pred" else k: v for k, v in output.items()}
        output.pop("deg")
        output.pop("model")
    except Exception as e:
        output = {
            "quality": None,
            "noisiness": None,
            "discontinuity": None,
            "coloration": None,
            "loudness": None,
            "error": str(e),
        }
    return output


def get_metrics_synthesized(file_path: str) -> dict:
    try:
        raw = subprocess.check_output(
            [
                # "conda",
                # "run",
                # "-n",
                # "nisqa",
                # "python",
                "python3",
                f"{NISQA_CWD}run.py",
                "--mode",
                "predict_file",
                "--pretrained_model",
                f"{NISQA_CWD}weights/nisqa_tts.tar",
                "--deg",
                file_path,
                "--output_dir",
                "/",
            ]
        )
        content = StringIO(raw.decode())
        output = next(csv.DictReader(content))
        output = {"naturalness" if k == "mos_pred" else k: v for k, v in output.items()}
        output.pop("deg")
        output.pop("model")
    except Exception as e:
        output = {
            "naturalness": None,
            "error": str(e),
        }
    return output


def get_metrics_batch(folder: str) -> tuple[list, list]:
    output = []
    error = []
    try:
        synthesized = get_metrics_synthesized_batch(folder)
    except RuntimeError as e:
        msg = {"synthesized batch analysis": str(e)}
        error.append(msg)
        synthesized = pd.DataFrame.from_records(
            [
                {
                    "file": f.as_posix(),
                    "naturalness": None,
                }
                for f in Path(folder).glob("*.wav")
            ]
        )
    try:
        transmitted = get_metrics_transmitted_batch(folder)
    except RuntimeError as e:
        msg = {"transmitted batch analysis": str(e)}
        error.append(msg)
        transmitted = pd.DataFrame.from_records(
            [
                {
                    "file": f.as_posix(),
                    "quality": None,
                    "noisiness": None,
                    "discontinuity": None,
                    "coloration": None,
                    "loudness": None,
                }
                for f in Path(folder).glob("*.wav")
            ]
        )
    try:
        combined = transmitted.merge(synthesized, on=['file'])
        # print(f"{combined=}")
        output = combined.to_dict("records")
        # print(f"{output=}")
        if error:
            [o.update({"log": error}) for o in output]
    except Exception as e:
        msg = {"speech analysis output merge failed": str(e)}
        error.append(msg)
    return output, error


def get_metrics_synthesized_batch(folder: str) -> pd.DataFrame:
    raw = subprocess.run(
        [
            # "conda",
            # "run",
            # "-n",
            # "nisqa",
            # "python",
            "python3",
            f"{NISQA_CWD}run.py",
            "--mode",
            "predict_dir",
            "--pretrained_model",
            f"{NISQA_CWD}weights/nisqa_tts.tar",
            "--data_dir",
            folder,
            "--output_dir",
            "/",
        ],
        capture_output=True,
    )
    raw_error = raw.stderr.decode()
    if raw_error and (
        error := (re.search(r"(\w+): (.*)$", raw_error, re.MULTILINE).group(2).strip())
    ):
        raise RuntimeError(error)

    content = StringIO(raw.stdout.decode())
    outputs = pd.read_csv(content, header=1)
    outputs.rename(
        columns={
            "deg": "file",
            "mos_pred": "naturalness",
        },
        inplace=True,
    )
    outputs.drop(columns=["model"], inplace=True)
    return outputs


def get_metrics_transmitted_batch(folder: str) -> pd.DataFrame:
    raw = subprocess.run(
        [
            # "conda",
            # "run",
            # "-n",
            # "nisqa",
            # "python",
            "python3",
            f"{NISQA_CWD}run.py",
            "--mode",
            "predict_dir",
            "--pretrained_model",
            f"{NISQA_CWD}weights/nisqa.tar",
            "--data_dir",
            folder,
            "--output_dir",
            "/",
        ],
        capture_output=True,
    )
    raw_error = raw.stderr.decode()
    if raw_error and (
        error := (re.search(r"(\w+): (.*)$", raw_error, re.MULTILINE).group(2).strip())
    ):
        raise RuntimeError(error)

    content = StringIO(raw.stdout.decode())
    outputs = pd.read_csv(content, header=1)
    outputs.rename(
        columns={
            "deg": "file",
            "mos_pred": "quality",
            "noi_pred": "noisiness",
            "dis_pred": "discontinuity",
            "col_pred": "coloration",
            "loud_pred": "loudness",
        },
        inplace=True,
    )
    outputs.drop(columns=["model"], inplace=True)
    return outputs
