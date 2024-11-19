import csv
import json
import subprocess
from io import StringIO

import pandas as pd

CWD = "NISQA/"


def process_speech(
    input_path: str,
    input_type: str,
):
    output = {"log": []}

    if input_type == "file":
        output.update(meta) if not (meta := get_metrics_transmitted(input_path)).get(
            "error"
        ) else output["log"].append({"transmitted speech analysis": meta["error"]})
        output.update(meta) if not (meta := get_metrics_synthesized(input_path)).get(
            "error"
        ) else output["log"].append({"synthesized speech analysis": meta["error"]})
    elif input_type == "folder":
        try:
            results, log = get_metrics_batch(input_path)
            output.update({"results": results})
            if log:
                output["log"].append({"speech analysis": log})
            return output
        except Exception as e:
            raise RuntimeError(str(e))
    else:
        raise TypeError("illegal task type")

    if output.get("log"):
        output["log"] = output.pop("log")
    else:
        output.pop("log")

    return output


def get_metrics_transmitted(file_path: str) -> dict:
    output = {}
    try:
        raw = subprocess.check_output(
            [
                # "conda",
                # "run",
                # "-n",
                # "nisqa",
                # "python",
                "python3",
                f"{CWD}run.py",
                "--mode",
                "predict_file",
                "--pretrained_model",
                f"{CWD}weights/nisqa.tar",
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
        return {"error": str(e)}
    return output


def get_metrics_synthesized(file_path: str) -> dict:
    output = {}
    try:
        raw = subprocess.check_output(
            [
                # "conda",
                # "run",
                # "-n",
                # "nisqa",
                # "python",
                "python3",
                f"{CWD}run.py",
                "--mode",
                "predict_file",
                "--pretrained_model",
                f"{CWD}weights/nisqa_tts.tar",
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
        return {"error": str(e)}
    return output


def get_metrics_batch(folder: str) -> tuple:
    output = []
    error = []

    synthesized, e = get_metrics_synthesized_batch(folder)
    if e:
        msg = {"synthesized analysis": e}
        print(f">> {folder}: {msg = }")
        error.append(msg)

    transmitted, e = get_metrics_transmitted_batch(folder)
    if e:
        msg = {"transmitted analysis": e}
        print(f">> {folder}: {msg = }")
        error.append(msg)

    try:
        combined = transmitted.merge(synthesized)
        output = combined.to_dict("records")
    except Exception as e:
        raise RuntimeError(json.dumps({"speech analysis failed": error}))

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
            f"{CWD}run.py",
            "--mode",
            "predict_dir",
            "--pretrained_model",
            f"{CWD}weights/nisqa_tts.tar",
            "--data_dir",
            folder,
            "--output_dir",
            "/",
        ],
        capture_output=True,
    )
    error = raw.stderr.decode()
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

    return outputs, error


def get_metrics_transmitted_batch(folder: str) -> pd.DataFrame:
    raw = subprocess.run(
        [
            # "conda",
            # "run",
            # "-n",
            # "nisqa",
            # "python",
            "python3",
            f"{CWD}run.py",
            "--mode",
            "predict_dir",
            "--pretrained_model",
            f"{CWD}weights/nisqa.tar",
            "--data_dir",
            folder,
            "--output_dir",
            "/",
        ],
        capture_output=True,
    )
    error = raw.stderr.decode()
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

    return outputs, error
