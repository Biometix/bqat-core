import csv
import json
import subprocess
from io import StringIO

import pandas as pd


def process_speech(
    input_path: str,
    input_type: str,
):
    output = {"log": {}}

    if input_type == "file":
        output.update(meta) if not (meta := get_metrics_transmitted(input_path)).get(
            "error"
        ) else output["log"].update({"transmitted speech analysis": meta["error"]})
        output.update(meta) if not (meta := get_metrics_synthesized(input_path)).get(
            "error"
        ) else output["log"].update({"synthesized speech analysis": meta["error"]})
    elif input_type == "folder":
        try:
            results, log = get_metrics_batch(input_path)
            output.update({"results": results})
            output["log"].update({"speech analysis": log})
            return output
        except Exception as e:
            raise RuntimeError(str(e))
    else:
        raise TypeError("illegal task type")

    if not output["log"]:
        output.pop("log")
    return output


def get_metrics_transmitted(file_path: str) -> dict:
    model = "weights/nisqa.tar"
    output = {}
    try:
        raw = subprocess.check_output(
            [
                "conda",
                "run",
                "-n",
                "nisqa",
                "python",
                "run.py",
                "--mode",
                "predict_file",
                "--pretrained_model",
                model,
                "--deg",
                file_path,
                "--output_dir",
                "/",
            ]
        )
        content = StringIO(raw.decode())
        output = next(csv.DictReader(content))
        output = {"Quality" if k == "mos_pred" else k: v for k, v in output.items()}
        output = {"Noisiness" if k == "noi_pred" else k: v for k, v in output.items()}
        output = {"Coloration" if k == "col_pred" else k: v for k, v in output.items()}
        output = {
            "Discontinuity" if k == "dis_pred" else k: v for k, v in output.items()
        }
        output = {"Loudness" if k == "loud_pred" else k: v for k, v in output.items()}
        output.pop("deg")
        output.pop("model")
    except Exception as e:
        return {"error": str(e)}
    return output


def get_metrics_synthesized(file_path: str) -> dict:
    model = "weights/nisqa_tts.tar"
    output = {}
    try:
        raw = subprocess.check_output(
            [
                "conda",
                "run",
                "-n",
                "nisqa",
                "python",
                "run.py",
                "--mode",
                "predict_file",
                "--pretrained_model",
                model,
                "--deg",
                file_path,
                "--output_dir",
                "/",
            ]
        )
        content = StringIO(raw.decode())
        output = next(csv.DictReader(content))
        output = {"Naturalness" if k == "mos_pred" else k: v for k, v in output.items()}
        output.pop("deg")
        output.pop("model")
    except Exception as e:
        return {"error": str(e)}
    return output


def get_metrics_batch(folder: str) -> tuple:
    output = []
    error = []
    try:
        synthesized = get_metrics_synthesized_batch(folder)
    except Exception as e:
        error.append({"synthesized analysis": str(e)})
        synthesized = False
    try:
        transmitted = get_metrics_transmitted_batch(folder)
    except Exception as e:
        error.append({"transmitted analysis": str(e)})
        transmitted = False

    if synthesized is not False and transmitted is not False:
        combined = synthesized.merge(transmitted)
        output = combined.to_dict("records")
    elif synthesized is not False and transmitted is False:
        output = synthesized.to_dict("records")
    elif synthesized is False and transmitted is not False:
        output = transmitted.to_dict("records")
    else:
        raise RuntimeError(json.dumps({"speech analysis failed": error}))

    return output, error


def get_metrics_synthesized_batch(folder: str) -> pd.DataFrame:
    raw = subprocess.run(
        [
            "conda",
            "run",
            "-n",
            "nisqa",
            "python",
            "run.py",
            "--mode",
            "predict_dir",
            "--pretrained_model",
            "weights/nisqa_tts.tar",
            "--data_dir",
            folder,
            "--output_dir",
            "/",
        ],
        capture_output=True,
    )
    if not raw.stderr:
        content = StringIO(raw.stdout.decode())
        metrics_synthesized = pd.read_csv(content, header=1)
        metrics_synthesized.rename(
            columns={
                "deg": "file",
                "mos_pred": "Naturalness",
            },
            inplace=True,
        )
        metrics_synthesized.drop(columns=["model"], inplace=True)

        return metrics_synthesized
    else:
        raise RuntimeError(raw.stderr.decode())


def get_metrics_transmitted_batch(folder: str) -> pd.DataFrame:
    raw = subprocess.run(
        [
            "conda",
            "run",
            "-n",
            "nisqa",
            "python",
            "run.py",
            "--mode",
            "predict_dir",
            "--pretrained_model",
            "weights/nisqa.tar",
            "--data_dir",
            folder,
            "--output_dir",
            "/",
        ],
        capture_output=True,
    )
    if not raw.stderr:
        content = StringIO(raw.stdout.decode())
        metrics_transmitted = pd.read_csv(content, header=1)
        metrics_transmitted.rename(
            columns={
                "deg": "file",
                "mos_pred": "Quality",
                "noi_pred": "Noisiness",
                "dis_pred": "Discontinuity",
                "col_pred": "Coloration",
                "loud_pred": "Loudness",
            },
            inplace=True,
        )
        metrics_transmitted.drop(columns=["model"], inplace=True)

        return metrics_transmitted
    else:
        raise RuntimeError(raw.stderr.decode())
