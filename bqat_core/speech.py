import csv
import pandas as pd
import subprocess
from io import StringIO


def process_speech(
    input_path: str,
    input_type: str,
):
    """_summary_

    Args:
        input_path (str): _description_
        input_type (str): _description_

    Returns:
        dict | list: _description_
    """
    output = {"log": {}}

    if input_type != "folder":
        output.update(meta) if not (meta := get_metrics_transmitted(input_path)).get(
            "error"
        ) else output["log"].update({"transmitted speech analysis": meta["error"]})
        output.update(meta) if not (meta := get_metrics_synthesized(input_path)).get(
            "error"
        ) else output["log"].update({"synthesized speech analysis": meta["error"]})
    else:
        return get_metrics_batch(input_path)

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


def get_metrics_batch(folder: str) -> list:
    output = {}
    try:
        synthesized = get_metrics_synthesized_batch(folder)
        transmitted = get_metrics_transmitted_batch(folder)
        combined = synthesized.merge(transmitted)
        output = combined.to_dict("records")
    except Exception as e:
        return {"error": str(e)}
    return output


def get_metrics_synthesized_batch(folder: str) -> pd.DataFrame:
    raw = subprocess.check_output(
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
        ]
    )
    content = StringIO(raw.decode())
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


def get_metrics_transmitted_batch(folder: str) -> pd.DataFrame:
    raw = subprocess.check_output(
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
        ]
    )
    content = StringIO(raw.decode())
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
