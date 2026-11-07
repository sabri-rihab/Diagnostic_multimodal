import mlflow
import csv
import yaml
from pathlib import Path

# MLFLOW_URI = "http://mlflow:5000"
MLFLOW_URI = "http://localhost:5000"

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "models" / "yolov8n_v1-4"


def log_yolo_to_mlflow():
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("brain-tumor-detection")

    with open(RESULTS_DIR / "args.yaml") as f:
        args = yaml.safe_load(f)

    with open(RESULTS_DIR / "results.csv") as f:
        rows = list(csv.DictReader(f))
    last = rows[-1]

    with mlflow.start_run(run_name="yolov8n_v1-4"):
        mlflow.log_params({
            "model": args["model"],
            "epochs": args["epochs"],
            "batch": args["batch"],
            "imgsz": args["imgsz"],
            "optimizer": args["optimizer"],
            "lr0": args["lr0"],
            "patience": args["patience"],
            "mosaic": args["mosaic"],
            "degrees": args["degrees"],
            "fliplr": args["fliplr"],
        })

        mlflow.log_metrics({
            "precision": float(last["metrics/precision(B)"].strip()),
            "recall": float(last["metrics/recall(B)"].strip()),
            "mAP50": float(last["metrics/mAP50(B)"].strip()),
            "mAP50-95": float(last["metrics/mAP50-95(B)"].strip()),
            "val_box_loss": float(last["val/box_loss"].strip()),
            "val_cls_loss": float(last["val/cls_loss"].strip()),
        })

        mlflow.log_artifact(str(RESULTS_DIR / "weights" / "best.pt"))
        mlflow.log_artifact(str(RESULTS_DIR / "results.csv"))
        mlflow.log_artifact(str(RESULTS_DIR / "args.yaml"))

        for img in RESULTS_DIR.glob("*.png"):
            mlflow.log_artifact(str(img))

    print("YOLO model logged to MLflow.")


if __name__ == "__main__":
    log_yolo_to_mlflow()

