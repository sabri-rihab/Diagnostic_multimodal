from airflow.decorators import dag, task
from datetime import datetime


# ROOT = "/opt/airflow"
ROOT = r"C:\Users\ycode\Desktop\2eme année YC\Breif_1(weather-app)\Diagnostic_multimodal"

# MLFLOW_URI = "http://mlflow:5000"
MLFLOW_URI = "http://localhost:5000"


@dag(
    dag_id="brain_tumor_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
)
def brain_tumor_pipeline():

    @task
    def validate_classification_data():
        from pathlib import Path

        models_dir = Path(ROOT) / "models"
        required = [
            models_dir / "fine_tuning_best.keras",
            models_dir / "feature_extraction_best.keras",
            models_dir / "results_fe.json",
            models_dir / "results_ft.json",
            models_dir / "confusion_matrix_fe.png",
            models_dir / "confusion_matrix_ft.png",
        ]

        missing = [str(f) for f in required if not f.exists()]
        if missing:
            raise FileNotFoundError(f"Missing: {missing}")
        return "classification data validated"

    @task
    def validate_detection_data():
        from pathlib import Path

        yolo_dir = Path(ROOT) / "models" / "yolov8n_v1-4"
        required = [
            yolo_dir / "weights" / "best.pt",
            yolo_dir / "results.csv",
            yolo_dir / "args.yaml",
        ]

        missing = [str(f) for f in required if not f.exists()]
        if missing:
            raise FileNotFoundError(f"Missing: {missing}")
        return "detection data validated"

    @task
    def log_classification_mlflow():
        import json
        import mlflow
        import mlflow.keras
        from pathlib import Path

        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("brain-tumor-classification")

        models_dir = Path(ROOT) / "models"

        for phase in ["fe", "ft"]:
            data = json.loads((models_dir / f"results_{phase}.json").read_text())

            with mlflow.start_run(run_name=data["run_name"]):
                mlflow.log_params(data["params"])
                mlflow.log_metrics(data["test_metrics"])

                cm_path = models_dir / f"confusion_matrix_{phase}.png"
                if cm_path.exists():
                    mlflow.log_artifact(str(cm_path))

        return "classification logged to mlflow"

    @task
    def log_detection_mlflow():
        import csv
        import yaml
        import mlflow
        from pathlib import Path

        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("brain-tumor-detection")

        yolo_dir = Path(ROOT) / "models" / "yolov8n_v1-4"

        with open(yolo_dir / "args.yaml") as f:
            args = yaml.safe_load(f)

        with open(yolo_dir / "results.csv") as f:
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

            mlflow.log_artifact(str(yolo_dir / "weights" / "best.pt"))
            mlflow.log_artifact(str(yolo_dir / "results.csv"))
            mlflow.log_artifact(str(yolo_dir / "args.yaml"))

            for img in yolo_dir.glob("*.png"):
                mlflow.log_artifact(str(img))

        return "detection logged to mlflow"

    @task
    def save_models():
        from pathlib import Path

        models_dir = Path(ROOT) / "models"
        artifacts = {
            "classification_fe": models_dir / "feature_extraction_best.keras",
            "classification_ft": models_dir / "fine_tuning_best.keras",
            "yolov8_best": models_dir / "yolov8n_v1-4" / "weights" / "best.pt",
        }

        summary = {}
        for name, path in artifacts.items():
            summary[name] = {"path": str(path), "exists": path.exists()}
        return summary

    v_cls = validate_classification_data()
    v_det = validate_detection_data()
    l_cls = log_classification_mlflow()
    l_det = log_detection_mlflow()
    save = save_models()

    v_cls >> v_det >> [l_cls, l_det] >> save


brain_tumor_pipeline()
