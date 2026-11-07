# 🧠 Multimodal Brain Tumor Diagnostic System

An end-to-end medical AI solution designed for automated brain tumor analysis using Deep Learning. The platform integrates two complementary approaches:
1. **Brain Tumor Classification** using transfer learning with **InceptionV3 / GoogLeNet** (Glioma, Meningioma, No Tumor, Pituitary).
2. **Brain Tumor Detection & Localization** using **YOLOv8n** with bounding box predictions.

The system is equipped with **MLflow** for experiment tracking, **Apache Airflow** for automated pipeline orchestration, **FastAPI** for model serving, and an interactive **Streamlit** web application for clinical review and inference.

---

## 🏗️ Architecture & Services

| Service | Port | Description |
|---|---|---|
| **Streamlit Dashboard** | `8501` | Interactive UI for metrics visualization and MRI inference |
| **Classification API** | `8001` | FastAPI endpoint for InceptionV3 tumor classification |
| **Detection API** | `8002` | FastAPI endpoint for YOLOv8 tumor detection & localization |
| **MLflow Tracking Server** | `5000` | Experiment tracking, hyperparameter logging, and model registry |
| **Airflow Webserver** | `8080` | Pipeline orchestration and workflow monitoring (SQLite backend) |

---

## 📁 Project Structure

```text
Diagnostic_multimodal/
├── airflow/
│   ├── dags/
│   │   └── training_pipeline.py     # Airflow DAG (@task-based pipeline)
│   └── requirements.airflow.txt     # Airflow container dependencies
├── Data/
│   ├── archive/                     # Raw MRI image datasets
│   ├── classification/              # Class-separated data
│   ├── split/                       # Train / Val / Test splits (70/15/15)
│   └── yolo_dataset/                # YOLO annotations and data.yaml
├── docker/
│   ├── Dockerfile.airflow           # Airflow service image
│   ├── Dockerfile.classification    # InceptionV3 FastAPI image
│   ├── Dockerfile.detection         # YOLOv8 FastAPI image
│   └── Dockerfile.streamlit         # Streamlit UI image
├── models/
│   ├── feature_extraction_best.keras # InceptionV3 Feature Extraction model
│   ├── fine_tuning_best.keras        # InceptionV3 Fine-Tuned model
│   ├── results_fe.json / ft.json    # Classification evaluation metrics
│   ├── confusion_matrix_*.png       # Confusion matrices
│   └── yolov8n_v1-4/                # YOLOv8n weights (best.pt), curves, and metrics
├── notebooks/
│   ├── 01_eda_classification.ipynb  # EDA & Classification training
│   ├── 02_eda_detection.ipynb       # EDA & Bounding box exploration
│   └── train_classification.ipynb   # Kaggle training notebook
├── src/
│   ├── api/
│   │   ├── classification_api.py    # FastAPI: POST /predict (Classification)
│   │   ├── detection_api.py         # FastAPI: POST /predict (YOLOv8 Detection)
│   │   ├── requirements.classification.txt
│   │   └── requirements.detection.txt
│   ├── detection/
│   │   └── log_yolo_mlflow.py       # MLflow logging script for YOLO metrics & artifacts
│   └── ui/
│       ├── app.py                   # Streamlit multi-page application
│       └── requirements.streamlit.txt
├── docker-compose.yml               # Multi-service container orchestration
├── map.md                           # Detailed implementation roadmap
└── README.md                        # Documentation
```

---

## 🚀 Running Locally (Without Docker)

### 1. Prerequisites
- Python 3.10+
- Virtual environment recommended (`python -m venv venv`)

Install requirements:
```bash
pip install -r src/api/requirements.classification.txt
pip install -r src/api/requirements.detection.txt
pip install -r src/ui/requirements.streamlit.txt
```

### 2. Launch MLflow Server
```bash
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow/mlflow.db --default-artifact-root ./mlflow/artifacts
```
Access the MLflow UI at: [http://localhost:5000](http://localhost:5000)

### 3. Log YOLO Model & Metrics to MLflow
```bash
python src/detection/log_yolo_mlflow.py
```

### 4. Start the APIs
In separate terminal windows from the project root:

- **Classification API:**
  ```bash
  uvicorn src.api.classification_api:app --host 0.0.0.0 --port 8001
  ```
  API Docs: [http://localhost:8001/docs](http://localhost:8001/docs)

- **Detection API:**
  ```bash
  uvicorn src.api.detection_api:app --host 0.0.0.0 --port 8002
  ```
  API Docs: [http://localhost:8002/docs](http://localhost:8002/docs)

### 5. Launch the Streamlit Dashboard
```bash
streamlit run src/ui/app.py --server.port 8501
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### 6. Run Airflow (Optional)
```bash
# In PowerShell:
$env:AIRFLOW_HOME = "$pwd\airflow"
airflow standalone
```
Open [http://localhost:8080](http://localhost:8080) to inspect and trigger `brain_tumor_pipeline`.

---

## 🐳 Running with Docker Compose

When running in containerized mode:

1. **Switch paths / URIs** in the following files to enable Docker network endpoints (uncomment Docker line, comment localhost line):
   - `src/api/classification_api.py` (`MODEL_PATH = Path("/app/models/fine_tuning_best.keras")`)
   - `src/api/detection_api.py` (`MODEL_PATH = Path("/app/models/yolov8n_v1-4/weights/best.pt")`)
   - `src/detection/log_yolo_mlflow.py` (`MLFLOW_URI = "http://mlflow:5000"`)
   - `src/ui/app.py` (`API_CLS = "http://api-classification:8001"`, `API_DET = "http://api-detection:8002"`)
   - `airflow/dags/training_pipeline.py` (`ROOT = "/opt/airflow"`, `MLFLOW_URI = "http://mlflow:5000"`)

2. **Start all services**:
   ```bash
   docker compose up --build
   ```

3. **Services will be available at**:
   - Streamlit: `http://localhost:8501`
   - Classification API: `http://localhost:8001/docs`
   - Detection API: `http://localhost:8002/docs`
   - MLflow Tracking: `http://localhost:5000`
   - Airflow UI: `http://localhost:8080` (Username: `admin`, Password: `admin`)

