# 🗺️ Project Roadmap — Brain Tumor Diagnostic System

> **Read this before starting:** Every task is designed to be done **once**.
> MLflow, Airflow, and Docker are integrated from the beginning — not added later.
> Follow the phases in order. Do not skip ahead.

---

## 📁 PHASE 0 — Project Structure & Docker Setup

> **Why first?** Everything else (training, APIs, Airflow, MLflow) runs inside containers.
> Setting up Docker now means you never have to rewrite code to "make it work in Docker later."

---

### TASK 0.1 — Define the folder structure

**Purpose:** Establish a clean, logical layout that all tools (Airflow, MLflow, FastAPI, Streamlit) will reference.

**Where:** Root of the project (`Diagnostic_multimodal/`)

**What to do:** Create the following folders manually or with a script:

```
Diagnostic_multimodal/
├── data/
│   ├── classification/        ← raw images for GoogLeNet (already in archive/)
│   └── detection/             ← raw images + YOLO annotations
├── notebooks/                 ← exploratory work (EDA, experiments)
├── src/
│   ├── classification/        ← all GoogLeNet code
│   ├── detection/             ← all YOLOv8 code
│   ├── api/                   ← FastAPI endpoints
│   └── ui/                    ← Streamlit app
├── airflow/
│   └── dags/                  ← Airflow DAG files
├── mlflow/                    ← MLflow tracking server config
├── models/                    ← saved model files (.h5, .pt)
├── docker/                    ← individual Dockerfiles per service
├── docker-compose.yml         ← orchestrates all services
└── map.md
```

**Why:** A clear structure prevents path confusion across services and makes Docker volume mounts straightforward.

---

### TASK 0.2 — Write `docker-compose.yml`

**Purpose:** Define and connect all services in one file so the entire system starts with `docker compose up`.

**Where:** `docker-compose.yml` at the root

**What to do:** Define the following services:

| Service | Image/Build | Purpose |
|---|---|---|
| `mlflow` | `ghcr.io/mlflow/mlflow` | Experiment tracking server |
| `postgres` | `postgres:15` | Backend DB for Airflow |
| `airflow` | custom build | Pipeline orchestration |
| `api-classification` | custom build | FastAPI for GoogLeNet |
| `api-detection` | custom build | FastAPI for YOLOv8 |
| `streamlit` | custom build | User interface |

**Key concepts to document:**
- `volumes:` — mounts local folders into containers so models/data are shared
- `networks:` — all services on the same Docker network can talk to each other by service name (e.g., `http://api-classification:8000`)
- `depends_on:` — ensures services start in the right order
- `environment:` — pass config (DB URLs, MLflow URI) as env variables

**Why:** Without this, each service is isolated. With it, Streamlit can call FastAPI, FastAPI can log to MLflow, and Airflow can trigger training — all automatically.

---

### TASK 0.3 — Write individual Dockerfiles

**Purpose:** Each service needs its own environment (different Python packages, different base images).

**Where:** `docker/` folder — one file per service

**What to do:** Create:
- `docker/Dockerfile.classification` — Python + TensorFlow + FastAPI
- `docker/Dockerfile.detection` — Python + Ultralytics (YOLOv8) + FastAPI
- `docker/Dockerfile.streamlit` — Python + Streamlit + requests
- `docker/Dockerfile.airflow` — Airflow + both ML libraries

**Key concept to document:**
- Multi-stage builds are optional here but good to know
- `COPY` vs `VOLUME` — Dockerfile copies code at build time; volumes mount data at run time

---

## 📊 PHASE 1 — Data Preparation (Classification)

> **Why before training?** You cannot train without clean, split, augmented data.
> MLflow is initialized here so every experiment is tracked from the first run.

---

### TASK 1.1 — Explore and validate the classification dataset

**Purpose:** Understand what you have before touching it. Catch corrupt files early.

**Where:** `notebooks/01_eda_classification.ipynb`

**What to do:**
1. Walk through `data/classification/` and list all files
2. For each file, use `try-except` to open it with PIL — log invalid files
3. Count images per class (folder name = class label)
4. Plot a `countplot` (seaborn/matplotlib) showing class distribution
5. Display one sample image per class

**Key concepts to document:**
- `try-except` for file integrity — why you can't trust file extensions alone
- Class imbalance — what it is, why it matters for training
- `countplot` — visual tool to detect imbalance

---

### TASK 1.2 — Split the dataset (Train / Val / Test)

**Purpose:** Separate data so the model is evaluated on data it has never seen.

**Where:** `src/classification/data_split.py`

**What to do:**
1. Collect all valid image paths per class
2. Split each class: 70% train, 15% val, 15% test (use `sklearn.model_selection.train_test_split` twice)
3. Copy files into:
   - `data/classification/train/<class>/`
   - `data/classification/val/<class>/`
   - `data/classification/test/<class>/`
4. Print counts per split per class

**Key concepts to document:**
- Stratified split — splitting per class ensures each split has proportional class representation
- Why you split before augmentation (augment only train, never val/test)

---

### TASK 1.3 — Augment the training data

**Purpose:** Artificially increase training set size and reduce class imbalance.

**Where:** `src/classification/augment.py`

**What to do:** For each image in `data/classification/train/`, apply and save:
1. Gaussian blur (`cv2.GaussianBlur`)
2. Gaussian noise (add random noise array)
3. Horizontal flip (`cv2.flip`)

Save augmented images in the same train folder with a suffix (`_blur`, `_noise`, `_flip`).

**Key concepts to document:**
- Data augmentation — why it helps generalization
- Only augment train — val/test must reflect real-world distribution
- Gaussian blur/noise — simulate imaging artifacts common in MRI scans

---

### TASK 1.4 — Build the TensorFlow data pipeline

**Purpose:** Load images efficiently in batches during training.

**Where:** `src/classification/dataset.py`

**What to do:**
1. Use `tf.keras.utils.image_dataset_from_directory()` for train, val, test
2. Set `image_size=(224, 224)`, `batch_size=32`, `shuffle=True` (train only)
3. Apply InceptionV3 preprocessing: `tf.keras.applications.inception_v3.preprocess_input`
4. Use `.cache().prefetch(tf.data.AUTOTUNE)` for performance

**Key concepts to document:**
- `image_dataset_from_directory` — auto-labels from folder names
- Normalization per model — each pretrained model expects a specific pixel range
- `prefetch` — loads next batch while GPU processes current one (pipeline parallelism)

---

## 🧠 PHASE 2 — GoogLeNet Classification Model

---

### TASK 2.1 — Build the model (Feature Extraction)

**Purpose:** Use a pretrained model as a fixed feature extractor, only train the new classification head.

**Where:** `src/classification/model.py`

**What to do:**
1. Load `InceptionV3` (GoogLeNet equivalent in Keras) with `include_top=False`, `weights='imagenet'`
2. Freeze all base layers: `base_model.trainable = False`
3. Add on top: `GlobalAveragePooling2D` → `Dense(256, relu)` → `Dropout(0.5)` → `Dense(num_classes, softmax)`
4. Compile with `Adam(lr=1e-3)`, `categorical_crossentropy`, metrics: `['accuracy']`

**Key concepts to document:**
- Transfer learning — reusing weights trained on ImageNet
- Feature extraction vs fine-tuning — frozen base = feature extractor
- `include_top=False` — removes the original classification head
- `GlobalAveragePooling2D` — reduces spatial dimensions before the dense layer

---

### TASK 2.2 — Train with MLflow tracking (Feature Extraction phase)

**Purpose:** Train the classification head and log everything to MLflow.

**Where:** `src/classification/train.py`

**What to do:**
1. Initialize MLflow: `mlflow.set_tracking_uri("http://mlflow:5000")`, `mlflow.set_experiment("brain-tumor-classification")`
2. Start a run: `with mlflow.start_run(run_name="feature_extraction"):`
3. Log hyperparameters: `mlflow.log_param("lr", 1e-3)`, batch size, epochs, frozen layers
4. Add callbacks: `EarlyStopping(patience=5)`, `ModelCheckpoint` saving best weights
5. Train: `model.fit(train_ds, validation_data=val_ds, epochs=30, callbacks=[...])`
6. Log metrics per epoch: accuracy, val_accuracy, loss, val_loss
7. Log the model: `mlflow.keras.log_model(model, "model")`

**Key concepts to document:**
- MLflow run — a single experiment execution with its own ID
- `mlflow.log_param` vs `mlflow.log_metric` — params are fixed config, metrics change per epoch
- `EarlyStopping` — stops training when val_loss stops improving (prevents overfitting)
- `ModelCheckpoint` — saves the best model weights during training

---

### TASK 2.3 — Evaluate the model

**Purpose:** Measure real performance on unseen test data.

**Where:** `src/classification/evaluate.py`

**What to do:**
1. Load best saved model
2. Run predictions on test set
3. Compute: accuracy, precision, recall, F1-score (`sklearn.metrics.classification_report`)
4. Plot confusion matrix (`sklearn.metrics.ConfusionMatrixDisplay`)
5. Log all metrics to the same MLflow run

**Key concepts to document:**
- Confusion matrix — shows which classes are confused with each other
- Precision vs Recall — precision = of predicted positives, how many are real; recall = of real positives, how many did we catch
- F1-score — harmonic mean of precision and recall, useful for imbalanced classes

---

### TASK 2.4 — Fine-tuning

**Purpose:** Unfreeze the last layers of the base model and retrain with a very low learning rate to squeeze out more accuracy.

**Where:** `src/classification/train.py` (add a second MLflow run)

**What to do:**
1. Start a new MLflow run: `run_name="fine_tuning"`
2. Unfreeze last 30 layers: `for layer in base_model.layers[-30:]: layer.trainable = True`
3. Recompile with `Adam(lr=1e-5)` (10x lower than before)
4. Train for 10–15 more epochs with same callbacks
5. Re-evaluate and log metrics
6. Compare the two runs in MLflow UI

**Key concepts to document:**
- Fine-tuning — adjusting pretrained weights slightly for your specific task
- Why low learning rate — large updates would destroy the pretrained weights
- MLflow comparison — use the UI to compare feature extraction vs fine-tuning runs side by side

---

## 🔍 PHASE 3 — YOLOv8 Detection Model

---

### TASK 3.1 — Explore and validate the detection dataset

**Purpose:** Verify every image has a matching annotation file before organizing data.

**Where:** `notebooks/02_eda_detection.ipynb`

**What to do:**
1. List all images in `data/detection/images/`
2. For each image, check if a `.txt` file with the same base name exists in `data/detection/labels/`
3. Log mismatches (image without label, label without image)
4. Display 5 sample images with bounding boxes drawn using the YOLO annotation coordinates

**Key concepts to document:**
- YOLO annotation format: `class_id cx cy w h` (all normalized 0–1 relative to image size)
- Bounding box drawing — convert normalized coords back to pixel coords for visualization

---

### TASK 3.2 — Organize data into YOLO structure

**Purpose:** YOLOv8 expects a specific folder layout to find images and labels.

**Where:** `src/detection/data_setup.py`

**What to do:**
1. Split valid image-label pairs: 70% train, 15% val, 15% test
2. Copy to:
   ```
   data/detection/
   ├── images/train/   images/val/   images/test/
   └── labels/train/   labels/val/   labels/test/
   ```
3. Skip any image without a matching label (and vice versa)
4. Count and print files per split

---

### TASK 3.3 — Validate annotation format

**Purpose:** Corrupt or malformed annotations will silently break training.

**Where:** `src/detection/validate_annotations.py`

**What to do:**
1. For each `.txt` file, read each line
2. Check: exactly 5 values, class_id is integer, cx/cy/w/h are floats in [0, 1]
3. Check class_id is within the expected number of classes
4. Log and skip invalid files

---

### TASK 3.4 — Create `data.yaml`

**Purpose:** YOLOv8 reads this file to know where data is and what classes exist.

**Where:** `data/detection/data.yaml`

**What to do:** Create the file:
```yaml
path: /app/data/detection
train: images/train
val: images/val
test: images/test
nc: <number_of_classes>
names: [<class1>, <class2>, ...]
```

**Key concept to document:**
- `data.yaml` — YOLOv8's configuration contract; paths must match what's mounted in Docker

---

### TASK 3.5 — Train YOLOv8 with MLflow tracking

**Purpose:** Train the detection model and log results to MLflow.

**Where:** `src/detection/train.py`

**What to do:**
1. Initialize MLflow: `mlflow.set_experiment("brain-tumor-detection")`
2. Start run: `with mlflow.start_run(run_name="yolov8_training"):`
3. Log params: model size (`yolov8n`), epochs, image size, batch size
4. Train:
   ```python
   model = YOLO("yolov8n.pt")
   results = model.train(data="data/detection/data.yaml", epochs=50, imgsz=640, batch=16,
                         augment=True, degrees=10, flipud=0.5, mosaic=1.0)
   ```
5. Log metrics: mAP@0.5, mAP@0.5:0.95, precision, recall
6. Log model artifact: `mlflow.log_artifact("runs/detect/train/weights/best.pt")`

**Key concepts to document:**
- YOLOv8 augmentation params — `mosaic`, `degrees`, `flipud` are set in the `train()` call
- mAP@0.5 — mean Average Precision at IoU threshold 0.5
- mAP@0.5:0.95 — averaged over multiple IoU thresholds (stricter, more complete metric)
- IoU (Intersection over Union) — measures overlap between predicted and ground truth box

---

### TASK 3.6 — Evaluate YOLOv8

**Purpose:** Measure generalization on the test set.

**Where:** `src/detection/evaluate.py`

**What to do:**
1. Load best weights: `YOLO("models/yolov8_best.pt")`
2. Run `model.val(data="data/detection/data.yaml", split="test")`
3. Log all metrics to MLflow
4. Save prediction samples (images with drawn boxes) as MLflow artifacts

---

## ⚙️ PHASE 4 — FastAPI Endpoints

> **Why before Streamlit?** Streamlit calls the APIs. APIs must exist first.
> Models are loaded once at startup — not per request.

---

### TASK 4.1 — Classification API

**Purpose:** Expose the GoogLeNet model as an HTTP endpoint.

**Where:** `src/api/classification_api.py`

**What to do:**
1. At startup, load the saved Keras model into memory
2. Define `POST /predict/classification`:
   - Accept: image file upload (`UploadFile`)
   - Preprocess: resize to 224×224, apply InceptionV3 normalization
   - Predict: `model.predict(img_array)`
   - Return: `{"class": "Glioma", "confidence": 0.97, "all_scores": {...}}`
3. Define `GET /health` returning `{"status": "ok"}`

**Key concepts to document:**
- FastAPI `UploadFile` — handles multipart form data (file uploads)
- Loading model at startup — avoids reloading on every request (performance)
- `@app.on_event("startup")` — runs code once when the server starts

---

### TASK 4.2 — Detection API

**Purpose:** Expose the YOLOv8 model as an HTTP endpoint.

**Where:** `src/api/detection_api.py`

**What to do:**
1. At startup, load `YOLO("models/yolov8_best.pt")`
2. Define `POST /predict/detection`:
   - Accept: image file upload
   - Run: `results = model(img)`
   - Extract: bounding boxes, class names, confidence scores
   - Return: `{"detections": [{"class": "tumor", "confidence": 0.89, "bbox": [x1,y1,x2,y2]}]}`
3. Define `GET /health`

---

## 🔄 PHASE 5 — Airflow Pipeline

> **Why after APIs?** The DAG orchestrates training steps. The code being orchestrated must already exist.

---

### TASK 5.1 — Configure Airflow

**Purpose:** Set up Airflow to use the project's DAG folder.

**Where:** `docker-compose.yml` (Airflow service config) + `airflow/` folder

**What to do:**
1. In `docker-compose.yml`, mount `./airflow/dags` into the Airflow container
2. Set env vars: `AIRFLOW__CORE__EXECUTOR=LocalExecutor`, `AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://...`
3. Initialize the DB on first run: `airflow db init`

**Key concepts to document:**
- DAG (Directed Acyclic Graph) — a workflow where tasks have dependencies but no cycles
- LocalExecutor — runs tasks as subprocesses on the same machine (vs CeleryExecutor for distributed)
- Airflow metadata DB — stores DAG runs, task states, logs (uses Postgres here)

---

### TASK 5.2 — Write the training DAG

**Purpose:** Automate the full pipeline from data validation to model saving.

**Where:** `airflow/dags/training_pipeline.py`

**What to do:** Create a DAG with these tasks in order:

```
validate_classification_data
        ↓
validate_detection_data
        ↓
train_classification          train_detection
        ↓                            ↓
evaluate_classification       evaluate_detection
        ↓                            ↓
              save_models_and_artifacts
```

Each task is a `PythonOperator` that calls the corresponding function from `src/`.

**What to configure:**
- `dag_id`, `schedule_interval=None` (manual trigger), `start_date`
- `on_failure_callback` — log error and send alert
- `retries=1`, `retry_delay=timedelta(minutes=5)`

**Key concepts to document:**
- `PythonOperator` — runs a Python function as an Airflow task
- `schedule_interval=None` — DAG only runs when manually triggered
- Task dependencies: `task_a >> task_b` means b runs after a
- XCom — how tasks pass small values to each other (e.g., model path)

---

## 🖥️ PHASE 6 — Streamlit Interface

---

### TASK 6.1 — Build the Streamlit app

**Purpose:** Give end users (doctors) a simple interface to upload brain images and get predictions.

**Where:** `src/ui/app.py`

**What to do:**
1. Title + description
2. File uploader: `st.file_uploader("Upload brain MRI", type=["jpg","png"])`
3. Two buttons: "Classify" and "Detect Tumor"
4. On "Classify":
   - POST image to `http://api-classification:8000/predict/classification`
   - Display: predicted class, confidence bar chart
5. On "Detect":
   - POST image to `http://api-detection:8000/predict/detection`
   - Draw bounding boxes on the image using PIL
   - Display annotated image + detection table

**Key concepts to document:**
- `st.file_uploader` returns a `BytesIO` object — pass directly to `requests.post(files=...)`
- Service-to-service communication — inside Docker network, use service names as hostnames
- `PIL.ImageDraw` — draw rectangles on images in Python

---

## ✅ PHASE 7 — Integration & Final Validation

---

### TASK 7.1 — End-to-end test

**Purpose:** Verify the entire system works together before submission.

**Where:** Terminal / browser

**What to do:**
1. Run `docker compose up --build`
2. Open MLflow UI at `http://localhost:5000` — verify experiments exist
3. Open Airflow UI at `http://localhost:8080` — trigger the DAG manually, watch it complete
4. Open Streamlit at `http://localhost:8501` — upload a test image, verify classification and detection both return results
5. Check FastAPI docs at `http://localhost:8000/docs` and `http://localhost:8001/docs`

---

### TASK 7.2 — Write `README.md`

**Purpose:** Document how to run the project for anyone who clones it.

**Where:** `README.md`

**What to include:**
- Project description (1 paragraph)
- Prerequisites (Docker, Docker Compose)
- How to run: `docker compose up --build`
- Service URLs table
- Folder structure explanation

---

## 🧭 Summary — Execution Order

```
Phase 0: Docker setup (structure + compose + Dockerfiles)
Phase 1: Classification data (explore → split → augment → pipeline)
Phase 2: GoogLeNet model (build → train+MLflow → evaluate → fine-tune)
Phase 3: YOLOv8 model (explore → organize → validate → yaml → train+MLflow → evaluate)
Phase 4: FastAPI (classification API → detection API)
Phase 5: Airflow (configure → write DAG)
Phase 6: Streamlit (build UI)
Phase 7: Integration test + README
```

> Each phase builds on the previous. Never go back to rewrite — the integrations are already in place.
