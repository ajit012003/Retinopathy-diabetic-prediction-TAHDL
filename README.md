# Retinopathy Diabetic Prediction and Screening using TAHDL

This repository contains my final‑year project on Diabetic Retinopathy (DR) prediction and screening using a Temporal Aware Hybrid Deep Learning (TAHDL) model. The model combines a multiscale CNN backbone with a 2‑layer LSTM and an attention mechanism to capture both spatial details in retinal fundus images and temporal patterns related to disease progression. [web:23][web:32]

---

## 1. Project overview

- **Goal:** Automatically classify retinal fundus images into 5 DR grades: No DR, Mild, Moderate, Severe and Proliferative. [web:29]  
- **Core idea:** Use multiscale convolutions to extract rich spatial features (vessels, microaneurysms, hemorrhages, exudates), then pass them through an LSTM with attention to better handle progression and subtle changes. [web:27][web:29]  
- **Use case:** Early screening support for ophthalmologists and automated DR grading on Kaggle EyePACS‑style datasets.

This implementation is based on the TAHDL architecture described in the Scientific Reports paper “A hybrid deep learning framework for early detection of diabetic retinopathy using retinal fundus images” (Scientific Reports, 2025, https://doi.org/10.1038/s41598-025-99309-w). [file:1]

---

## 2. Datasets

This repo does **not** ship any images. You must download the data separately from Kaggle and place it under `Datasets/`. The code supports **two** compatible datasets; you can use either one. [web:23][web:32]

### 2.1 Dataset A – Diabetic Retinopathy Detection Resized (gzuidhof)

- **Kaggle link:** https://www.kaggle.com/datasets/gzuidhof/diabetic-retinopathy-detection-resized [web:23]  
- **What it is:** Pre‑resized EyePACS images (e.g. 512×512) plus CSV files with 5‑class DR labels (`diagnosis` from 0 to 4). [web:23][web:31]  
- **Minimum required for this project:**
  - `train_images_512/` (train images)  
  - `train.csv` (image IDs and labels)

**Optional but not compulsory:**

- `test_images_512/`  
- `test.csv`  

Expected structure:
PROJECT_DIR/
Datasets/
train_images_512/
train_images_512/
<train images>.png/.jpg/...
train.csv/
train.csv
test_images_512/ # optional
test_images_512/
<test images> # optional
test.csv/ # optional
test.csv

 
The training pipeline only needs `train_images_512` and `train.csv`. The script will automatically perform an internal **80:20 split** of the rows in `train.csv` into train and validation sets (stratified by label). [web:23] If `test_images_512` and `test.csv` exist, you can additionally evaluate or run inference on them, but they are not required for the basic training run.

### 2.2 Dataset B – Diabetic Retinopathy Detection (original competition data)

- **Kaggle link:** https://www.kaggle.com/competitions/diabetic-retinopathy-detection/data [web:32]  
- **What it is:** Original EyePACS competition data (raw, higher‑resolution fundus images) with `train.zip` and `trainLabels.csv` (labels 0–4). [web:29][web:32]

To use this dataset:

1. Unzip `train.zip` into `Datasets/train/`.  
2. Place `trainLabels.csv` into `Datasets/trainLabels/`.  
3. In `dr_tahdl_model_using_dl.py`, adjust the paths at the top, for example:

PROJECT_DIR = r"/path/to/your/project"
DATASET_DIR = os.path.join(PROJECT_DIR, "Datasets")

For Dataset B (original EyePACS)
TRAIN_IMG_DIR = os.path.join(DATASET_DIR, "train")
TRAIN_CSV = os.path.join(DATASET_DIR, "trainLabels", "trainLabels.csv")

Optional if you prepare a separate test set
TEST_IMG_DIR = os.path.join(DATASET_DIR, "test") # optional
TEST_CSV = os.path.join(DATASET_DIR, "test.csv") # optional


The script will read `trainLabels.csv`, auto‑detect the filename and label columns (`image`/`id_code`, `level`/`diagnosis` etc.) and again create an internal **80:20** train/validation split from this training data. [web:29][web:32]

You are free to choose **either Dataset A (Resized)** or **Dataset B (Original)** depending on your hardware and goals.

---

## 3. TAHDL architecture (what the model does)

### 3.1 Preprocessing

For each fundus image, the pipeline applies:

- Resize to **224×224×3**. [file:1]  
- Color normalization using ImageNet mean and standard deviation:  
  - mean = (0.485, 0.456, 0.406)  
  - std  = (0.229, 0.224, 0.225) [web:27]  
- **CLAHE** (Contrast Limited Adaptive Histogram Equalization) on the L‑channel in LAB color space to enhance local contrast while avoiding over‑amplification of noise. [file:1][web:27]  
- Optional circular cropping around the retinal region so that the black corners and background do not dominate the image. [web:27]

### 3.2 Multiscale CNN backbone

The CNN backbone is responsible for spatial feature extraction, following the structure described in the TAHDL paper: [file:1]

- Initial base layers:
  - Conv(3×3, 32) → BatchNorm → ReLU  
  - Conv(3×3, 64) → BatchNorm → ReLU → MaxPool(2×2)

- Then three **parallel branches** operate on the same feature map:
  - Branch 1: 3×3 convolutions → 128 channels → 256 channels → MaxPool(2×2)  
  - Branch 2: 5×5 convolutions → 128 channels → 256 channels → MaxPool(2×2)  
  - Branch 3: 7×7 convolutions → 128 channels → 256 channels → MaxPool(2×2)

- Outputs from the three branches are concatenated along the channel dimension (total = 256 × 3 channels), followed by:
  - Adaptive average pooling to get a 1×1 spatial map  
  - Flatten → Linear → BatchNorm → ReLU  

The result is a fixed‑length embedding vector for each image (for example 256‑D or 1024‑D depending on configuration). This multiscale design helps the model capture both fine lesions and broader structures. [file:1]

### 3.3 Temporal modeling with LSTM and attention

The temporal part is designed so that the model can, in principle, work with sequences of images (e.g. multiple visits per patient). In the current version, the code mainly treats each image as a sequence of length 1, but the LSTM and attention blocks are implemented in a generic way so they can be extended to patient‑wise sequences later. [file:1]

- A 2‑layer LSTM is applied on the sequence of embeddings:
  - Input size = embedding dimension from the CNN  
  - Hidden size ≈ 256 (typical setting; can also be 512 depending on configuration)  
  - `num_layers = 2`  
  - `dropout = 0.5` between layers  
  - `batch_first = True`

- An additive attention layer then takes the LSTM outputs over time and learns a weight for each time step. It produces:
  - A context vector (weighted sum of LSTM outputs)  
  - Attention weights that indicate which time steps are more important

Even in single‑image mode, this block is active and keeps the code compatible with sequence mode planned for future work. [file:1]

### 3.4 Classification head

The final classification head operates on the context vector from attention:

- Two fully‑connected “sub‑heads” (e.g. 256‑D each) are applied and then concatenated.  
- The combined feature passes through:
  - Dropout(0.5)  
  - Linear → ReLU  
  - Dropout(0.5)  
  - Linear → 5 logits (for the 5 DR classes)

The model is trained with standard cross‑entropy loss. To handle class imbalance, class weights and/or a WeightedRandomSampler can be used. L2 regularization (weight decay = 0.01) is applied in the optimizer, as described in the paper. [file:1]

---

## 4. Training setup and GPU note

### 4.1 Hyperparameters

The default training configuration follows the paper:

- Image size: 224×224  
- Batch size: 32  
- Epochs: 50  
- Optimizer: Adam  
- Learning rate: 0.001  
- Weight decay (L2): 0.01  
- Dropout: 0.5 in LSTM and classifier layers  
- Loss: CrossEntropyLoss (with optional class weights and sampling for imbalance) [file:1]

The script:

- Loads the training CSV (`train.csv` for the resized dataset or `trainLabels.csv` for the original dataset).  
- Detects which column holds the filenames (e.g. `id_code`, `image`, `image_id`) and which holds labels (e.g. `diagnosis`, `level`, `label`). [web:23][web:32]  
- Automatically creates an **80:20** train/validation split with stratification per class. [file:1]

### 4.2 GPU requirements

This project is designed to run on a **GPU machine** (for example an NVIDIA RTX series card):

- The combination of multiscale CNN + 2‑layer LSTM + attention on tens of thousands of DR images is compute‑heavy and memory demanding. [file:1]  
- On a modern GPU (e.g. RTX 3080/3090 or better), training with batch size 32 and 50 epochs is realistic and training time is acceptable. [web:52]  
- On CPU, the code will still execute, but training will be very slow, especially on the full Kaggle datasets.

If you do not have a local GPU, the project can be run on services like Google Colab or any institutional GPU cluster.

---



After a successful run, the script will save:

- Model checkpoints in a `models` directory (for example `backend/models/...`).  
- Run configuration, logs and plots under a `results` directory.

---

## 5. Expected performance

In the original TAHDL paper, the authors report approximately:

- Around 97–98% accuracy on DRIVE  
- Around 94% accuracy on Kaggle DR (EyePACS)  
- Around 96–97% accuracy on larger combined EyePACS‑style datasets [file:1]

This project uses the same main ideas (image size, multiscale CNN, LSTM+attention, CLAHE, learning rate, batch size, epochs and L2 regularization), but the exact numbers on your runs can differ because of:

- Using the pre‑resized Kaggle dataset instead of raw images  
- Differences in train/validation splitting and augmentation  
- Hardware, random seed and other implementation details [web:23][web:29]

You can compare your training logs (accuracy, precision, recall, F1‑score) against these reference values and discuss them in your report or presentation.

---





