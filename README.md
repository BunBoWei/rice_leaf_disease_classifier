# Rice Leaf Disease Classification

A computer vision project for CSCI935 to classify five common rice leaf diseases using PyTorch and an EfficientNetV2 model.

## Description

This project implements a complete deep learning pipeline to classify images of rice leaves into one of five disease categories: **Brown Spot**, **Leaf Scald**, **Rice Blast**, **Rice Tungro**, and **Sheath Blight**.

The solution is designed to be robust against different backgrounds (field vs. white) by using an automated Region of Interest (ROI) cropping step to isolate the leaf before classification. The training process includes modern techniques like transfer learning, class weighting, targeted data augmentations, and CutMix regularization.

## 🔧 Setup and Installation

Follow these steps to set up the project environment.

### 1\. Prerequisites

Make sure you have Python installed on your machine. This project was developed using **Python 3.12**. You can check your version by running:

```bash
python3 --version
```

### 2\. Clone the Repository

```bash
git clone <your-repo-url>
cd <your-repo-name>
```

### 3\. Install Dependencies

All required packages are listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

### 4\. Dataset Setup

The dataset for this project must be downloaded and structured correctly before running any scripts.

  * **Step 4.1: Download the dataset**
    The dataset, "Dhan-Shomadhan", is available [here](https://uowmailedu-my.sharepoint.com/personal/wanqing_uow_edu_au/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fwanqing%5Fuow%5Fedu%5Fau%2FDocuments%2F2025%2DCSCI935%2DSP25%2FGroup%20Project&ga=1).  

  * **Step 4.2: Extract and arrange the dataset**
    After downloading, extract the compressed file. You must then arrange the extracted folders to match the following structure within your main project folder:

    _IMPORTANT: You might want to modify the subfolders' names as there might be trailing spaces or typos._    

<!-- end list -->
```
your-project-folder/
├── Dhan-Shomadhan/
│   ├── White Background/
│   │   ├── Brown Spot/
│   │   │   └── image_001.jpg
│   │   ├── Leaf Scald/
│   │   └── ... (other diseases)
│   └── Field Background/
│       ├── Brown Spot/
│       └── ... (other diseases)
├── train.py
├── eval.py
├── prepare_data.py
└── ... (all other .py files)
```

## 🚀 Workflow: How to Run

The project is designed to be run in a specific order. Each step generates files needed by the next.

### Step 1: Prepare the Dataset

This is the first and most important step. This script scans the original images, creates the data splits, and preprocesses the images for training.

  * **Command:**
    ```bash
    python prepare_data.py
    ```
  * **What it does:**
      * Automatically deletes any old `crops/` folder to ensure a fresh run.
      * Scans the `Dhan-Shomadhan` source directory.
      * Creates 5 different stratified train/validation/test splits and saves them as `.txt` files in the `Dhan-Shomadhan/splits/` folder.
      * Applies the ROI segmentation to every image, creating 224x224 cropped versions in the `Dhan-Shomadhan/crops/` folder.

### Step 2: Train the Models

After preparing the data, you need to train one model for each of the 5 runs.

  * **Command (run for each run number):**
    ```bash
    # Train the model for the first run
    python train.py --run 1

    # Repeat for other runs
    python train.py --run 2
    # ... and so on for runs 3, 4, 5
    ```
  * **What it does:**
      * Trains the **EfficientNetV2** model on the data corresponding to the specified run number.
      * Implements a two-phase training strategy (warm-up followed by fine-tuning).
      * Uses class weighting, label smoothing, and CutMix to improve robustness.
      * Saves the best-performing model checkpoint to the `Dhan-Shomadhan/results/run_X/` folder.

### Step 3: Evaluate the Models

Once a model is trained, you can evaluate its performance on the test set.

  * **Command (run for each trained model):**
    ```bash
    # Evaluate the model from the first run
    python eval.py --run 1

    # Repeat for other runs
    python eval.py --run 2
    # ... and so on
    ```
  * **What it does:**
      * Loads the best saved model for the specified run.
      * Evaluates it on three scenarios: `white_background`, `field_background`, and `mixed_background`.
      * Prints a detailed breakdown of accuracy and F1-score for each disease to the console.
      * Saves a **confusion matrix heatmap** as a `.png` image for each scenario.
      * Saves the overall metrics to a `metrics.csv` file inside the `Dhan-Shomadhan/results/run_X/` folder.

### Step 4: Aggregate Final Results

After all 5 runs have been trained and evaluated, this final script will summarize the overall performance.

  * **Command:**
    ```bash
    python aggregate.py
    ```
  * **What it does:**
      * Reads the `metrics.csv` file from all 5 run folders.
      * Calculates the **mean and standard deviation** for accuracy and F1-score across all runs.
      * Prints a final summary table to the console, which you can use in your report.

-----

## 📊 Making a Single Prediction

You can use the `predict.py` script for two main purposes: to classify a brand new image with a trained model, or to test and debug the model's performance on a specific image from your dataset.

  * **Command:**
    ```bash
    python predict.py --model_path "Dhan-Shomadhan/results/run_1/best_model.pt" --image_path "path/to/your/image.jpg"
    ```
  * **What it does:**
      * Loads the specified trained model.
      * Applies the same ROI cropping and transformations to the single image.
      * Prints the predicted disease and the model's confidence score.
