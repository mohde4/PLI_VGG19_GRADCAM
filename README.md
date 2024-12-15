# Comparing Imaging Interpretability  using GRAD-CAM and PLI
This Python-based program systematically evaluates and compares the performance of the Pixel-Level Interpretability (PLI) model with Grad-CAM, using publicly available COVID-19 chest radiograph datasets. The program is tailored to assess interpretability, diagnostic accuracy, and computational efficiency. It is implemented in Python with deep learning frameworks like TensorFlow or PyTorch and additional tools for visualization and statistical analysis.

#  Project Description: Enhanced Interpretability Diagnosis with PLI and Grad-CAM Interpretability Using VGG19
This project focuses on enhancing interpretability in AI-driven medical diagnostics using chest X-ray images. It leverages deep learning techniques and interpretability methods to provide visual explanations of model predictions. The primary components of this project are:

1. Pixel-Level Interpretability (PLI)
PLI is a fine-grained interpretability method designed to highlight the importance of each pixel in an image concerning the model's prediction. By attributing relevance scores at the pixel level, PLI provides highly detailed heatmaps that allow clinicians to understand exactly which regions of an X-ray contributed to the diagnostic outcome. This method is particularly valuable in medical imaging, where precise localization of abnormalities is critical.

2. Gradient-weighted Class Activation Mapping (Grad-CAM)
Grad-CAM is a widely-used interpretability technique that generates class-discriminative heatmaps, highlighting the regions of an image that are most influential in the model's decision-making process. Unlike PLI, Grad-CAM produces more general heatmaps that capture broader feature areas, making it suitable for identifying larger regions of interest in X-ray images.

3. VGG19 Model
, a convolutional neural network (CNN) architecture pre-trained on ImageNet, serves as the backbone of the pneumonia classification model. VGG19 is known for its depth and simplicity, making it effective for extracting complex features from medical images. The model is fine-tuned to classify chest X-ray images as either pneumonia-positive or normal.

Workflow
Data Preprocessing: Chest X-ray images are resized, normalized, and augmented to ensure robustness.
Model Training: The VGG19 model is trained on  normal and infected datasets, optimizing for accuracy and generalization.
Interpretability Generation: After training, PLI and Grad-CAM are applied to generate heatmaps for a given query X-ray image.
Visualization: The generated heatmaps are overlaid on the original image to provide visual explanations of the model's decision-making process.
Purpose
The primary goal of this project is to enhance the interpretability of AI models in medical imaging, ensuring that predictions can be understood and trusted by healthcare professionals. By comparing PLI and Grad-CAM, this study aims to determine which method provides more clinically relevant explanations, ultimately contributing to safer and more transparent AI-assisted diagnostics.


## 1. Overview
This repository supports:
- **Preprocessing**: Scripts to handle real-world medical imaging datasets.
- **Training**: Implementation of the PLI model using both simulated and real-world data.
- **Heatmap Generation**: Scripts to produce interpretability maps.
- **Reproducibility**: Steps to validate metrics and replicate results from the manuscript.

---

## 2. Requirements
To run the scripts, you need the following dependencies:
- **Python >= 3.8**
- **TensorFlow >= 2.x**
- **NumPy**
- **OpenCV**
- **Matplotlib**
- **Pandas**
- **Scikit-learn**
## 3. Dataset Information
The study utilizes publicly available chest X-ray datasets. Download the datasets from:



## 4. Preprocessing steps include:
- **Resizing images to 224x224 pixels.**
- **Normalizing pixel values.**
- **Reducing noise using Gaussian filtering.**
- **Applying data augmentation techniques such as rotation and flipping.**
## To preprocess the data, run:
Install dependencies using:
```bash
pip install -r requirements.txt
python preprocessing/preprocess_real_data.py --input_path data/raw --output_path data/processed
python train_real_data.py --data_path data/processed --batch_size 64 --epochs 30 --learning_rate 0.0001
python train_simulation.py --data_path data/simulation --batch_size 32 --epochs 20
python generate_pli_heatmaps.py --model_path models/pli_model.h5 --image_path data/processed/sample_image.jpg --output_path results/
python evaluate_metrics.py --data_path data/processed --model_path models/pli_model.h5
python generate_calibration_curve.py --data_path data/processed --model_path models/pli_model.h5
Adjust parameters such as learning rates, epochs, and fuzzification thresholds in configs/parameters.json. Example parameters include:
{
  "learning_rate": 0.0001,
  "batch_size": 64,
  "epochs": 30,
  "fuzzification_threshold": 0.5
}
python validate_model.py --data_path data/test --model_path models/pli_model.h5
python generate_logs.py --output_path logs/
```
## 6. Troubleshooting
- **Dataset Issues: Ensure datasets are downloaded and paths are specified correctly in the scripts.**
- **Memory Errors: Reduce batch size or use a GPU for training to handle large datasets.**
- **Script Errors: Check if required dependencies are installed and compatible with your system.**

## 7. System Requirements for Running the Model
To run the Pixel-Level Interpretability (PLI) model effectively, the following system requirements are recommended:

# Hardware Requirements
Processor:

Minimum: Quad-core CPU (e.g., Intel Core i5, AMD Ryzen 5)
Recommended: Octa-core CPU or better (e.g., Intel Core i7/i9, AMD Ryzen 7/9)
Graphics Processing Unit (GPU):

Minimum: NVIDIA GTX 1060 (6GB VRAM) or equivalent
Recommended: NVIDIA RTX 3080 (10GB VRAM) or higher for faster computation during training and inference
Memory (RAM):

Minimum: 16 GB
Recommended: 32 GB or more for handling large medical imaging datasets
Storage:

Minimum: 256 GB SSD
Recommended: 1 TB SSD for fast read/write access to datasets and intermediate results
# Monitor Resolution:

Full HD (1920 x 1080) minimum for visualizing heatmaps and results
Software Requirements
Operating System:

Linux (Ubuntu 20.04 or later preferred)
Windows 10/11 64-bit
macOS (if GPU support is available)
Programming Environment:

Python 3.8 or higher
Required Libraries and Frameworks:

TensorFlow 2.6 or later / PyTorch 1.10 or later (for deep learning components)
NumPy, Pandas (for data manipulation)
Matplotlib, Seaborn (for visualization)
Scikit-learn (for preprocessing and additional machine learning utilities)
Fuzzy Logic Libraries: Scikit-fuzzy or custom implementations
# Visualization Tools:

Jupyter Notebook or JupyterLab for running and analyzing experiments
Optional (Hardware Acceleration):

CUDA Toolkit (for NVIDIA GPUs)
cuDNN library
# Dataset Requirements
Image Formats: DICOM, PNG, or JPEG
Resolution: Standardized to 512x512 pixels for preprocessing
Storage: Approximately 50 GB of free disk space for datasets, logs, and outputs
Performance Recommendations
For real-time or high-throughput scenarios, consider using a server-grade GPU (e.g., NVIDIA A100) and at least 64 GB RAM.
Cloud-based solutions, such as Google Colab Pro, AWS EC2 (with GPU instances), or Azure ML, are suitable alternatives for scaling computational resources as needed.

# Initialize local git repository if not done
git init

# Add remote repository
git remote add origin https://github.com/mohde4/PLI_GRADCAM_VGG19.git

# Add files
git add .

# Commit changes
git commit -m "Initial commit with code and README"

# Push to GitHub
git push -u origin main
