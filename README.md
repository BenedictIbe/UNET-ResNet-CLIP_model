# UNET-ResNet-CLIP_model

Lung X-Ray Classification and Segmentation with UNet, ResNet, and CLIP

Table of Contents
Overview
Objectives
Methodology
Data Preparation
Model Architectures
Training Process
Evaluation Metrics
Results and Insights
Future Work
Getting Started
Clone the Repository
Install Dependencies
Run the Notebook
Acknowledgments
License

Overview
This project leverages UNet, ResNet, and CLIP models to classify and segment lung X-ray images, specifically identifying COVID-19, Non-COVID, and Normal cases. This research aims to improve diagnostic accuracy in identifying lung infections using medical imaging.

Objectives
Classification: Accurately categorize lung X-rays into COVID-19, Non-COVID, and Normal classes.

Segmentation: Identify and localize infected regions within COVID-19 X-ray images.

Model Comparison: Evaluate performance across different model architectures for both classification and segmentation tasks.

Methodology

Data Preparation

Dataset: Utilizes the COVID-QU-Ex dataset with original X-ray images, lung masks, and infection masks.

Preprocessing: Images are resized, normalized, and augmented to improve model generalization.

Data Organization: Dataframe structure includes paths to images, infection masks, and corresponding class labels.
Model Architectures

UNet: Employed for segmenting infected lung regions, trained to highlight specific infection areas.

ResNet: Acts as a feature extractor, providing robust embeddings for classification.

CLIP: Fine-tuned to classify images based on visual and textual labels (COVID-19, Non-COVID, Normal).
Training Process

Segmentation Training: UNet was trained on COVID-19 images with infection masks, reaching an Intersection over Union (IoU) of 0.8.

Classification Training: ResNet and CLIP were fine-tuned for classification tasks, utilizing transfer learning for adaptation to X-ray data.
Evaluation Metrics
Accuracy, Precision, Recall: Used to assess classification performance.
IoU (Intersection over Union): Key metric for evaluating segmentation accuracy.
Results and Insights

Segmentation: UNet successfully segmented infected areas in COVID-19 X-rays, achieving high IoU scores.

Classification: ResNet and CLIP accurately classified images, with ResNet slightly outperforming due to better feature extraction.

Model Comparison: Each model demonstrated strengths; UNet for segmentation and ResNet & CLIP for robust classification, suggesting that a hybrid approach may be optimal.
Future Work

Expanded Classification: Include more respiratory infection classes.

Model Optimization: Adapt models for real-time diagnostic use.

Weak Supervision: Explore using bounding boxes as a training aid without full segmentation labels.


Getting Started
Clone the Repository

git clone https://github.com/your-username/UNet-ResNet-CLIP-XRay-Classification.git

Install Dependencies

Ensure all packages in requirements.txt are installed:


pip install -r requirements.txt

Run the Notebook
Open unet-resnet-clip.ipynb in Jupyter Notebook or Jupyter Lab.
Follow the steps in the notebook to replicate training and evaluation processes.
Acknowledgments
COVID-QU-Ex Dataset: For providing the dataset used in this research.
OpenAI CLIP, UNet, and ResNet: For pre-trained models and architectures that facilitated this research.
License
This project is licensed under the MIT License. See the LICENSE file for details.


