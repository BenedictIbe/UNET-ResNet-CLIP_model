# UNET-ResNet-CLIP_model

UNet-ResNet-CLIP for Lung X-Ray Classification and Segmentation
Overview
This project combines the power of UNet, ResNet, and CLIP models to classify and segment lung X-ray images, focusing on distinguishing between COVID-19, Non-COVID, and Normal cases. This research addresses the challenge of accurate detection and classification of lung conditions using X-ray imagery, aiming to enhance diagnostic accuracy and efficiency in medical imaging analysis.

Objectives
Classification: To accurately categorize X-ray images as COVID-19, Non-COVID, or Normal.
Segmentation: To identify and localize infected areas in COVID-19 X-ray images, using advanced segmentation techniques.
Model Comparison: To evaluate performance across various model architectures, including UNet for segmentation and ResNet & CLIP for feature extraction and classification.
Methodology
Data Preparation
Dataset: The COVID-QU-Ex dataset, consisting of original images, lung masks, and infection masks, is utilized.
Preprocessing: Images are resized, normalized, and augmented to enhance model generalization.
DataFrame Structure: Processed images are organized into a dataframe with paths to images, infection masks, and class labels.
Models
UNet: For segmenting infected lung regions, UNet was fine-tuned to learn the distinct infection patterns in COVID-19 X-rays.
ResNet: ResNet was leveraged for its feature extraction capabilities, providing robust feature embeddings for classification.
CLIP: Fine-tuned to classify images, CLIP assists in associating visual features with textual labels (COVID-19, Non-COVID, Normal).
Training Process
Segmentation Model Training: UNet was trained on COVID-19 images with infection masks, achieving a segmentation IoU of up to 0.8.
Classification Model Training: ResNet and CLIP were fine-tuned to classify the segmented images, taking advantage of pre-trained weights and transfer learning techniques to adapt to X-ray imaging data.
Evaluation Metrics
Accuracy, Precision, Recall: Used to evaluate the performance across classification tasks.
IoU (Intersection over Union): For segmentation tasks, measuring how well the segmented area overlaps with the ground truth infection mask.
Results and Insights
Segmentation: UNet achieved high accuracy in segmenting infected areas on COVID-19 X-rays, particularly when combined with infection masks.
Classification: ResNet and CLIP provided promising results in distinguishing COVID-19 from Non-COVID and Normal cases, with ResNet outperforming in certain scenarios due to its specificity in feature extraction.
Comparison: Evaluating UNet, ResNet, and CLIP highlighted the strengths of each model for specific tasks (segmentation vs. classification), suggesting potential for hybrid approaches in medical imaging applications.
Future Work
Integration of More Classes: Further exploration of other respiratory infections to expand the classification capability.
Model Optimization: Streamlining models for faster, real-time diagnostic support.
Weakly Supervised Learning: Utilizing bounding box data without full segmentation labels to improve model learning efficiency.
How to Use
Clone the Repository:

bash
Copy code
git clone https://github.com/your-username/UNet-ResNet-CLIP-XRay-Classification.git
Install Dependencies:
Ensure all packages listed in the requirements.txt are installed.

Run the Notebook:
Open and execute the notebook to replicate the training and evaluation steps.

Acknowledgments
COVID-QU-Ex Dataset: For providing the high-quality dataset critical to this research.
OpenAI CLIP Model, UNet, ResNet: For the pre-trained models that facilitated faster training and improved accuracy.
