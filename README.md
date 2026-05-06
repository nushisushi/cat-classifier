# Can CLIP Distinguish Your Cat? Transfer Learning for Individual Cat Recognition in Household Images

This repository contains the code, data, outputs, and rendered notebooks for my final project in DATASCI 447: Statistical Machine Learning II at Emory University.

## Project Overview

Models trained on large image datasets such as ImageNet are designed for broad visual recognition, but distinguishing between individual household cats is a more fine-grained task. This project explores whether pretrained CLIP image embeddings can distinguish between two specific cats, Tig and Buster, using a small custom household image dataset.

The project asks whether transfer learning can pick out features that help distinguish the cats without needing to train a new deep model from scratch. It also explores whether segmentation can help with harder images where both cats appear in the same photo.

## Research Questions

1. Can pretrained CLIP embeddings distinguish between two individual household cats well enough for a simple classifier to separate them accurately?
2. How well can a model trained on broad visual data adapt to a very small custom dataset?
3. Can segmentation help extend individual cat recognition to multi-cat images?

## Repository Structure

```text
cat_project/
├── data/
│   └── raw/
│       ├── tig/
│       ├── buster/
│       └── both/
├── outputs/
│   ├── class_metrics.png
│   ├── confusion_matrix_gradient.png
│   ├── cv_accuracy_boxplot.png
│   ├── embedding_pca.png
│   ├── embedding_similarity_heatmap.png
│   ├── metrics_summary.txt
│   ├── mistakes.txt
│   └── multicat/
├── cat_classifier_baseline.ipynb
├── cat_classifier_baseline.html
├── sam_multicat_extension.ipynb
├── sam_multicat_extension.html
└── .gitignore
```

## Data

The dataset contains household images of two cats, Tig and Buster. The baseline classifier uses solo images of each cat collected across different poses, lighting conditions, distances, and backgrounds. The multi-cat extension uses additional images containing both cats.

The current baseline dataset contains 60 solo-cat images split evenly between the two cats, with 44 images used for training and 16 images held out for validation.

## Methods

The project has two main stages.

First, I built a baseline classifier using CLIP. I used a pretrained CLIP model to turn each cat photo into a high-dimensional embedding. Then, I trained a logistic regression classifier on these embeddings to decide whether each image was Tig or Buster.

Second, I began developing a SAM-based multi-cat extension. In this pipeline, the Segment Anything Model (SAM) generates candidate masks and cropped regions from images containing both cats. These cropped regions are then passed through a pretrained feature extractor and classifier to assign cat labels to detected regions.

## Results

The CLIP embedding baseline achieved 100% accuracy on the held-out validation set, correctly classifying all 16 validation images. Repeated stratified 5-fold cross-validation with 20 repeats produced a mean accuracy of 98.3%. The confusion matrix showed no errors on the current validation split, and the PCA visualization showed separation between the two cats in the learned embedding space.

These results suggest that pretrained visual models can pick up useful identity-related features, even with a small household dataset. However, because the dataset is small, the results should be interpreted as encouraging early evidence rather than a final measure of how well the method would work in general.

The SAM-based multi-cat extension uses SAM ViT-B to generate candidate masks from 12 both-cat images, applies non-max suppression to reduce duplicate regions, and classifies selected crops with a ResNet18 feature extractor and logistic regression. This extension is interpreted qualitatively because the project does not include hand-labeled bounding boxes or masks for formal detection metrics.

## How to Reproduce

1. Clone this repository.
2. Install the required dependencies:
```
pip install torch torchvision scikit-learn matplotlib pillow numpy
pip install git+https://github.com/openai/CLIP.git
pip install git+https://github.com/facebookresearch/segment-anything.git
```
3. Open `cat_classifier_baseline.ipynb`.
4. Run the notebook cells in order to reproduce the baseline CLIP embedding classifier.
5. View the generated results in the `outputs/` folder.
6. Open `sam_multicat_extension.ipynb` to inspect the exploratory multi-cat segmentation extension.

The rendered HTML files are included so that the analysis can be viewed without rerunning the notebooks.

## Notes on Model Files

Large model checkpoint files are not included in this repository. If using the SAM extension, download the required SAM checkpoint separately and place it in the project folder before running the SAM notebook.

## References

McAlister, K. (2026). DATASCI 447: Statistical Machine Learning II course lecture materials. Emory University.

Murphy, K. P. (2023). *Probabilistic Machine Learning: Advanced Topics*. MIT Press.

Prince, S. J. D. (2023). *Understanding Deep Learning*. MIT Press.

## Author

Anushka Basu
Department of Data and Decision Sciences
Emory University Class of 2026
