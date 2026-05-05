# Can CLIP Distinguish Your Cat? Transfer Learning for Individual Cat Recognition in Household Images

This repository contains the code, data, outputs, and rendered notebooks for my final project in DATASCI 447: Statistical Machine Learning II at Emory University.

## Project Overview

Models trained on large image datasets such as ImageNet are designed for broad visual recognition, but distinguishing between individual household cats is a more fine-grained task. This project explores whether pretrained CLIP image embeddings can distinguish between two specific cats, Tig and Buster, using a small custom household image dataset.

The project asks whether transfer learning can extract identity-relevant visual features without training a new deep model from scratch. It also begins to explore whether segmentation-based preprocessing can help with more difficult images where both cats appear in the same scene.

## Research Questions

1. Can pretrained CLIP embeddings distinguish between two individual household cats well enough for a simple classifier to separate them accurately?
2. How effectively can a model trained on broad visual data adapt to a very small custom dataset?
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
````

## Data

The dataset contains household images of two cats, Tig and Buster. The baseline classifier uses solo images of each cat collected across different poses, lighting conditions, distances, and backgrounds. The multi-cat extension uses additional images containing both cats.

The current baseline dataset contains 40 images split evenly between the two cats, with 30 images used for training and 10 images used for validation.

## Methods

The project has two main stages.

First, I built a CLIP-based baseline classifier. Each cat image was encoded using a pretrained CLIP image model, producing a high-dimensional embedding for each photo. I then trained a logistic regression classifier on these embeddings to classify each image as Tig or Buster.

Second, I began developing a SAM-based multi-cat extension. In this pipeline, the Segment Anything Model (SAM) generates candidate masks and cropped regions from images containing both cats. These cropped regions are then passed through a pretrained feature extractor and classifier to assign cat labels to detected regions.

## Results

The CLIP embedding baseline achieved 100% accuracy on the held-out validation set, correctly classifying all 10 validation images. The confusion matrix showed no errors on the current validation split, and the PCA visualization showed separation between the two cats in the learned embedding space.

These results suggest that pretrained visual representations can capture meaningful identity-related features even in a small household dataset. However, because the dataset is small, the results should be interpreted as encouraging preliminary evidence rather than a final generalization measure.

The SAM-based multi-cat extension is still in progress and has not yet been fully benchmarked, but it points toward a possible scene-level recognition pipeline for images containing multiple cats.

## How to Reproduce

1. Clone this repository.
2. Open `cat_classifier_baseline.ipynb`.
3. Run the notebook cells in order to reproduce the baseline CLIP embedding classifier.
4. View the generated results in the `outputs/` folder.
5. Open `sam_multicat_extension.ipynb` to inspect the in-progress multi-cat segmentation extension.

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
