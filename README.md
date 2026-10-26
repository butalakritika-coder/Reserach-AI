# AI Adoption among Indian Establishments: A Data-Driven & Machine Learning Approach

## Overview

This project investigates whether technological, digital, and workforce-related characteristics can help predict AI adoption among Indian establishments. It combines research on AI adoption with a machine-learning classification approach using establishment-level data.

## Research Motivation

AI adoption varies across businesses. Previous studies identify factors related to technology, organisational readiness, and the external environment. This project examines measurable establishment characteristics and compares multiple machine-learning models for predicting AI adoption.

## Dataset

**Source:** World Bank Enterprise Survey — Follow-up on Artificial Intelligence 2026, India.

* Original dataset: 1,355 establishments and 132 variables.
* Analytical dataset: 1,354 rows after preprocessing.
* Target variable: `ai_adopt1` — AI adopter/non-adopter.
* Selected predictors: 14 technological, digital, software-use, and workforce-related variables.

The dataset includes establishments of different sizes; results should not be interpreted as applying exclusively to SMEs.

Dataset catalog: https://microdata.worldbank.org/catalog/8552/data-dictionary

## Methodology

The workflow consists of:

1. Data loading and preprocessing.
2. Selection of the target and 14 predictor variables.
3. Stratified 80:20 train-test split.
4. Training and comparison of four classifiers.
5. Evaluation using accuracy, precision, recall, F1-score, and ROC-AUC.
6. Stratified 5-fold cross-validation.
7. Random Forest feature-importance analysis.

### Machine-Learning Models

* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting

## Results

On the held-out test set, Gradient Boosting achieved the highest accuracy and F1-score among the four tested models. Random Forest achieved the highest ROC-AUC on the test set.

The Random Forest feature-importance analysis ranked specialised finance software, daily computer usage, and digital finance platforms among the most influential predictors in this model.

These results describe predictive associations; they do not establish causality.

Detailed result tables and plots are available in the `results/` directory.

## Repository Structure

```text
Reserach-AI/
├── PPT/
├── Research papers/
├── src/
│   └── main.py
├── results/
│   ├── cleaned_data.csv
│   ├── model_comparison.csv
│   ├── model_comparison.png
│   ├── cross_validation_results.csv
│   ├── cross_validation_comparison.png
│   ├── feature_importance.png
│   └── confusion_matrix_*.png
├── README.md
├── requirements.txt
└── .gitignore
```

## Setup

Use Python 3.10 or later.

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

Run the main implementation from the repository root:

```bash
python src/main.py
```

**Dataset note:** The source dataset is not included in this repository by default. Place the required data file in the expected location before running the script. Check the paths configured in `src/main.py`.

## Limitations

* Results depend on the available variables and the dataset's coverage.
* Feature importance indicates predictive contribution, not causal influence.
* The dataset includes establishments of different sizes, so SME-specific conclusions require separate analysis.

## Project Context

Research-based academic project — B.Tech Computer Science and Engineering, Manipal University Jaipur.

**Student:** Kritika Butala
**Supervisor:** Dr. Amit Kumar Gupta
