# Tidal Disruption Event (TDE) Classification

**Authors:**  
Julian Go  
Gabriel Masangkay  

## Overview

This project builds a machine learning pipeline to detect **Tidal Disruption Events (TDEs)** from time-domain astronomical observations.

The model analyzes **multi-band light curve data** from astronomical surveys and identifies objects that are likely to be TDEs based on their temporal and statistical characteristics.

Kaggle Competition:  
https://www.kaggle.com/competitions/mallorn-astronomical-classification-challenge

---

## Project Overview

A **Tidal Disruption Event (TDE)** occurs when a star passes close to a supermassive black hole and is torn apart by tidal forces. The resulting debris produces a luminous flare observable across multiple photometric filters.

This project aims to:

- Extract meaningful features from raw **multi-band time-series light curves**
- Handle **severe class imbalance** (~5% TDE events)
- Train an ensemble machine learning classifier
- Optimize predictions to maximize the **F1 score**

The final model uses a **CatBoost-based ensemble** combined with neural networks and KNN models.

---

## Feature Engineering

Features are extracted from astronomical light curves and include:

### Global Features
- Flux variability statistics
- Peak-to-peak amplitude
- Integrated signal energy

### Per-Band Features
For each filter (`u, g, r, i, z, y`):

- Mean flux
- Standard deviation
- Skewness and kurtosis
- Percentile features
- Signal-to-noise statistics

### Light Curve Shape
- Rise slope
- Decay slope
- Asymmetry between rise and decay
- Power-law decay approximation

### Cross-Band Features
- Peak time alignment across filters
- Color differences
- Amplitude ratios
- Band lag features

### Cosmological Correction
- Rest-frame duration using redshift

These features capture both **statistical properties** and **astrophysical characteristics** of transient events.

---

## Modeling Approach

The project uses an ensemble consisting of:

- **CatBoost (primary model)**
- Temporal specialist CatBoost model
- Statistical specialist CatBoost model
- Multi-layer perceptron (MLP)
- K-Nearest Neighbors (KNN)

Predictions are combined using **weighted probability blending**.

Cross-validation uses **GroupKFold** to prevent data leakage between objects.

---

## Installation

### 1. Clone the Repository

```
git clone https://github.com/juliangoph/mallorn-astronomical-classification.git
```

### 2. Install Dependencies

Recommended: **Python 3.9+**

```
pip install -r requirements.txt
```

### 3. Download Competition Data

Download the dataset from Kaggle:

https://www.kaggle.com/competitions/mallorn-astronomical-classification-challenge/data

Place the data inside `\data`

---

## Running the Notebook

Start Jupyter:

```
jupyter notebook
```

Open the notebook and run all cells sequentially. The notebook will:

1. Combine lightcurve files
2. Generate engineered features
3. Train ensemble models
4. Optimize prediction thresholds
5. Produce a submission file

---

## Output

The final output is:

```
submission.csv
```

This file contains predictions for the Kaggle competition.

---

## Competition Metric

The competition uses the **F1 score**, which balances precision and recall for rare event detection.