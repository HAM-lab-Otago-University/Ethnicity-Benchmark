# Ethnicity-Benchmark
# 🧠 Benchmarking Ethnicity-Related Bias in MRI-Based Cognitive Prediction

This repository contains the analysis and modelling scripts for the study:

> **Khakpoor, F.L., van der Vliet, W., Deng, J., & Pat, N. (2025).  
> When Brain Models Aren’t Universal: Benchmarking of Ethnic Bias in MRI-Based Cognitive Prediction Across Modalities**

The project investigates how predictive models trained on MRI-derived features generalize across ethnic groups, using data from the **Adolescent Brain Cognitive Development (ABCD)** study.
Preprint: https://www.biorxiv.org/cgi/content/short/2025.11.12.688133v1
---

## 🔍 Overview

We systematically benchmarked **ethnicity-related bias** in MRI-based models predicting cognitive functioning, focusing on:
- **81 unimodal** MRI feature sets (e.g., structural MRI, diffusion MRI, task-based response, functional connectivity)  
- **11 multimodal** stacked models integrating predictions across modalities  
- **Four training strategies:**
  1. All (majority White American)
  2. RandWA-only (randomly selected White Americans)
  3. AA-only (African Americans)
  4. Balanced AA + RandWA  
- **Incremental oversampling analyses** to test how increasing African American representation affects model accuracy and fairness.

All preprocessing steps (fMRIPrep-based pipelines, GLM modeling, parcellation, and feature extraction) are documented and available in a separate repository:  
👉 [HAM-lab-Otago-University/ADHD-MultiModal](https://github.com/HAM-lab-Otago-University/ADHD-MultiModal)

---

## 📂 Repository Contents

| File | Description |
|------|--------------|
| `PLS_Unimodal_matchedGroups.py` / `.sl` | Runs Partial Least Squares (PLS) regression on unimodal features across matched training/test groups |
| `PLS_Unimodal_IncrOverSampling_WA.py` / `.sl` | Implements incremental oversampling of African American participants for unimodal models |
| `RF_Multimodal_MatchedGroups.py` / `.sl` | Random Forest stacked modelling combining multimodal MRI phenotypes |
| `RF_Multimodal_IncrOverSampling_EqualWA.py` / `.sl` | Performs incremental oversampling for multimodal models |
| `Analysis-Plot.ipynb` | Generates figures: prediction errors (MAE), Ethnicity Bias Index (EBI), accuracy–bias correlations, and incremental sampling curves |

---

## ⚙️ Dependencies

```bash
Python >= 3.9  
NumPy  
pandas  
scikit-learn  
matplotlib  
seaborn  
joblib
