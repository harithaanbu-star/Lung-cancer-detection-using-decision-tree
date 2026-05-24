# Lung Cancer Prediction using Decision Tree
# 🚀 Day 4 — AI Internship Journey

## 📌 Project Overview
This repository contains my Day 5 learning progress during my AI Internship.
I worked on a **Lung Cancer Prediction** project using a medical survey dataset
of 309 patient records with 16 features, covering:
- Exploratory Data Analysis
- Label Encoding for Categorical Features
- Multiple Visualization Techniques
- Outlier Detection and Treatment using Winsorization
- Decision Tree Classification Model
- Model Evaluation with Confusion Matrix and Classification Report

---

## 🛠 Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy
- Jupyter Notebook

---

## 📊 Topics Practiced

### ✅ Dataset Overview
- Loaded Lung Cancer Survey dataset with 309 rows and 16 columns
- Explored structure using head, tail, info, describe, and shape
- Dataset had no missing values — all 309 entries complete
- 14 integer columns and 2 object columns — GENDER and LUNG_CANCER
- Most symptom features are binary encoded as 1 or 2

**Statistical Summary:**
- Age ranges from 21 to 87 years, mean age around 62.6
- All symptom columns (SMOKING, ANXIETY, etc.) have values either 1 or 2
- LUNG_CANCER target column has YES and NO string values

### ✅ Exploratory Data Analysis

**Line Plot:**
- Plotted AGE column as a line chart to observe distribution pattern

**Histogram:**
- Plotted histogram of AGE column
- Most patients are concentrated between ages 54 and 73
- Very few patients below 40 or above 80

**Distribution Plot:**
- Used seaborn distplot on AGE column
- Showed normal bell-curve distribution centered around age 62

**Pairplot:**
- Used seaborn pairplot with LUNG_CANCER as hue
- Visualized relationships between all feature pairs
- Color-coded by cancer status for pattern identification

**Bar Chart:**
- Plotted SMOKING vs LUNG_CANCER to observe smoking impact
- Visual comparison of smoking behavior with cancer diagnosis

**Pie Chart:**
- Plotted GENDER column as pie chart
- Showed proportion of male and female patients in dataset
- Note: pie chart on raw encoded values created many wedge segments

### ✅ Label Encoding
Applied Label Encoding to convert categorical columns to numeric:

| Column | Encoding |
|--------|----------|
| GENDER | F → 0, M → 1 |
| LUNG_CANCER | NO → 0, YES → 1 |

After encoding, all 16 columns became fully numeric,
ready for machine learning model training

### ✅ Visualization Errors Encountered

**Error 1 — Distribution Plot Subplot Mismatch**
- Used 4x3 grid for 16 columns — maximum allowed is 12 plots
- Plot failed at 13th feature — ValueError raised
- Lesson: Grid rows × cols must be greater than or equal to total features

**Error 2 — Boxplot Subplot Mismatch**
- Used 7x2 grid for 15 features — maximum allowed is 14 plots
- Plot failed at 15th feature — ValueError raised
- Lesson: Always verify rows × cols matches total number of features

### ✅ Outlier Treatment
- Detected outliers in AGE column using visual inspection and boxplot
- Applied **Winsorization** with 5% limits on both tails
- Extreme age values at both ends replaced with boundary values
- Visualized before and after comparison using bar chart
- Original AGE vs Winsorized AGE plotted for clear comparison

---

## 🤖 Machine Learning Model — Decision Tree Classifier

### Model Configuration

| Parameter | Value |
|-----------|-------|
| Algorithm | Decision Tree Classifier |
| Criterion | Entropy (Information Gain) |
| Train Split | 70% — 216 samples |
| Test Split | 30% — 93 samples |
| Random State | 42 |
| Tree Visualization | max_depth=3, filled=True |

### Target Variable
- `LUNG_CANCER` — YES or NO
- Encoded as 1 (Cancer) and 0 (No Cancer)

### Features Used
All 15 columns except LUNG_CANCER:
- GENDER, AGE, SMOKING, YELLOW_FINGERS, ANXIETY
- PEER_PRESSURE, CHRONIC DISEASE, FATIGUE, ALLERGY
- WHEEZING, ALCOHOL CONSUMING, COUGHING
- SHORTNESS OF BREATH, SWALLOWING DIFFICULTY, CHEST PAIN

---

## 📈 Model Performance

### Accuracy Score
- **Overall Accuracy: 92.47%**

### Confusion Matrix

| | Predicted No Cancer | Predicted Cancer |
|--|--|--|
| **Actual No Cancer** | 4 | 3 |
| **Actual Cancer** | 4 | 82 |

### Confusion Matrix Interpretation

| Metric | Value | Meaning |
|--------|-------|---------|
| True Negatives (TN) | 4 | Correctly predicted No Cancer |
| False Positives (FP) | 3 | No Cancer predicted as Cancer |
| False Negatives (FN) | 4 | Cancer predicted as No Cancer |
| True Positives (TP) | 82 | Correctly predicted Cancer |

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0 — No Cancer | 0.50 | 0.57 | 0.53 | 7 |
| 1 — Cancer | 0.96 | 0.95 | 0.96 | 86 |
| Accuracy | | | 0.92 | 93 |
| Macro Average | 0.73 | 0.76 | 0.75 | 93 |
| Weighted Average | 0.93 | 0.92 | 0.93 | 93 |

### Visualization
- Confusion Matrix plotted as heatmap using Seaborn with Blues colormap
- Actual vs Predicted values plotted as line chart
- Decision Tree structure visualized with figsize 150x100 and max_depth=3

---

## 📁 Dataset Description

| Property | Detail |
|----------|--------|
| Total Records | 309 rows |
| Total Features | 16 columns |
| Missing Values | None |
| Target Variable | LUNG_CANCER |
| Target Classes | YES (Cancer) / NO (No Cancer) |
| Numeric Features | 14 integer, 2 object |

### Feature Description

| Feature | Description |
|---------|-------------|
| GENDER | Patient gender — M or F |
| AGE | Patient age in years |
| SMOKING | Smoking habit — 1 or 2 |
| YELLOW_FINGERS | Yellow finger discoloration — 1 or 2 |
| ANXIETY | Anxiety symptoms — 1 or 2 |
| PEER_PRESSURE | Peer pressure factor — 1 or 2 |
| CHRONIC DISEASE | Existing chronic disease — 1 or 2 |
| FATIGUE | Fatigue symptom — 1 or 2 |
| ALLERGY | Allergy presence — 1 or 2 |
| WHEEZING | Wheezing symptom — 1 or 2 |
| ALCOHOL CONSUMING | Alcohol consumption — 1 or 2 |
| COUGHING | Coughing symptom — 1 or 2 |
| SHORTNESS OF BREATH | Breathing difficulty — 1 or 2 |
| SWALLOWING DIFFICULTY | Swallowing issue — 1 or 2 |
| CHEST PAIN | Chest pain symptom — 1 or 2 |
| LUNG_CANCER | Target — YES or NO |

---

## 🔍 Key Observations

| Observation | Detail |
|-------------|--------|
| Dataset Size | 309 rows × 16 columns |
| Missing Values | Zero — clean dataset |
| Age Range | 21 to 87 years |
| Most Common Age | 54 to 73 years |
| Outlier Treated | AGE — Winsorized at 5% |
| Model Used | Decision Tree with Entropy |
| Model Accuracy | 92.47% |
| Cancer Detection Rate | 82 out of 86 cancer cases correctly detected |
| Class Imbalance | Only 7 No Cancer cases vs 86 Cancer cases in test set |

---

## ⚠️ Errors Encountered and Lessons Learned

**Error 1 — Distribution Plot Grid Too Small**
- 4x3 grid used for 16 features — supports only 12 subplots
- ValueError at 13th subplot position
- Fix: Use 4x4 or 6x3 grid for 16 features

**Error 2 — Boxplot Grid Too Small**
- 7x2 grid used for 15 features — supports only 14 subplots
- ValueError at 15th subplot position
- Fix: Use 8x2 or 5x3 grid for 15 features

**Warning — Deprecated distplot Function**
- sns.distplot is deprecated from seaborn v0.14.0
- Replacement: Use sns.histplot or sns.displot instead

**Lesson:** Always calculate rows × cols before plotting subplots
and use updated seaborn functions for distribution plots

---

## 📈 Learning Outcome

Through this project, I learned:
- How to perform EDA on a medical survey dataset
- How to apply Label Encoding to binary categorical features
- How to visualize data distributions using histogram, distplot, and pairplot
- How to detect and treat outliers in age data using Winsorization
- How to build a Decision Tree with Entropy criterion for medical prediction
- How to interpret Confusion Matrix in healthcare context
- How to read Precision, Recall, and F1-Score from Classification Report
- How class imbalance affects model performance on minority class
- How to correctly size matplotlib subplot grids to avoid ValueError
- Importance of using updated seaborn functions to avoid deprecation warnings

---

## 🎯 Internship Progress

This is **Day 5** of my AI and Machine Learning journey.
I will continue uploading my daily learning progress,
projects, and experiments throughout the internship.

---

## ⭐ Connect With Me

Feel free to explore the repository and follow my learning journey.
