# Multi-Stage Ensemble and Deep Learning for Fake News Detection: Reproduction Repository

This repository contains the complete source code, configuration files, and documentation required to reproduce the results presented in the paper **"Multi-Stage Ensemble and Deep Learning for Fake News Detection: A Hierarchical Feature Fusion Approach"**.

The system implements a novel three-stage hierarchical ensemble that fuses 10,034 handcrafted linguistic features (lexical, syntactic, semantic, stylometric) with deep sequence representations to achieve state-of-the-art accuracy in automated credibility assessment.

---

## 📑 Table of Contents
1. [Repository Structure](#repository-structure)
2. [Dataset Information](#dataset-information)
3. [Environment Setup](#environment-setup)
4. [Reproduction Instructions](#reproduction-instructions)
5. [Generating Figures & Results](#generating-figures--results)
6. [License & Attribution](#license--attribution)

---

## 📂 Repository Structure

```text
├── data/
│   ├── processed/              # Final preprocessed and augmented datasets
│   ├── preprocessor.py         # 3-stage text cleaning pipeline
│   └── advanced_feature_engineering.py # Extraction of 4-level linguistic features
├── models/
│   ├── traditional_models.py   # Stage 1 base classifiers (RF, SVM, NB)
│   ├── neural_network.py       # Stage 3 deep learning (CNN, LSTM, Hybrid)
│   └── multi_stage_ensemble.py # Final meta-learner and stacking architecture
├── evaluation/
│   ├── statistical_testing.py  # McNemar's tests and confidence intervals
│   └── error_analysis.py       # False positive/negative profiling
├── results/
│   ├── ieee_figures/           # High-resolution figures for the main manuscript
│   └── supplementary_figures/  # Extended plots for supplementary materials
├── paper/
│   ├── fake_news_detection_ieee.md # Main manuscript
│   └── supplementary_materials.md  # Supplementary tables and figures
├── generate_ieee_figures.py    # Script to regenerate all publication figures
├── requirements.txt            # Explicit dependency version pinning
└── README.md                   # This reproduction guide
```

---

## 📊 Dataset Information

The study utilizes a balanced binary fake news corpus combining ~44,898 articles (23,481 fake, 21,417 real), structurally identical to the widely used **ISOT Fake News Dataset**.

**Accessing the Data:**
Due to licensing and size constraints, the raw dataset is not bundled directly in this repository. To reproduce the study:
1. Download the ISOT Fake News Dataset from [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) or the original University of Victoria repository.
2. Place the `True.csv` and `Fake.csv` files into the `data/raw/` directory.
3. The automated preprocessing pipeline will automatically merge, shuffle, clean, and stratify the data into the required `data/processed/cleaned_dataset.csv` format.

*Note: The dataset is subject to its original creators' licensing. Please cite the original authors when using the ISOT dataset.*

---

## ⚙️ Environment Setup

This project is tested across macOS (M-series and Intel), Linux (Ubuntu 22.04), and Windows 11. It requires **Python 3.10+**.

**1. Clone the repository:**
```bash
git clone https://github.com/your-username/fake-news-detection-ensemble.git
cd fake-news-detection-ensemble
```

**2. Create a virtual environment:**
```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
.\venv\Scripts\activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Download necessary NLTK data:**
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger'); nltk.download('vader_lexicon')"
```

---

## 🚀 Reproduction Instructions

To run the complete pipeline from raw data to the final meta-ensemble model evaluation:

**Step 1: Data Preprocessing & Feature Engineering**
```bash
# Cleans text, extracts Lexical/Syntactic/Semantic/Stylometric features, and generates TF-IDF
python data/advanced_feature_engineering.py
```

**Step 2: Train Individual Models (Stages 1 & 2)**
```bash
# Trains Random Forest, SVM, Naive Bayes, and Gradient Boosting
python models/traditional_models.py
```

**Step 3: Train Deep Learning Models (Stage 3)**
```bash
# Trains CNN, LSTM, and the Hybrid feature-sequence architecture
python models/neural_network.py
```

**Step 4: Train Meta-Ensemble and Evaluate**
```bash
# Fuses predictions from previous stages into the final logistic regression meta-learner
python models/multi_stage_ensemble.py
```
*Note: Depending on your hardware, Step 2 and 3 may take several hours. A GPU is strongly recommended for Step 3.*

---

## 📈 Generating Figures & Results

To perfectly recreate the figures used in the main manuscript and the supplementary materials, run the dedicated figure generation script. This script loads the trained models and test sets, computes all necessary metrics, and outputs publication-ready, non-cropped `.png` files.

```bash
python generate_ieee_figures.py
```

The figures will be saved in:
- Main Paper Figures: `results/ieee_figures/`
- Supplementary Figures: `results/supplementary_figures/`

---

## 📄 License & Attribution

**Code License:** MIT License
You are free to use, modify, and distribute this code for academic and commercial purposes, provided proper attribution is given.

**Citation:**
If you use this codebase, methodology, or the resulting models in your research, please cite our paper:

```bibtex
@article{fakenews_ensemble_2026,
  title={Multi-Stage Ensemble and Deep Learning for Fake News Detection: A Hierarchical Feature Fusion Approach},
  author={Medin, et al.},
  journal={IEEE Transactions on Knowledge and Data Engineering (Submitted)},
  year={2026}
}
```