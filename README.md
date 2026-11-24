# Thyroid Disease Classification with Machine Learning and Streamlit

A multi-class classifier that predicts a patient's thyroid condition (**Negative, Hyperthyroid, Hypothyroid, Binding Protein, Other**) from routine blood tests and patient history, with a Streamlit app (ThyroCare AI) that predicts for a new patient.

> **Disclaimer:** this is a learning project trained on an old clinical dataset. It is not a medical diagnostic tool.

![App screenshot](images/app_screenshot.png)

## Dataset

- UCI Thyroid Disease database, in the Kaggle file `thyroidDF.csv` ([Kaggle: emmanuelfwerr/thyroid-disease-data](https://www.kaggle.com/datasets/emmanuelfwerr/thyroid-disease-data), CC0 public domain).
- 9,172 patients and 31 columns: age, sex, yes/no medical history, five blood tests (TSH, T3, TT4, T4U, FTI) and the diagnosis code.
- Class balance after cleaning (9,070 patients): Negative 6,690, Other 1,096, Hypothyroid 667, Binding Protein 376, Hyperthyroid 241. Always predicting Negative would score 73.8% accuracy.

## Workflow

1. **Target preparation:** grouped 32 diagnosis codes into five classes by the first letter (A-D hyperthyroid, E-H hypothyroid, I-J binding protein, `-` negative, the rest Other).
2. **Cleaning:** dropped `patient_id`, the original target and `TBG` (96% missing); removed 102 duplicate rows (checked after dropping the ID); replaced impossible ages with the median.
3. **Feature assessment:** chi-square (with Cramer's V) for categorical columns and ANOVA for numeric columns, run before filling missing values; dropped `lithium`, `query_on_thyroxine` and `I131_treatment`.
4. **Missing values:** a missing lab value means the test was not done, so labs were filled with the median and the `*_measured` flags were kept; blank sex was filled with the most common value.
5. **Encoding and transformation:** t/f to 1/0, sex to 1/0, one-hot encoding of referral source, log transform of the skewed TSH.
6. **Split and scaling:** stratified 80/20 split; the six numeric columns were scaled with a scaler fitted on the training set only.
7. **Imbalance:** random oversampling of the training set only (5,352 rows per class), so the test set keeps its real proportions.
8. **Models:** kNN (k=7), Gaussian Naive Bayes (`var_smoothing=0.1`) and Decision Tree (entropy, `max_depth=8`), each trained with and without oversampling.

## Results

| Model | Train accuracy | Test accuracy |
|---|---|---|
| kNN (k=7) | 0.895 | 0.872 |
| Gaussian Naive Bayes | 0.835 | 0.830 |
| Decision Tree (depth 8) | 0.967 | 0.939 |

Baseline (always Negative): 0.738.

Two problems found and fixed along the way: the default Naive Bayes scored 0.159 (fixed with `var_smoothing=0.1`), and a Decision Tree with no depth limit scored 0.997 on training and 0.933 on test (fixed with `max_depth=8`).

Accuracy alone hides how the small classes do, so see the per-class reports in the notebook (macro precision, recall and F1).

## Streamlit app

The notebook saves `thyroid_model_bundle.joblib` (six models, scaler, column order, medians and metrics). The app loads it, applies the same preprocessing to the entered patient values, runs all six models and shows each prediction and a final vote.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Use consistent lab values: in this dataset FTI = TT4 / T4U.

## Project structure

```
.
├── README.md
├── requirements.txt
├── app.py                          # Streamlit app (ThyroCare AI)
├── thyroid_model_bundle.joblib     # saved models and preprocessing
├── notebooks/
│   └── Thyroid_ML_clean.ipynb      # full pipeline
├── data/
│   └── thyroidDF.csv               # dataset (or download from Kaggle)
└── images/
    └── app_screenshot.png
```

## Limitations

- Old (1980s) data from one clinic, so it may not match other populations.
- The small classes have few patients (48 Hyperthyroid patients in the test set).
- The "Other" class mixes several situations, and some labels are noisy.
- Median imputation and the log transform were applied before the split (a pipeline would be stricter).

## Author

Niranjana Rajesh · www.linkedin.com/in/niranjana-rajesh-0610ab292· krniranjanarajesh@gmail.com
