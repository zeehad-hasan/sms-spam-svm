# SMS Spam Classification with SVM

A machine-learning project that classifies SMS messages as **spam** or **ham (legitimate)** using TF-IDF feature extraction and a Linear Support Vector Machine.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Overview

This project builds an end-to-end pipeline for detecting spam SMS messages. It covers exploratory data analysis, text preprocessing, feature extraction with TF-IDF, model training and tuning, evaluation, and a reusable prediction helper.

**Pipeline:**

```
Raw SMS → Data Cleaning → Text Preprocessing → TF-IDF → Linear SVM
        → Hyperparameter Tuning → Evaluation → Save Model → Custom Prediction
```

---

## Results

Evaluated on a held-out test set (20% of the dataset, stratified):

| Metric              | Value  |
| ------------------- | ------ |
| Test accuracy       | ~0.985 |
| Precision (spam)    | ~0.98  |
| Recall (spam)       | ~0.89  |
| F1-score (spam)     | ~0.94  |
| ROC-AUC             | ~0.99  |

> The dataset is imbalanced (~87% ham / ~13% spam), so **F1 on the spam class** is the primary metric rather than accuracy. A trivial "always predict ham" classifier already reaches ~87% accuracy.

---

## Project Structure

```
sms-spam-svm/
├── SMS_Spam_SVM.ipynb     # main notebook (data → model → evaluation)
├── requirements.txt       # Python dependencies
├── .gitignore             # files excluded from version control
├── README.md              # this file
├── data/                  # dataset (git-ignored, auto-downloaded on first run)
└── models/                # saved model artefacts (git-ignored)
```

---

## Dataset

**SMS Spam Collection** — 5,574 English SMS messages labelled as *ham* or *spam*.

- UCI Machine Learning Repository: <https://archive.ics.uci.edu/dataset/228/sms+spam+collection>
- Kaggle mirror: <https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset>

The notebook downloads the dataset automatically on the first run and caches it under `data/`. No manual upload is required.

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/sms-spam-svm.git
cd sms-spam-svm
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the notebook

```bash
jupyter notebook SMS_Spam_SVM.ipynb
```

Or open it directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-username>/sms-spam-svm/blob/main/SMS_Spam_SVM.ipynb)

---

## Methodology

### 1. Data quality checks
- Missing values, duplicate rows, and duplicate messages removed
- Class distribution inspected

### 2. Exploratory data analysis
- Count plots and word-count distributions by class
- Word clouds for spam vs. ham vocabulary

### 3. Text preprocessing
- Lower-casing
- Removing URLs, email addresses, digits, and punctuation
- Stop-word removal — **negation words kept** (`not`, `never`, `dont`, …) so *"you have not won"* is not reduced to *"you have won"*
- Lemmatization with `WordNetLemmatizer`
- Dropping 1-character tokens

### 4. Feature extraction
- **TF-IDF** vectorization
- Unigrams + bigrams (`ngram_range=(1, 2)`)
- `sublinear_tf=True` to dampen the effect of repeated terms
- `max_features` and `ngram_range` tuned along with the classifier

### 5. Model selection
Three baselines are cross-validated on the training set before committing to an SVM:

| Model                | Notes                                          |
| -------------------- | ---------------------------------------------- |
| MultinomialNB        | Classic text-classification baseline           |
| LogisticRegression   | Strong linear model with calibrated probabilities |
| **LinearSVC**        | Fast and strong on sparse high-dimensional data |

### 6. Hyperparameter tuning
- `GridSearchCV` over `tfidf__max_features`, `tfidf__ngram_range`, `tfidf__sublinear_tf`, and `svm__C`
- 5-fold stratified cross-validation
- Scored on **F1** (not accuracy)

### 7. Evaluation
- Classification report (precision / recall / F1)
- Confusion matrix
- ROC curve + AUC
- Precision-Recall curve + average precision
- 5-fold cross-validation on the training set

### 8. Model persistence
- The full pipeline (vectorizer + classifier) saved with `joblib` to `models/sms_spam_svm.joblib`

---

## Usage Example

After running the notebook, the saved model can be loaded and used to classify new messages:

```python
import joblib

model = joblib.load("models/sms_spam_svm.joblib")

def predict_sms(text):
    prediction = model.predict([text])[0]
    return "Spam" if prediction == 1 else "Ham"

print(predict_sms("Congratulations! You have won a free iPhone. Click here now."))
# -> Spam

print(predict_sms("Hi, are we meeting tomorrow at 3 PM?"))
# -> Ham
```

---

## Limitations

- SMS text is short and informal; lemmatization helps less than on longer documents.
- `LinearSVC` does not produce calibrated probabilities — use `CalibratedClassifierCV` if probabilities are needed.
- The dataset is from 2011, so its vocabulary is dated (*"txt"*, *"claim"*). Performance on modern messages is untested.
- Class imbalance is handled through metric choice only. `class_weight="balanced"` is a natural next experiment.

---

## Possible Next Steps

- Try `SVC(kernel="rbf")` on a reduced feature set, or `SGDClassifier` for very large corpora.
- Replace TF-IDF with word embeddings (Word2Vec / GloVe / Sentence-Transformers).
- Fine-tune a small transformer (DistilBERT) and compare against the SVM.
- Deploy behind a FastAPI endpoint or a Streamlit demo.
- Add unit tests for `clean_text()` and a GitHub Actions workflow that runs the notebook end-to-end.

---

## License

Released under the MIT License. See `LICENSE` for details.

---

## Acknowledgements

- Dataset: Almeida, T. A., Gómez Hidalgo, J. M., & Yamakami, A. (2011). *Contributions to the Study of SMS Spam Filtering: New Collection and Results.*
- Built with [scikit-learn](https://scikit-learn.org/), [NLTK](https://www.nltk.org/), and [pandas](https://pandas.pydata.org/).