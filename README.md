# 🔍 AI-Powered Fake News Detection System

A machine learning system that classifies news articles as **Real** or **Fake** using NLP techniques — built entirely from scratch in Python without using sklearn's classifiers.

---

## 📊 Model Performance

| Model | Accuracy | F1-Score | AUC-ROC |
|---|---|---|---|
| Logistic Regression | 97.27% | 97.33% | 99.43% |
| Naive Bayes | 94.06% | 94.06% | 98.25% |
| Random Forest | 98.31% | 98.34% | 99.87% |
| **Ensemble (Voting)** | **95.89%** | **95.88%** | **99.56%** |

> Trained on 44,898 real-world articles from the Kaggle Fake News Dataset

---

## 🧠 How It Works
```
Raw Text
   ↓
Text Preprocessing (clean, tokenize, remove stopwords, stem)
   ↓
TF-IDF Vectorization + Linguistic Features
   ↓
3 ML Models (Logistic Regression + Naive Bayes + Random Forest)
   ↓
Ensemble Voting Classifier
   ↓
REAL or FAKE verdict with confidence score
```

---

## 🔬 NLP Pipeline

- **Text Cleaning** — removes URLs, emails, numbers, punctuation
- **Stopword Removal** — filters out common words like "the", "and"
- **Stemming** — reduces words to root form (running → run)
- **TF-IDF Vectorizer** — built from scratch with bigram support
- **Linguistic Features** — 11 features including:
  - CAPS word ratio
  - Exclamation mark count
  - Unique word ratio
  - Average sentence length
  - Quote count

---

## 💡 Example Predictions
```
Article: "Scientists at Oxford published findings in Nature Medicine..."
→ REAL  ✓  Confidence: 97.3%

Article: "BOMBSHELL!!! SECRET government LEAK proves vaccines contain MICROCHIPS!!!"
→ FAKE  ✓  Confidence: 97.8%
```

---

## ⚙️ Setup & Installation
```bash
# 1. Clone the repository
git clone https://github.com/rutvi2305/fake-news-detector
cd fake-news-detector

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add dataset
# Download from Kaggle and place Fake.csv and True.csv in data/ folder
# https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

# 5. Run
python fake_news_detection.py
```

---

## 📁 Project Structure
```
fake-news-detector/
├── data/
│   ├── Fake.csv          # Kaggle dataset (not included)
│   └── True.csv          # Kaggle dataset (not included)
├── fake_news_detection.py # Main ML pipeline
├── requirements.txt       # Dependencies
└── README.md             # This file
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10+ | Core language |
| NumPy | Numerical computations |
| Pandas | Data loading and processing |
| Custom TF-IDF | Feature extraction (from scratch) |
| Custom ML Models | Classification (from scratch) |

---

## 📚 Models Built From Scratch

- **Logistic Regression** — Mini-batch SGD with L2 regularization
- **Naive Bayes** — Multinomial with Laplace smoothing  
- **Random Forest** — Bootstrap sampling with Gini impurity splits
- **Voting Ensemble** — Soft voting (LR 45% + RF 30% + NB 25%)

---

## 👩‍💻 Author

**Rutvi** — [github.com/rutvi2305](https://github.com/rutvi2305)