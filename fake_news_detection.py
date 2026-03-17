# -*- coding: utf-8 -*-
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
"""
AI-Powered Fake News Detection System
======================================
A complete NLP pipeline for detecting and classifying news articles as real or fake.
Includes: preprocessing, feature extraction, model training, evaluation, and prediction.
"""

import numpy as np
import pandas as pd
import re
import string
import warnings
import os
import pickle
from collections import Counter

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. TEXT PREPROCESSING
# ─────────────────────────────────────────────

class TextPreprocessor:
    """Handles all NLP preprocessing steps."""

    STOPWORDS = {
        'a','an','the','and','or','but','in','on','at','to','for','of','with',
        'by','from','up','about','into','through','during','is','are','was',
        'were','be','been','being','have','has','had','do','does','did','will',
        'would','could','should','may','might','shall','can','need','dare',
        'ought','used','it','its','this','that','these','those','i','me','my',
        'we','our','you','your','he','him','his','she','her','they','them',
        'their','what','which','who','whom','not','no','nor','so','yet','both',
        'either','neither','each','few','more','most','other','some','such',
        'than','too','very','just','also','there','then','now','only','s','t'
    }

    def __init__(self, remove_stopwords=True, stemming=True):
        self.remove_stopwords = remove_stopwords
        self.stemming = stemming

    def _simple_stem(self, word):
        """Rule-based stemmer (no external dependency)."""
        suffixes = ['ing','tion','ness','ment','able','ible','ous','ive','ful','less','ly','ed','er','est']
        for suf in suffixes:
            if len(word) > len(suf) + 3 and word.endswith(suf):
                return word[:-len(suf)]
        return word

    def clean(self, text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', ' URL ', text)
        text = re.sub(r'\S+@\S+', ' EMAIL ', text)
        text = re.sub(r'\d+', ' NUM ', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.STOPWORDS and len(t) > 2]
        if self.stemming:
            tokens = [self._simple_stem(t) for t in tokens]
        return ' '.join(tokens)

    def extract_features_text(self, text):
        """Extract linguistic features from raw text."""
        if not isinstance(text, str) or len(text) == 0:
            return {}
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        caps_words = len(re.findall(r'\b[A-Z]{2,}\b', text))
        exclamations = text.count('!')
        questions = text.count('?')
        quotes = len(re.findall(r'"[^"]*"', text))
        return {
            'char_count': len(text),
            'word_count': len(words),
            'sentence_count': max(len(sentences), 1),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'avg_sentence_length': len(words) / max(len(sentences), 1),
            'caps_ratio': caps_words / max(len(words), 1),
            'exclamation_count': exclamations,
            'question_count': questions,
            'quote_count': quotes,
            'unique_word_ratio': len(set(words)) / max(len(words), 1),
            'punctuation_ratio': sum(1 for c in text if c in string.punctuation) / max(len(text), 1),
        }


# ─────────────────────────────────────────────
# 2. FEATURE EXTRACTION (TF-IDF from scratch)
# ─────────────────────────────────────────────

class TFIDFVectorizer:
    """TF-IDF vectorizer built from scratch."""

    def __init__(self, max_features=5000, ngram_range=(1, 2), min_df=2):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.vocabulary_ = {}
        self.idf_ = {}

    def _get_ngrams(self, tokens):
        ngrams = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            for i in range(len(tokens) - n + 1):
                ngrams.append(' '.join(tokens[i:i+n]))
        return ngrams

    def fit(self, corpus):
        doc_freq = Counter()
        n_docs = len(corpus)
        all_ngrams_per_doc = []
        for doc in corpus:
            tokens = doc.split()
            ngrams = self._get_ngrams(tokens)
            all_ngrams_per_doc.append(ngrams)
            for ng in set(ngrams):
                doc_freq[ng] += 1

        # Filter by min_df and select top features by doc freq
        filtered = {t: f for t, f in doc_freq.items() if f >= self.min_df}
        top_terms = sorted(filtered.items(), key=lambda x: -x[1])[:self.max_features]
        self.vocabulary_ = {term: idx for idx, (term, _) in enumerate(top_terms)}

        # Compute IDF
        for term, _ in top_terms:
            self.idf_[term] = np.log((1 + n_docs) / (1 + doc_freq[term])) + 1

        return self

    def transform(self, corpus):
        rows = []
        for doc in corpus:
            tokens = doc.split()
            ngrams = self._get_ngrams(tokens)
            tf_counts = Counter(ngrams)
            total = max(len(ngrams), 1)
            row = np.zeros(len(self.vocabulary_))
            for term, idx in self.vocabulary_.items():
                if term in tf_counts:
                    tf = tf_counts[term] / total
                    row[idx] = tf * self.idf_.get(term, 1.0)
            # L2 normalize
            norm = np.linalg.norm(row)
            if norm > 0:
                row /= norm
            rows.append(row)
        return np.array(rows)

    def fit_transform(self, corpus):
        return self.fit(corpus).transform(corpus)


# ─────────────────────────────────────────────
# 3. MODELS
# ─────────────────────────────────────────────

class LogisticRegressionScratch:
    """Logistic Regression with SGD, L2 regularization."""

    def __init__(self, lr=0.1, epochs=100, C=1.0, batch_size=64):
        self.lr = lr
        self.epochs = epochs
        self.C = C  # inverse regularization strength
        self.batch_size = batch_size
        self.weights = None
        self.bias = 0.0
        self.loss_history = []

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for epoch in range(self.epochs):
            indices = np.random.permutation(n_samples)
            epoch_loss = 0
            for start in range(0, n_samples, self.batch_size):
                batch_idx = indices[start:start + self.batch_size]
                Xb, yb = X[batch_idx], y[batch_idx]
                z = Xb @ self.weights + self.bias
                pred = self._sigmoid(z)
                error = pred - yb
                grad_w = (Xb.T @ error) / len(yb) + self.weights / (self.C * n_samples)
                grad_b = np.mean(error)
                self.weights -= self.lr * grad_w
                self.bias -= self.lr * grad_b
                eps = 1e-9
                epoch_loss += -np.mean(yb * np.log(pred + eps) + (1 - yb) * np.log(1 - pred + eps))
            self.loss_history.append(epoch_loss)

            if epoch % 20 == 0:
                acc = np.mean(self.predict(X) == y)
                print(f"  Epoch {epoch:3d}/{self.epochs} | Loss: {epoch_loss:.4f} | Train Acc: {acc:.4f}")
        return self

    def predict_proba(self, X):
        z = X @ self.weights + self.bias
        prob = self._sigmoid(z)
        return np.column_stack([1 - prob, prob])

    def predict(self, X):
        return (self._sigmoid(X @ self.weights + self.bias) >= 0.5).astype(int)


class NaiveBayesScratch:
    """Multinomial-style Naive Bayes for text classification."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Laplace smoothing
        self.class_log_prior_ = {}
        self.feature_log_prob_ = {}
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        for c in self.classes_:
            mask = y == c
            self.class_log_prior_[c] = np.log(mask.sum() / n_samples)
            feature_counts = X[mask].sum(axis=0) + self.alpha
            self.feature_log_prob_[c] = np.log(feature_counts / feature_counts.sum())
        return self

    def predict_proba(self, X):
        log_probs = []
        for c in self.classes_:
            lp = self.class_log_prior_[c] + X @ self.feature_log_prob_[c]
            log_probs.append(lp)
        log_probs = np.column_stack(log_probs)
        # Softmax
        log_probs -= log_probs.max(axis=1, keepdims=True)
        probs = np.exp(log_probs)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class RandomForestScratch:
    """Simplified Random Forest using decision stumps."""

    def __init__(self, n_estimators=50, max_features=0.3, max_depth=3):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.trees = []

    class DecisionTree:
        def __init__(self, max_depth=3):
            self.max_depth = max_depth
            self.tree = None

        def _gini(self, y):
            if len(y) == 0: return 0
            p = np.bincount(y, minlength=2) / len(y)
            return 1 - np.sum(p ** 2)

        def _best_split(self, X, y, features):
            best = {'gain': -1, 'feat': None, 'thresh': None}
            parent_gini = self._gini(y)
            for feat in features:
                thresholds = np.percentile(X[:, feat], [25, 50, 75])
                for thresh in thresholds:
                    left = y[X[:, feat] <= thresh]
                    right = y[X[:, feat] > thresh]
                    if len(left) == 0 or len(right) == 0:
                        continue
                    gain = parent_gini - (len(left)*self._gini(left) + len(right)*self._gini(right)) / len(y)
                    if gain > best['gain']:
                        best = {'gain': gain, 'feat': feat, 'thresh': thresh}
            return best

        def _build(self, X, y, depth):
            if depth == 0 or len(set(y)) == 1 or len(y) < 5:
                return {'leaf': True, 'pred': np.bincount(y, minlength=2) / len(y)}
            n_feats = max(1, int(np.sqrt(X.shape[1])))
            features = np.random.choice(X.shape[1], n_feats, replace=False)
            split = self._best_split(X, y, features)
            if split['feat'] is None:
                return {'leaf': True, 'pred': np.bincount(y, minlength=2) / len(y)}
            mask = X[:, split['feat']] <= split['thresh']
            return {
                'leaf': False, 'feat': split['feat'], 'thresh': split['thresh'],
                'left': self._build(X[mask], y[mask], depth-1),
                'right': self._build(X[~mask], y[~mask], depth-1)
            }

        def fit(self, X, y):
            self.tree = self._build(X, y, self.max_depth)
            return self

        def _pred_one(self, node, x):
            if node['leaf']:
                return node['pred']
            if x[node['feat']] <= node['thresh']:
                return self._pred_one(node['left'], x)
            return self._pred_one(node['right'], x)

        def predict_proba(self, X):
            return np.array([self._pred_one(self.tree, x) for x in X])

    def fit(self, X, y):
        n = X.shape[0]
        for i in range(self.n_estimators):
            idx = np.random.choice(n, n, replace=True)
            tree = self.DecisionTree(self.max_depth)
            tree.fit(X[idx], y[idx])
            self.trees.append(tree)
            if (i+1) % 10 == 0:
                print(f"  Tree {i+1}/{self.n_estimators} built")
        return self

    def predict_proba(self, X):
        probs = np.mean([t.predict_proba(X) for t in self.trees], axis=0)
        return probs

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


# ─────────────────────────────────────────────
# 4. ENSEMBLE VOTING CLASSIFIER
# ─────────────────────────────────────────────

class VotingEnsemble:
    """Soft-voting ensemble of multiple classifiers."""

    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights or [1.0] * len(models)

    def predict_proba(self, X):
        total = np.zeros((X.shape[0], 2))
        for model, w in zip(self.models, self.weights):
            total += w * model.predict_proba(X)
        total /= sum(self.weights)
        return total

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


# ─────────────────────────────────────────────
# 5. EVALUATION
# ─────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Compute classification metrics."""
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    tp = np.sum((preds == 1) & (y_test == 1))
    tn = np.sum((preds == 0) & (y_test == 0))
    fp = np.sum((preds == 1) & (y_test == 0))
    fn = np.sum((preds == 0) & (y_test == 1))

    accuracy  = (tp + tn) / len(y_test)
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)

    # AUC (trapezoidal approximation)
    thresholds = np.linspace(0, 1, 100)
    tprs, fprs = [], []
    for t in thresholds:
        p = (proba >= t).astype(int)
        tprs.append(np.sum((p == 1) & (y_test == 1)) / max(np.sum(y_test == 1), 1))
        fprs.append(np.sum((p == 1) & (y_test == 0)) / max(np.sum(y_test == 0), 1))
    auc = np.trapezoid(tprs[::-1], fprs[::-1])

    print(f"\n{'='*50}")
    print(f"  {model_name} Results")
    print(f"{'='*50}")
    print(f"  Accuracy  : {accuracy:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"              Predicted")
    print(f"              Real  Fake")
    print(f"  Actual Real  {tn:4d}  {fp:4d}")
    print(f"  Actual Fake  {fn:4d}  {tp:4d}")
    print(f"{'='*50}")

    return {'accuracy': accuracy, 'precision': precision,
            'recall': recall, 'f1': f1, 'auc': auc}


# ─────────────────────────────────────────────
# 6. PIPELINE
# ─────────────────────────────────────────────

class FakeNewsDetector:
    """End-to-end fake news detection pipeline."""

    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.tfidf = TFIDFVectorizer(max_features=3000, ngram_range=(1, 2))
        self.lr_model = LogisticRegressionScratch(lr=0.05, epochs=80, C=1.0)
        self.nb_model = NaiveBayesScratch(alpha=0.5)
        self.rf_model = RandomForestScratch(n_estimators=30, max_depth=4)
        self.ensemble = None
        self.feature_names = None

    def _make_features(self, texts, fit=False):
        """Combine TF-IDF + linguistic features."""
        cleaned = [self.preprocessor.clean(t) for t in texts]
        if fit:
            tfidf_feats = self.tfidf.fit_transform(cleaned)
        else:
            tfidf_feats = self.tfidf.transform(cleaned)
        ling_feats = np.array([
            list(self.preprocessor.extract_features_text(t).values())
            for t in texts
        ])
        # Normalize linguistic features
        if fit:
            self._ling_mean = ling_feats.mean(axis=0)
            self._ling_std = ling_feats.std(axis=0) + 1e-8
        ling_feats = (ling_feats - self._ling_mean) / self._ling_std
        return np.hstack([tfidf_feats, ling_feats])

    def fit(self, texts, labels):
        """Train all models."""
        print("\n[1/4] Preprocessing & extracting features...")
        X = self._make_features(texts, fit=True)
        y = np.array(labels)
        print(f"      Feature matrix: {X.shape}")

        print("\n[2/4] Training Logistic Regression...")
        self.lr_model.fit(X, y)

        print("\n[3/4] Training Naive Bayes...")
        # NB needs non-negative features — shift TF-IDF (already ≥0) + ling
        X_nb = np.clip(X, 0, None)
        self.nb_model.fit(X_nb, y)

        print("\n[4/4] Training Random Forest...")
        self.rf_model.fit(X, y)

        self.ensemble = VotingEnsemble(
            [self.lr_model, self.nb_model, self.rf_model],
            weights=[0.45, 0.25, 0.30]
        )
        print("\n✓ Training complete!")
        return self

    def predict(self, texts):
        X = self._make_features(texts)
        X_nb = np.clip(X, 0, None)
        # Use ensemble but NB needs clipped X
        lr_prob = self.lr_model.predict_proba(X)
        nb_prob = self.nb_model.predict_proba(X_nb)
        rf_prob = self.rf_model.predict_proba(X)
        proba = (0.45 * lr_prob + 0.25 * nb_prob + 0.30 * rf_prob)
        preds = proba.argmax(axis=1)
        return preds, proba

    def predict_single(self, text):
        preds, proba = self.predict([text])
        label = "FAKE" if preds[0] == 1 else "REAL"
        confidence = proba[0][preds[0]]
        return {
            'label': label,
            'confidence': float(confidence),
            'real_prob': float(proba[0][0]),
            'fake_prob': float(proba[0][1]),
        }

    def evaluate(self, texts, labels):
        X = self._make_features(texts)
        X_nb = np.clip(X, 0, None)
        y = np.array(labels)
        results = {}
        results['Logistic Regression'] = evaluate_model(self.lr_model, X, y, "Logistic Regression")
        results['Naive Bayes'] = evaluate_model(self.nb_model, X_nb, y, "Naive Bayes")
        results['Random Forest'] = evaluate_model(self.rf_model, X, y, "Random Forest")

        # Ensemble
        class _Ens:
            def __init__(self, lr, nb, rf):
                self.lr, self.nb, self.rf = lr, nb, rf
            def predict_proba(self, X):
                return (0.45*self.lr.predict_proba(X)
                       + 0.25*self.nb.predict_proba(X)
                       + 0.30*self.rf.predict_proba(X))
            def predict(self, X):
                return self.predict_proba(X).argmax(axis=1)

        ens = _Ens(self.lr_model, self.nb_model, self.rf_model)
        results['Ensemble'] = evaluate_model(ens, X, y, "Ensemble (Voting)")
        return results

    def save(self, path='fake_news_model.pkl'):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"✓ Model saved to {path}")

    @staticmethod
    def load(path='fake_news_model.pkl'):
        with open(path, 'rb') as f:
            return pickle.load(f)


# ─────────────────────────────────────────────
# 7. DATASET GENERATION + DEMO
# ─────────────────────────────────────────────

def generate_demo_dataset(n_real=400, n_fake=400):
    """Generate a realistic synthetic dataset for demonstration."""
    np.random.seed(42)

    real_templates = [
        "Scientists at {uni} have published research in {journal} showing that {finding}. The study, which involved {n} participants, found statistically significant results. Lead researcher Dr. {name} stated the findings need further peer review.",
        "The {gov} announced new policy measures on {topic} following months of consultation with experts. The legislation, passed with {votes} votes, aims to address {issue} by {year}. Critics and supporters offered mixed reactions.",
        "{company} reported quarterly earnings of ${amount} billion, {dir} analyst expectations. CFO {name} cited {reason} as contributing factors. The company's stock {moved} following the announcement.",
        "Health officials confirmed {n} new cases of {disease} in {region}, bringing the total to {total}. The {org} has deployed response teams and is working with local authorities to contain the outbreak.",
        "A new study published in {journal} suggests that {habit} may {effect} the risk of {condition}. Researchers analyzed data from {n} people over {years} years, controlling for age, diet, and other factors.",
    ]

    fake_templates = [
        "BREAKING: SECRET documents LEAKED proving {conspiracy}!!! The mainstream media won't tell you THIS. Sources inside the {fake_org} confirm what we've known all along. SHARE before they DELETE this!!!",
        "SHOCKING: {celebrity} caught in massive {scandal} cover-up! An anonymous insider reveals the TRUTH they've been hiding from you. This will BLOW YOUR MIND! The {institution} doesn't want you to see this.",
        "CURE for {disease} DISCOVERED but BIG {industry} is SUPPRESSING it!! Doctors HATE this one trick that {effect}. Natural healers have known for centuries what the establishment refuses to admit.",
        "{politician} CAUGHT on tape plotting to {action}!!! Anonymous whistleblowers risk their lives to expose this BOMBSHELL. The deep state is {conspiracy2}. Wake up sheeple before it's too late!!!",
        "PROOF that {event} was STAGED by {group}!!! Crisis actors and paid operatives orchestrated the whole thing. The photos don't lie — see for yourself what the controlled media is hiding from you!",
    ]

    fill = {
        'uni': ['Harvard', 'MIT', 'Stanford', 'Oxford', 'Johns Hopkins'],
        'journal': ['Nature', 'Science', 'NEJM', 'The Lancet', 'JAMA'],
        'finding': ['early intervention improves outcomes', 'the treatment showed efficacy', 'dietary patterns affect health'],
        'n': ['1,200', '5,000', '800', '2,400'],
        'name': ['Smith', 'Johnson', 'Patel', 'Chen', 'Williams'],
        'gov': ['Federal Government', 'Parliament', 'Senate', 'Ministry'],
        'topic': ['healthcare', 'climate policy', 'taxation', 'infrastructure'],
        'votes': ['234', '178', '301'],
        'issue': ['rising costs', 'inequality', 'emissions'],
        'year': ['2025', '2026', '2027'],
        'company': ['Apple', 'Microsoft', 'Google', 'Amazon'],
        'amount': ['12.4', '8.7', '23.1', '5.9'],
        'dir': ['beating', 'missing', 'meeting'],
        'reason': ['strong demand', 'cost efficiencies', 'market expansion'],
        'moved': ['rose 3%', 'fell 2%', 'remained flat'],
        'disease': ['influenza', 'COVID variant', 'RSV', 'norovirus'],
        'region': ['the Northeast', 'the Pacific Coast', 'Europe', 'Southeast Asia'],
        'total': ['1,200', '450', '3,400'],
        'org': ['WHO', 'CDC', 'NHS', 'ECDC'],
        'habit': ['regular exercise', 'Mediterranean diet', 'adequate sleep', 'stress reduction'],
        'effect': ['reduce', 'increase', 'have no effect on'],
        'condition': ['heart disease', 'diabetes', 'cognitive decline', 'hypertension'],
        'years': ['10', '15', '20'],
        'conspiracy': ['global currency reset is imminent', 'vaccines contain tracking chips', '5G is mind control'],
        'fake_org': ['WHO', 'Deep State', 'Bilderberg Group', 'Shadow Government'],
        'celebrity': ['A-list actor', 'pop star', 'tech billionaire'],
        'scandal': ['financial fraud', 'secret society', 'mind control program'],
        'institution': ['government', 'media', 'elite'],
        'industry': ['PHARMA', 'FOOD', 'TECH', 'MEDIA'],
        'politician': ['High-ranking official', 'World leader', 'Globalist puppet'],
        'action': ['destroy the economy', 'enslave the population', 'rig elections'],
        'conspiracy2': ['watching your every move', 'planning the great reset'],
        'event': ['recent disaster', 'mass shooting', 'economic crash'],
        'group': ['government agents', 'globalist elites', 'shadow operatives'],
    }

    def fill_template(template):
        for key, options in fill.items():
            placeholder = '{' + key + '}'
            if placeholder in template:
                template = template.replace(placeholder, np.random.choice(options))
        return template

    real_articles = [fill_template(np.random.choice(real_templates)) for _ in range(n_real)]
    fake_articles = [fill_template(np.random.choice(fake_templates)) for _ in range(n_fake)]

    texts = real_articles + fake_articles
    labels = [0] * n_real + [1] * n_fake

    idx = np.random.permutation(len(texts))
    return [texts[i] for i in idx], [labels[i] for i in idx]


def train_test_split_manual(texts, labels, test_size=0.2, seed=42):
    np.random.seed(seed)
    n = len(texts)
    idx = np.random.permutation(n)
    split = int(n * (1 - test_size))
    train_idx, test_idx = idx[:split], idx[split:]
    return ([texts[i] for i in train_idx], [labels[i] for i in train_idx],
            [texts[i] for i in test_idx],  [labels[i] for i in test_idx])


# ─────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("   AI-Powered Fake News Detection System")
    print("   Training on REAL Kaggle Dataset")
    print("=" * 60)

    # ── Load real dataset ──────────────────────────────────────
    print("\n[Dataset] Loading Fake and Real news CSV files...")
    import pandas as pd

    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    fake_df = pd.read_csv(os.path.join(base_dir, 'data', 'Fake.csv'))
    real_df = pd.read_csv(os.path.join(base_dir, 'data', 'True.csv'))

    fake_df['label'] = 1
    real_df['label'] = 0

    df = pd.concat([fake_df, real_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    df['content'] = df['title'].fillna('') + ' ' + df['text'].fillna('')

    texts  = df['content'].tolist()
    labels = df['label'].tolist()

    print(f"  Total articles : {len(texts)}")
    print(f"  Real articles  : {labels.count(0)}")
    print(f"  Fake articles  : {labels.count(1)}")

    # ── Split dataset ──────────────────────────────────────────
    X_train, y_train, X_test, y_test = train_test_split_manual(
        texts, labels, test_size=0.2
    )
    print(f"  Train size : {len(X_train)}")
    print(f"  Test size  : {len(X_test)}")

    # ── Train ──────────────────────────────────────────────────
    detector = FakeNewsDetector()
    detector.fit(X_train, y_train)

    # ── Evaluate ───────────────────────────────────────────────
    print("\n\n--- EVALUATION ON REAL TEST DATA ---")
    results = detector.evaluate(X_test, y_test)

    print("\n\n--- COMPARISON TABLE ---")
    print(f"{'Model':<25} {'Accuracy':>10} {'F1-Score':>10} {'AUC-ROC':>10}")
    print("-" * 57)
    for name, r in results.items():
        print(f"{name:<25} {r['accuracy']:>10.4f} {r['f1']:>10.4f} {r['auc']:>10.4f}")

    # ── Live predictions ───────────────────────────────────────
    print("\n\n--- LIVE PREDICTIONS ---")
    test_articles = [
        {
            "text": "Scientists at Oxford University confirmed a breakthrough "
                    "cancer treatment showing 80% success rate in clinical trials. "
                    "The peer-reviewed study was published in Nature Medicine.",
            "expected": "REAL"
        },
        {
            "text": "BREAKING!!! Government secretly putting mind control chemicals "
                    "in tap water!!! SHARE before they DELETE this!!! "
                    "Anonymous insider EXPOSES the TRUTH!!!",
            "expected": "FAKE"
        },
        {
            "text": "The Federal Reserve held interest rates steady on Wednesday, "
                    "citing mixed economic signals. Fed Chair noted inflation "
                    "has eased but remains above the 2% target.",
            "expected": "REAL"
        },
        {
            "text": "SHOCKING: BIG PHARMA suppressing natural cure for diabetes! "
                    "Doctors HATE this one trick! The deep state doesn't want "
                    "you to know. WAKE UP SHEEPLE!!!",
            "expected": "FAKE"
        },
    ]

    print(f"\n{'─'*65}")
    for i, article in enumerate(test_articles, 1):
        result = detector.predict_single(article['text'])
        match  = "✓" if result['label'] == article['expected'] else "✗"
        print(f"\n[Article {i}] {article['text'][:80]}...")
        print(f"  Prediction : {result['label']} {match}  (Expected: {article['expected']})")
        print(f"  Confidence : {result['confidence']*100:.1f}%")
        print(f"  Real Prob  : {result['real_prob']*100:.1f}%  "
              f"|  Fake Prob: {result['fake_prob']*100:.1f}%")

    # ── Save model ─────────────────────────────────────────────
    detector.save('fake_news_model.pkl')
    print("\n✓ Done! Model saved.")


if __name__ == '__main__':
    main()