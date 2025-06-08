import math
from collections import Counter


class Matrix(list):
    @property
    def shape(self):
        if not self:
            return (0, 0)
        return (len(self), len(self[0]))

class TfidfVectorizer:
    """Minimal TF-IDF vectorizer implementation."""

    def __init__(self):
        self.vocabulary_ = {}
        self.idf_ = {}
        self._fitted = False

    def fit_transform(self, data):
        tokens_list = [self._tokenize(d) for d in data]
        all_tokens = set(t for tokens in tokens_list for t in tokens)
        self.vocabulary_ = {t: i for i, t in enumerate(sorted(all_tokens))}
        doc_count = {t: 0 for t in self.vocabulary_}
        for tokens in tokens_list:
            for t in set(tokens):
                doc_count[t] += 1
        n_docs = len(data)
        self.idf_ = {t: math.log(n_docs / (1 + c)) + 1 for t, c in doc_count.items()}
        self._fitted = True
        matrix = Matrix([self._vectorize(tokens) for tokens in tokens_list])
        return matrix

    def transform(self, data):
        if not self._fitted:
            raise ValueError("Vectorizer not fitted")
        tokens_list = [self._tokenize(d) for d in data]
        matrix = Matrix([self._vectorize(tokens) for tokens in tokens_list])
        return matrix

    def _tokenize(self, text):
        return text.lower().split()

    def _vectorize(self, tokens):
        vec = [0.0] * len(self.vocabulary_)
        counts = Counter(tokens)
        total = len(tokens) or 1
        for t, c in counts.items():
            if t in self.vocabulary_:
                idx = self.vocabulary_[t]
                tf = c / total
                vec[idx] = tf * self.idf_.get(t, 0.0)
        return vec
