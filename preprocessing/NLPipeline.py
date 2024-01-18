import pandas as pd
import numpy as np
from preprocess import *
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

class PreprocessDataTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, preprocessing_funcs, column='text', **kwargs):
        self.preprocessing_funcs = preprocessing_funcs
        self.column = column
        self.kwargs = kwargs

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        processed_data = preprocess_data(X, self.preprocessing_funcs, self.column, **self.kwargs)
        return processed_data

# Create a custom transformer class for vectorization
class VectorizationTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, vectorizer='tf-idf', vector=None, **kwargs):
        self.vectorizer = vectorizer
        self.kwargs = kwargs
        self.vector = vector

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            data = X.copy()
        elif isinstance(X, pd.Series):
            data = pd.DataFrame({'prep': X})
        else:
            data = pd.DataFrame({'prep': [X]})
        self.vector, _ = vectorization(data, vectorizer=self.vectorizer, **self.kwargs)
        return self

    def transform(self, X):
        vectorized_data = vectorization_transform(X, self.vectorizer, self.vector, **self.kwargs)
        return vectorized_data
    
def NLP_helper(preprocessing_funcs, vectorizer='tf-idf', **kwargs):
    pipeline = Pipeline([
        ('preprocess', PreprocessDataTransformer(preprocessing_funcs=preprocessing_funcs, **kwargs)),
        ('vectorize', VectorizationTransformer(vectorizer, **kwargs)),
    ])
    return pipeline