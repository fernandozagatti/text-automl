import sys
sys.path.append('../preprocessing/')

import pandas as pd
import numpy as np
from preprocess import *
from bertopic import BERTopic
import os
import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class TopicModelingWithBERTopic(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.preprocessed_data = None
        self.preprocessed_data_list = None
        self.topic_model = None
    
    def preprocess_data(self, data, target_column='review_text', stopwords=None, language='english', 
                        verbose=0, **kwargs):
        preprocessing_functions = ['transform_to_lowercase', 'remove_special_characters']
        
        if stopwords:
            preprocessing_functions.append('remove_stopwords')

        self.preprocessed_data = preprocess_data(data=data, 
                                                 preprocessing_funcs=preprocessing_functions, 
                                                 language=language, 
                                                 column=target_column, 
                                                 verbose=verbose, **kwargs)
        
        self.preprocessed_data_list = self.preprocessed_data['prep'].tolist()
        
        return self.preprocessed_data_list
    
    def fit(self, data, target_column='review_text', num_topics=None, verbose=0, **kwargs):
            
        self.kwargs = kwargs
        if self.preprocessed_data == None:
            self.preprocess_data(data=data, target_column=target_column, verbose=verbose, **kwargs)
            
        if verbose > 0:
            print('Fitting the model...')
        
        self.topic_model = BERTopic(language="multilingual").fit(self.preprocessed_data_list)
        if type(num_topics) is int:
            self.topic_model.reduce_topics(self.preprocessed_data_list, nr_topics=num_topics)
            
        if verbose > 0:
            print('Done!\n')
        return self.topic_model
    
    def transform(self, data=None, verbose=0, **kwargs):
        if self.topic_model is None:
            raise ValueError("Model has not been fitted yet. Call the 'fit' method first.")
        
        if data is not None:
            preprocessed_data_list = self.preprocess_data(data, target_column='review_text', verbose=verbose, **kwargs)
            if verbose > 0:
                print('Transforming the data...')
            topics, probs = self.topic_model.transform(preprocessed_data_list)
        else:
            if verbose > 0:
                print('Transforming the data...')
            topics, probs = self.topic_model.transform(self.preprocessed_data_list)
            
        if verbose > 0:
            print('Done!\n')
        return topics, probs
    
    def fit_transform(self, data, target_column='review_text', num_topics=None, verbose=0, **kwargs):
        self.fit(data, target_column, num_topics, verbose=verbose, **kwargs)
        self.transform(verbose=verbose, **kwargs)
        return self.topic_model
    
    def get_topic_info(self, num_words=10):
        if self.topic_model is None:
            raise ValueError("Model has not been fitted yet. Call the 'fit' method first.")
        
        return self.topic_model.get_topic_info()
    
    def get_document_info(self):
        if self.topic_model is None:
            raise ValueError("Model has not been fitted yet. Call the 'fit' method first.")
            
        return self.topic_model.get_document_info(self.preprocessed_data_list)
    
    def visualize_topics(self):
        if self.topic_model is None:
            raise ValueError("Model has not been fitted yet. Call the 'fit' method first.")
        
        return self.topic_model.visualize_topics()