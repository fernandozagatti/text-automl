import pandas as pd
import numpy as np
from lda_class import TopicModelingWithLDA
from bertopic_class import TopicModelingWithBERTopic

class AutoTM(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.topic_model = None
        
    def TopicModeling(self, data, topic_model, target_column, stopwords=None, language='english', 
                      num_topics=None, limit=30, start=2, step=2, verbose=0, **kwargs):
        
        if verbose > 0:
            print('================================= RUNNING TOPIC MODELING ================================\n')
        
        if topic_model == 'lda':
            self.topic_model = TopicModelingWithLDA()
            self.topic_model.fit(data=data, target_column=target_column, stopwords=stopwords, 
                                 language=language, verbose=verbose, num_topics=num_topics, 
                                 limit=limit, start=start, step=step, **kwargs)
                
        elif topic_model == 'bertopic':
            self.topic_model = TopicModelingWithBERTopic()
            self.topic_model.fit_transform(data=data, target_column=target_column, stopwords=stopwords, 
                                           language=language, verbose=verbose, num_topics=num_topics, **kwargs)
        else:
            raise ValueError(f"The model '{topic_model}' is not valid, use: 'lda' or 'bertopic'.\n")
            
        if verbose > 0:
            print('=================================== END TOPIC MODELING ==================================\n')
        return self.topic_model
    
    def get_topic_info(self, num_words=10):
        if self.topic_model is None:
            raise ValueError("Model has not been fitted yet. Call the 'TopicModeling' method first.")
        
        return self.topic_model.get_topic_info(num_words=num_words)
    
    def get_document_info(self):
        if self.topic_model is None:
            raise ValueError("Model has not been fitted yet. Call the 'TopicModeling' method first.")
        
        return self.topic_model.get_document_info()
    
    def visualize_topics(self):
        if self.topic_model is None:
            raise ValueError("Model has not been fitted yet. Call the 'TopicModeling' method first.")
        
        return self.topic_model.visualize_topics()