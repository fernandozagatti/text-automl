import sys
sys.path.append('../preprocessing/')

import pandas as pd
import numpy as np
from preprocess import *
import gensim.corpora as corpora
from gensim.models.ldamodel import LdaModel
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models
import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 

class TopicModelingWithLDA(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.target_column = None
        self.preprocessed_data = None
        self.topic_model = None
        self.text_list = None
        self.corpora_dict = None
        self.corpus = None
    
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
        
        self.text_list = self.preprocessed_data['prep'].str.split()
        self.corpora_dict = corpora.Dictionary(self.text_list)
        self.corpus = [self.corpora_dict.doc2bow(text) for text in self.text_list]

        
    def fit(self, data, target_column='review_text', stopwords=None, language='english', verbose=0, 
            num_topics=None, limit=30, start=2, step=2, **kwargs):
        
        self.target_column = target_column

        if self.preprocessed_data == None:
            self.preprocess_data(data=data, target_column=target_column, stopwords=stopwords, language=language, 
                                 verbose=verbose, **kwargs)
            self.preprocessed_data = self.preprocessed_data.reset_index()
        
        if num_topics is None:
            coherence_value = 0
            model = None
            topics = 0

            if verbose > 0:
                print('Searching for the best number of topics...\n')

            for num_topics in range(start, limit, step):
                lda_model = LdaModel(corpus=self.corpus,
                                     id2word=self.corpora_dict,
                                     num_topics=num_topics,
                                     random_state=42)

                coherencemodel = CoherenceModel(model=lda_model, texts=self.text_list, 
                                                dictionary=self.corpora_dict, coherence='c_v')
                coherencemodel = coherencemodel.get_coherence()

                if verbose > 0:
                    print(f'Test with {num_topics} topics...')

                if coherencemodel > coherence_value:
                    coherence_value = coherencemodel
                    best_num_topics = num_topics

            if verbose > 0:
                print(f'\nBest test is with {best_num_topics} topics!\n')
            num_topics = best_num_topics
            
        if verbose > 0:
            print('Fitting the model...')
        
        self.topic_model = LdaModel(corpus=self.corpus,
                                    id2word=self.corpora_dict,
                                    num_topics=num_topics, 
                                    random_state=42,
                                    chunksize=100,
                                    passes=10,
                                    per_word_topics=True,
                                    alpha = 0.9,
                                    eta = 0.3)

        if verbose > 0:
            print('Done!\n')
        return self.topic_model
    
    
    def get_topic_info(self, num_words=10):
        if self.topic_model is None:
            raise ValueError("Model has not been fitted yet. Call the 'fit' method first.")
        
        x=self.topic_model.show_topics(num_topics=self.topic_model.num_topics, num_words=num_words,formatted=False)
        topics_words = [(tp[0], [wd[0] for wd in tp[1]]) for tp in x]

        for topic,words in topics_words:
            print(str(topic)+ ":"+ str(words))
        
        print('\nLog perplexity: ', self.topic_model.log_perplexity(self.corpus))

        coherence_model_lda = CoherenceModel(model=self.topic_model, texts=self.text_list, 
                                             dictionary=self.corpora_dict, coherence='c_v')
        
        coherence_lda = coherence_model_lda.get_coherence()
        print(f'Coherence score: {coherence_lda}\n')
        
        return topics_words
    
    
    def get_document_info(self):
        if self.topic_model is None:
            raise ValueError("Model has not been fitted yet. Call the 'fit' method first.")
            
        probability = self.topic_model[self.corpus[:]]

        data = []
        for sublist in probability:
            row_dict = {col: val for col, val in sublist[0]}
            data.append(row_dict)

        data = pd.DataFrame(data)
        data = pd.concat([self.preprocessed_data[self.target_column], data], axis=1)
        
        return data
        
    
    def visualize_topics(self):
        if self.topic_model is None:
            raise ValueError("Model has not been fitted yet. Call the 'fit' method first.")
        
        vis = pyLDAvis.gensim_models.prepare(self.topic_model, self.corpus, self.corpora_dict)
        pyLDAvis.enable_notebook()
        return pyLDAvis.display(vis)