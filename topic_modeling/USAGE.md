# Meta-AutoNLP Prototype

**Using the _Topic modeling_**

---

## How use _Topic modeling_ :

[Go to Example Notebook](../Example%20notebooks/Topic%20modeling%20example.ipynb)

```python
import sys
sys.path.append('../topic_modeling/')

from autotm import AutoTM

topic_model = AutoTM()
```

where, the class **AutoTM** have 4 specific methods: 

* **TopicModeling**: Fit your model with _"LDA"_ or _"BERTopic"_.
* **get_topic_info**: Get information of all the generated topics.
* **get_document_info**: Get information of all the documents in DataFrame, printing the text and the probability of belonging to a certain topic.
* **visualize_topics**: Visual method to analyze the model.

The methods that can be used, as well as the hyperparameters, will be explained below.

## TopicModeling

Used to fit your model with _"LDA"_ or _"BERTopic"_ :

```python
topic_model.TopicModeling(data=df, target_column='review_text', topic_model='lda')
```

Hyperparameters: 

* **data**: <pandas.DataFrame> Your DataFrame.
* **target_column**: <str> Name of the column with the texts that must be preprocessed.
* **topic_model**: <str> Model that you want to use, can be 'lda' or 'bertopic'.
* **\*\*kwargs**: Additional hyperparameters.

Additional hyperparameters:

* **stopwords**: <bool, default=False> If _True_, the list will be obtained by the NLTK library through the assigned language; else, stopwords will not be removed.
    * Example: stopwords = True 
* **language**: <str, default='english'> Add the language you want to use for stopwords removal, the list will be obtained by the NLTK library.
    * Example: language ='portuguese' 
* **num_topics**: <int, default=None> Number of topics that the model will have, manually defined.
    * Example: num_topics = 20
* **start**: <int, default=2> Used only for LDA. If no value is set for "num_topics", values starting from the number specified here will be tested.
    * Example: start = 4
* **limit**: <int, default=30> Used only for LDA. If no value is set for "num_topics", values up to the limit specified here will be tested.
    * Example: limit = 20  
* **step**: <int, default=2> Used only for LDA. If no value is set for "num_topics", this is the steps that the tests will take, from start to limit.
    * Example: step = 3
* **verbose**: <int, default=0> If the number is greater than 0, it will print extra info about the processing.
    * Example: verbose = 1

## get_topic_info

Used to get information of all the generated topics:

```python
topic_model.get_topic_info()
```

Additional hyperparameters:

* **num_words**: <int, default=10> = Number of the most representative words of the topic that will be printed.
    * Example: num_words = 15

## get_document_info

Used to get information of all the documents in DataFrame, printing the text and the probability of belonging to a certain topic:

```python
topic_model.get_document_info()
```

## visualize_topics

Visual representation of topics:

```python
topic_model.visualize_topics()
```