#Standard libraries
import numpy as np
import pandas as pd
import string
import re
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings("ignore")

#NLP
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
#nltk.download('stopwords')


def clean(text):
    '''
    Performs text preprocessing operations to clean the given text data:
    - Converts the input text to a string.
    - Converts all characters to lowercase.
    - Removes non-alphanumeric characters.
    - Removes stopwords from the text.
    - Lemmatizes words in the text.

    Parameters:
    text (str): The input text data to be cleaned.

    Returns:
    str: The cleaned text after applying the specified preprocessing steps.
    '''
    assert isinstance(text, str), 'Enter your input as a string'
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    stop = stopwords.words('english')
    text = [w for w in text.split() if not w in stop]
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in text]
    text = [ word for word in text if len(word) > 1 ]
    text = ' '.join(text)
    return text

def list_count(lst):
  '''
    Counts the occurrences of unique elements in the given list.

    Parameters:
    input: list --> list): Input list containing elements to be counted.

    Returns:
    list)count: A list of dictionaries where each dictionary contains a unique element
          from the input list as the key and its corresponding count as the value.
          The list is sorted in descending order based on element counts.
    '''
    assert isinstance(lst, list), 'Enter your input as a list'
    dic_counter = collections.Counter()
    for x in lst:
        dic_counter[x] += 1
    dic_counter = collections.OrderedDict(
                     sorted(dic_counter.items(),
                     key=lambda x: x[1], reverse=True))
    list_count = [ {key:value} for key,value in dic_counter.items() ]
    return list_count

def ner_features(lst_dics_tuples, tag):
    '''
    Analyzes Named Entity Recognition (NER) features from a list of dictionaries containing tuples.
    Counts occurrences of specified entity types.

    Parameters:
    lst_dics_tuples (list): A list of dictionaries where each dictionary contains tuples representing
                            named entities and their corresponding types.
    tag (str): The entity type for which the count needs to be retrieved.

    Returns:
    int: The count of occurrences of the specified entity type (`tag`) across all dictionaries.
         Returns 0 if the input list (`lst_dics_tuples`) is empty.
    '''
    assert isinstance(lst_dics_tuples, list), 'Enter the named entities as a list of dictionaries'
    assert isinstance(tag, list), 'Enter tage information as a string'

    if len(lst_dics_tuples) > 0:
        tag_type = []
        for dic_tuples in lst_dics_tuples:
            for tuple in dic_tuples:
                type, n = tuple[1], dic_tuples[tuple]
                tag_type = tag_type + [type]*n
                dic_counter = collections.Counter()
                for x in tag_type:
                    dic_counter[x] += 1
        return dic_counter[tag]
    else:
        return 0