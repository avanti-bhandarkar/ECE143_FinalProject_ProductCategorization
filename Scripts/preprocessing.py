#Standard libraries
import numpy as np
import pandas as pd
import string
import re
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings("ignore")

#From our utils.py
from utils import clean

#NLP
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
#nltk.download('stopwords')

# Load data
df = pd.read_csv('.../makeup_original.csv') #change path as required
df.drop(df.filter(regex="Unname"),axis=1, inplace=True)
df.dropna()

# Clean data

df['cleaned_desc'] = df['description'].apply(lambda x: clean(x))

# Remove common cosmetic ingredients and colorants
color_cleaned = []
from itertools import groupby
tokens = (x for line in open('colorants.txt') for x in line.split())
for key, it in groupby(tokens, lambda x: x[0].isdigit()):
    color_cleaned.append(str.join(' ', it))

ing_cleaned = []
from itertools import groupby
tokens = (x for line in open('ingredients.txt') for x in line.split())
for key, it in groupby(tokens, lambda x: x[0].isdigit()):
    ing_cleaned.append(str.join(' ', it))

ingredients_to_remove = color_cleaned + ing_cleaned

df['cleaned_desc'] = df.cleaned_desc.replace(ingredients_to_remove, '')

# Save dataframe as a new CSV for downstream tasks
df.to_csv('.../cleaned_makeup.csv',index=False) #change path as required
