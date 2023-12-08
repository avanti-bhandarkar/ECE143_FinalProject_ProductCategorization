import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import gensim
from gensim.models import LdaMulticore
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
import gensim.corpora as corpora

import pyLDAvis
import pyLDAvis.gensim

# Load data
df = pd.read_csv('.../cleaned_makeup.csv')
df.drop(df.filter(regex="Unname"),axis=1, inplace=True)
df.dropna()

#Topic modelling using LDA

corpus = [row.split() for row in df['cleaned_desc']]
dic = corpora.Dictionary(corpus)
bow_corpus = [dic.doc2bow(doc) for doc in corpus]

lda_model = LdaMulticore(bow_corpus, num_topics = 10, id2word = dic, passes = 10, workers = 8)

# Compute Perplexity - lower is better.
perplexity = lda_model.log_perplexity(bow_corpus)
print('\nPerplexity:', perplexity)

# Compute Coherence Score
coherence_lda = CoherenceModel(model=lda_model, texts=corpus, dictionary=dic, coherence='c_v')
coherence = coherence_lda.get_coherence()
print('\nCoherence Score: ', coherence)

#Generate and display LDA inline
lda_vis = pyLDAvis.gensim.prepare(lda_model, bow_corpus, dic)

#Save LDA visualization as html file
pyLDAvis.save_html(lda_vis,'LDAvis.html')
