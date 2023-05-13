from flask import *  
import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import nltk
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from bs4 import BeautifulSoup
import re,string,unicodedata
from keras.preprocessing import text, sequence
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from string import punctuation
from nltk import pos_tag
from nltk.corpus import wordnet
import keras
from keras.models import Sequential
from keras.layers import Dense,Embedding,LSTM,Dropout
from keras.callbacks import ReduceLROnPlateau
import tensorflow as tf
nltk.download('stopwords')






#importing the new model
new_model = tf.keras.models.load_model('model/saved_model/my_model')

#setting stopwords and punctuation
stopping = set(stopwords.words('english'))
punctuating = list(string.punctuation)
stopping.update(punctuating)

#Remove html tags
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()
      
#Removing the square brackets
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)
# Removing URL's
    def remove_between_square_brackets(text):
        return re.sub(r'http\S+', '', text)
#Removing the stopwords from text
def remove_stopwords(text):
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stopping:
            final_text.append(i.strip())
    return " ".join(final_text)
#Removing the noisy text
def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    text = remove_stopwords(text)
    return text
   
text = text.apply(str)
text=text.apply(denoise_text)

def get_corpus(text):
    words = []
    for i in text:
        for j in i.split():
            words.append(j.strip())
    return words
corpus = get_corpus(text)
corpus[:5]

from collections import Counter
counter1 = Counter(corpus)
most_common = counter1.most_common(10)
most_common = dict(most_common)
most_common

from sklearn.feature_extraction.text import CountVectorizer
def get_top_text_ngrams(corpus, n, g):
    vec1 = CountVectorizer(ngram_range=(g, g)).fit(corpus)
    bag_of_words1 = vec1.transform(corpus)
    sum_words1 = bag_of_words1.sum(axis=0) 
    words_freq1 = [(word, sum_words1[0, idx]) for word, idx in vec1.vocabulary_.items()]
    words_freq1 =sorted(words_freq1, key = lambda x: x[1], reverse=True)
    #check the return
    return words_freq1[:n]
      
material = text

max_features = 10000
maxlen = 300

tokenizer = text.Tokenizer(num_words=max_features)
#tokenizer.fit_on_texts(x_train)
#tokenized_train = tokenizer.texts_to_sequences(x_train)
#x_train = sequence.pad_sequences(tokenized_train, maxlen=maxlen)
#tokenized_test = tokenizer.texts_to_sequences(x_test)
#X_test = sequence.pad_sequences(tokenized_test, maxlen=maxlen)

tokenizer.fit_on_texts(material)
tokenized_test = tokenizer.texts_to_sequences(material)
testingMaterial = sequence.pad_sequences(tokenized_test, maxlen=maxlen)

EMBEDDING_FILE = "artifacts/glove.twitter.27B.100d.txt"
file = open(EMBEDDING_FILE, encoding="utf-8")
def get_coefs(word, *arr): 
    return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open('glove.twitter.27B.100d.txt', 'r', encoding='utf-8'))
      
all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

max_features = 10000
tokenizer= text.Tokenizer()
word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
#change below line if computing normal stats is too slow
embedding_matrix = embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
      
threshold = 0.5
predict_x= new_model.predict(testingMaterial) 
predict_x[predict_x <=threshold] = 0
predict_x[predict_x > (1 - threshold)] = 1
result = predict_x[:10]