from flask import *
import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud,STOPWORDS
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
import json


app = Flask(__name__) 

model = tf.keras.models.load_model(r"C:\Users\Drdr\Desktop\Fake News Detection\extension\server\erast.h5")

#route to the popup.html
@app.route('/')  
def customer():  
   return render_template('popup.html')


@app.route('/predict', methods=['POST'])
def prediction():
   if request.method == 'POST':
      recieved = request.form.to_dict()

      #swaps the key and value pairs in the recieved dictionary
      text_swap = {v: k for k, v in recieved.items()}

      capture = text_swap['']

      #df1 = pd.DataFrame({"text": [capture]})
      
      x = 1

      return ('', 204)
      if x == 1:
         return render_template('true.html')



@app.route('/success', methods = ['POST', 'GET'])  
def print_data():  
   if request.method == 'POST':  
      result = request.form
      #return render_template("true.html",result = result)
      #return render_template("false.html", result = result)
      return redirect(url_for('cool_form'))

@app.route('/true', methods = ['POST', 'GET'])  
def cool_form():  
   #if request.method == 'POST':  
      #result = request.form
      #return render_template("true.html",result = result)
   return render_template('false.html')
   return ('', 204)      
   
   
if __name__ == '__main__':  
   app.run(debug = True)  