#!/usr/bin/env python
# coding: utf-8

# # ML Pipeline Preparation
# Follow the instructions below to help you create your ML pipeline.
# ### 1. Import libraries and load data from database.
# - Import Python libraries
# - Load dataset from database with [`read_sql_table`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql_table.html)
# - Define feature and target variables X and Y

# In[1]:


# import libraries
from sqlalchemy import create_engine
import nltk


nltk.download(['punkt', 'wordnet'])
nltk.download('stopwords')


from sklearn.svm import SVC
import re
import numpy as np
import pandas as pd

from nltk.tokenize import word_tokenize,WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pickle


from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer


import warnings
warnings.filterwarnings('ignore')


# In[2]:


# load data from database
engine = create_engine('sqlite:///DisasterResponse.db')
df = pd.read_sql('SELECT * FROM DisasterResponse', con = engine)
df.head()


# ### 2. Write a tokenization function to process your text data

# In[3]:


X = df['message']
y = df.drop(['id','message','original', 'genre'], axis = 1)
y.shape


# In[4]:


def tokenize(text):
    # get tokens from text
    tokens= WhitespaceTokenizer().tokenize(text)
    lemmatizer= WordNetLemmatizer()
    
    # clean tokens
    processed_tokens=[]
    for token in tokens:
        token=lemmatizer.lemmatize(token).lower().strip('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
        token=re.sub(r'\[[^.,;:]]*\]','', token)
        
        # add token to compiled list if not empty
        if token !='':
            processed_tokens.append(token)
    return processed_tokens


# ### 3. Build a machine learning pipeline
# This machine pipeline should take in the `message` column as input and output classification results on the other 36 categories in the dataset. You may find the [MultiOutputClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) helpful for predicting multiple target variables.

# In[5]:


pipeline = Pipeline([
    ('cvect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
])


# ### 4. Train pipeline
# - Split data into train and test sets
# - Train pipeline

# In[6]:


# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el pipeline
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)


# ### 5. Test your model
# Report the f1 score, precision and recall for each output category of the dataset. You can do this by iterating through the columns and calling sklearn's `classification_report` on each.

# In[7]:


y_pred_df = pd.DataFrame(y_pred, columns = y_test.columns)
y_pred_df.head()


# ### 6. Improve your model
# Use grid search to find better parameters. 

# In[8]:


y_pred_df.shape 


# In[10]:


for column in y_test.columns:
    print('column_name: {}\n'.format(column))
    print(classification_report(y_test[column], y_pred_df[column]))


# In[11]:


pipeline.get_params()


# In[9]:


parameters = {

    'clf__estimator__n_estimators': [10, 15],
    'clf__estimator__min_samples_split': [2, 4]
}

cv = GridSearchCV(pipeline, param_grid=parameters, verbose=2,)


# In[10]:


cv.fit(X_train, y_train)


# In[11]:


cv.cv_results_


# In[12]:


print(cv.best_params_)


# In[13]:


optimised_model = cv.best_estimator_
print(cv.best_estimator_)


# ### 7. Test your model
# Show the accuracy, precision, and recall of the tuned model.  
# 
# Since this project focuses on code quality, process, and  pipelines, there is no minimum performance metric needed to pass. However, make sure to fine tune your models for accuracy, precision and recall to make your project stand out - especially for your portfolio!

# In[14]:


y_pred = optimised_model.predict(X_train)

for column in y_test.columns:
    print('column_name: {}\n'.format(column))
    print(classification_report(y_test[column], y_pred_df[column]))


# ### 8. Try improving your model further. Here are a few ideas:
# * try other machine learning algorithms
# * add other features besides the TF-IDF

# In[18]:



pipeline = Pipeline([
    ('cvect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf',MultiOutputClassifier(SVC()))
])


# ### 9. Export your model as a pickle file

# In[19]:


pickle.dump(optimised_model, open('model.pkl', 'wb'))


# ### 10. Use this notebook to complete `train.py`
# Use the template file attached in the Resources folder to write a script that runs the steps above to create a database and export a model based on a new dataset specified by the user.

# In[ ]:




