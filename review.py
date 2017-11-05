import pandas as pd
import re
import numpy as np
from nltk.corpus import stopwords
from bs4 import BeautifulSoup as bs
from sklearn.feature_extraction.text import CountVectorizer as cv
from sklearn.naive_bayes import MultinomialNB as NB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# review to list of useful words
def review_to_words(raw_review):
	reviews = re.sub(r'[^A-Za-z]'," ",bs(raw_review,"lxml").get_text())
	words = reviews.lower().split()
	stops = set(stopwords.words('english'))
	words = [w for w in words if not w in stops] 
	joint_tokens = ":".join(words)
	return joint_tokens, reviews

# loading dataset
train = pd.read_csv('labeledTrainData.tsv', header=0, delimiter='\t', quoting=3)

tokens, reviews = [],[]

# preprocessing data => converting reviews to token list of words
for rev in range (0,train.shape[0]):
	t, r = review_to_words(train["review"][rev])	
	tokens.append(t)			# token list
	reviews.append(r)			# seperate reviews

vocabulary = 5000				# max features
vectorizer = cv(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = vocabulary)

X = vectorizer.fit_transform(reviews).toarray()
Y = train["sentiment"]

validation_size = 0.20
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X,Y,test_size=validation_size)

#classifier = DecisionTreeClassifier()		#DTC
#classifier = SVC()				#SVM
#classifier = KNeighborsClassifier()		#KNN
classifier = NB(alpha=2) 			#alpha=0 means no laplace smoothing
classifier.fit(X_train, np.array(Y_train))

predictions = classifier.predict(X_validation)
print("Accuracy: "+ accuracy_score(Y_validation, predictions))
print("Confusion Matrix: "+ confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
