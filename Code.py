#Importing required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Reading the dataset
dataset = pd.read_csv('Amezon_Reviews.tsv', delimiter = '\t', quoting = 3)

# Preprocessing
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
print(corpus)
    
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#Clasiification
#Logsistic Regression model
from sklearn.linear_model import LogisticRegression
lr_classifier = LogisticRegression()
lr_classifier.fit(X_train, y_train)
y_pred_lr = lr_classifier.predict(X_test)

# Fitting Naive Bayes
from sklearn.naive_bayes import GaussianNB
NB_classifier = GaussianNB()
NB_classifier.fit(X_train, y_train)
y_pred_NB = NB_classifier.predict(X_test)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
rf_classifier.fit(X_train, y_train)
y_pred_rf = rf_classifier.predict(X_test)

#Support Vector
from sklearn.svm import SVC
SVC_classifier = SVC(kernel = 'rbf')
SVC_classifier.fit(X_train, y_train)
y_pred_SVC = SVC_classifier.predict(X_test)

#KNN model
from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn_classifier.fit(X_train, y_train)
y_pred_knn = knn_classifier.predict(X_test)

#Confusion Matrices
from sklearn.metrics import confusion_matrix, accuracy_score
cm_lr = confusion_matrix(y_test, y_pred_lr)
cm_NB = confusion_matrix(y_test, y_pred_NB)
cm_RandFor = confusion_matrix(y_test, y_pred_rf)
cm_SVC = confusion_matrix(y_test, y_pred_SVC)
cm_knn = confusion_matrix(y_test, y_pred_knn)
print(cm_lr)
accuracy_score(y_test, y_pred_lr)
print(cm_NB)
accuracy_score(y_test, y_pred_NB)
print(cm_RandFor)
accuracy_score(y_test, y_pred_rf)
print(cm_SVC)
accuracy_score(y_test, y_pred_SVC)
print(cm_knn)
accuracy_score(y_test, y_pred_knn)

#CAP Analysis
total = len(y_test) 
one_count = np.sum(y_test) 
zero_count = total - one_count 
lm_lr = [y for _, y in sorted(zip(y_pred_lr, y_test), reverse = True)]
lm_NB = [y for _, y in sorted(zip(y_pred_NB, y_test), reverse = True)] 
lm_SVC = [y for _, y in sorted(zip(y_pred_SVC, y_test), reverse = True)] 
lm_RandFor = [y for _, y in sorted(zip(y_pred_rf, y_test), reverse = True)] 
lm_knn = [y for _, y in sorted(zip(y_pred_knn, y_test), reverse = True)]
x = np.arange(0, total + 1) 
y_lr = np.append([0], np.cumsum(lm_lr)) 
y_NB = np.append([0], np.cumsum(lm_NB)) 
y_SVC = np.append([0], np.cumsum(lm_SVC)) 
y_RandFor = np.append([0], np.cumsum(lm_RandFor)) 
y_knn = np.append([0], np.cumsum(lm_knn))

plt.figure(figsize = (10, 10))
plt.title('CAP Curve Analysis')
plt.plot([0, total], [0, one_count], c = 'k', linestyle = '--', label = 'Random Model')
plt.plot([0, one_count, total], [0, one_count, one_count], c = 'grey', linewidth = 2, label = 'Perfect Model') 
plt.plot(x, y_lr, c = 'r', label = 'LR classifier', linewidth = 2)
plt.plot(x, y_SVC, c = 'y', label = 'SVC', linewidth = 2)
plt.plot(x, y_NB, c = 'b', label = 'Naive Bayes', linewidth = 2)
plt.plot(x, y_RandFor, c = 'c', label = 'Rand Forest', linewidth = 2)
plt.plot(x, y_knn, c = 'g', label = 'KNN classifier', linewidth = 2)
plt.legend()