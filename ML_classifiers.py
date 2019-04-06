#========Implementing machine learning algorithms on dataset=======#
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings
warnings.filterwarnings("ignore")
import os

#=============Importing the dataset========================+#
data = pd.read_csv('D:/University/Y3/Dissertation/input/HAM10000_metadata.csv')
data.sex = [1 if each == 'female'else 0 for each in data.sex]

image = pd.read_csv('D:/University/Y3/Dissertation/input/hmnist_28_28_L.csv')

#=============Preprocessing dataset===========================#

data.dx = [1 if each == 'bkl'else 0 for each in data.dx]

data['age'].dropna()
data = data.dropna()
data['localization'] = pd.Categorical(data['localization'])
new_categ_df = pd.get_dummies(data['localization'],prefix ='local')
new_data_frame = pd.concat([data,new_categ_df],axis=1)

data = new_data_frame
new_data_frame['dx_type'] = pd.Categorical(new_data_frame['dx_type'])
new_categ_df = pd.get_dummies(new_data_frame['dx_type'],prefix ='dx_type')
new_data =  pd.concat([new_data_frame,new_categ_df],axis=1)
new_data_frame = pd.concat([data,new_categ_df],axis=1)
data = new_data_frame
data = data.drop(['localization','lesion_id','image_id','dx_type'],axis=1)

x = data.drop('dx',axis=1)
y = data['dx']

#=========Splitting dataset into train test=================================#

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=111)

#======Importing machine learning classifiers==================================#

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

svc_Linear=SVC(kernel='linear')
svc_RBf=SVC()
Rf=RandomForestClassifier()
LR=LogisticRegression()
knc= KNeighborsClassifier(n_neighbors=49)
dtc=DecisionTreeClassifier(min_samples_split=7,random_state=111)
nb=GaussianNB()
models={'Linear SVM':svc_Linear,'SVM with RBF kernel':svc_RBf,"Random Forest":Rf,"Logistic Regression":LR,"KNN":knc,"Decision Tree":dtc,'Naive Bayes':nb}

#=========Implementing  Performance Measures calculation===========#

def train_classifier(models,X_train,y_train):
	models.fit(X_train,y_train)

def predict_class(models,test_data):
	return models.predict(test_data)


accuracy=[]
precision=[]
recall=[]
F1_score=[]
for k,v in models.items():
    train_classifier(v,X_train,y_train)
    pred=predict_class(v,X_test)
    accuracy.append((accuracy_score(y_test,pred)))
    precision.append((precision_score(y_test,pred)))
    recall.append((recall_score(y_test,pred)))
    F1_score.append((f1_score(y_test,pred)))

    
#============Final results dataframe======================================#

results=pd.DataFrame({"Models":["Linear SVM",'SVM with RBF kernel',"Random Forest","Logistic Regression","KNN","Decision Tree",'Naive Bayes'],
                      "Accuracy":accuracy,"Precision":precision,"Recall":recall,"F1_score":F1_score})
    
print(results)
