import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
pima = pd.read_csv('C:/Users/mathu/Downloads/diabetes.csv')
#print(pima['Outcome'].unique())
print(pima.head())
#print(pima.info())
#print(dataset.describe())
print(pima.groupby('Outcome').size())
#EDA
'''dataset_copy = dataset.copy(deep = True)
dataset_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = dataset_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
print(dataset_copy.isnull().sum())
dataset_copy['Glucose'].fillna(dataset_copy['Glucose'].mean(), inplace = True)
dataset_copy['BloodPressure'].fillna(dataset_copy['BloodPressure'].mean(), inplace = True)
dataset_copy['SkinThickness'].fillna(dataset_copy['SkinThickness'].median(), inplace = True)
dataset_copy['Insulin'].fillna(dataset_copy['Insulin'].median(), inplace = True)
dataset_copy['BMI'].fillna(dataset_copy['BMI'].median(), inplace = True)'''

X_train, X_test, y_train, y_test = train_test_split(pima.loc[:, pima.columns != 'Outcome'], pima['Outcome'], stratify=pima['Outcome'],random_state=42)

print(pima.loc[:,pima.columns !='Outcome'])
print(y_train.value_counts())
print(y_test.value_counts())

feature_name=list(X_train.columns)
class_name=list(y_train.unique())
print(feature_name)
print(class_name)

acc=[]
for i in range(1,200,10):
 model = RandomForestClassifier(n_estimators=i)
 model.fit(X_train, y_train)
 y_predicted = model.predict(X_test)
 a=metrics.accuracy_score(y_test, y_predicted)
 acc.append(a)
print(acc)
#ploting tree
plt.plot(range(1,200,10), acc)
plt.show()
'''from sklearn import tree
plt.figure(figsize=(10,15))
tree.plot_tree(clf,filled=True)
plt.show()'''

from sklearn.model_selection import GridSearchCV
gd = GridSearchCV(model,{'max_depth':[10,20,30,40,50,60],'criterion':['gini','entropy']},cv=8)
gd=gd.fit(X_train,y_train)
print(gd.best_params_)
print(gd.best_score_)
y_pred=gd.predict(X_test)
print("Accuracy : ",metrics.accuracy_score(y_test,y_pred))
