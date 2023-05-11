# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Get the data and use label encoder to change all the values.
2. Classify the training data and the data.
3. Calculate the accuracy score,confusion matrix and classification report.
4. Then program predicts the Logistic regression.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: G.K Pravinrajj
RegisterNumber:  212222240080

import pandas as pd
data=pd.read_csv('/content/Placement_Data (1).csv')
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1) #removes the specified row or cols
data1.head() 

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x
y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver="liblinear") 
lr.fit(x_train,y_train)
y_pred= lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1= classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
*/
```

## Output:
### Placement data:
![image](https://github.com/Pravinrajj/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/117917674/14e00375-09eb-4552-942b-d9b5582474ac)

### Salary Data:
![image](https://github.com/Pravinrajj/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/117917674/5caaaac1-8aee-4c11-9313-9865a33818ae)

### isNULL():
![image](https://github.com/Pravinrajj/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/117917674/a491fa72-3f6d-419d-806a-cf8e888dd0de)

### Checking for Duplicates:
![image](https://github.com/Pravinrajj/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/117917674/c5f0fd85-1da5-42e1-a4db-0538dfd77d16)

### Print data:
![image](https://github.com/Pravinrajj/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/117917674/63491ed7-6cf1-486c-98e7-22b909d047d3)

### Data status:
![image](https://github.com/Pravinrajj/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/117917674/4919af56-820f-451b-be3e-bfe91f8a959c)

### y_prediction Array:
![image](https://github.com/Pravinrajj/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/117917674/307f1844-30b2-48d2-b448-16a9b4e11c6b)

### Accuracy Score:
![image](https://github.com/Pravinrajj/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/117917674/bee68668-dd75-4923-b279-b7364527daa6)

### Confusion matrix:
![image](https://github.com/Pravinrajj/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/117917674/c13a84cb-d3fc-4826-8807-e7f7e5689c3a)

### Classification Report:
![image](https://github.com/Pravinrajj/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/117917674/1e519947-da62-48bd-a52a-65b3693e1981)

### Prediction of LR:
![image](https://github.com/Pravinrajj/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/117917674/c1708d4e-e674-4df3-9653-925736fa18e6)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
