# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
### Algorithm for the Logistic Regression Model to Predict Placement Status of Students:

1. **Import Libraries**  
   - Import the necessary Python libraries, including pandas, sklearn modules, and others.

2. **Load Dataset**  
   - Read the dataset (`Placement_Data.csv`) into a DataFrame using pandas.

3. **Inspect Data**  
   - Display the first few rows of the dataset to understand its structure.

4. **Preprocess Data**  
   - Create a copy of the dataset for processing.
   - Drop irrelevant columns (`sl_no`, `salary`) to simplify analysis.
   - Check for missing values and duplicates in the data.
   - Encode categorical variables into numerical format using `LabelEncoder`.

5. **Prepare Features (X) and Target (Y)**  
   - Define the features (`X`) by excluding the target variable (`status`).
   - Set the target variable (`Y`) as the encoded `status` column.

6. **Split Data**  
   - Split the dataset into training (80%) and testing (20%) sets using `train_test_split`.

7. **Build Logistic Regression Model**  
   - Initialize the logistic regression model using the `liblinear` solver.
   - Train the model on the training dataset (`x_train`, `y_train`).

8. **Make Predictions**  
   - Use the trained model to predict the placement status for the test set (`x_test`).

9. **Evaluate Model**  
   - Calculate the model's accuracy using `accuracy_score`.
   - Generate a confusion matrix to assess prediction errors.
   - Create a classification report to summarize performance metrics like precision, recall, and F1-score.

10. **Predict Placement Status for New Data**  
    - Input a sample student’s feature values to predict their placement status using the trained model.

11. **Output Results**  
    - Display the accuracy score, confusion matrix, classification report, and prediction result for the sample input.
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Pandidharan.G.R
RegisterNumber:212222040111
**
import pandas as pd
data=pd.read_csv("/content/Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

*/
```

## Output:
## 1.Placement Data
![ml exp 4 t](https://github.com/Rama-Lekshmi/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118541549/d343df92-e10c-416a-bbeb-469c70f9317b)
## 2.Salary Data
![Screenshot 2023-05-14 181224](https://github.com/Rama-Lekshmi/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118541549/4dec9e7f-b24f-4947-8ed8-eb77166acd02)
## 3.Checking the null function()
![ml exp 3 th](https://github.com/Rama-Lekshmi/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118541549/2bf8d777-40f9-4a15-819f-f740a88dc01b)
## 4.Data Duplicate
![ml exp 4 f](https://github.com/Rama-Lekshmi/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118541549/24435259-ce8d-415e-a46e-515b3ce5bcd5)
## 5.Print Data
![Screenshot 2023-05-14 181531](https://github.com/Rama-Lekshmi/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118541549/b61b0a25-84b6-41c7-952d-09f2099f6354)
![ml exp 4 s](https://github.com/Rama-Lekshmi/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118541549/1af99856-990c-46db-adab-2fcb5646787c)
## 6.Data Status
![ml exp 4 se](https://github.com/Rama-Lekshmi/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118541549/8d74f3ac-a247-4bce-82a4-b2bc27cdeb80)
## 7.y_prediction array
![ml exp 4 ei](https://github.com/Rama-Lekshmi/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118541549/6d21b45c-3716-42ea-bfd8-0e6d4bd96875)
## 8.Accuracy value
![ml exp 4 n](https://github.com/Rama-Lekshmi/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118541549/f13d398f-7b71-49bd-b787-2ae8b6c86f49)
## 9.Confusion matrix
![ml exp 4 te](https://github.com/Rama-Lekshmi/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118541549/bea05d71-7a26-41df-b4af-84dc2d0788f6)
## 10.Classification Report
![ml exp 4 ele](https://github.com/Rama-Lekshmi/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118541549/495dae98-1d9d-4fcf-bb3c-20e259485d07)
## 11.Prediction of LR
![ml exp twe](https://github.com/Rama-Lekshmi/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118541549/17ef5d1d-79dc-41cb-a9d1-b05eff1c61dd)




## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
