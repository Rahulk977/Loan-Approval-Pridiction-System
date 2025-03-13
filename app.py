import numpy as np
import pandas as pd 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


loan_dataset=pd.read_csv('/content/loan_dataset.csv')

loan_dataset.head()
loan_dataset.shape
loan_dataset.describe()

loan_dataset.isnull().sum()
loan_dataset=loan_dataset.dropna()
loan_dataset.isnull().sum()

loan_dataset.replace({"Loan_Status":{'N':0,'Y':1}},inplace=True)

loan_dataset['Dependents'].value_counts()

loan_dataset=loan_dataset.replace(to_replace='3+',value=4)

loan_dataset['Dependents'].value_counts()

sns.countplot(x='Education',hue='Loan_Status',data=loan_dataset)

sns.countplot(x="Married",hue="Loan_Status",data=loan_dataset)

sns.countplot(x="Credit_History",hue="Loan_Status",data=loan_dataset)

loan_dataset.replace({"Married":{'No':0,'Yes':1},
                      "Gender":{"Male":1,"Female":0},
                      "Property_Area":{"Rural":0,"Semiurban":1,"Urban":2},
                      "Self_Employed":{"No":0,"Yes":1},
                      "Education":{"Graduate":1,"Not Graduate":0}
                      },inplace=True)

X=loan_dataset.drop(columns=['Loan_ID','Loan_Status'],axis=1)
Y=loan_dataset['Loan_Status']

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,random_state=2)
print(X.shape,X_train.shape,X_test.shape)

classifier=svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)
x_train_pridiction=classifier.predict(X_train)
training_data_accuracy=accuracy_score(x_train_pridiction,Y_train)

print('Accuracy on training data : ', training_data_accuracy)

x_testing_pridiction=classifier.predict(X_test)
testing_data_accuracy=accuracy_score(x_testing_pridiction,Y_test)
print('Accuracy on training data : ', testing_data_accuracy)

input_data = (1, 0, 0, 1, 0, 4166, 7210.0, 184.0, 360.0, 1.0, 2)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

feature_names = ["Gender", "Married", "Dependents", "Education", "Self_Employed", 
                 "ApplicantIncome", "CoapplicantIncome", "LoanAmount", 
                 "Loan_Amount_Term", "Credit_History", "Property_Area"]

input_data_df = pd.DataFrame(input_data_reshaped, columns=feature_names)
prediction = classifier.predict(input_data_df)

if prediction[0] == 0:
    print("loan is not approved")
else:
    print("loan is approved")