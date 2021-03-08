import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

data = pd.read_csv("C://Users//Administrator//Desktop//ML//loan.csv")
# print(data.shape)
# print(data.head())
# print(data.dtypes)

for col in ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]:
    data[col] = pd.to_numeric(data[col], errors='coerce')
    data = data.dropna(subset=[col])
    data[col] = data[col].astype(int)

for col in ["Dependents", "Gender", "Married", "Education", "Self_Employed", "Property_Area", "Loan_Status"]:
    data[col] = data[col].astype('category')

#print(data.dtypes)

# label encoding
data["Gender_cat"] = data["Gender"].cat.codes
data["Married_cat"] = data["Married"].cat.codes
data["Dependents_cat"] = data["Dependents"].cat.codes
data["Education_cat"] = data["Education"].cat.codes
data["Self_Employed_cat"] = data["Self_Employed"].cat.codes
data["Property_Area_cat"] = data["Property_Area"].cat.codes
data["Loan_Status_cat"] = data["Loan_Status"].cat.codes
# print(data.dtypes)

final_data = data.select_dtypes(include=['integer', 'float']).copy()

# 6print("\n")
#print(final_data.dtypes)

# data splicing
feature_names = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History',
                 'Gender_cat', 'Married_cat', 'Dependents_cat', 'Education_cat', 'Self_Employed_cat',
                 'Property_Area_cat']
x = final_data[feature_names]
y = final_data['Loan_Status_cat']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

svm = SVC(kernel="poly", degree=5)
svm.fit(x_train, y_train)
print('accuracy on training data:  {:.2f}'.format(svm.score(x_train, y_train)))
print('accuracy on testing data:  {:.2f}'.format(svm.score(x_test, y_test)))
