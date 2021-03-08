import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


# providing headers as the dataset does not contain any headers
headers = ["buying", "maintenance", "doors", "persons", "lug_boot", "safety", "result"]
data = pd.read_csv("C://Users//Administrator//Desktop//ML//cars.csv", header=None, names=headers, na_values="?")

# changing objects datatype to category, to enable label encoding
for col in ["buying", "maintenance", "lug_boot", "safety", "result"]:
    data[col] = data[col].astype('category')

# replacing textual data entry (data cleaning)
cleanup_door_persons = {"doors": {"5more": 5},
                        "persons": {"more": 5}}

data.replace(cleanup_door_persons, inplace=True)

# object to int type
for col in ["doors", "persons"]:
    data[col] = data[col].astype('int')


# label encoding
data["buying_cat"] = data["buying"].cat.codes
data["maintenance_cat"] = data["maintenance"].cat.codes
data["lug_cat"] = data["lug_boot"].cat.codes
data["safety_cat"] = data["safety"].cat.codes
data["result_cat"] = data["result"].cat.codes

# selecting only the integer type data columns for feeding it into the learning model
final_data = data.select_dtypes(include=['integer']).copy()

# data splicing
feature_names = ['doors', 'persons', 'buying_cat', 'maintenance_cat', 'lug_cat', 'safety_cat']
x = final_data[feature_names]
y = final_data['result_cat']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

svm = SVC(kernel="poly", degree=3)
svm.fit(x_train, y_train)
print('accuracy on training data:  {:.2f}'.format(svm.score(x_train, y_train)))
print('accuracy on testing data:  {:.2f}'.format(svm.score(x_test, y_test)))

