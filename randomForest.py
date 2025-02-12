# Random forest code for my CIS class at College of DuPage

from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd

praticeDataset = datasets.load_iris()

print(praticeDataset.target_names)
print(praticeDataset.feature_names)

X, y = datasets.load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3)

data = pd.DataFrame({'sepallength': praticeDataset.data})

