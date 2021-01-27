import pickle
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import azureml.train.automl
import azureml.core


df = pd.read_csv('../Data/wine_data.csv')

print(df.head(10))

X = df.drop('Cultivar', axis=1).values
y = df.Cultivar.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=70, random_state=1)

# Normalising
scaler = StandardScaler().fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

filepath = "./models/automl_wine/model.pkl"

with open(filepath, "rb") as f:
    model = pickle.load(f, encoding='latin1')

y_pred = model.predict(X_test)

print(f'{classification_report(y_test, y_pred)}\n')
print(f'Accuracy: \u001b[32;1m{accuracy_score(y_test, y_pred) * 100}\u001b[0m \n')
print(f'Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n')
