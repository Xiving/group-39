import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# import data
data = pd.read_csv('./train.csv')
print("Data length: " + str(len(data)))

# split features and target
features = data.iloc[:, 1:55]
target = data.iloc[:, 55]

# split test and train data
mask = np.random.rand(len(features)) < 0.7
x_train = features[mask]
y_train = target[mask]
x_test = features[~mask]
y_test = target[~mask]

# fit a model
random_forrest = RandomForestClassifier(n_estimators=1000)
random_forrest.fit(x_train, y_train)

# test model
y_predicted = random_forrest.predict(x_test)
print("Accuracy:" + str(accuracy_score(y_test, y_predicted)) + "\n")

importance = random_forrest.feature_importances_
feature_names = list(data.columns.values[1:56])

feature_importance = list(zip(feature_names, importance))
feature_importance_sorted = sorted(feature_importance, key=lambda tup: tup[1], reverse=True)

print("Importance features sorted")

for el in feature_importance_sorted:
  print('%-35s %-21s' % (str(el[0]), str(el[1])))

