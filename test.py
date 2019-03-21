import sys

import pandas as pd
from sklearn import metrics, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.tree import DecisionTreeClassifier

# constants
relevant_features = [
    'Elevation',
    'Horizontal_Distance_To_Roadways',
    'Horizontal_Distance_To_Fire_Points',
    'Horizontal_Distance_To_Hydrology',
    'Vertical_Distance_To_Hydrology',
    'Hillshade_9am',
    'Aspect',
    'Hillshade_3pm',
    'Hillshade_Noon',
    'Wilderness_Area4',
    'Slope']

target_feature = ['Cover_Type']

cols_to_norm = [
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am",
    "Hillshade_3pm",
    "Hillshade_Noon",
    "Horizontal_Distance_To_Fire_Points"]


# A function used for getting the AUC when having multiple features
def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)

    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)

    return roc_auc_score(y_test, y_pred, average=average)


# import and split data
data = pd.read_csv('./train.csv')
train, test = train_test_split(data, test_size=0.2)

# normalize the data of certain features
scaler = StandardScaler().fit(data[cols_to_norm])
train[cols_to_norm] = scaler.transform(train[cols_to_norm])
test[cols_to_norm] = scaler.transform(test[cols_to_norm])

# split the training data into feautures and target
X = train[relevant_features]
y = train[target_feature].values.ravel()


#
# Training and printing the results of multiple models using cross validation (10 folds)
#

# Random forrest (n = 10, 30, 50, 70, 90)
for i in range(10, 100, 20):
    rfc = RandomForestClassifier(n_estimators=i)
    rfc.fit(X.values, y)
    scores = cross_val_score(rfc, train[relevant_features], train[target_feature].values.ravel(), cv=10)
    print("rfc n=" + str(i) + " mean: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    y_pred = rfc.predict(test[relevant_features])
    y_true = test[target_feature]

    print("rfc n=" + str(i) + ", accuracy: " + str(metrics.accuracy_score(y_pred, y_true)))
    print("rfc n=" + str(i) + ", precision: " + str(multiclass_roc_auc_score(y_pred, y_true)))


# Decision Tree
dtc = DecisionTreeClassifier()
dtc.fit(X.values, y)
scores = cross_val_score(dtc, train[relevant_features], train[target_feature].values.ravel(), cv=10)
print("dtc mean: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

y_pred = dtc.predict(test[relevant_features])
y_true = test[target_feature]

print("dtc accuracy: " + str(metrics.accuracy_score(y_pred, y_true)))
print("dtc precision: " + str(multiclass_roc_auc_score(y_pred, y_true)))

# Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X.values, y)
scores = cross_val_score(gnb, train[relevant_features], train[target_feature].values.ravel(), cv=10)
print("gnb mean: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

y_pred = gnb.predict(test[relevant_features])
y_true = test[target_feature]

print("gnb accuracy: " + str(metrics.accuracy_score(y_pred, y_true)))
print("gnb precision: " + str(multiclass_roc_auc_score(y_pred, y_true)))

# Support Vector Machine (1 vs rest)
SVC = svm.SVC(decision_function_shape='ovr', gamma='auto')
SVC.fit(X.values, y)
print("support vectors: " + str(len(SVC.support_vectors_)))
scores = cross_val_score(SVC, train[relevant_features], train[target_feature].values.ravel(), cv=10)
print("SVC 1vAll mean: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

y_pred = SVC.predict(test[relevant_features])
y_true = test[target_feature]

print("SVC 1vAll accuracy: " + str(metrics.accuracy_score(y_pred, y_true)))
print("SVC 1vAll precision: " + str(multiclass_roc_auc_score(y_pred, y_true)))

# KN Neighbour
for i in range(1, 20):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X.values, y)
    scores = cross_val_score(knn, train[relevant_features], train[target_feature].values.ravel(), cv=10)
    print("KNN n=" + str(i) + " mean: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    y_pred = knn.predict(test[relevant_features])
    y_true = test[target_feature]

    print("KNN n=" + str(i) + ", accuracy: " + str(metrics.accuracy_score(y_pred, y_true)))
    print("KNN n=" + str(i) + ", precision: " + str(multiclass_roc_auc_score(y_pred, y_true)))

sys.exit(0)
