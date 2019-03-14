import sys

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# import data
data = pd.read_csv('./train.csv')

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

# cross-validation
random_forrest = RandomForestClassifier(n_estimators=100)
scores = cross_val_score(random_forrest, data[relevant_features], data[target_feature].values.ravel(), cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# Used to identify important features
# importance = random_forrest.feature_importances_
# feature_names = list(data.columns.values[1:56])
#
# feature_importance = list(zip(feature_names, importance))
# feature_importance_sorted = sorted(feature_importance, key=lambda tup: tup[1], reverse=True)
#
# print("Importance features sorted")
#
# for el in feature_importance_sorted:
#     print('%-35s %-21s' % (str(el[0]), str(el[1])))

sys.exit(0)
