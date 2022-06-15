# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 11:57:02 2021

@author: aleks
"""
import copy
import pandas as pd
from IRD import X_df
import numpy as np
X_df = X_df.sort_values(by = ["Goal"])[5300:]
justX = X_df[:]
justX = justX.drop(["Goal", "X", "Y"], axis = 1)
y = X_df["Goal"].values
X = justX.values

"""
from woe import iv_woe
iv, woe = iv_woe(data = X_df, target = 'Goal', bins=10, show_woe = True)
print(iv)
print(woe)
"""


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 123, stratify = y)





from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
tree = DecisionTreeClassifier(random_state = 1)
params_depth = [1, 2, 3, 4, 5, 6, 7, 8]
params_pcc = [0, 0.001, 0.1, 0.15, 0.2]
params_min_sample_split = [2, 4, 5, 10]
min_samples_leaf = [2, 4, 8, 10, 20, 30]
params = {'max_depth': params_depth,
          'ccp_alpha': params_pcc,
          'criterion': ["gini", "entropy"], 
          'min_samples_split': params_min_sample_split,
          'min_samples_leaf': min_samples_leaf}
grid = GridSearchCV(estimator = tree, param_grid = params, cv = 10, scoring = "roc_auc")
gs = grid.fit(X_train, y_train)

clf = gs.best_estimator_
clf.fit(X_train, y_train)
from sklearn. model_selection import cross_val_score
scores = cross_val_score(estimator = clf, X = X_train, y = y_train, cv = 10, n_jobs = 1)
print(np.mean(scores), np.std(scores))
print(clf.score(X_test, y_test))

from sklearn.model_selection import learning_curve
train_sizes, train_scores, test_scores = learning_curve(estimator = clf, X = X_train, y = y_train, train_sizes = np.linspace(0.1, 1, 10), cv=10, n_jobs = 1)
train_mean = np.mean(train_scores, axis = 1)
train_std = np.std(train_scores, axis = 1)
test_mean = np.mean(test_scores, axis = 1)
test_std = np.std(test_scores, axis = 1)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
plt.plot(train_sizes, train_mean, color = 'b', marker = 'o', markersize = 5, label = "Dokładnosc uczenia")
plt.fill_between(train_sizes, train_mean+train_std, train_mean - train_std, alpha = 0.15, color = 'b')

plt.plot(train_sizes, test_mean, color = 'g', marker = 'o', markersize = 5, label = "Dokładnosc walidacji")
plt.fill_between(train_sizes, test_mean+test_std, test_mean - test_std, alpha = 0.15, color = 'g')
plt.grid()
ax.set_xlabel("Liczba próbek uczących")
ax.set_ylabel("Dokładnosc")
plt.legend(loc = "lower right")

plt.ylim([0.6, 0.9])
plt.savefig("krzywa_uczenia1.png")






from sklearn.model_selection import validation_curve

param_range = params_depth
train_scores, test_scores = validation_curve(estimator = tree, X = X_train, y = y_train, param_name = 'max_depth',param_range=param_range, cv=10)
train_mean = np.mean(train_scores, axis = 1)
train_std = np.std(train_scores, axis = 1)
test_mean = np.mean(test_scores, axis = 1)
test_std = np.std(test_scores, axis = 1)
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
plt.plot(params_depth, train_mean, color = 'b', marker = 'o', markersize = 5, label = "Dokładnosc uczenia")
plt.fill_between(params_depth, train_mean+train_std, train_mean - train_std, alpha = 0.15, color = 'b')

plt.plot(params_depth, test_mean, color = 'g', marker = 'o', markersize = 5, label = "Dokładnosc walidacji")
plt.fill_between(params_depth, test_mean+test_std, test_mean - test_std, alpha = 0.15, color = 'g')
plt.grid()
plt.legend(loc = "lower right")
ax.set_xlabel("Parametr")
ax.set_ylabel("Dokładnosc")
plt.ylim([0.6,  0.9])
plt.savefig("krzywa_walidacji1.png")


from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
dot_data = export_graphviz(clf, filled = True, rounded = True, class_names = ["Brak gola", "Gol"], feature_names = list(justX.columns))
graph = graph_from_dot_data(dot_data)
graph.write_png("drzewo.png")

import sklearn.metrics as metrics
# calculate the fpr and tpr for all thresholds of the classification
probs = clf.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

# method I: plt
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.savefig("ROC1.png")



feat_labels = justX.columns


importances = clf.feature_importances_
indicies = np.argsort(importances)[::-1]


for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f+1, 30, feat_labels[indicies[f]], importances[indicies[f]]))
    
plt.title("Istotnosć cech")
plt.bar(range(X_train.shape[1]), importances[indicies], align = 'center')
plt.xticks(range(X_train.shape[1]), feat_labels[indicies], rotation = 90)
plt.xlim([-1, X_train.shape[1]])
plt.show()

y_pred1 = clf.predict(X_test)




from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(clf, X_test, y_test)
plt.savefig("macierz konfuzji.png")



from sklearn.ensemble import BaggingClassifier 
bag = BaggingClassifier(base_estimator = clf, n_estimators = 500, bootstrap = True, n_jobs = -1, random_state = 1, bootstrap_features = False)
max_samples = [0.05, 0.1, 0.2, 0.5]
max_features = [0.05, 0.1, 0.2, 0.5, 1]
params = {'max_samples': max_samples,
          'max_features': max_features}

grid = GridSearchCV(estimator = bag, param_grid = params, cv = 10, scoring = "roc_auc", n_jobs = -1)
gs2 = grid.fit(X_train, y_train)
clf2 = gs2.best_estimator_
clf2.fit(X_train, y_train)
from sklearn. model_selection import cross_val_score
scores = cross_val_score(estimator = clf2, X = X_train, y = y_train, cv = 10, n_jobs = 1)
print(np.mean(scores), np.std(scores))
print(clf2.score(X_test, y_test))


import sklearn.metrics as metrics
# calculate the fpr and tpr for all thresholds of the classification
probs = clf2.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

# method I: plt
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.savefig("ROC4.png")

