# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 15:03:11 2021

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
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 40, stratify = y)


from IRD_model import clf
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn. model_selection import cross_val_score
params_ada = [0.000001, 0.00001, 0.0001, 0.001, 0.1, 1]
ada = AdaBoostClassifier(base_estimator = clf, n_estimators = 500, random_state = 1)
params = {"learning_rate": params_ada}
grid_ada = GridSearchCV(estimator = ada, param_grid = params, cv = 10, scoring = "roc_auc", n_jobs = -1)
gs_ada = grid_ada.fit(X_train, y_train)
clf_ada = gs_ada.best_estimator_
clf_ada.fit(X_train, y_train)
scores_boost = cross_val_score(estimator = clf_ada, X = X_train, y = y_train, cv = 10, n_jobs = 1)
print(np.mean(scores_boost), np.std(scores_boost))
print(clf_ada.score(X_test, y_test))





from sklearn.model_selection import validation_curve
param_range = params_ada
train_scores, test_scores = validation_curve(estimator = ada, X = X_train, y = y_train, param_name = 'learning_rate', param_range=param_range, cv=10, n_jobs = -1)
train_mean = np.mean(train_scores, axis = 1)
train_std = np.std(train_scores, axis = 1)
test_mean = np.mean(test_scores, axis = 1)
test_std = np.std(test_scores, axis = 1)





import matplotlib.pyplot as plt
fig, ax = plt.subplots()
plt.plot(params_ada, train_mean, color = 'b', marker = 'o', markersize = 5, label = "Dokładnosc uczenia")
plt.fill_between(params_ada, train_mean+train_std, train_mean - train_std, alpha = 0.15, color = 'b')

plt.plot(params_ada, test_mean, color = 'g', marker = 'o', markersize = 5, label = "Dokładnosc walidacji")
plt.fill_between(params_ada, test_mean+test_std, test_mean - test_std, alpha = 0.15, color = 'g')
plt.grid()
plt.legend(loc = "lower right")
ax.set_xlabel("Parametr")
ax.set_ylabel("Dokładnosc")
plt.xscale("log")
plt.ylim([0.4,  1])
plt.savefig("krzywa_walidacji2.png")


import sklearn.metrics as metrics
# calculate the fpr and tpr for all thresholds of the classification
probs = clf_ada.predict_proba(X_test)
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
plt.savefig("ROC2.png")

feat_labels = justX.columns


importances = clf_ada.feature_importances_
indicies = np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f+1, 30, feat_labels[indicies[f]], importances[indicies[f]]))
    
plt.title("Istotnosć cech")
plt.bar(range(X_train.shape[1]), importances[indicies], align = 'center')
plt.xticks(range(X_train.shape[1]), feat_labels[indicies], rotation = 90)
plt.xlim([-1, X_train.shape[1]])
plt.show()

y_pred2 = clf_ada.predict(X_test)
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(clf_ada, X_test, y_test)
plt.savefig("macierz konfuzji.png")
