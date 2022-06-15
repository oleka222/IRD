# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 15:02:55 2021

@author: aleks
"""
import matplotlib as plt
from IRD_model import clf, clf2
from IRD_model_boost import clf_ada
from IRD_las_losowy import forest

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from IRD import X_df


X_df = X_df.sort_values(by = ["Goal"])[5300:]
justX = X_df[:]
justX = justX.drop(["Goal", "X", "Y"], axis = 1)
y = X_df["Goal"].values
X = justX.values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 40, stratify = y)

fig, ax = plt.subplots()
clf_labels = ["Drzewo", "AdaBoost", "Las", "Bagging"]
colors = ["black", "orange", "blue", "green"]
linestyles = [':', '--', '-.', '-']
all_clf = [clf, clf2, clf_ada, forest]
for i, label, clr, ls in zip(all_clf, clf_labels, colors, linestyles):
    y_pred = i.fit(X_train, y_train).pradict_proba(X_test[:, 1])
    fpr, tpr, tresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color = clr, linestyle = ls, label = '%s (Obszar AUC = %0.2f)'% (label, roc_auc))
plt.legend(loc = "lower right")
plt.plot([0,1], [0,1], linestyle = '--', color = 'gray')
plt.xlim([0,1])
plt.ylim([0,1])
plt.grid(alpha = 0.5)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
plt.savefig("Roc_ost.png")