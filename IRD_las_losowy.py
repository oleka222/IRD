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





from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 500, ccp_alpha=0, max_depth=4, min_samples_leaf=10,
                       random_state=1)


forest.fit(X_train, y_train)
from sklearn. model_selection import cross_val_score
scores = cross_val_score(estimator = forest, X = X_train, y = y_train, cv = 10, n_jobs = -1)
print(np.mean(scores), np.std(scores))
print(forest.score(X_test, y_test))



from sklearn.model_selection import learning_curve
train_sizes, train_scores, test_scores = learning_curve(estimator = forest, X = X_train, y = y_train, train_sizes = np.linspace(0.1, 1, 10), cv=10, n_jobs = -1)
train_mean = np.mean(train_scores, axis = 1)
train_std = np.std(train_scores, axis = 1)
test_mean = np.mean(test_scores, axis = 1)
test_std = np.std(test_scores, axis = 1)


import matplotlib.pyplot as plt

import sklearn.metrics as metrics
# calculate the fpr and tpr for all thresholds of the classification
probs = forest.predict_proba(X_test)
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
plt.savefig("ROC3.png")

plt.show()

feat_labels = justX.columns

importances = forest.feature_importances_
indicies = np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f+1, 30, feat_labels[indicies[f]], importances[indicies[f]]))
    
plt.title("Istotnosć cech")
plt.bar(range(X_train.shape[1]), importances[indicies], align = 'center')
plt.xticks(range(X_train.shape[1]), feat_labels[indicies])
plt.xlim([-1, X_train.shape[1]])
plt.savefig("czynniki_istotne")

y_pred3 = forest.predict(X_test)
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(forest, X_test, y_test)
plt.savefig("macierz konfuzji.png")
