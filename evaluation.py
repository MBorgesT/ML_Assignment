from sklearn.neighbors import KNeighborsClassifier as kNC
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import AdaBoostClassifier as AdaC
from sklearn.ensemble import GradientBoostingClassifier as GBC

from sklearn.metrics import accuracy_score as acc
from sklearn.model_selection import cross_val_score

import numpy as np

from utils import test_model_min


nfolds = 5

# EVALUATION

# kNN Evaluation 
def knn_classifier_err(k,X_train,y_train,X_test,y_test):
    model=kNC(n_neighbors=k).fit(X_train,y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    return np.array([ 1 - acc(y_test,y_test_pred),
                      1 - acc(y_train,y_train_pred)])


def knn_classifier_auc(k, X, y):#
    model=kNC(n_neighbors=k)
    mean_auc = test_model_min(model, X, y, n_tests=10)
    # print('nbitches')
    # print(mean_auc)
    # print(np.array([mean_auc]))
    return np.array([mean_auc])


# Decision Tree Evaluation
def dt_classifier_score(X_train,y_train,X_test,y_test, max_depth=None, min_samples_split=2):
    # print(max_depth)
    model=DTC(
        criterion='entropy', 
        max_depth=max_depth, 
        min_samples_split=min_samples_split
        )
    model.fit(X_train,y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    return np.array([ 1 - acc(y_test,y_test_pred),
                      1 - acc(y_train,y_train_pred)])

# Decision Tree Evaluation
def dt_classifier_auc(X, y, max_depth=None, min_samples_split=2):
    model=DTC(criterion='entropy', max_depth=max_depth, min_samples_split=min_samples_split)
    mean_score = np.mean(cross_val_score(model, X, y, scoring='roc_auc', cv=nfolds))
    # print('bitch')
    # print(mean_score)
    # print(np.array([mean_score]))
    return np.array([mean_score])


def get_auc_gbc_trees(m, X, y):
    model=GBC(n_estimators=m, learning_rate=0.1,max_depth=3)
    return np.mean(cross_val_score(model, X, y, scoring='roc_auc', cv=nfolds))


def get_auc_ada_trees(m, X, y):
    model=AdaC(n_estimators=m)
    return np.mean(cross_val_score(model, X, y, scoring='roc_auc', cv=nfolds))


# def get_auc_tree_depth(X, y, max_depth=1, min_samples_split=2):
#     model=DTC(max_depth=max_depth, min_samples_split=min_samples_split)
#     return np.mean(cross_val_score(model, X, y, scoring='roc_auc', cv=nfolds))


# def get_auc_tree_sample(X, y, sample=2, max_depth=None):
#     model=DTC(max_depth=max_depth, min_samples_split=sample)
#     return np.mean(cross_val_score(model, X, y, scoring='roc_auc', cv=nfolds))