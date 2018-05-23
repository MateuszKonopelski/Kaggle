import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt

from sklearn.decomposition import PCA
import sklearn as sk
import sklearn.ensemble
import sklearn.linear_model
import sklearn.naive_bayes
import sklearn.discriminant_analysis


############################################################


DIR = '../input/'
train_set = pd.read_csv(DIR + 'train.csv')
print('Import finished')

X =  train_set.drop('label', axis=1)
X = X/255.0
labels = train_set['label']
del train_set


############################################################

def reduce_byPCA(df, n):
    pca = PCA(n_components=n)
    pca.fit(df)
    df = pca.transform(df)

    columns = ['pca{}'.format(x) for x in range(n)]

    return pd.DataFrame(df, columns=columns)


############################################################


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit

MLA_AT_columns = ['pca','best1_score', 'best1_param', 'best2_score','best2_param', 'best3_score', 'best3_param']
MLA_AT = pd.DataFrame(columns=MLA_AT_columns)


def GridSearch_method(model, model_name, params):

    n_pca = 150
    X_pca = reduce_byPCA(X, n_pca)

    cv_split = ShuffleSplit(n_splits=4, test_size=.25, train_size=.75, random_state=8)

    clf =  GridSearchCV(model, params, cv=cv_split, return_train_score=False).fit(X_pca, labels)

    gs_results = pd.DataFrame(clf.cv_results_).loc[:, ['mean_test_score', 'rank_test_score', 'params']].sort_values \
        (by='rank_test_score')

    MLA_AT.loc[model_name, 'pca'] = n_pca
    for rank in [1, 2, 3]:
        MLA_AT.loc[model_name, 'best{}_score'.format(rank)] = clf.cv_results_['mean_test_score'][rank -1]
        MLA_AT.loc[model_name, 'best{}_param'.format(rank)] = str(clf.cv_results_['params'][rank - 1])

    print(clf.best_params_)
    return gs_results

############################################################

params = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
         'multi_class': ['ovr', 'multinomial'],
         'solver': ['newton-cg'],
         'tol' : [0.1, 10],
         'penalty' : ['l2']}

LCV = GridSearch_method(model=sk.linear_model.LogisticRegression(),
                        model_name='sk.linear_model.LogisticRegression',
                        params=params)

LCV.to_csv('LCV.csv', index=False)
MLA_AT.to_csv('MLA_AT_LCV.csv', index=False
