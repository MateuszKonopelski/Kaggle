# SET UP ENVIRONMENT
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import sklearn as sk

DIR = '../input/'
train_set = pd.read_csv(DIR + 'train.csv')
print('Import finished')

X = train_set.drop('label', axis=1)
X = X / 255.0
labels = train_set['label']
del train_set

print('X and labels created')


def reduce_byPCA(df, n):
    pca = PCA(n_components=n)
    pca.fit(df)
    df = pca.transform(df)

    columns = ['pca{}'.format(x) for x in range(n)]

    return pd.DataFrame(df, columns=columns)


print('MLA list created.')

print('\nPredicting Test data:')

for n in [5, 10, 15, 20, 25, 50, 75, 100]:
    X_train = reduce_byPCA(X, n)

    X_test = pd.read_csv(DIR + 'test.csv')
    X_test = X_test / 255.0
    X_test = reduce_byPCA(X_test, n)
    ImageId = np.arange(1, len(X_test.index) + 1)

    model = sk.neighbors.KNeighborsClassifier()

    Label = model.fit(X_train, labels).predict(X_test)
    prediction = pd.concat([pd.Series(ImageId), pd.Series(Label)], axis=1, ignore_index=True)
    prediction.columns = ['ImageId', 'Label']
    prediction_name = 'KNN_default_pca{}.csv'.format(n)
    prediction.to_csv(prediction_name, index=False)
    print('\tKNN with pca={} and default parameters has been created and exported.'.format(n))