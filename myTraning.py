import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle

def data_split(data,ratio):
    np.random.seed(42)
    shuffiled = np.random.permutation(len(data))
    test_set_size =int(len(data) * ratio)
    test_indices = shuffiled[:test_set_size]
    train_indices =shuffiled[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]

if __name__ == '__main__':

    df = pd.read_csv('Data.csv')
    train, test = data_split(df, 0.2)
    X_train = train[['Fever', 'bodyPain', 'age', 'runnyNose', 'diffBreath']].to_numpy()
    X_test = test[['Fever', 'bodyPain', 'age', 'runnyNose', 'diffBreath']].to_numpy()
    Y_train = train[['infactionProb']].to_numpy().reshape(2080, )
    Y_test = test[['infactionProb']].to_numpy().reshape(519, )
    clf = LogisticRegression()
    clf.fit(X_train, Y_train)

    # open a file, where you ant to store the data
    file = open('model.pkl', 'wb')

    # dump information to that file
    pickle.dump(clf, file)
    file.close()


