from sklearn.base import BaseEstimator
import os
os.environ["OMP_NUM_THREADS"] = "1"
from lasagne.easy import SimpleNeuralNet
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import theano


class Regressor(BaseEstimator):

    def __init__(self):

        self.clf = Pipeline([
            ('scaler', StandardScaler()),
            ('neuralnet', SimpleNeuralNet(nb_hidden_list=[100],
                                          is_classification=False,
                                          max_nb_epochs=30,
                                          batch_size=256,
                                          learning_rate=1.)),
        ])

    def fit(self, X, y):
        X = X.astype(theano.config.floatX)
        self.clf.fit(X, y)
        return self

    def predict(self, X):
        X = X.astype(theano.config.floatX)
        return self.clf.predict(X)
