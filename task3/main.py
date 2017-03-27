import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import matplotlib.pyplot as plt

class uprls:
    def __init__(self, n_visible = 21, n_hidden=21, l1=0.0, l2=0.0, sparse_param=0.0, sparse_penalty=0.0, grad_coef=0.1):

        self.n_visible = n_visible
        self.n_hidden = n_hidden

        self.W_hid = theano.shared(value=np.ones((self.n_visible, self.n_hidden), dtype=theano.config.floatX), borrow=True)
        self.W_vis = theano.shared(value=np.ones(self.n_hidden, dtype=theano.config.floatX), borrow=True)

        self.b_vis = theano.shared(value=1.0, borrow=True)
        self.b_hid = theano.shared(value=np.ones(n_hidden, dtype=theano.config.floatX), borrow=True)

        self.grad_coeff = theano.shared(value=grad_coef, borrow=True)


        self.X = T.dmatrix(name='X')
        self.y = T.vector(name='y')

        self.l1 = l1
        self.l2 = l2
        self.sparse_param = sparse_param
        self.sparse_penalty=sparse_penalty
        self._params = [self.W_hid, self.b_hid, self.W_vis, self.b_vis]

        cost, updates = self.get_updates()

        self.training_step = theano.function([self.X, self.y], cost, updates=updates)
        self.hidden_values = theano.function([self.X], self.get_hidden_values())
        self.predict = theano.function([self.X], self.get_predictions())
        self.SMAPE = theano.function([self.X, self.y], self.calculate_SMAPE())
        self.grad_vector = theano.function([self.X, self.y], T.grad(self.calculate_loss(), self._params))
        self.loss_value = theano.function([self.X, self.y], self.calculate_loss())

    def get_hidden_values(self):
        return T.nnet.sigmoid(T.dot(self.X, self.W_hid) + self.b_hid)

    def get_predictions(self):
        #return T.nnet.sigmoid(T.dot(self.get_hidden_values(), self.W_vis) + self.b_vis)
        return T.dot(self.get_hidden_values(), self.W_vis) + self.b_vis

    def _regularization(self):
        return self.l1 * (T.sum(T.abs_(self.W_hid)) / (self.n_visible * self.n_hidden)
                               + T.sum(T.abs_(self.W_vis)) / (self.n_hidden * self.n_visible))\
               + self.l2 * (T.sum(T.sqr(self.W_hid)) / (self.n_visible * self.n_hidden)
                      + T.sum(T.sqr(self.W_vis)) / (self.n_hidden * self.n_visible))

    def _sparsity_regularization(self):
        a = np.array([self.sparse_param for i in np.linspace(1, self.n_hidden, self.n_hidden)])
        return self.sparse_penalty * T.nnet.categorical_crossentropy(T.mean(self.get_hidden_values(), axis=0), a)

    def calculate_SMAPE(self):
        predictions = self.get_predictions()
        return 200*T.mean(T.abs_(predictions - self.y)/(T.abs_(predictions) + T.abs_(self.y)))

    def calculate_loss(self):
             return self.calculate_SMAPE() + self._regularization() + self._sparsity_regularization()
            #return T.mean((self.y - self.get_predictions())**2)

    def get_updates(self):
        cost = self.calculate_loss()
        grad = T.grad(cost, self._params)
        norm = T.sum([vect.norm(2) for vect in grad])
        # updates = [(p, p - grad_coeff * grad) for p, grad in zip(self._params, grad)]
        updates = [(p, p - self.grad_coeff * grad) for p, grad in zip(self._params, grad)]
        return cost, updates

    def fit(self, X, y):
        print 'fitting'
        last_cost = -1
        for iter in xrange(30000):
            #index_iter = np.random.randint(len(y), size=int(len(y)*0.02))
            #new_cost = self.training_step(X[index_iter], y[index_iter])
            new_cost = self.training_step(X, y)
            if iter % 1000 == 0:
                print 'Training iteration {0}, loss {1}'.format(iter, new_cost)
            if new_cost != -1 and np.abs(new_cost - last_cost) < 0.0:
                print 'early stop on iteration ' + str(iter) + 'required accuracy achieved'
                print "last_cost = ", last_cost
                break
            if new_cost > 1.05 * last_cost and last_cost != -1:
                self.grad_coeff.set_value(self.grad_coeff.get_value()/5.0)
                print 'iteration ' + str(iter) + ' grad_coef: ' + str(self.grad_coeff.get_value())
            last_cost = new_cost


def SMAPE(target, prediction):
    return 200*np.mean(np.abs(target-prediction)/(np.abs(target) + np.abs(prediction)))


train = pd.read_csv("train.tsv")
test = pd.read_csv("test.tsv")
sample_submission = pd.read_csv("sample_submission.tsv")
train = train.sample(frac=0.1, random_state=42)

mms = MinMaxScaler()
X = mms.fit_transform(train.drop(['Num', 'y'], axis=1).as_matrix())
y = train['y'].values


print np.max(y)
def test_estimator(estimator):
    def do_stuff(X, y):
        tscv = TimeSeriesSplit(n_splits=3)
        score = []
        for train_index, test_index in tscv.split(X):
            estimator.fit(X[train_index], y[train_index])
            score.append(estimator.SMAPE(X[test_index], y[test_index]))
            print score[-1]
        return np.mean(score)
    return do_stuff(X, y),


print '\n', test_estimator(uprls(n_visible=X.shape[1], n_hidden=15000, grad_coef=100.0))
#for n in range(100, 3100, 100):
#    print '\n', test_estimator(uprls(n_visible=X.shape[1], n_hidden=n))

