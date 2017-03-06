from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

boston = load_boston()
X_train, X_test, Y_train, Y_test = train_test_split(boston.data, boston.target, test_size=0.25, random_state=42)

class leave:
    def __init__(self, X, Y):
        min_err = None
        min_err_split = None
        for i in range(len(X[0])):
            barriers = np.linspace(np.min(X[:, i]), np.max(X[:, i]), 100)
            for barrier in barriers:
                new_err, mean1, mean2 = self.error(X, Y, i, barrier)
                if not min_err or min_err > new_err:
                    min_err = new_err
                    self.mean1 = mean1
                    self.mean2 = mean2
                    min_err_split = (i, barrier)
        self.my_split = min_err_split
        self.my_error = min_err
        self.my_children = []

    def error(self, X, Y, index, barrier):
        left = [Y[i] for i in range(len(Y)) if X[i][index] <= barrier]
        right = [Y[i] for i in range(len(Y)) if X[i][index] > barrier]
        mean1 = np.mean(left)
        mean2 = np.mean(right)
        return 1.0 / len(Y) * (np.sum(map(lambda x: (x - mean1) ** 2, left))
                               + np.sum(map(lambda x: (x - mean2) ** 2, right))), mean1, mean2

    def split(self, X, Y):
        return np.array([X[i] for i in range(len(Y)) if X[i][self.my_split[0]] <= self.my_split[1]]), \
               np.array([Y[i] for i in range(len(Y)) if X[i][self.my_split[0]] <= self.my_split[1]]), \
                np.array([X[i] for i in range(len(Y)) if X[i][self.my_split[0]] > self.my_split[1]]), \
               np.array([Y[i] for i in range(len(Y)) if X[i][self.my_split[0]] > self.my_split[1]])

    def get_direction(self, elem):
        if elem[self.my_split[0]] <= self.my_split[1]:
            return 'left'
        else:
            return 'right'



class tree:
    def __init__(self, max_depth=5):
        self.first_leave = None
        self.max_depth = max_depth

    def fit(self, X, Y, cur_depth=0, parrent_leave=None):
        new_leave = leave(X, Y)
        if not parrent_leave:
            self.first_leave = new_leave
        else:
            parrent_leave.my_children.append(new_leave)
        if cur_depth < self.max_depth:
            X_left, Y_left, X_right, Y_right = new_leave.split(X, Y)
            if len(X_left) == 0 or len(X_right) == 0:
                new_leave.my_children = None
            else:
                self.fit(X_left, Y_left, cur_depth+1, new_leave)
                self.fit(X_right, Y_right, cur_depth+1, new_leave)
        else:
            new_leave.my_children = None

    def predict(self, X, cur_leave=None):
        return np.array([self.elem_value(elem) for elem in X])

    def elem_value(self, elem, cur_leave=None):
        if not cur_leave:
            cur_leave = self.first_leave
        direction = cur_leave.get_direction(elem)
        if cur_leave.my_children:
            if direction == 'left':
                return self.elem_value(elem, cur_leave.my_children[0])
            else:
                return self.elem_value(elem, cur_leave.my_children[1])
        else:
            if direction == 'left':
                return cur_leave.mean1
            else:
                return cur_leave.mean2


train_loss = []
test_loss = []
N = 25

for i in range(N+1)[2:]:
    print i
    model = tree(i)
    model.fit(X_train, Y_train)
    test_prediction =  model.predict(X_test)
    train_prediction = model.predict(X_train)
    train_loss.append(
        np.nanmean(map(lambda i: (train_prediction[i] - Y_train[i]) ** 2, range(len(Y_train)))))
    test_loss.append(
        np.nanmean(map(lambda i: (test_prediction[i] - Y_test[i]) ** 2, range(len(Y_test)))))

print test_loss
plt.plot(range(N+1)[2:], train_loss, label='train loss')
plt.plot(range(N+1)[2:], test_loss, label='test loss')
plt.legend()
plt.show()


