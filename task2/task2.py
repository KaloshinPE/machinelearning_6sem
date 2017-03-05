import numpy as np
from sklearn import tree
import pydotplus
from IPython.display import Image


def get_data():
    f = open('data/german.data-numeric.txt', 'r')
    data = []
    for line in f:
        new_line = []
        for elem in line.split(' '):
            if elem != '' and elem != '\n':
                new_line.append(elem)
        data.append(new_line)
    return np.array(data)[:, :-1], np.array(data)[:, -1]


def loss(true, predicted):
    loss = 0
    for i in range(len(true)):
        if true[i] == 2 and predicted[i] == 1:
            loss += 5
        if true[i] == 1 and predicted[i] == 2:
            loss += 1
    return loss


features, classes = get_data()
model = tree.DecisionTreeRegressor()
model.fit(features, classes)
dot_data = tree.export_graphviz(model, out_file="small_tree.out",
                         class_names=classes,
                         filled=True, rounded=True,
                         special_characters=True)

graph = pydotplus.graphviz.graph_from_dot_file("small_tree.out")
Image(graph.create_png())



