import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

k_max = 30
N = 100000
num_of_klasses = 4 # not changeble

average = [10*np.random.random(2) for i in range(num_of_klasses)]
disp = [2*np.random.random() for i in range(num_of_klasses)]

Data, klasses = [], []
for i in range(N):
    klasses.append(np.random.choice(range(num_of_klasses)))
    Data.append(np.random.normal(average[klasses[-1]], disp[klasses[-1]]))
Data = np.array(Data)

plt.subplot(221)
colors = ['r', 'g', 'b', 'y']
klass_diff = [[], [], [], []]
for i in range(len(Data)):
    klass_diff[klasses[i]].append(Data[i])
for i in range(len(klass_diff)):
    plt.scatter(np.array(klass_diff[i])[:,:1], np.array(klass_diff[i])[:,1:], color=colors[i], alpha=0.2)

print 'here'
k_acc = []
for i in range(k_max+1)[1:]:
    print i
    Classifier = KNeighborsClassifier(i)
    k_acc.append(np.mean(cross_val_score(Classifier, Data, klasses, cv=5)))


plt.subplot(222)
plt.plot(range(k_max+1)[1:], k_acc)
plt.xlabel('K nearest neighbours', fontsize=20)
plt.ylabel('accuracy', fontsize=20)
plt.show()