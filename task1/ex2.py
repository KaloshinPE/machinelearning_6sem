from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn import naive_bayes

cancer = datasets.load_breast_cancer()
digits = datasets.load_digits()

classifiers = [naive_bayes.BernoulliNB(), naive_bayes.MultinomialNB(), naive_bayes.GaussianNB()]
means = [[], []]
for i in range(3):
    means[0].append(cross_val_score(classifiers[i], cancer.data, cancer.target, cv=5).mean())
    means[1].append(cross_val_score(classifiers[i], digits.data, digits.target, cv=5).mean())

print means