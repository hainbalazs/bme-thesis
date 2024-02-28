from boolean_rule_cg_global import *
from aix360.algorithms.rbm import FeatureBinarizer
from sklearn import metrics

data = pd.read_csv("../../datasets/Iris.csv")

data = data.drop(data[data["Species"] == "Iris-setosa"].index)

fb = FeatureBinarizer(numThresh=2, colCateg=[], negations=True, returnOrd=True)

X = data.drop(columns=["Id", "Species"])
X_bin, X_std = fb.fit_transform(X)

Y = data["Species"].map(lambda x: 1 if x == "Iris-versicolor" else 0).astype(int)

def explain_with_BRCG(X, y, CNF=False, lambda0=1e-2, lambda1=1e-2, verbose=True):
    # Instantiate BRCG with small complexity penalty
    br_model = BooleanRuleCG(lambda0, lambda1, CNF=False)
    # Train, print, and evaluate model
    br_model.fit(X, y)
    if verbose:
        print('Training accuracy:', metrics.accuracy_score(y, br_model.predict(X)))
    if br_model.CNF:
        print('Predict Y=0 if ANY of the following rules are satisfied, otherwise Y=1:')
    else:
        print('Predict Y=1 if ANY of the following rules are satisfied, otherwise Y=0:')
    print(br_model.explain()['rules'])
    return br_model

rules = explain_with_BRCG(X_bin, Y)

rules.explain()['rules']
