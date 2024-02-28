import os
import numpy as np
import pandas as pd
import cvxpy as cvx
import cylp
from itertools import product
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin

from beam_search import beam_search, beam_search_K1


class BooleanRuleCG(BaseEstimator, ClassifierMixin):
    """BooleanRuleCG is a directly interpretable supervised learning method
    for binary classification that learns a Boolean rule in disjunctive
    normal form (DNF) or conjunctive normal form (CNF) using column generation (CG).
    AIX360 implements a heuristic beam search version of BRCG that is less
    computationally intensive than the published integer programming version [#NeurIPS2018]_.

    References:
        .. [#NeurIPS2018] `S. Dash, O. Günlük, D. Wei, "Boolean decision rules via
           column generation." Neural Information Processing Systems (NeurIPS), 2018.
           <https://papers.nips.cc/paper/7716-boolean-decision-rules-via-column-generation.pdf>`_
    """
    def __init__(self,
        lambda0=0.001,
        lambda1=0.001,
        omegaFP=1,
        omegaFN=1,
        CNF=False,
        iterMax=100,
        K=10,
        D=10,
        B=16,
        eps=1e-6,
        solver='CBC',
        verbose=False,
        silent=False):
        """
        Args:
            lambda0 (float, optional): Complexity - fixed cost of each clause
            lambda1 (float, optional): Complexity - additional cost for each literal
            omegaFP (float, optional): Cost of each false positive "clause" in a sample
            omegaFN (float, optional): Cost of each false negative "clause" in a sample
            CNF (bool, optional): CNF instead of DNF
            iterMax (int, optional): Column generation - maximum number of iterations
            K (int, optional): Column generation - maximum number of columns generated per iteration
            D (int, optional): Column generation - maximum degree
            B (int, optional): Column generation - beam search width
            eps (float, optional): Numerical tolerance on comparisons
            solver (str, optional): Linear programming - solver
            verbose (bool, optional): Linear programming - verboseness
            silent (bool, optional): Silence overall algorithm messages
        """
        # Complexity parameters
        self.lambda0 = lambda0      # fixed cost of each clause
        self.lambda1 = lambda1      # additional cost per literal
        # Confusion tuning parameters
        self.omegaFP = omegaFP
        self.omegaFN = omegaFN
        # CNF instead of DNF
        self.CNF = CNF
        # Column generation parameters
        self.iterMax = iterMax      # maximum number of iterations
        self.K = K                  # maximum number of columns generated per iteration
        self.D = D                  # maximum degree
        self.B = B                  # beam search width
        # Numerical tolerance on comparisons
        self.eps = eps
        # Linear programming parameters
        self.solver = solver        # solver
        self.verbose = verbose      # verboseness
        # Silence output
        self.silent = silent

    def solve_with_aix360(self, X, y):
        self.solver = "ECOS"
        """Fit model to training data.

               Args:
                   X (DataFrame): Binarized features with MultiIndex column labels
                   y (array): Binary-valued target variable
               Returns:
                   BooleanRuleCG: Self
               """
        if not self.silent:
            print('Learning {} rule with complexity parameters lambda0={}, lambda1={}' \
                  .format('CNF' if self.CNF else 'DNF', self.lambda0, self.lambda1))
        if self.CNF:
            # Flip labels for CNF
            y = 1 - y
        # Positive (y = 1) and negative (y = 0) samples
        P = np.where(y > 0.5)[0]
        Z = np.where(y < 0.5)[0]
        nP = len(P)
        n = len(y)

        # Initialize with empty and singleton conjunctions, i.e. X plus all-ones feature
        # Feature indicator and conjunction matrices
        z = pd.DataFrame(np.eye(X.shape[1], X.shape[1] + 1, 1, dtype=int), index=X.columns)
        A = np.hstack((np.ones((X.shape[0], 1), dtype=int), X))
        # Iteration counter
        self.it = 0

        # Formulate master LP
        # Variables
        w = cvx.Variable(A.shape[1], nonneg=True)
        xi = cvx.Variable(nP, nonneg=True)
        # Objective function (no penalty on empty conjunction)
        cs = self.lambda0 + self.lambda1 * z.sum().values
        cs[0] = 0
        obj = cvx.Minimize(cvx.sum(xi * self.omegaFN) / n + cvx.sum(A[Z, :] * w * self.omegaFP) / n + cs * w)
        # Constraints
        constraints = [xi + A[P, :] * w >= 1]

        # Solve problem
        prob = cvx.Problem(obj, constraints)
        prob.solve(solver=self.solver, verbose=self.verbose)
        if not self.silent:
            print('Initial LP solved')

        # Extract dual variables
        r = np.ones_like(y, dtype=float) / n
        r[P] = -constraints[0].dual_value

        # Beam search for conjunctions with negative reduced cost
        # Most negative reduced cost among current variables
        UB = np.dot(r, A) + cs
        UB = min(UB.min(), 0)
        v, zNew, Anew = beam_search(r, X, self.lambda0, self.lambda1,
                                    K=self.K, UB=UB, D=self.D, B=self.B, eps=self.eps)

        while (v < -self.eps).any() and (self.it < self.iterMax):
            # Negative reduced costs found
            self.it += 1
            if not self.silent:
                print('Iteration: {}, Objective: {:.4f}'.format(self.it, prob.value))

            # Add to existing conjunctions
            z = pd.concat([z, zNew], axis=1, ignore_index=True)
            A = np.concatenate((A, Anew), axis=1)

            # Reformulate master LP
            # Variables
            w = cvx.Variable(A.shape[1], nonneg=True)
            # Objective function
            cs = np.concatenate((cs, self.lambda0 + self.lambda1 * zNew.sum().values))
            obj = cvx.Minimize(cvx.sum(xi * self.omegaFN) / n + cvx.sum(A[Z, :] * w * self.omegaFP) / n + cs * w)
            # Constraints
            constraints = [xi + A[P, :] * w >= 1]

            # Solve problem
            prob = cvx.Problem(obj, constraints)
            prob.solve(solver=self.solver, verbose=self.verbose)
            print("\nAfter this iteration the optimal value is", prob.value)
            print("\nAfter this iteration the optimal w is", w.value)
            # Extract dual variables
            r[P] = -constraints[0].dual_value

            # Beam search for conjunctions with negative reduced cost
            # Most negative reduced cost among current variables
            UB = np.dot(r, A) + cs
            UB = min(UB.min(), 0)
            v, zNew, Anew = beam_search(r, X, self.lambda0, self.lambda1,
                                        K=self.K, UB=UB, D=self.D, B=self.B, eps=self.eps)

        # Save generated conjunctions and LP solution
        self.z = z
        self.wLP = w.value
        sum1 = np.sum(xi.value * self.omegaFN) / n
        sum2 = np.sum(A[Z, :] * w.value * self.omegaFP) / n
        sum3 = np.sum(cs * w.value)
        r = np.full(nP, 1. / n)
        self.w = beam_search_K1(r, pd.DataFrame(1 - A[P, :]), 0, A[Z, :].sum(axis=0) / n + cs,
                                UB=r.sum(), D=100, B=2 * self.B, eps=self.eps, stopEarly=False)[1].values.ravel()
        if len(self.w) == 0:
            self.w = np.zeros_like(self.wLP, dtype=int)

        return sum1, sum2, sum3


    def fit(self, X, y):
        """
        # preprocessing
        local_wbm, local_obj = self.solve_with_aix360(X, y)
        # z = pd.DataFrame(np.array(list(product([0, 1], repeat=X.shape[1]))).T, index=X.columns)
        z = pd.DataFrame(columns=X.columns)
        i = 0
        for possible_clause in product([0, 1], repeat=X.shape[1]):
            if self.lambda0 + self.lambda1 * np.array(possible_clause).sum() < local_obj:
                z.loc[i] = np.array(possible_clause)
                i += 1
                print(i)

        org_size = 2 ** X.shape[1]
        print(f"Search space has been narrowed by {(1 - i/org_size * 100)}% (from {org_size} to {i})")
        """
        """Fit model to training data.

        Args:
            X (DataFrame): Binarized features with MultiIndex column labels
            y (array): Binary-valued target variable
        Returns:
            BooleanRuleCG: Self
        """
        if not self.silent:
            print('Learning {} rule with complexity parameters lambda0={}, lambda1={}'\
                  .format('CNF' if self.CNF else 'DNF', self.lambda0, self.lambda1))
        if self.CNF:
            # Flip labels for CNF
            y = 1 - y
        # Positive (y = 1) and negative (y = 0) samples
        P = np.where(y > 0.5)[0]
        Z = np.where(y < 0.5)[0]
        nP = len(P)
        n = len(y)

        # Initialize with empty and singleton conjunctions, i.e. X plus all-ones feature
        # Feature indicator and conjunction matrices
        z = pd.DataFrame(np.array(list(product([0, 1], repeat=X.shape[1]))).T, index=X.columns)
        A = 1 - (np.dot(1 - X, z) > 0)
        #A = np.hstack((np.ones((X.shape[0],z.shape[1]-X.shape[1]), dtype=int), X))

        # Iteration counter
        self.it = 0

        # Formulate master LP
        # Variables
        w = cvx.Variable(A.shape[1], integer=True)
        xi = cvx.Variable(nP, integer=True)
        # Objective function (no penalty on empty conjunction)
        cs = self.lambda0 + self.lambda1 * z.sum().values
        cs[0] = 0
        obj = cvx.Minimize(cvx.sum(xi * self.omegaFN) / n + cvx.sum(A[Z,:] * w * self.omegaFP) / n + cs * w)
        # Constraints
        constraints = [xi + A[P,:] * w >= 1, xi >= 0, xi <= 1, w >= 0, w <= 1]

        # Solve problem
        prob = cvx.Problem(obj, constraints)
        prob.solve(solver=self.solver, verbose=self.verbose)
        if not self.silent:
            print('Initial LP solved')

        # Extract variables
        sum1 = np.sum(xi.value * self.omegaFN) / n
        sum2 = np.sum(A[Z, :] * w.value * self.omegaFP) / n
        sum3 = np.sum(cs * w.value)

        # Save generated conjunctions and LP solution
        self.z = z
        self.wLP = w.value
        print(prob.value)
        print(self.wLP)

        self.w = self.wLP

        print(f"self.w \n {self.w} \n")
        if len(self.w) == 0:
            self.w = np.zeros_like(self.wLP, dtype=int)

        return sum1, sum2, sum3

    def get_master_solution(self, wlp, local_complexity, l0, l1):
        eps = 1e-4  # custom approximation for each dataset will probably be needed
        ref = np.delete(wlp, np.where(wlp < eps))  # removing values very close to 0
        ref[::-1].sort()  # sorting in desc order

        complexity = 0
        i = 0
        while complexity < local_complexity:
            clause_idx = np.where(wlp == ref[i])[0].item()
            np.delete(wlp, clause_idx)
            z_sum = self.z.loc[:, clause_idx].sum()
            complexity += l0 + l1 * z_sum
            i += 1

        thresh = ref[i]
        return np.array([1 if x > thresh else 0 for x in wlp])

    def get_local_complexity(self, wbm, l0, l1):
        eps = 1e-4
        complexity = 0
        for clause_idx, clause in enumerate(wbm):
            if clause > eps:
                z_sum = self.z.loc[:, clause_idx].sum()
                complexity += l0 + l1 * z_sum

        return complexity


    def compute_conjunctions(self, X):
        """Compute conjunctions of features as specified in self.z.

        Args:
            X (DataFrame): Binarized features with MultiIndex column labels
        Returns:
            array: A -- Conjunction values
        """
        try:
            A = 1 - (np.dot(1 - X, self.z) > 0) # Changed matmul to dot, because failed on some machines
        except AttributeError:
            print("Attribute 'z' does not exist, please fit model first.")
        return A

    def predict(self, X):
        """Predict class labels.

        Args:
            X (DataFrame): Binarized features with MultiIndex column labels
        Returns:
            array: y -- Predicted labels
        """
        # Compute conjunctions of features
        A = self.compute_conjunctions(X)
        # Predict labels
        if self.CNF:
            # Flip labels since model is actually a DNF for Y=0
            return 1 - (np.dot(A, self.w) > 0)
        else:
            return (np.dot(A, self.w) > 0).astype(int)

    def explain(self, maxConj=None, prec=2):
        """Return rules comprising the model.

        Args:
            maxConj (int, optional): Maximum number of conjunctions to show
            prec (int, optional): Number of decimal places to show for floating-value thresholds
        Returns:
            Dictionary containing

            * isCNF (bool): flag signaling whether model is CNF or DNF
            * rules (list): selected conjunctions formatted as strings
        """
        # Selected conjunctions
        z = self.z.loc[:, self.w > 0.5]
        truncate = (maxConj is not None) and (z.shape[1] > maxConj)
        nConj = maxConj if truncate else z.shape[1]

        """
        if self.CNF:
            print('Predict Y=0 if ANY of the following rules are satisfied, otherwise Y=1:')
        else:
            print('Predict Y=1 if ANY of the following rules are satisfied, otherwise Y=0:')
        """

        # Sort conjunctions by increasing order
        idxSort = z.sum().sort_values().index[:nConj]
        # Iterate over sorted conjunctions
        conj = []
        for i in idxSort:
            # MultiIndex of features participating in rule i
            idxFeat = z.index[z[i] > 0]
            # String representations of features
            strFeat = idxFeat.get_level_values(0) + ' ' + idxFeat.get_level_values(1)\
                + ' ' + idxFeat.get_level_values(2).to_series()\
                .apply(lambda x: ('{:.' + str(prec) + 'f}').format(x) if type(x) is float else str(x))
            # String representation of rule
            strFeat = strFeat.str.cat(sep=' AND ')
            conj.append(strFeat)

        return {
            'isCNF': self.CNF,
            'rules': conj
        }
