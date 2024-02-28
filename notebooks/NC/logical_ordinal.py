import string
import numpy as np


class LogicalOrdinalValue:
    class Term:
        def __init__(self, expr):
            parts = expr.split()
            if len(parts) != 3:
                raise ValueError

            self.value = float(parts[2])
            self.op = parts[1]
            self.attr = parts[0]

        def get_expr(self):
            return f"{self.attr} {self.op} {self.value}"

    def __init__(self, expr, landmarks):
        print(expr.split())
        if expr.split()[0] not in landmarks.keys():
            self.term = expr
            self.type = "categorical"
            self.logical_v = None
            self.le = self.name_cat_value(expr)
        else:
            self.term = self.Term(expr)
            self.type = "ordinal"
            self.logical_v = self.v(self.term, landmarks[self.term.attr]).astype(int)
            self.le = f"{self.term.attr}_{self.v2bin(self.logical_v)}"

    def v(self, term, landmark):
        value = term.value
        op = term.op
        # ezen meg mindig kell dolgozni....
        result = np.zeros(19)
        for i, thr in enumerate(landmark):
            if op == ">":
                if value < landmark[i]:
                    result[i] = 1
            elif op == "<=":
                if value >= landmark[i]:
                    result[i] = 1
            else:  # =
                if value < landmark[i] and i != 0:
                    result[i - 1] = 1
                    break

        return result

    def v2bin(self, arr):
        return np.sum((2 ** np.arange(arr.shape[0])) * arr).astype(int)

    def name_cat_value(self, expr):
        # naming: attr_c_[in]v
        parts = expr.split()
        op = "i" if len(parts) > 1 and (parts[1] == "==" or parts[1] == "not") else "n"
        if len(parts) == 3:
            c = parts[2].translate(str.maketrans('', '', string.punctuation))
        else:
            c = "0"
        return f"{parts[0]}_c_{op}{c}"
