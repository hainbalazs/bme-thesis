from logical_ordinal import *
from pyeda.inter import *

class TransformLogical:
    def __init__(self, ruleset, landmark):
        self.landmark = landmark
        self.transformed_rules = []
        for rule in ruleset:
            terms = rule.split(" AND ")
            self.transformed_rules.append([LogicalOrdinalValue(term, landmark) for term in terms])

    def get_expr(self):
        result = ""
        for rule in self.transformed_rules:
            for terms in rule:
                result += terms.le
                result += " & "
            result = result.rstrip(" & ")
            result += " | "
        return result.rstrip(" | ")

    def add_ord_impl(self, x, y, const_list):
        dim = len(x.logical_v)
        x_and_y = np.bitwise_and(x.logical_v, y.logical_v)
        mask = np.zeros(dim).astype(int)
        mask[0] = 1
        mask[-1] = 1
        if np.count_nonzero(x_and_y) == 0:
            const_list.append(expr(f"{x.le} <=> ~ {y.le}"))
        elif np.count_nonzero(np.bitwise_and(x_and_y, mask)) != 0:
            if x.v2bin(x.logical_v) < y.v2bin(y.logical_v):
                const_list.append(expr(f"{x.le} => {y.le}"))

    def make_constraints(self):
        cat_constraints = dict()
        ord_constraints = []
        flat_rules = list(np.concatenate(self.transformed_rules).flat)
        # could be n*logn
        for x in flat_rules:
            if x.type == "categorical":
                attr = x.term.split()[0]
                op = x.le.split('_')[-1].startswith("i")
                if op:
                    if attr not in cat_constraints:
                        cat_constraints[attr] = list()
                    cat_constraints[attr].append(x.le)
            else:
                for y in flat_rules:
                    if y.type == "ordinal" and x.term.attr == y.term.attr and x != y:
                        self.add_ord_impl(x, y, ord_constraints)

        onehot_list = []
        for onehot in cat_constraints.values():
            if len(onehot) > 1:
                onehot_list.append(OneHot0(*onehot))

        onehot_expr = And(*onehot_list)
        impl_expr = And(*ord_constraints)
        return And(onehot_expr, impl_expr)

    def get_expr_w_constraints(self):
        main_expr = expr(self.get_expr())
        constraints = self.make_constraints()
        return And(main_expr, constraints)
