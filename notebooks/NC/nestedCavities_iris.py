from boolean_rule_cg_global import *
from aix360.algorithms.rbm import FeatureBinarizer
from pyeda.inter import *
from logical_ordinal import *
from transform_logical import *
from sklearn.metrics import accuracy_score


class NestedCavities:
    def __init__(self, dataset, cat_list, drop_list, target_n, target_v, FPcost, FNcost, complexity, complexity_rate,
                 max_iter):
        self.df = dataset
        self.categories = cat_list
        self.to_drop = drop_list
        self.target_n = target_n
        self.target_v = target_v
        self.c_fp = FPcost
        self.c_fn = FNcost
        self.c = complexity
        self.c_rate = complexity_rate
        self.max_iter = max_iter

    def process(self, data, cat_list, drop_list, target_n, target_v, pos):
        """
        FeatureBinarizer throws an exception if the dataset which it needs to binarize consist of only 2 or less rows.
        The workaround is to inject a dummy sample to the dataset, then remove it after binarization.
        """
        trunkAtEnd = 0
        if len(data) < 3:
            dummy_dims = 3 - len(data)
            trunkAtEnd = dummy_dims
            df_padded = pd.DataFrame(columns=data.columns)
            for i in range(dummy_dims):
                df_padded.loc[i] = np.zeros(len(data.columns))
            data = pd.concat([df_padded, data])

        self.fb = FeatureBinarizer(colCateg=cat_list, numThresh=2, negations=True, returnOrd=True)
        X = data.drop(columns=drop_list)
        X_bin, X_std = self.fb.fit_transform(X)

        if pos:
            Y = data[target_n].map(lambda x: 1 if x == target_v else 0).astype(int)
        else:
            Y = data[target_n].map(lambda x: 1 if x != target_v else 0).astype(int)

        if trunkAtEnd > 0:
            return X_bin[trunkAtEnd:], Y[trunkAtEnd:]

        return X_bin, Y

    def fit_cavity(self, bin_v, target_v, complexity, FPcost, FNcost):

        br_model = BooleanRuleCG(complexity, complexity, omegaFP=FPcost, omegaFN=FNcost, CNF=False)
        # Train, print, and evaluate model
        sum1, sum2, sum3 = br_model.fit(bin_v, target_v)
        with open("sum_logger.txt", "a+") as sum_logger:
            sum_logger.write("New iteration \n")
            sum_logger.write(str(sum1) + ', ' + str(sum2) + ', ' + str(sum3) + '\n')

        return br_model

    def filter_data(self, dataset, bin_v, rules, pos):

        target = 1 if pos else 0
        filtered_df = pd.DataFrame(columns=dataset.columns)

        for i, sample in bin_v.iterrows():
            if rules.predict(sample) == target:
                filtered_df.loc[i] = dataset.loc[i]

        return filtered_df

    def subtract_rules(self, ruleset, neg_rules):
        sub_rules = TransformLogical(neg_rules, self.fb.thresh)
        if ruleset is None:
            return sub_rules.get_expr()
        else:
            return Xor(ruleset, sub_rules.get_expr(), simplify=False)

    def good_enough(self, prev_set, current_set, n_iter, end):
        # gondolkodni kell rajta:
        """
              - detektalni mikor ugyanazt rakjuk ki es be
                - osszevetni az elozo iteracioval az eredmeny halmazt, ha ugyanaz vege
              - osszes eredetileg pozitivkent jelolt mintat kivalasztottuk (valoszinuleg tulilleszt)
                - szamoljuk meg az algoritmus elejen hany pozitiv van
                - ha annyi pozitiv van a mi halmazunkban, es az osszes pozitiv akkor vege
              - szabalyok komplexitasara megkotes?
                - szabalyok komplexitasa ~ aix
              """

        if n_iter == self.max_iter or current_set.equals(prev_set) or end:
            return True
        else:
            return False

    def evaluate(self, filtered_data, stats_holder, dnf_rules):
        # processing the Y values [dataset - predicted]
        Y_pred = []
        for i, row in self.df.iterrows():
            if i in filtered_data.index:
                Y_pred.append(1)
            else:
                Y_pred.append(0)

        Y_org = self.df[self.target_n].map(lambda x: 1 if x == self.target_v else 0).astype(int)

        # evaluating confusion values
        conf_matrix = confusion_matrix(Y_org, Y_pred)
        conf = []
        for i, count in enumerate(conf_matrix.flatten()):
            conf.append(count)
        stats_holder["confusion"].append(conf)

        # evaluating accuracy
        stats_holder["accuracy"].append(accuracy_score(Y_org, Y_pred))

        # evaluating complexity
        c = 0
        for phase in dnf_rules:
            for conj in phase:
                c += 1
                terms = conj.split(" AND ")
                for term in terms:
                    c += 1
        stats_holder["complexity"].append(c)

        return stats_holder

    def learn(self):
        dnf_ruleset = []
        tr_ruleset = None
        model_stats = dict()
        model_stats["accuracy"] = []
        model_stats["complexity"] = []
        model_stats["confusion"] = []
        prev_samples = pd.DataFrame()
        pos_class_samples = pd.DataFrame(columns=self.df.columns)
        dataset = self.df
        n_iter = 1
        end = False
        while not self.good_enough(prev_samples, pos_class_samples, n_iter, end):
            # getting all the positives + some false positives
            tr_X, tr_Y = self.process(dataset, self.categories, self.to_drop, self.target_n, self.target_v, True)

            print(f"Phase {n_iter} / I.")

            rules = self.fit_cavity(tr_X, tr_Y, complexity=self.c, FNcost=self.c_fn, FPcost=self.c_fp)
            filtered_data = self.filter_data(dataset, tr_X, rules, True)
            removed_data1 = self.filter_data(dataset, tr_X, rules, False)

            dnf_ruleset.append(rules.explain()['rules'])
            tr_ruleset = self.subtract_rules(tr_ruleset, rules.explain()['rules'])
            self.evaluate(pd.concat([pos_class_samples, filtered_data]), model_stats, dnf_ruleset)

            if filtered_data.groupby(by=self.target_n).size()[self.target_v] == len(filtered_data):
                print("Filtered out all the negative values")
                end = True

            self.c *= 0.8
            if not end:
                # removing all the negatives + some false negatives
                tr_X, tr_Y = self.process(filtered_data, self.categories, self.to_drop, self.target_n,
                                          self.target_v, False)

                print(f"Phase {n_iter} / II.")

                rules = self.fit_cavity(tr_X, tr_Y, complexity=self.c, FNcost=self.c_fn, FPcost=self.c_fp)
                filtered_data2 = self.filter_data(filtered_data, tr_X, rules, False)
                removed_data2 = self.filter_data(filtered_data, tr_X, rules, True)

                dnf_ruleset.append(rules.explain()['rules'])
                tr_ruleset = self.subtract_rules(tr_ruleset, rules.explain()['rules'])

                # end of the iteration
                prev_samples = pos_class_samples
                pos_class_samples = pd.concat([pos_class_samples, filtered_data2])
                self.c *= 0.5
                dataset = removed_data2
                self.evaluate(pos_class_samples, model_stats, dnf_ruleset)
                n_iter += 1

                if self.target_v not in removed_data2[self.target_n].unique():
                    print("Filtered out all the negative values, finishing.")
                    end = True

            else:
                pos_class_samples = pd.concat([pos_class_samples, filtered_data])

        return pos_class_samples, dnf_ruleset, tr_ruleset, model_stats



def rule_set_transform(rules):
    conjunction = 0
    literal = 0
    for rule in rules:
        conjunction += 1
        literal_list = rule.split("AND")
        for l in literal_list:
            literal += 1

    return conjunction, literal


def calc_dnf_complexity(rules, lambda0=1, lambda1=1):
    if len(rules) == 0:
        return 0

    if type(rules[0][0]) is str:
        conj, lit = rule_set_transform(rules)

    complexity = conj * lambda0 + lit * lambda1
    return complexity


def calc_conf_props(y_, y_pred_):
    conf_matrix = confusion_matrix(y_, y_pred_).flatten()
    tpr = conf_matrix[3]
    fpr = conf_matrix[1]
    tnr = conf_matrix[0]
    fnr = conf_matrix[2]

    return tpr, fpr, tnr, fnr


def compare_aix(data, cat_list, drop_list, target_n, target_v):
    # binarizing
    fb = FeatureBinarizer(colCateg=cat_list, numThresh=2, negations=True, returnOrd=True)
    X = data.drop(columns=drop_list)
    X_bin, X_std = fb.fit_transform(X)
    Y = data[target_n].map(lambda x: 1 if x == target_v else 0).astype(int)

    # training
    brcg_metrics = {'accuracy': [], 'complexity': [], 'dnf': [], 'tp': [], 'fp': [], 'tn': [], 'fn': [], 'lambda': []}
    number_of_models_to_train = 6
    lambda_values = np.linspace(0.000001, 0.01, number_of_models_to_train)

    for i, l in enumerate(np.flip(lambda_values)):
        print("iter:", i)

        br_model = BooleanRuleCG(l, l, omegaFP=1, omegaFN=1, CNF=False)
        sum1, sum2, sum3 = br_model.solve_with_aix360(X_bin, Y)
        y_pred = br_model.predict(X_bin)
        rules = br_model.explain()['rules']
        complexity = calc_dnf_complexity(rules)
        accuracy = accuracy_score(Y, y_pred)
        true_pos, false_pos, true_neg, false_neg = calc_conf_props(Y, y_pred)

        brcg_metrics['accuracy'].append(accuracy)
        brcg_metrics['complexity'].append(complexity)
        brcg_metrics['dnf'].append(rules)
        brcg_metrics['tp'].append(true_pos)
        brcg_metrics['fp'].append(false_pos)
        brcg_metrics['tn'].append(true_neg)
        brcg_metrics['fn'].append(false_neg)
        brcg_metrics['lambda'].append(f"l0, l1 = {l}")

        with open("sum_logger.txt", "a+") as sum_logger:
            sum_logger.write("New iteration \n")
            sum_logger.write("lambda: " + str(l) + "\n")
            sum_logger.write(str(sum1) + ', ' + str(sum2) + ', ' + str(sum3) + '\n')

    return brcg_metrics
