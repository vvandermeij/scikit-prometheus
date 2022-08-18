from sklearn import tree
from skprometheus.metrics import MetricRegistry
from sklearn.metrics import confusion_matrix
from prometheus_client import Gauge


class DecisionTreeClassifier(tree.DecisionTreeClassifier):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for i in ['TP', 'FP', 'TN', 'FN']:
            MetricRegistry.add_gauge(
                f"{i}_gauge_train",
                f"Gauge value of {i} on training data",
                additional_labels=("score", "data_set")
            )

    def fit(self, X, y):
        super().fit(X, y)
        y_pred = self.predict(X)

        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        MetricRegistry.TN_gauge_train(score='TN', data_set='train').set(tn)
        MetricRegistry.FP_gauge_train(score='FP', data_set='train').set(fp)
        MetricRegistry.FN_gauge_train(score='FN', data_set='train').set(fn)
        MetricRegistry.TP_gauge_train(score='TP', data_set='train').set(tp)

        return self
