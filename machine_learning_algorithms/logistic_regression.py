
from sklearn.linear_model import LogisticRegression


def logistic_regression_process(trains, targets, *args, **kwargs):
    if 'sample_weight' in kwargs:
        sample_weight = kwargs.pop('sample_weight')
        lr = LogisticRegression(**kwargs)
        lr.fit(trains, targets, sample_weight=sample_weight)
    else:
        lr = LogisticRegression(**kwargs)
        lr.fit(trains, targets)
    return lr
