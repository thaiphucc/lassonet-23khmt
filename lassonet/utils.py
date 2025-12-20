from itertools import zip_longest
from typing import TYPE_CHECKING, Iterable, List

import scipy.stats
import torch

def eval_on_path(model, path, X_test, y_test, *, score_function=None):
    if score_function is None:
        score_fun = model.score # ClassifierMixin.score = mean accuracy
    else:
        assert callable(score_function)

        def score_fun(X_test, y_test):
            return score_function(y_test, model.predict(X_test))

    score = []
    for save in path:
        model.load(save)
        score.append(score_fun(X_test, y_test))
    return score