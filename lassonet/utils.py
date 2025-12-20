from itertools import zip_longest
from typing import TYPE_CHECKING, Iterable, List

import scipy.stats
import torch

def eval_on_path(model, path, X_test, y_test, *, score_function=None):
    """
    Đánh giá mô hình trên toàn bộ regulariztion path (danh sách các snapshot).
    
    Args:
        model: Đối tượng LassoNetClassifier (hoặc tương tự) có phương thức load() và predict()/score().
        path: Danh sách các dictionary chứa trạng thái mô hình (state_dict) và thông tin khác.
        X_test: Dữ liệu kiểm tra (features).
        y_test: Nhãn kiểm tra.
        score_function: Hàm đánh giá tùy chỉnh (y_true, y_pred) -> float. 
                        Nếu None, sử dụng phương thức `model.score(X, y)`.
    
    Returns:
        score: Danh sách các điểm số (accuracy, v.v.) tương ứng với từng snapshot trong path.
    """
    if score_function is None:
        score_fun = model.score # ClassifierMixin.score = mean accuracy
    else:
        def score_fun(X_test, y_test):
            return score_function(y_test, model.predict(X_test))

    score = []
    for save in path:
        model.load(save)
        score.append(score_fun(X_test, y_test))
    return score