from itertools import zip_longest
from typing import TYPE_CHECKING, Iterable, List

import scipy.stats
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC

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

def eval_binary_metrics(dataset, lasso_sparse, path_sparse, X_test_selected, y_test, device):
    print("\n--- Binary Classification Evaluation ---")
        
    # Lấy model tốt nhất từ path_sparse (model đã retrained với lambda=0)
    best_model_state = path_sparse[0]
    lasso_sparse.load(best_model_state)
    
    # Dự đoán
    y_pred = lasso_sparse.predict(X_test_selected)
    X_test_tensor = torch.from_numpy(X_test_selected).to(device)
    lasso_sparse.model.eval()
    with torch.no_grad():
        logits = lasso_sparse.model(X_test_tensor)
        # Nếu output dim = 1 (Binary/BCEWithLogitsLoss), dùng sigmoid
        if logits.shape[1] == 1:
            probs = torch.sigmoid(logits).cpu().numpy()
            y_prob = probs.flatten()
        else:
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            y_prob = probs[:, 1]

        # Debug Data for ROC
        # print(f"y_test unique: {np.unique(y_test)}")
        # print(f"y_prob stats: min={y_prob.min()}, max={y_prob.max()}, mean={y_prob.mean()}")
        # print(f"y_test shape: {y_test.shape}, y_prob shape: {y_prob.shape}")

def calculate_and_plot_metrics(y_test, y_pred, y_prob=None, dataset=None, method_name="", roc_collection=None, eval_binary=False):
    # Metrics
    y_test = y_test.astype(int).flatten()
    y_pred = y_pred.astype(int).flatten()

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")

    if eval_binary:
        print(f'{y_prob.tolist() = }')
        prec = precision_score(y_test, y_pred, average='binary', pos_label=1)
        rec = recall_score(y_test, y_pred, average='binary', pos_label=1)
        f1 = f1_score(y_test, y_pred, average='binary', pos_label=1)
        
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1 Score: {f1:.4f}")

        if dataset and y_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label=1)
            roc_auc = auc(fpr, tpr)
            
            if roc_collection is not None:
                roc_collection.append({
                    'fpr': fpr,
                    'tpr': tpr,
                    'auc': roc_auc,
                    'method': method_name if method_name else "LassoNet" # Default for eval_binary_metrics
                })

            file_suffix = f"_{method_name.lower()}_roc_curve" if method_name else "_roc_curve"
            filename = f"{dataset}{file_suffix}.png"
            title = f"{method_name} ROC" if method_name else "ROC"

            # plt.figure(figsize=(8, 6))
            # plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            # plt.xlabel('False Positive Rate')
            # plt.ylabel('True Positive Rate')
            # plt.title(title)
            # plt.legend(loc="lower right")
            # plt.savefig(filename)
            # print(f"{title} curve saved to {filename}")
            # plt.close()
    return acc

def plot_comparison_roc(roc_collection, dataset):
    if not roc_collection:
        return

    plt.figure(figsize=(10, 8))
    
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']
    
    for i, data in enumerate(roc_collection):
        method = data['method']
        fpr = data['fpr']
        tpr = data['tpr']
        roc_auc = data['auc']
        color = colors[i % len(colors)]
        
        plt.plot(fpr, tpr, color=color, lw=2, label=f'{method} (area = {roc_auc:.2f})')
        
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc="lower right")
    filename = f'{dataset}_comparison_roc_curve.png'
    plt.savefig(filename)
    print(f"Comparison ROC curve saved to {filename}")
    plt.close()

def eval_binary_metrics(dataset, lasso_sparse, path_sparse, X_test_selected, y_test, device, roc_collection=None):
    print("\n--- Binary Classification Evaluation ---")
        
    # Lấy model tốt nhất từ path_sparse (model đã retrained với lambda=0)
    best_model_state = path_sparse[0]
    lasso_sparse.load(best_model_state)
    
    # Dự đoán
    y_pred = lasso_sparse.predict(X_test_selected)
    X_test_tensor = torch.from_numpy(X_test_selected).to(device)
    lasso_sparse.model.eval()
    with torch.no_grad():
        logits = lasso_sparse.model(X_test_tensor)
        # Nếu output dim = 1 (Binary/BCEWithLogitsLoss), dùng sigmoid
        if logits.shape[1] == 1:
            probs = torch.sigmoid(logits).cpu().numpy()
            y_prob = probs.flatten()
        else:
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            y_prob = probs[:, 1]

    calculate_and_plot_metrics(y_test, y_pred, y_prob, dataset, method_name="Decoder Network", roc_collection=roc_collection, eval_binary=True)

def train_extra_trees(X_train, y_train, X_test, y_test, dataset=None, roc_collection=None, eval_binary=False):
    print("\n--- ExtraTreesClassifier Downstream ---")
    clf = ExtraTreesClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    # Calculate probabilities for ROC
    y_prob = None
    if eval_binary:
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(X_test)
            if probs.shape[1] > 1:
                y_prob = probs[:, 1]
            else:
                if clf.classes_[0] == 1:
                    y_prob = np.ones(y_test.shape[0])
                else:
                    y_prob = np.zeros(y_test.shape[0])

    acc = calculate_and_plot_metrics(y_test, y_pred, y_prob, dataset, method_name="ExtraTrees", roc_collection=roc_collection, eval_binary=eval_binary)
    return clf, acc

def train_svc(X_train, y_train, X_test, y_test, dataset=None, roc_collection=None, eval_binary=False):
    print("\n--- SVC Downstream ---")
    clf = SVC(kernel='rbf', probability=True, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    # Calculate probabilities for ROC
    y_prob = None
    if eval_binary:
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(X_test)
            if probs.shape[1] > 1:
                y_prob = probs[:, 1]
            else:
                if clf.classes_[0] == 1:
                    y_prob = np.ones(y_test.shape[0])
                else:
                    y_prob = np.zeros(y_test.shape[0])

    acc = calculate_and_plot_metrics(y_test, y_pred, y_prob, dataset, method_name="SVC", roc_collection=roc_collection, eval_binary=eval_binary)
    return clf, acc