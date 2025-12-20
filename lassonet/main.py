
import sys
import os
import pickle
# Thêm thư mục cha vào sys.path để giải quyết việc import nếu chạy trực tiếp từ thư mục lassonet
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lassonet.trainer import LassoNetClassifier
from lassonet.data_utils import load_dataset
from lassonet.utils import eval_on_path
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import matplotlib.pyplot as plt
import torch
import uuid
import time
import glob
import re

device = "cuda" if torch.cuda.is_available() else "cpu"
PLOT_AVAILABLE = True
# Các tham số thí nghiệm được fix cứng (Hardcoded Experiment Parameters)
BATCH_SIZE = 256
EPOCHS = 1000
LR = 1e-3
PATIENCE = 10
dataset = "MNIST"
K = 50 # Mục tiêu của chúng ta: chọn 50 đặc trưng

# Biến debug để tìm lỗi tước khi chạy train thật
DEBUGGING = False

# Hàm hỗ trợ tính toán độ chính xác (accuracy)
def score_function(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def save_artifacts(estimator, X, y, prefix):
    path = estimator.path_results_
    if not path:
        return
        
    # Lưu file pickle
    with open(f"{prefix}_path.pkl", "wb") as f:
        pickle.dump(path, f)
    
    # Đánh giá và Vẽ biểu đồ (Eval and Plot)
    if PLOT_AVAILABLE:
        try:
            accuracies = eval_on_path(estimator, path, X, y, score_function=score_function)
            n_features = [step['k'] for step in path]
            lambdas = [step['lambda'] for step in path]

            fig, ax1 = plt.subplots(figsize=(10, 6))
            color = 'tab:blue'
            ax1.set_xlabel('Number of Features')
            ax1.set_ylabel('Classification Accuracy', color=color)
            ax1.plot(n_features, accuracies, marker='o', color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.grid(True)
            
            ax2 = ax1.twinx()
            color = 'tab:red'
            ax2.set_ylabel('Lambda', color=color)
            ax2.plot(n_features, lambdas, marker='x', linestyle='--', color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            
            plt.title(f'Regularization Path: M={estimator.M}')
            plt.savefig(f"{prefix}.png")
            plt.close(fig)
            print(f"Saved artifacts to {prefix}.png/pkl")
        except Exception as e:
            print(f"Failed to plot/save artifacts: {e}")

def _load_dataset():
    (X_train, y_train), (X_test, y_test) = load_dataset(dataset)
    if DEBUGGING:
        X_train = X_train[:20]
        y_train = y_train[:20]
        X_test = X_test[:20]
        y_test = y_test[:20]
    return (X_train, y_train), (X_test, y_test)

def tune_M_downstream():
    """
    Huấn luyện downstream trên các (M, tập đặc trưng) được chọn bởi tune_M(), để đánh giá accuracy cho phân lớp
    """
    (X_train, y_train), (X_test, y_test) = _load_dataset()
    
    data_dim = X_train.shape[1]
    hidden_dim = (data_dim // 3,)
    
    # K là biến toàn cục
    print(f"Target features K={K}")

    # Tìm các file pkl (Glob)
    # Định dạng: tune_M_{M_val}_{run_id}_path.pkl
    pkl_files = glob.glob("run_cv_3/tune_M_*_path.pkl")
    
    if not pkl_files:
        print("No pkl files found.")
        return

    print(f"Found {len(pkl_files)} pkl files: {pkl_files}")

    # Grid search các giá trị M
    target_Ms = {5, 10, 15}
    processed_Ms = set()
    files_to_process = []
    
    for pkl_file in sorted(pkl_files):
        match = re.search(r"tune_M_(\d+)_", pkl_file)
        if match:
             M_val = int(match.group(1))
             if M_val in target_Ms and M_val not in processed_Ms:
                 files_to_process.append(pkl_file)
                 processed_Ms.add(M_val)
    
    print(f"Filtered files to process (Target M={target_Ms}): {files_to_process}")

    done_Ms = set()

    for pkl_file in files_to_process:
        print(f"\nProcessing {pkl_file}...")
        
        # Lấy M từ tên file
        # tune_M_1_5811b978_path.pkl
        match = re.search(r"tune_M_(\d+)_", pkl_file)
        M_val = int(match.group(1))
        
        if M_val in done_Ms:
            continue
            
        print(f"Extracted M={M_val}")
        done_Ms.add(M_val)
        
        # Load đường dẫn (path)
        with open(pkl_file, "rb") as f:
            path = pickle.load(f)
            
        if not path:
            print("Empty path, skipping.")
            continue
            
        # Chọn đặc trưng
        desired_save = None
        for save in path:
            theta = save['theta']
            mask = (np.linalg.norm(theta, axis=0) > 1e-5)
            k_feat = mask.sum()
            if k_feat <= K:
                desired_save = save
                SELECTED_FEATURES = mask
                break
        
        if desired_save is None:
            # Dự phòng
            desired_save = path[-1]
            theta = desired_save['theta']
            SELECTED_FEATURES = (np.linalg.norm(theta, axis=0) > 1e-5)
            
        selected_count = SELECTED_FEATURES.sum()
        print(f"Selected {selected_count} features.")
        
        if selected_count == 0:
             print("No features selected, skipping.")
             continue

        # Lấy tập con dữ liệu
        X_train_selected = X_train[:, SELECTED_FEATURES]
        X_test_selected = X_test[:, SELECTED_FEATURES]

        # Huấn luyện lại từ đầu
        # Downstream Learner: decoder
        # "chạy lasso_sparse lần nữa (huấn luyện lại)"
        lasso_sparse = LassoNetClassifier(
            M=M_val,
            hidden_dims=(3, ) if DEBUGGING else hidden_dim, 
            verbose=False, 
            device=device,
            epochs=1 if DEBUGGING else EPOCHS,
            batch_size=BATCH_SIZE,
            optim_lr=LR
        )
        
        print("Retraining with selected features...")
        
        path_sparse = lasso_sparse.path(
            X_train_selected,
            y_train,
            validation_split=0.125,
            lambda_seq=[0]
        )

        # Đánh giá mô hình trên tập kiểm tra
        scores = eval_on_path(lasso_sparse, path_sparse[:1], X_test_selected, y_test)
        print("Test accuracy (retrained):", scores[0])


def tune_M():
    """
    Cross-Validation 3-fold để lựa chọn model lựa chọn đặc trưng (LassoNet) tốt nhất 
    bằng Grid-Search trên siêu tham số M
    """
    (X_train, y_train), (X_test, y_test) = _load_dataset()
    
    data_dim = X_train.shape[1]
    hidden_dim = (data_dim // 3,)
    
    # 1. Định nghĩa estimator
    model = LassoNetClassifier(
        hidden_dims=(3, ) if DEBUGGING else hidden_dim, 
        epochs=1 if DEBUGGING else EPOCHS,
        device=device,
        optim_lr=LR,
        verbose=False, # Giảm độ chi tiết (verbosity) cho grid search
        batch_size=BATCH_SIZE,
        patience=PATIENCE
    )

    # 2. Định nghĩa custom scorer
    # Chúng ta muốn chọn giá trị M mang lại độ chính xác tốt nhất tại K feature
    def scorer(estimator, X, y):
        path = estimator.path_results_
        
        # Lưu các artifact cho fold này
        # Tạo ID duy nhất
        run_id = str(uuid.uuid4())[:8]
        M_val = estimator.M
        prefix = f"tune_M_{M_val}_{run_id}"
        save_artifacts(estimator, X, y, prefix)
        
        best_step = None
        
        selected_step = None
        for step in path:
            if step['k'] <= K:
                selected_step = step
                break
        
        if not path:
             # Nếu path rỗng (mô hình không hội tụ hoặc không tìm thấy giải pháp thưa), trả về 0.0
             return 0.0

        # Bây giờ chúng ta cần đánh giá bước cụ thể này trên X, y được cung cấp (đây là tập validation trong CV)
        # Load trọng số
        estimator.load(selected_step)
        
        # Tính điểm
        return estimator.score(X, y)

    # 3. Thiết lập GridSearchCV
    param_grid = {
        'M': [5, 10, 15]
    }
    
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        scoring=scorer,
        n_jobs=1, # GPU có thể xung đột nếu chạy song song
        verbose=3
    )
    
    print("Starting GridSearchCV for M...")
    grid.fit(X_train, y_train)
    
    print("Best params:", grid.best_params_)
    print("Best CV Accuracy at K={}: {:.4f}".format(K, grid.best_score_))
    
    return grid.best_params_

def main():
    """
    Với các tham số trên (biến toàn cục), chọn đặc trưng trên LassoNet => huấn luyện decoder (downstream learner)  
    """
    (X_train, y_train), (X_test, y_test) = _load_dataset()
    if DEBUGGING:
        X_train = X_train[:50]
        y_train = y_train[:50]

    print(f"Train data shape: {X_train.shape}")
    print(f"Train label shape: {y_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Test label shape: {y_test.shape}")
    
    input_dim = X_train.shape[1]
    output_dim = 10 # 10 lớp cho MNIST
    data_dim = X_test.shape[1]
    # 1 Tầng ẩn: k/3 (như trong bài báo)
    hidden_dim = (data_dim // 3,)
    
    print("Initializing LassoNetClassifier...")
    model = LassoNetClassifier(
        hidden_dims=(3,) if DEBUGGING else hidden_dim, 
        epochs=1 if DEBUGGING else EPOCHS,
        M=10, 
        device=device,
        optim_lr=LR,
        verbose=True,
        batch_size=BATCH_SIZE,
        patience=PATIENCE
    )
    print("Training model...")
    # fit trả về self, path trả về path_results
    path = model.path(X_train, y_train, validation_split=0.125)
    
    print("Training complete.")
    
    print("Evaluating regularization path...")
    
    
    accuracies = eval_on_path(model, path, X_test, y_test, score_function=score_function)
    n_features = [step['k'] for step in path]

    for k, acc in zip(n_features, accuracies):
        print(f"Features: {k}, Accuracy: {acc:.4f}")

    # Plotting
    K = 50 # Số feature mục tiêu. Hoặc lặp để tìm số tốt nhất.
    
    # Lưu
    desired_save = None
    for save in path:
        # Kiểm tra tính thưa (sparsity)
        # theta có shape (input_dim, output_dim)
        theta = save['theta']
        mask = (np.linalg.norm(theta, axis=0) > 1e-5)
        k = mask.sum()
        
        if k <= K:
            desired_save = save
            SELECTED_FEATURES = mask
            break
    
    # for step in path:
    #     # Load state
    #     model.model.layers.load_state_dict(step['W'])
    #     # 'theta' is numpy array, need to convert to tensor and put in skip.weight
    #     model.model.skip.weight.data = torch.from_numpy(step['theta']).to(model.device)
        
    #     # Predict
    #     model.model.eval()
    #     with torch.no_grad():
    #         logits = model.model(X_test_t)
    #         predictions = torch.argmax(logits, dim=1).numpy()
        
    #     acc = accuracy_score(y_test, predictions)
        
    #     n_features.append(step['k'])
    #     accuracies.append(acc)
    #     print(f"Features: {step['k']}, Accuracy: {acc:.4f}")

    # Plotting
    if PLOT_AVAILABLE:
        lambdas = [step['lambda'] for step in path]
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        color = 'tab:blue'
        ax1.set_xlabel('Number of Features')
        ax1.set_ylabel('Classification Accuracy', color=color)
        ax1.plot(n_features, accuracies, marker='o', color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True)
        
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Lambda', color=color)
        ax2.plot(n_features, lambdas, marker='x', linestyle='--', color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title('Regularization Path: Accuracy & Lambda vs Features')
        plt.savefig('regularization_path.png')
        print("Biểu đồ đã được lưu vào regularization_path.png")

            
    if desired_save is None:
        print("Không tìm thấy model với <= {} features".format(K))
        # Fallback về model cuối cùng
        desired_save = path[-1]
        theta = desired_save['theta']
        SELECTED_FEATURES = (np.linalg.norm(theta, axis=0) > 1e-5)

    print("Number of selected features:", SELECTED_FEATURES.sum())

    # Chọn các đặc trưng từ dữ liệu huấn luyện và kiểm tra
    X_train_selected = X_train[:, SELECTED_FEATURES]
    X_test_selected = X_test[:, SELECTED_FEATURES]

    lasso_sparse = LassoNetClassifier(
        M=10,
        hidden_dims=hidden_dim, # Sử dụng cấu hình hidden dims gốc
        verbose=True,
        device=device,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        optim_lr=LR
    )
    
    print("Retraining with selected features...")
    
    path_sparse = lasso_sparse.path(
        X_train_selected,
        y_train,
        validation_split=0.125,
        lambda_seq=[0]
    )

    # Đánh giá mô hình trên tập kiểm tra
    scores = eval_on_path(lasso_sparse, path_sparse[:1], X_test_selected, y_test, score_function=score_function)
    print("Test accuracy (retrained):", scores[0])

    # Lưu với đường dẫn
    with open(f"{dataset}_path.pkl", "wb") as f:
        pickle.dump(path_sparse, f)

if __name__ == "__main__":
    main()
    # tune_M()
    # tune_M_downstream()