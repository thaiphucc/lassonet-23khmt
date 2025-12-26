import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from functools import partial
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from .model import LassoNet
from .prox import inplace_prox, inplace_group_prox

class LassoNetClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_dims=(100,), M=10, path_multiplier=0.02, 
                 lambda_start=1e-4, optim_lr=1e-3, epochs=100, 
                 device='cpu', verbose=False, patience=10, batch_size=None):
        """
        Args:
            hidden_dims: Tuple kích thước các lớp ẩn (không bao gồm input/output).
            M: Hệ số phân cấp (siêu tham số) - Hierarchy multiplier.
            path_multiplier: epsilon (bước tăng lambda).
            lambda_start: Cường độ phạt ban đầu (lambda ban đầu).
            optim_lr: Tốc độ học (Learning rate).
            epochs: Số lượng epoch (B) cho mỗi vòng lặp con (inner loop). Có thể là int hoặc tuple (init, path).
            device: 'cpu' hoặc 'cuda'.
            verbose: In tiến trình.
            patience: Số epoch đợi cải thiện trước khi dừng sớm (early stopping) (mặc định 10).
        """
        self.hidden_dims = hidden_dims
        self.M = M
        self.path_multiplier = path_multiplier
        # self.lambda_start = lambda_start
        self.lambda_start = 'auto'
        self.optim_lr = optim_lr
        self.device = device
        self.verbose = verbose
        self.patience = patience
        self.batch_size = batch_size
        self.epochs = epochs
        
        if isinstance(epochs, int):
            self.epoch_init = epochs
            self.epoch_path = epochs
        else:
            self.epoch_init, self.epoch_path = epochs

        # Xử lý optimize
        # Định nghĩa factory tại đây, nhưng việc khởi tạo xảy ra trong fit/path
        self.optim_init = partial(torch.optim.Adam, lr=optim_lr)
        self.optim_path = partial(torch.optim.SGD, lr=optim_lr, momentum=0.9)

        self.model = None
        self.path_results_ = []

    def load(self, state_snapshot):
        """
        Load a state snapshot (dict) into the model.
        """
        self.model.layers.load_state_dict(state_snapshot['W'])
        self.model.skip.weight.data = torch.from_numpy(state_snapshot['theta']).to(self.device)

    # Dự đoán sau khi huấn luyện mô hình. Trả về mảng numpy
    def predict(self, X):
        self.model.eval()
        X_t = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            output = self.model(X_t)
            if output.shape[1] > 1:
                return output.argmax(dim=1).cpu().numpy()
            else:
                return (output > 0).float().cpu().numpy()

    def path(self, X, y, lambda_seq=None, validation_split=0.2):
        """
        Run the regularization path and return the results.
        """
        self.fit(X, y, lambda_seq=lambda_seq, validation_split=validation_split)
        return self.path_results_

    def fit(self, X, y, lambda_seq=None, validation_split=0.2):
        """
        Triển khai Algorithm 1: Training LassoNet
        Quá trình validation được thực hiện dọc theo đường dẫn (path) để chọn mô hình tốt nhất.
        """
        self.path_results_ = []

        # 1. Data Prep & Validation Split
        if validation_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split)
        else:
            X_train, X_val, y_train, y_val = X, None, y, None

        input_dim = X.shape[1]
        X_t = torch.as_tensor(X_train, dtype=torch.float32).to(self.device)
        
        unique_y = np.unique(y)
        if len(unique_y) > 2 and np.all(unique_y == np.arange(len(unique_y))):
             # Multiclass
             output_dim = len(unique_y)
             y_t = torch.as_tensor(y_train, dtype=torch.long).to(self.device)
             # Đã có softmax trong CrossEntropyLoss nên không cần layer softmax trong model
             criterion = nn.CrossEntropyLoss()
             self.mode = 'multiclass'
        else:
             # Binary hoặc Regression - code giả định thiết lập binary/regression
             output_dim = 1
             y_t = torch.as_tensor(y_train, dtype=torch.float32).reshape(-1, 1).to(self.device)
             criterion = nn.BCEWithLogitsLoss() # Tạm thời giả định phân loại nhị phân
             self.mode = 'binary'
        mode = self.mode 
        
        # Puts data on device for validation
        X_val_t = None
        y_val_t = None
        if X_val is not None:
             X_val_t = torch.as_tensor(X_val, dtype=torch.float32).to(self.device)
             if mode == 'multiclass':
                 y_val_t = torch.as_tensor(y_val, dtype=torch.long).to(self.device)
             else:
                 y_val_t = torch.as_tensor(y_val, dtype=torch.float32).reshape(-1, 1).to(self.device)


        # 2. Initialize Model using the PROVIDED class
        # dims format: [input_dim, hidden_1, ..., hidden_n, output_dim]
        dims = [input_dim] + list(self.hidden_dims) + [output_dim]
        self.model = LassoNet(*dims).to(self.device)

        # 3. Warm Start
        # Khởi tạo và huấn luyện mạng FFNN trên hàm mất mát không có phạt
        if self.verbose: print("Running Initial Training...")
        
        optimizer = self.optim_init(self.model.parameters())

        # --- Algorithm Step 2: Warm Start ---
        self._train(
            X_t, y_t, X_val_t, y_val_t, 
            optimizer, criterion, 
            self.epoch_init, 
            lambda_val=0, 
            mode=mode
        )

        
        best_val_score = -np.inf 
        best_state = None

        # Optimizer for path
        optimizer = self.optim_path(self.model.parameters())

        if self.lambda_start == 'auto':
            # chia cho 10 cho lần huấn luyện đầu tiên
            self.lambda_start_ = (
                self.model.lambda_start(M=self.M)
                / optimizer.param_groups[0]["lr"]
                / 10
            )
            print(f"lambda_start = {self.lambda_start_:.2e}")

        if lambda_seq is None:
            def lambda_generator():
                lam = self.lambda_start_
                while lam <= 1e4:
                    lam = (1 + self.path_multiplier) * lam
                    yield lam, True 
            
            path_iterator = lambda_generator()
        else:
            def lambda_generator():
                for lam in lambda_seq:
                    yield lam, False
            path_iterator = lambda_generator()

        k = input_dim # Start with all features active for auto-path logic
        
        for lam, check_sparsity in path_iterator:
            val_score, val_loss = self._train(
                X_t, y_t, X_val_t, y_val_t, 
                optimizer, criterion, 
                self.epoch_path, 
                lambda_val=lam, 
                mode=mode
            )

            # --- Algorithm Step 11: Update k ---
            with torch.no_grad():
                theta_norms = self.model.skip.weight.norm(p=2, dim=0)
                current_k = (theta_norms > 1e-5).sum().item()

            # Đối với auto-path: chỉ lưu nếu k giảm
            # Đối với manual path: luôn lưu
            
            should_save = False
            if check_sparsity:
                # Auto path logic
                if current_k < k:
                    should_save = True
                    k = current_k
                
                # Check termination condition for auto path
                if k == 0:
                    should_save = True
                    break 
            else:
                should_save = True

            if should_save:
                if self.verbose:
                    print(f"Features active: {current_k} (lambda={lam:.5f}) Val Score: {val_score:.4f}")

                state_snapshot = {
                    'k': current_k,
                    'lambda': lam,
                    'theta': copy.deepcopy(self.model.skip.weight.detach().cpu().numpy()),
                    'W': copy.deepcopy(self.model.layers.state_dict()),
                    'val_acc': val_score,
                    'val_loss': val_loss
                }
                self.path_results_.append(state_snapshot)
                
                if best_state is None or val_score > best_state['val_acc']:
                    best_state = state_snapshot

        
        # Load best model
        if best_state is not None:
             if self.verbose: print(f"Best model found with k={best_state['k']}, val_acc={best_state['val_acc']}")
             self.model.layers.load_state_dict(best_state['W'])
             pass 

        return self

    def _train(self, X, y, X_val, y_val, optimizer, criterion, epochs, lambda_val=None, mode=None):
        """
        Vòng lặp huấn luyện bên trong (Inner training loop) với tùy chọn validation và dừng sớm (early stopping).
        """
        best_inner_val = np.inf
        patience_counter = 0
        final_val_score = 0
        final_val_loss = 0

        n_train = len(X)
        batch_size = self.batch_size
        if batch_size is None:
            batch_size = n_train
            randperm = torch.arange
        else:
            randperm = torch.randperm
        batch_size = min(batch_size, n_train)

        def validation_obj(criterion):
            with torch.no_grad():
                out_val = self.model(X_val)
                return out_val, criterion(out_val, y_val).item() + lambda_val * self.model.l1_regularization_skip().item()

        # --- Algorithm Step 6: For b in 1...B ---
        for epoch in range(epochs):
            indices = randperm(n_train)
            self.model.train()
            def train_one_step(X_batch, y_batch):
                # --- Algorithm Step 7 & 8: Gradient & Update ---
                self._train_step(X_batch, y_batch, optimizer, criterion)
                
                # Cập nhật Proximal nếu lambda được cung cấp
                # --- Algorithm Step 9: Hierarchical Proximal Update ---
                if lambda_val is not None:
                    with torch.no_grad():
                        penalty = optimizer.param_groups[0]['lr'] * lambda_val
                        inplace_prox(
                            self.model.skip, 
                            self.model.layers[0], 
                            penalty, 
                            0,
                            self.M
                        )

            for i in range(n_train // batch_size):
                # không lấy batch không đầy đủ
                batch = indices[i * batch_size : (i + 1) * batch_size]
                train_one_step(X[batch], y[batch])
            
            # Early Stopping Check (only if Validation Data provided)
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    out_val, val_loss = validation_obj(criterion)
                    if mode == 'multiclass':
                        pred = out_val.argmax(dim=1)
                        acc = (pred == y_val).float().mean().item()
                    else:
                        pred = (out_val > 0).float()
                        acc = (pred == y_val).float().mean().item()
                    
                    val_score = acc
                
                final_val_score = val_score
                final_val_loss = val_loss

                if final_val_loss < best_inner_val: # Improvement
                    best_inner_val = final_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if self.patience is not None and patience_counter >= self.patience:
                    # Early Stop
                    # if self.verbose: print(f"Early separating at epoch {epoch}")
                    break
        
        return final_val_score, final_val_loss

    def _train_step(self, X, y, optimizer, criterion):
        optimizer.zero_grad()
        output = self.model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
