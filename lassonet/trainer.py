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
            hidden_dims: Tuple of hidden layer sizes (excluding input/output).
            M: Hierarchy multiplier (hyperparameter).
            path_multiplier: epsilon (step size for increasing lambda).
            lambda_start: Initial penalty strength.
            optim_lr: Learning rate.
            epochs: Number of epochs (B) per inner loop. Can be int or tuple (init, path).
            device: 'cpu' or 'cuda'.
            verbose: Print progress.
            patience: Number of epochs to wait for improvement before early stopping (default 10).
        """
        self.hidden_dims = hidden_dims
        self.M = M
        self.path_multiplier = path_multiplier
        self.lambda_start = lambda_start
        self.optim_lr = optim_lr
        self.device = device
        self.verbose = verbose
        self.patience = patience
        self.batch_size = batch_size
        
        if isinstance(epochs, int):
            self.epoch_init = epochs
            self.epoch_path = epochs
        else:
            self.epoch_init, self.epoch_path = epochs

        # Handle optimizers
        # We define the factories here, but instantiation happens in fit/path
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

    # Predict after training model. Return numpy array
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
        Implements Algorithm 1: Training LassoNet
        Validation is performed along the path to select the best model.
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
             criterion = nn.CrossEntropyLoss()
             mode = 'multiclass'
        else:
             # Binary or Regression - user code assumed binary/regression setup
             output_dim = 1
             y_t = torch.as_tensor(y_train, dtype=torch.float32).reshape(-1, 1).to(self.device)
             criterion = nn.BCEWithLogitsLoss() # Assuming binary classif for now
             mode = 'binary'

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
        # Initialize and train FFNN on loss without penalty
        if self.verbose: print("Running Warm Start...")
        
        optimizer = self.optim_init(self.model.parameters())

        # If lambda_seq is provided, we might skip warm start or use it?
        # Standard: Warm start is useful to initialize weights.
        
        # --- Algorithm Step 2: Warm Start ---
        self._train(
            X_t, y_t, X_val_t, y_val_t, 
            optimizer, criterion, 
            self.epoch_init, 
            lambda_val=None, 
            mode=mode
        )

        
        best_val_score = -np.inf 
        best_state = None

        # Optimizer for path
        optimizer = self.optim_path(self.model.parameters())

        if lambda_seq is None:
            def lambda_generator():
                lam = self.lambda_start
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

            # For auto-path: only save if k decreases
            # For manual path: always save
            
            should_save = False
            if check_sparsity:
                # Auto path logic
                if current_k < k:
                    should_save = True
                    k = current_k
                
                # Check termination condition for auto path
                if k == 0:
                    should_save = True # Save the 0 feature model? Usually yes.
                    # But if we break immediately?
                    # The original logic saved if current_k < k.
                    break 
            else:
                should_save = True

            if should_save:
                if self.verbose:
                    print(f"Features active: {current_k} (lambda={lam:.5f}) Val Acc: {val_score:.4f}")

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

    def _train(self, X, y, X_val, y_val, optimizer, criterion, epochs, lambda_val=None, mode='binary'):
        """
        Inner training loop with optional validation and early stopping.
        """
        best_inner_val = -np.inf
        patience_counter = 0
        final_val_score = 0
        final_val_loss = 0

        # Prepare DataLoader if batch_size is set
        loader = None
        if self.batch_size is not None:
            dataset = TensorDataset(X, y)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # --- Algorithm Step 6: For b in 1...B ---
        for epoch in range(epochs):
            
            def train_one_step(X_batch, y_batch):
                # --- Algorithm Step 7 & 8: Gradient & Update ---
                self._train_step(X_batch, y_batch, optimizer, criterion)
                
                # Proximal update if lambda provided
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

            if loader is not None:
                for X_batch, y_batch in loader:
                    train_one_step(X_batch, y_batch)
            else:
                train_one_step(X, y)
            
            # Early Stopping Check (only if Validation Data provided)
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    out_val = self.model(X_val)
                    val_loss = criterion(out_val, y_val).item()
                    if mode == 'multiclass':
                        pred = out_val.argmax(dim=1)
                        acc = (pred == y_val).float().mean().item()
                    else:
                        pred = (out_val > 0).float()
                        acc = (pred == y_val).float().mean().item()
                    
                    val_score = acc
                self.model.train()
                
                final_val_score = val_score
                final_val_loss = val_loss

                if val_score > best_inner_val: # Improvement
                    best_inner_val = val_score
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if self.patience is not None and patience_counter >= self.patience:
                    # Early Stop
                    # if self.verbose: print(f"Early separating at epoch {epoch}")
                    break
        
        return final_val_score, final_val_loss

    def _train_step(self, X, y, optimizer, criterion):
        self.model.train()
        optimizer.zero_grad()
        output = self.model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()