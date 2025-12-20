
import sys
import os
import pickle
# Add the parent directory to sys.path to resolve imports if running from lassonet directly
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
# Hardcoded Experiment Parameters
BATCH_SIZE = 256
EPOCHS = 1000
LR = 1e-3
PATIENCE = 10
dataset = "MNIST"
K = 50 # Our goal: select 50 features

# Biến debug để tìm lỗi tước khi chạy train thật
DEBUGGING = False

# helper to compute accuracy
def score_function(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def save_artifacts(estimator, X, y, prefix):
    path = estimator.path_results_
    if not path:
        return
        
    # Save pickle
    with open(f"{prefix}_path.pkl", "wb") as f:
        pickle.dump(path, f)
    
    # Eval and Plot
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
    return (X_train, y_train), (X_test, y_test)

def tune_M_downstream():
    (X_train, y_train), (X_test, y_test) = _load_dataset()
    
    data_dim = X_train.shape[1]
    hidden_dim = (data_dim // 3,)
    
    # K is global
    print(f"Target features K={K}")

    # Glob pkl files
    # Format: tune_M_{M_val}_{run_id}_path.pkl
    pkl_files = glob.glob("run_cv_3/tune_M_*_path.pkl")
    
    if not pkl_files:
        print("No pkl files found.")
        return

    print(f"Found {len(pkl_files)} pkl files: {pkl_files}")

    # Grid search Ms
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
        
        # Parse M from filename
        # tune_M_1_5811b978_path.pkl
        match = re.search(r"tune_M_(\d+)_", pkl_file)
        M_val = int(match.group(1))
        
        if M_val in done_Ms:
            continue
            
        print(f"Extracted M={M_val}")
        done_Ms.add(M_val)
        
        # Load path
        with open(pkl_file, "rb") as f:
            path = pickle.load(f)
            
        if not path:
            print("Empty path, skipping.")
            continue
            
        # Select features
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
            # Fallback
            desired_save = path[-1]
            theta = desired_save['theta']
            SELECTED_FEATURES = (np.linalg.norm(theta, axis=0) > 1e-5)
            
        selected_count = SELECTED_FEATURES.sum()
        print(f"Selected {selected_count} features.")
        
        if selected_count == 0:
             print("No features selected, skipping.")
             continue

        # Subset data
        X_train_selected = X_train[:, SELECTED_FEATURES]
        X_test_selected = X_test[:, SELECTED_FEATURES]

        # Train from scratch
        # Downstream Learner: decoder
        # "run lasso_sparse again (retrain)"
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

        # Evaluate the model on the test data
        scores = eval_on_path(lasso_sparse, path_sparse[:1], X_test_selected, y_test)
        print("Test accuracy (retrained):", scores[0])


def tune_M():
    (X_train, y_train), (X_test, y_test) = _load_dataset()
    
    data_dim = X_train.shape[1]
    hidden_dim = (data_dim // 3,)
    
    # 1. Define estimator
    model = LassoNetClassifier(
        hidden_dims=(3, ) if DEBUGGING else hidden_dim, 
        epochs=1 if DEBUGGING else EPOCHS,
        device=device,
        optim_lr=LR,
        verbose=False, # Reduce verbosity for grid search
        batch_size=BATCH_SIZE,
        patience=PATIENCE
    )

    # 2. Define custom scorer
    # We want to pick value of M that yields best accuracy at K features
    def scorer(estimator, X, y):
        path = estimator.path_results_
        
        # Save artifacts for this fold
        # Generate unique ID
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
             # If path is empty (model didn't converge or find sparse solutions), return 0
             return 0.0

        # Now we need to evaluate this specific step on the provided X, y (which is validation set in CV)
        # Load the weights
        estimator.load(selected_step)
        
        # Score
        return estimator.score(X, y)

    # 3. Setup GridSearchCV
    param_grid = {
        'M': [5, 10, 15]
    }
    
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        scoring=scorer,
        n_jobs=1, # GPU might conflict if parallel
        verbose=3
    )
    
    print("Starting GridSearchCV for M...")
    grid.fit(X_train, y_train)
    
    print("Best params:", grid.best_params_)
    print("Best CV Accuracy at K={}: {:.4f}".format(K, grid.best_score_))
    
    return grid.best_params_

def main():
    (X_train, y_train), (X_test, y_test) = _load_dataset()
    if DEBUGGING:
        X_train = X_train[:50]
        y_train = y_train[:50]

    print(f"Train data shape: {X_train.shape}")
    print(f"Train label shape: {y_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Test label shape: {y_test.shape}")
    
    input_dim = X_train.shape[1]
    output_dim = 10 # 10 classes for MNIST
    data_dim = X_test.shape[1]
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
    # fit returns self, path returns path_results
    path = model.path(X_train, y_train, validation_split=0.125)
    
    print("Training complete.")
    
    print("Evaluating regularization path...")
    
    
    accuracies = eval_on_path(model, path, X_test, y_test, score_function=score_function)
    n_features = [step['k'] for step in path]

    for k, acc in zip(n_features, accuracies):
        print(f"Features: {k}, Accuracy: {acc:.4f}")

    # Plotting
    K = 50 # Target features. Or iterate to find best.
    
    desired_save = None
    for save in path:
        # Check sparsity
        # theta is (input_dim, output_dim)
        theta = save['theta']
        mask = (np.linalg.norm(theta, axis=1) > 1e-5) # Axis 1 because skipped layers output is dim 0? 
    
        
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
        print("Plot saved to regularization_path.png")

            
    if desired_save is None:
        print("Could not find a model with <= {} features".format(K))
        # Fallback to last
        desired_save = path[-1]
        theta = desired_save['theta']
        SELECTED_FEATURES = (np.linalg.norm(theta, axis=0) > 1e-5)

    print("Number of selected features:", SELECTED_FEATURES.sum())

    # Select the features from the training and test data
    X_train_selected = X_train[:, SELECTED_FEATURES]
    X_test_selected = X_test[:, SELECTED_FEATURES]

    lasso_sparse = LassoNetClassifier(
        M=10,
        hidden_dims=hidden_dim, # Using original hidden dims configuration
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

    # Evaluate the model on the test data
    scores = eval_on_path(lasso_sparse, path_sparse[:1], X_test_selected, y_test, score_function=score_function)
    print("Test accuracy (retrained):", scores[0])

    # Save the path
    with open(f"{dataset}_path.pkl", "wb") as f:
        pickle.dump(path_sparse, f)
    # Skipping file save for now unless requested, or just print.


if __name__ == "__main__":
    main()
    # tune_M()
    # tune_M_downstream()