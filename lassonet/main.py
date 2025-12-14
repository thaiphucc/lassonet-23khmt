
import sys
import os

# Add the parent directory to sys.path to resolve imports if running from lassonet directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lassonet.trainer import LassoNetClassifier
from lassonet.data_utils import load_dataset
from lassonet.utils import eval_on_path
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
PLOT_AVAILABLE = True
    
# helper to compute accuracy
def score_function(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def main():
    # Hardcoded Experiment Parameters
    BATCH_SIZE = 256
    EPOCHS = 1000
    LR = 1e-3
    PATIENCE = 10
    dataset = "MNIST"

    (X_train, y_train), (X_test, y_test) = load_dataset(dataset)
    # X_train = X_train[:50]
    # y_train = y_train[:50]

    print(f"Train data shape: {X_train.shape}")
    print(f"Train label shape: {y_train.shape}")
    
    input_dim = X_train.shape[1]
    output_dim = 10 # 10 classes for MNIST
    data_dim = X_test.shape[1]
    hidden_dim = (data_dim // 3,)
    
    print("Initializing LassoNetClassifier...")
    model = LassoNetClassifier(
        hidden_dims=hidden_dim, 
        epochs=EPOCHS,
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
    # Select the features
    import pickle
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