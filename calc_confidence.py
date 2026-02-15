import torch
import numpy as np
import dflow

def predict_with_confidence(model, loss_func, device='cpu', num_seeds=5):
    """
    Instead of a single prediction, running the optimization multiple times 
    with different random seeds.
    
    Returns:
        - final_x1: The generated points from the last run.
        - stats: A dictionary with mean_loss and std_dev.
    """
    losses = []
    print(f"\n--- Calculating Confidence (Running {num_seeds} times) ---")
    
    for i in range(num_seeds):
        

        hist_x0, hist_x1 = dflow.dflow(model, loss_func, steps=20, batch_size=500, 
                                       device=device, optmz_steps=300)
        
        
        final_x1 = torch.tensor(hist_x1[-1], device=device)
        final_loss = loss_func(final_x1).item()
        losses.append(final_loss)

    
    losses = np.array(losses)
    mean_loss = np.mean(losses)
    std_dev = np.std(losses)
    
    confidence_score = std_dev
    
    print(f"Mean Loss: {mean_loss:.2e} | Std Dev: {std_dev:.2e}")
    print(f"Computed Confidence Score: {confidence_score:.4f}")
    
    return hist_x1[-1], {"mean": mean_loss, "std": std_dev}