import dflow
import circle_losses
import visualizations
import torch

def main():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available(): 
        device = "mps"
    else:
        device = "cpu"
    
    print(f"Using device: {device}")
    
    #step 1 is to train the fm model
    trained_fm_model = dflow.fm_training(device=device)

    tasks = {
        "Standard_Circle": circle_losses.calc_circle_loss,
        "Donut_Ring":      circle_losses.donut_loss,
        "Crescent_Moon":   circle_losses.crescent_loss,
        "Repulsion":     circle_losses.repulsion_loss 
    }

    print(f"\n--- Starting Batch Generation for {len(tasks)} shapes ---\n")

    for name, loss_func in tasks.items():
        print(f"--> Processing: {name}")
        
        # optimize
        hist_x0, hist_x1 = dflow.dflow(trained_fm_model, loss_func=loss_func, device=device)

        filename = f"dflow_{name}.gif"
        
        # visualize
        visualizations.visualise_dual_optimization(hist_x0, hist_x1, filename=filename)
        print(f"--> Finished {name}! Saved to {filename}\n")

    print("All animations generated successfully.")



if __name__ == "__main__":
    main()


