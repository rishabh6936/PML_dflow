import dflow
import circle_losses
import visualizations
import calc_confidence  
import torch
import numpy as np
import os

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)


def main():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu" 
    
    print(f"Using device: {device}")
    
    required_folders = ["dflow_gifs", "explain_dflow", "path_dflow"]
    for folder in required_folders:
        os.makedirs(folder, exist_ok=True)
        print(f"Ensured folder exists: {folder}")
    trained_fm_model = dflow.fm_training(device=device)

    
    tasks = {
        "Standard_Circle": circle_losses.calc_circle_loss,
        "Donut_Ring":      circle_losses.donut_loss,
        "Crescent_Moon":   circle_losses.crescent_loss,
        "Repulsion":     circle_losses.repulsion_loss 
    }

    print(f"\n--- MILESTONE 3 EVALUATION ---\n")

    results_table = []

    for name, loss_func in tasks.items():
        print(f"--> Processing: {name}")
        
        # DFlow
        hist_x0, hist_x1 = dflow.dflow(trained_fm_model, loss_func=loss_func, device=device)

        visualizations.visualise_dual_optimization(hist_x0, hist_x1, filename=f"dflow_gifs/dflow_{name}.gif")
        visualizations.visualise_trajectory(hist_x0, filename=f"explain_dflow/explain_{name}.png")
        
        _, stats = calc_confidence.predict_with_confidence(
            trained_fm_model, loss_func, device=device, num_seeds=5
        )
        final_optimized_x0 = hist_x0[-1] 
        visualizations.visualise_integration_path(trained_fm_model, final_optimized_x0, device=device, filename=f"path_dflow/flow_path_{name}.png")

 
        results_table.append({
            "Shape": name,
            "Confidence": stats['std'],             
            "MeanLoss": stats['mean']     
        })
        print(f"--> Finished {name}!\n")

   
    print("\n" + "="*50)
    print(f"{'SHAPE':<20} | {'CONFIDENCE':<12} | {'MEAN LOSS':<10}")
    print("-" * 50)
    for res in results_table:
        print(f"{res['Shape']:<20} | {res['Confidence']:.4f}   | {res['MeanLoss']:.2e}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()