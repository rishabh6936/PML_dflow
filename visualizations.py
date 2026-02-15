import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import torch

def visualise_dual_optimization(history_x0, history_x1, filename="dflow_dual_process.gif"):
    """
    Animating the Input (x0) and Output (x1) side-by-side gif.
    """
    print(f"Generating animation: {filename}...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor('white') # dark background
    
    # Left Plot for x0 
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-4, 4)
    ax1.set_facecolor('black')
    ax1.set_title("Input ($x_0$): Finding the Optimized Input", color='white')
    scat1 = ax1.scatter([], [], s=5, c='magenta', alpha=0.6, edgecolors='none')
    ax1.grid(True, alpha=0.2)
    
    # Right plot for x1 
    ax2.set_xlim(-5, 5)
    ax2.set_ylim(-5, 5)
    ax2.set_facecolor('black')
    ax2.set_title("Output ($x_1$): Forming the Circle", color='white')
    scat2 = ax2.scatter([], [], s=5, c='cyan', alpha=0.6, edgecolors='none')
    
    circle = plt.Circle((0, 0), 2.0, color='red', fill=False, linestyle='--', alpha=0.5, linewidth=2)
    ax2.add_artist(circle)
    
    def update(frame_idx):
        scat1.set_offsets(history_x0[frame_idx])
        scat2.set_offsets(history_x1[frame_idx])
        return scat1, scat2

    ani = animation.FuncAnimation(fig, update, frames=len(history_x0), interval=30, blit=True)
    ani.save(filename, writer='pillow', fps=20)
    print(f"Saved to {filename}")
    plt.close()

def visualise_trajectory(history_x0, filename="explanation_trajectory.png"):
    """
    Visualizes the optimization trajectory of latent inputs (x0).
    Draws lines showing how the optimizer pushes points to satisfy the constraint.
    """
    print(f"Generating explanation plot: {filename}...")
    
   
    trajectory = np.stack(history_x0)
    steps, batch_size, _ = trajectory.shape
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    
    circle = plt.Circle((0, 0), 2.0, color='black', fill=False, linestyle='--', linewidth=2, label="Target Radius")
    ax.add_artist(circle)
    
    
    indices = np.linspace(0, batch_size-1, 10, dtype=int)
    
    for idx in indices:
        path = trajectory[:, idx, :] # Shape [Steps, 2]
        
        # gray line
        ax.plot(path[:, 0], path[:, 1], color='gray', alpha=0.5, linewidth=1)

        marker_interval = 1
        ax.scatter(path[::marker_interval, 0], path[::marker_interval, 1], 
                   color='blue', s=1, alpha=0.5)
        
        # red and green
        ax.scatter(path[0, 0], path[0, 1], color='red', s=40, zorder=3, label='Start ($x_0$ init)' if idx==indices[0] else "")
        ax.scatter(path[-1, 0], path[-1, 1], color='green', s=40, zorder=3, label='End ($x_0$ final)' if idx==indices[0] else "")

    ax.set_title("D-Flow Explanation: Optimization Trajectories\n(How noise $x_0$ is updated?)")
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.savefig(filename, dpi=150)
    plt.close()


def visualise_integration_path(model, final_x0, device='cpu', filename="integration_flow.png"):
    """
    Visualizes the physical flow of particles from t=0 to t=1.
    Shows the Green Dot (Optimized Start) moving to the Circle (Target).
    """
    print(f"Generating integration flow plot: {filename}...")
    
    model.eval()
    num_particles = 30
    x = torch.tensor(final_x0[:num_particles], device=device)
    
    steps = 20
    dt = 1.0 / steps
    
    trajectory = [x.detach().cpu().numpy()]
    
    # Run ODE Integration
    for i in range(steps):
        t_input = torch.full((num_particles,), i/steps, device=device)
        v = model(x, t_input)
        x = x + v * dt
        trajectory.append(x.detach().cpu().numpy())
    
    trajectory = np.array(trajectory) # Convert to numpy for plotting
    

    fig, ax = plt.subplots(figsize=(8, 8))
    

    circle = plt.Circle((0, 0), 2.0, color='black', fill=False, linestyle='--', linewidth=2, label="Target Manifold")
    ax.add_artist(circle)
    

    for i in range(num_particles):
        path = trajectory[:, i, :]
        
        ax.plot(path[:, 0], path[:, 1], color='blue', alpha=0.3, linewidth=1)
        ax.scatter(path[:, 0], path[:, 1], color='blue', s=5, alpha=0.3)
        
        ax.scatter(path[0, 0], path[0, 1], color='green', s=30, zorder=3, label='Optimized Input ($t=0$)' if i==0 else "")
        
        ax.scatter(path[-1, 0], path[-1, 1], color='purple', s=30, zorder=3, label='Final Prediction ($t=1$)' if i==0 else "")

    ax.set_title("Integration Flow: From Optimized Input to Target")
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.savefig(filename, dpi=150)
    plt.close()