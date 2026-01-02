import matplotlib.pyplot as plt
import warnings
import numpy as np
import matplotlib.animation as animation


def visualise_dual_optimization(history_x0, history_x1, filename="dflow_dual_process_crescent.gif"):
    """
    Animating the Input (x0) and Output (x1) side-by-side gif.
    """
    print("Generating dual visualization...")
    

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor('black') # dark background for the whole image
    
    # Left Plot for x0 
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-4, 4)
    ax1.set_facecolor('black')
    ax1.set_title("Input ($x_0$): Finding the Optimized Input", color='white')
    scat1 = ax1.scatter([], [], s=5, c='magenta', alpha=0.6, edgecolors='none')
    ax1.grid(True, alpha=0.2)
    
    # right plot for x1 
    ax2.set_xlim(-5, 5)
    ax2.set_ylim(-5, 5)
    ax2.set_facecolor('black')
    ax2.set_title("Output ($x_1$): Forming the Circle", color='white')
    scat2 = ax2.scatter([], [], s=5, c='cyan', alpha=0.6, edgecolors='none')
    
    circle = plt.Circle((0, 0), 2.0, color='white', fill=False, linestyle='--', alpha=0.5, linewidth=2)
    ax2.add_artist(circle)
    
    # Frame Update Function
    def update(frame_idx):
        # update x0
        data_x0 = history_x0[frame_idx]
        scat1.set_offsets(data_x0)
        
        # update x1
        data_x1 = history_x1[frame_idx]
        scat2.set_offsets(data_x1)
        
        return scat1, scat2

    # creating animation
    ani = animation.FuncAnimation(fig, update, frames=len(history_x0), interval=30, blit=True, repeat = True)
    
    ani.save(filename, writer='pillow', fps=10)
    print(f"Animation saved to {filename}")
    plt.close()
