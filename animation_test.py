import mne
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
from topograph import get_psds, plot_topomap
import os
import argparse

def create_eeg_animation(edf_file='1.edf', num_frames=50, interval=100, 
                         save_gif=False, output_filename=None, channels=None):
    """
    Create an EEG topographic animation from an EDF file.
    
    Parameters:
    -----------
    edf_file : str
        Path to the EDF file to animate
    num_frames : int
        Number of chunks to split the data into (frames in the animation)
    interval : int
        Interval between frames in milliseconds
    save_gif : bool
        Whether to save the animation as a GIF
    output_filename : str
        Name of the output file if saving
    channels : list
        List of channel indices to use, if None, use default range (2:16)
    """
    print(f"Loading EDF file: {edf_file}")
    data = mne.io.read_raw_edf(edf_file)
    raw_data = data.get_data()
    ch_names = data.ch_names
    
    # Select channels
    if channels is None:
        ch_indices = list(range(2, min(16, len(raw_data))))
    else:
        ch_indices = channels
        
    ch_data = raw_data[ch_indices, :]
    ch_names_selected = [ch_names[i] for i in ch_indices]
    
    print(f"Animating {len(ch_indices)} channels: {ch_names_selected}")
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Split data into chunks
    chunk_data = np.array_split(ch_data, num_frames, axis=1)
    
    # Initialize variables
    cbar_axis = None  # Will hold the colorbar axis
    title = f'EEG Topography Animation - {os.path.basename(edf_file)}'
    progress_text_obj = None  # We'll create this in the update function
    
    # Animation update function
    def update(frame):
        nonlocal cbar_axis, progress_text_obj
        
        # Calculate power for this chunk
        pwrs, _ = get_psds(chunk_data[frame])
        
        # Clear colorbar if it exists
        if cbar_axis is not None:
            cbar_axis.remove()
            cbar_axis = None
            
        # Clear previous plot
        ax.clear()
        
        # Create new plot with colorbar only on first frame
        _, new_cbar = plot_topomap(pwrs, ax, fig, channel_labels=ch_names_selected, 
                     draw_cbar=(frame==0), title=title)
        if new_cbar is not None:
            cbar_axis = new_cbar
        
        # Add progress indicator
        progress = f"Frame: {frame+1}/{num_frames}"
        progress_text_obj = ax.text(0.02, 0.02, progress, transform=ax.transAxes, 
                               fontsize=10, color='black', bbox=dict(facecolor='white', alpha=0.7))
        
        print(f"Rendering frame {frame+1}/{num_frames}", end="\r")
        return ax,
    
    # Create the animation
    ani = FuncAnimation(fig, update, frames=num_frames, blit=False, interval=interval)
    
    if save_gif:
        if output_filename is None:
            output_filename = f"topograph_animation_{os.path.splitext(os.path.basename(edf_file))[0]}.gif"
        
        print(f"\nSaving animation to {output_filename}...")
        ani.save(output_filename, writer='pillow', fps=1000/interval)
        print(f"Animation saved to {output_filename}")
    
    plt.tight_layout()
    plt.show()
    
    return ani

if __name__ == "__main__":
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Create EEG topographic animation from EDF file')
    parser.add_argument('edf_file', nargs='?', default='1.edf', 
                        help='Path to EDF file (default: 1.edf)')
    parser.add_argument('--frames', type=int, default=50,
                        help='Number of frames in the animation (default: 50)')
    parser.add_argument('--interval', type=int, default=100,
                        help='Interval between frames in milliseconds (default: 100)')
    parser.add_argument('--save', action='store_true',
                        help='Save the animation as a GIF')
    parser.add_argument('--output', '-o', type=str, 
                        help='Output filename for the animation GIF')
    parser.add_argument('--channels', type=str,
                        help='Comma-separated list of channel indices to use')
    
    args = parser.parse_args()
    
    # Process channel indices if provided
    channels = None
    if args.channels:
        try:
            channels = [int(i) for i in args.channels.split(',')]
        except ValueError:
            print("Error: Invalid channel indices. Please provide comma-separated integers.")
            import sys
            sys.exit(1)
    
    # Create animation
    ani = create_eeg_animation(
        edf_file=args.edf_file,
        num_frames=args.frames,
        interval=args.interval,
        save_gif=args.save,
        output_filename=args.output,
        channels=channels
    )