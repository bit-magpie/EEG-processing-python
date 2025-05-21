"""
EEG Channel Locator using the 10-5 system coordinates from electrode_positions_labeled.csv.
This script provides functions to load electrode positions and create visualizations
of electrode layouts for EEG topographic mapping.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon

# File path for the electrode positions
ELECTRODE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                             "resources/electrode_positions_labeled.csv")

def load_electrode_positions(filepath=ELECTRODE_FILE):
    """
    Load electrode positions from the CSV file.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file containing electrode positions.
        
    Returns:
    --------
    dict
        Dictionary with electrode labels as keys and (x, y) coordinates as values.
    """
    try:
        # Read CSV file, handle comment lines starting with '//'
        df = pd.read_csv(filepath, comment='/')
        
        # Normalize coordinates to range [-1, 1] for consistency with topographic plotting
        # Original coordinates are in range [0, 1]
        # First shift to center at (0.5, 0.5)
        df['x_norm'] = df['x'] - 0.5
        df['y_norm'] = df['y'] - 0.5
        
        # Flip y-axis to match conventional EEG orientation (nose at top)
        df['y_norm'] = -df['y_norm']
        
        # Scale to range [-1, 1]
        max_radius = np.max(np.sqrt(df['x_norm']**2 + df['y_norm']**2))
        df['x_norm'] = df['x_norm'] / max_radius
        df['y_norm'] = df['y_norm'] / max_radius
        
        # Convert to dictionary
        positions = {row['label']: (row['x_norm'], row['y_norm']) for _, row in df.iterrows()}
        
        print(f"Loaded {len(positions)} electrode positions from {filepath}")
        return positions
    except Exception as e:
        print(f"Error loading electrode positions: {e}")
        return {}

def get_channel_positions(channel_labels, positions_map=None):
    """
    Retrieves 2D coordinates for a list of channel labels.
    
    Parameters:
    -----------
    channel_labels : list of str
        A list of EEG channel labels.
    positions_map : dict, optional
        Dictionary with electrode labels as keys and (x, y) coordinates as values.
        If None, positions are loaded from the default file.
        
    Returns:
    --------
    tuple
        (found_positions, not_found_labels)
        found_positions : dict
            Dictionary with {label: (x,y)} for channels found in positions_map.
        not_found_labels : list
            List of labels not found.
    """
    if positions_map is None:
        positions_map = load_electrode_positions()
    
    found_positions = {}
    not_found_labels = []
    
    # Create a case-insensitive version of the positions_map for lookup
    upper_case_positions_map = {key.upper(): positions_map[key] for key in positions_map}
    
    for label in channel_labels:
        if label.upper() in upper_case_positions_map:
            found_positions[label] = upper_case_positions_map[label.upper()]
        else:
            not_found_labels.append(label)
    
    if not_found_labels:
        print(f"Warning: {len(not_found_labels)} channel labels were not found: {not_found_labels}")
    
    return found_positions, not_found_labels

def plot_all_electrodes(positions=None, show_labels=True, fontsize=8, marker_size=50):
    """
    Plot all electrode positions with head outline and labels.
    
    Parameters:
    -----------
    positions : dict, optional
        Dictionary with electrode labels as keys and (x, y) coordinates as values.
        If None, positions are loaded from the default file.
    show_labels : bool, optional
        Whether to display electrode labels.
    fontsize : int, optional
        Font size for electrode labels.
    marker_size : int, optional
        Size of electrode markers.
        
    Returns:
    --------
    tuple
        (fig, ax) - Matplotlib figure and axis objects
    """
    if positions is None:
        positions = load_electrode_positions()
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Extract coordinates and labels
    labels = list(positions.keys())
    coords = np.array([positions[label] for label in labels])
    
    # Draw head outline
    head = Circle((0, 0), radius=1.0, edgecolor='black', facecolor='whitesmoke', 
                 linewidth=2, alpha=0.3, zorder=1)
    ax.add_patch(head)
    
    # Draw nose
    nose = Polygon([(0, 1.0), (-0.1, 1.1), (0.1, 1.1)], closed=True, 
                  edgecolor='black', facecolor='lightgray', zorder=2)
    ax.add_patch(nose)
    
    # Plot electrodes
    ax.scatter(coords[:, 0], coords[:, 1], c='blue', s=marker_size, alpha=0.7, zorder=3)
    
    # Add labels
    if show_labels:
        for label, (x, y) in positions.items():
            ax.text(x, y, label, fontsize=fontsize, ha='center', va='center', 
                   bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.1', 
                            edgecolor='none'), zorder=4)
    
    # Set plot properties
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.set_title('10-5 System Electrode Positions', fontsize=14)
    ax.axis('off')
    
    plt.tight_layout()
    return fig, ax

def plot_selected_electrodes(channel_labels, positions=None, show_labels=True, 
                            fontsize=10, marker_size=80, highlight_color='red'):
    """
    Plot selected electrode positions with special highlighting.
    
    Parameters:
    -----------
    channel_labels : list of str
        Labels of electrodes to highlight.
    positions : dict, optional
        Dictionary with electrode labels as keys and (x, y) coordinates as values.
        If None, positions are loaded from the default file.
    show_labels : bool, optional
        Whether to display electrode labels.
    fontsize : int, optional
        Font size for electrode labels.
    marker_size : int, optional
        Size of electrode markers.
    highlight_color : str, optional
        Color for highlighted electrodes.
        
    Returns:
    --------
    tuple
        (fig, ax) - Matplotlib figure and axis objects
    """
    if positions is None:
        positions = load_electrode_positions()
    
    # Get positions for selected channels
    selected_positions, not_found = get_channel_positions(channel_labels, positions)
    
    # First plot all electrodes in gray
    fig, ax = plot_all_electrodes(positions, show_labels=False, marker_size=30)
    
    # Extract coordinates for selected electrodes
    if selected_positions:
        selected_labels = list(selected_positions.keys())
        selected_coords = np.array([selected_positions[label] for label in selected_labels])
        
        # Highlight selected electrodes
        ax.scatter(selected_coords[:, 0], selected_coords[:, 1], 
                  c=highlight_color, s=marker_size, alpha=0.9, zorder=5)
        
        # Add labels for selected electrodes
        if show_labels:
            for label, (x, y) in selected_positions.items():
                ax.text(x, y, label, fontsize=fontsize, ha='center', va='center',
                       color='black', fontweight='bold',
                       bbox=dict(facecolor='white', alpha=0.85, boxstyle='round,pad=0.2',
                                edgecolor=highlight_color, linewidth=2), zorder=6)
    
    ax.set_title(f'Selected Electrodes ({len(selected_positions)} of {len(channel_labels)} found)', 
                fontsize=14)
    
    return fig, ax

if __name__ == "__main__":
    # When run as a script, demonstrate the functionality
    print("EEG Channel Locator - 10-5 System")
    
    # Load electrode positions
    positions = load_electrode_positions()
    
    # Example 1: Plot all electrodes
    fig1, ax1 = plot_all_electrodes(positions)
    fig1.savefig("all_electrodes.png", dpi=300, bbox_inches='tight')
    
    # Example 2: Plot selected electrodes (international 10-20 system subset)
    international_1020 = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 
                         'T7', 'C3', 'Cz', 'C4', 'T8', 
                         'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2']
    
    fig2, ax2 = plot_selected_electrodes(international_1020, positions)
    fig2.savefig("selected_electrodes.png", dpi=300, bbox_inches='tight')
    
    plt.show()
