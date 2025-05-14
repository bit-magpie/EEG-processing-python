import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# --- 10-5 System Channel Positions (VERY Illustrative Subset) ---
# IMPORTANT: The 10-5 system includes over 300 electrode positions.
# This dictionary is a very small, illustrative subset for demonstration purposes only.
# For comprehensive 10-5 system coverage, you should:
# 1. Obtain a standard electrode location file (e.g., .sfp, .elc, .loc, .ced, .csv)
#    from EEG software (MNE, EEGLAB, FieldTrip), hardware manufacturers,
#    or reputable neuroscience resources.
# 2. Write a parser function to load these coordinates into a dictionary.
#
# Example (conceptual) for loading from a CSV:
# def load_locations_from_csv(filepath):
#     positions = {}
#     with open(filepath, 'r') as f:
#         reader = csv.DictReader(f) # Assuming first row is header (e.g., label,X,Y,Z)
#         for row in reader:
#             try:
#                 # Adjust key names ('label', 'X', 'Y') based on your CSV file
#                 positions[row['label']] = (float(row['X']), float(row['Y']))
#             except (KeyError, ValueError) as e:
#                 print(f"Skipping row due to error: {row} - {e}")
#     return positions
#
# # Then, you would replace the dictionary below with:
# # TEN_FIVE_POSITIONS = load_locations_from_csv('path/to/your/full_10_5_coordinates.csv')

TEN_FIVE_POSITIONS = {
    # Midline
    'Nz': (0, 1.05),    # Nasion (approximate, slightly above unit circle for visibility)
    'Fpz': (0, 0.9),
    'AFz': (0, 0.65),
    'Fz': (0, 0.4),
    'FCz': (0, 0.15),
    'Cz': (0, 0),
    'CPz': (0, -0.15),
    'Pz': (0, -0.4),
    'POz': (0, -0.65),
    'Oz': (0, -0.9),
    'Iz': (0, -1.05),   # Inion (approximate, slightly below unit circle for visibility)

    # Left Hemisphere (Illustrative)
    'Fp1': (-0.30, 0.90),
    'AF7': (-0.65, 0.65), 'AF3': (-0.30, 0.65),
    'F9': (-0.95, 0.35), 'F7': (-0.70, 0.40), 'F5': (-0.50, 0.40), 'F3': (-0.30, 0.40), 'F1': (-0.15, 0.40),
    'FT9': (-0.98, 0.10), 'FT7': (-0.75, 0.15), 
    'FC5': (-0.50, 0.15), 'FC3': (-0.30, 0.15), 'FC1': (-0.15, 0.15),
    'T7': (-0.80, 0), # Often T3 in 10-20
    'C5': (-0.50, 0), 'C3': (-0.30, 0), 'C1': (-0.15, 0),
    'TP9': (-0.98, -0.10), 'TP7': (-0.75, -0.15),
    'CP5': (-0.50, -0.15), 'CP3': (-0.30, -0.15), 'CP1': (-0.15, -0.15),
    'P9': (-0.95, -0.35), 'P7': (-0.70, -0.40), 'P5': (-0.50, -0.40), 'P3': (-0.30, -0.40), 'P1': (-0.15, -0.40),
    'PO7': (-0.65, -0.65), 'PO3': (-0.30, -0.65),
    'O1': (-0.30, -0.90),

    # Right Hemisphere (Illustrative - mirrored from left for simplicity here)
    'Fp2': (0.30, 0.90),
    'AF8': (0.65, 0.65), 'AF4': (0.30, 0.65),
    'F10': (0.95, 0.35), 'F8': (0.70, 0.40), 'F6': (0.50, 0.40), 'F4': (0.30, 0.40), 'F2': (0.15, 0.40),
    'FT10': (0.98, 0.10), 'FT8': (0.75, 0.15),
    'FC6': (0.50, 0.15), 'FC4': (0.30, 0.15), 'FC2': (0.15, 0.15),
    'T8': (0.80, 0), # Often T4 in 10-20
    'C6': (0.50, 0), 'C4': (0.30, 0), 'C2': (0.15, 0),
    'TP10': (0.98, -0.10),'TP8': (0.75, -0.15),
    'CP6': (0.50, -0.15), 'CP4': (0.30, -0.15), 'CP2': (0.15, -0.15),
    'P10': (0.95, -0.35), 'P8': (0.70, -0.40), 'P6': (0.50, -0.40), 'P4': (0.30, -0.40), 'P2': (0.15, -0.40),
    'PO8': (0.65, -0.65), 'PO4': (0.30, -0.65),
    'O2': (0.30, -0.90),
    
    # Additional illustrative points for density
    'Fpz': (0, 0.9), 'Oz': (0, -0.9),
    'CzA': (0, 0.05), # Slightly anterior to Cz for illustration
    'CzP': (0, -0.05),# Slightly posterior to Cz
}

def get_channel_positions(channel_labels, positions_map):
    """
    Retrieves 2D coordinates for a list of channel labels from a given positions map.

    Args:
        channel_labels (list of str): A list of EEG channel labels.
        positions_map (dict): A dictionary where keys are channel labels (str)
                              and values are (x, y) coordinates (tuples).

    Returns:
        tuple: (found_positions, not_found_labels)
               found_positions (dict): {label: (x,y)} for channels found in positions_map.
               not_found_labels (list): List of labels not found.
    """
    found_positions = {}
    not_found_labels = []
    
    # Create a case-insensitive version of the positions_map keys for lookup
    # Stores original label from map to preserve its casing if needed later,
    # but lookup is done via uppercase.
    upper_case_positions_map = {key.upper(): positions_map[key] for key in positions_map}

    for label in channel_labels:
        if label.upper() in upper_case_positions_map:
            found_positions[label] = upper_case_positions_map[label.upper()]
        else:
            not_found_labels.append(label)
            
    if not_found_labels:
        print(f"Warning: The following channel labels were not found in the provided map: {not_found_labels}")
        
    return found_positions, not_found_labels

if __name__ == "__main__":
    print("--- EEG Channel Location Visualizer (10-5 System Subset) ---")
    print("IMPORTANT: The channel coordinates in this script are a VERY SMALL, illustrative subset.")
    print("For full 10-5 coverage, the TEN_FIVE_POSITIONS dictionary needs to be populated")
    print("by loading data from a comprehensive electrode location file.\n")

    # Example: Test with a mix of channels, including some potentially not in our subset
    test_labels = [
        'Fpz', 'Cz', 'Oz', 'Pz', 'Fz',
        'Fp1', 'Fp2', 'AF7', 'AF8',
        'T7', 'T8', 'C3', 'C4',
        'P7', 'P8', 'O1', 'O2',
        'FCz', 'CPz', 'POz', 'Nz', 'Iz',
        'NonExistentChannel1', 'F7', 'F8'
    ]
    
    print(f"Requesting positions for: {test_labels}\n")
    
    channel_coords, not_found = get_channel_positions(test_labels, TEN_FIVE_POSITIONS)
    
    if not channel_coords:
        print("No channel positions found for the given labels. Cannot generate plot.")
    else:
        print(f"\nFound positions for: {list(channel_coords.keys())}")

        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 1. Draw Head Outline (approximated by an ellipse)
        # Scaled to roughly fit coordinates between -1 and 1
        head_width = 2.0 
        head_height = 2.1 # Slightly taller than wide
        head_outline = Ellipse((0, 0), width=head_width, height=head_height, angle=0, 
                               edgecolor='black', facecolor='whitesmoke', lw=1.5, zorder=0)
        ax.add_patch(head_outline)

        # 2. Add simple anatomical markers (Nose, Ears)
        # Nose (pointing upwards along positive Y)
        nose_tip_y = head_height / 2
        ax.plot([0, 0], [nose_tip_y, nose_tip_y + 0.15], color='black', lw=1.5, zorder=1) # Bridge
        ax.plot([-0.07, 0, 0.07], [nose_tip_y +0.05, nose_tip_y, nose_tip_y+0.05], color='black', lw=1.5, zorder=1) # Nostrils area


        # Ears (simple ellipses on the sides)
        ear_width = 0.15
        ear_height = 0.4
        left_ear_x = -head_width / 2 
        right_ear_x = head_width / 2
        
        left_ear = Ellipse((left_ear_x, 0), width=ear_width, height=ear_height, angle=0, 
                           edgecolor='black', facecolor='lightgray', zorder=1)
        right_ear = Ellipse((right_ear_x, 0), width=ear_width, height=ear_height, angle=0, 
                            edgecolor='black', facecolor='lightgray', zorder=1)
        ax.add_patch(left_ear)
        ax.add_patch(right_ear)

        # 3. Plot Channel Positions and Labels
        x_coords = [pos[0] for pos in channel_coords.values()]
        y_coords = [pos[1] for pos in channel_coords.values()]
        labels_to_plot = list(channel_coords.keys())
        
        ax.scatter(x_coords, y_coords, color='blue', s=60, zorder=5, label="Channel Positions")
        for i, txt_label in enumerate(labels_to_plot):
            ax.text(x_coords[i] + 0.02, y_coords[i] + 0.02, txt_label, fontsize=8, 
                    color='darkred', zorder=6)
            
        # 4. Set Plot Properties
        ax.set_xlim(-head_width/2 - 0.3, head_width/2 + 0.3)
        ax.set_ylim(-head_height/2 - 0.3, head_height/2 + 0.3)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title("EEG Channel Locations (10-5 System - Illustrative Subset)", fontsize=14)
        ax.set_xlabel("X Coordinate (Left <-> Right)", fontsize=10)
        ax.set_ylabel("Y Coordinate (Posterior <-> Anterior)", fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.6, zorder=-1)
        ax.legend()
        
        plt.tight_layout()
        plt.show()
