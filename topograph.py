import numpy as np
import scipy.interpolate
from scipy import signal
from matplotlib import patches
import matplotlib.pyplot as plt
from eeg_channel_locator import get_channel_positions, TEN_FIVE_POSITIONS

# Commented out for clarity - This function is still in use in the main program to 
# calculate power spectral density values for topographic plotting
def get_psds(data, fs=128, f_range=[0.5, 30]):
    '''
    Calculate signal power using Welch method.

    Input: data- mxn matrix (m: number of channels, n: samples of signals)
           fs- Sampling frequency (default 128Hz)
           f_range- Frequency range (default 0.5Hz to 30Hz)
    Output: Power values and PSD values
    '''
    powers = []
    psds = list()
    for sig in data:
        freq, psd = signal.welch(sig, fs)
        idx = np.logical_and(freq >= f_range[0], freq <= f_range[1])
        powers = np.append(powers, sum(psd[idx]))
        psds.append(psd[idx])
    
    return powers, psds

def plot_topomap(data, ax, fig, channel_labels, draw_cbar=True, title=None):
    '''
    Plot topographic plot of EEG data using the standard 10-5 system electrode positions.
    
    Input: data - 1D array of values for each channel
           ax - Matplotlib subplot object
           fig - Matplotlib figure object to draw colormap
           channel_labels - List of EEG channel labels corresponding to data points
                          (must match the order of data)
           draw_cbar - Visualize color bar in the plot
           title - Optional title for the plot
    '''
    N = 300  # Grid size for interpolation
    
    # Get positions from 10-5 system using eeg_channel_locator
    positions, not_found = get_channel_positions(channel_labels, TEN_FIVE_POSITIONS)
    
    if not positions:
        print("Error: No channel positions found in 10-5 system.")
        return ax
        
    if not_found:
        print(f"Warning: {len(not_found)} channels not found in 10-5 system: {not_found}")
        # Continue with the channels we have positions for
    
    # Extract coordinates and corresponding data
    x = []
    y = []
    filtered_data = []
    filtered_labels = []
    
    # Match data with positions, assuming same order
    for i, label in enumerate(channel_labels):
        if i < len(data) and label in positions:
            pos = positions[label]
            x.append(pos[0])
            y.append(pos[1])
            filtered_data.append(data[i])
            filtered_labels.append(label)
    
    if len(filtered_data) == 0:
        print("Error: No matching channels with positions found.")
        return ax
        
    print(f"Plotting {len(filtered_data)} channels out of {len(channel_labels)} provided.")
    
    # Standard 10-5 coordinate system
    xy_center = [0, 0]  # Center of the head
    radius = 1.0  # Typical radius for normalized coordinates
    
    # Set up interpolation grid
    xi = np.linspace(-1.5, 1.5, N)
    yi = np.linspace(-1.5, 1.5, N)
    
    # First, create synthetic points around the edge of the head to improve interpolation
    # This helps ensure that the interpolation covers the entire head area
    n_border_points = 32  # Number of points to add around the circle
    border_angles = np.linspace(0, 2*np.pi, n_border_points, endpoint=False)
    
    # Calculate mean and std of the data for realistic border values
    data_mean = np.mean(filtered_data)
    data_std = np.std(filtered_data) if len(filtered_data) > 1 else 0.1 * data_mean
    
    # Add synthetic points at the border of the head circle
    x_border = [radius * 0.98 * np.cos(angle) for angle in border_angles]
    y_border = [radius * 0.98 * np.sin(angle) for angle in border_angles]
    
    # Generate values for the border based on nearest electrodes
    border_values = []
    for i in range(n_border_points):
        # Find the closest electrode
        min_dist = float('inf')
        closest_idx = 0
        for j, (ex, ey) in enumerate(zip(x, y)):
            dist = (x_border[i] - ex)**2 + (y_border[i] - ey)**2
            if dist < min_dist:
                min_dist = dist
                closest_idx = j
        
        # Use value from closest electrode with small decay
        if min_dist < float('inf'):
            decay = min(1.0, 0.8 + 0.2 * np.exp(-min_dist * 3))
            border_values.append(filtered_data[closest_idx] * decay)
        else:
            border_values.append(data_mean)
    
    # Combine original electrodes with synthetic border points
    x_combined = np.concatenate([x, x_border])
    y_combined = np.concatenate([y, y_border])
    data_combined = np.concatenate([filtered_data, border_values])
    
    # Interpolate using the enhanced dataset
    zi = scipy.interpolate.griddata(
        (x_combined, y_combined), data_combined, 
        (xi[None, :], yi[:, None]), 
        method='cubic'
    )
    
    # Add a second pass with 'linear' method to fill remaining NaN values
    mask_nans = np.isnan(zi)
    if np.any(mask_nans):
        zi_linear = scipy.interpolate.griddata(
            (x_combined, y_combined), data_combined,
            (xi[None, :], yi[:, None]),
            method='linear'
        )
        zi[mask_nans] = zi_linear[mask_nans]
    
    # If there are still NaNs, try nearest interpolation as a final resort
    mask_nans = np.isnan(zi)
    if np.any(mask_nans):
        zi_nearest = scipy.interpolate.griddata(
            (x_combined, y_combined), data_combined,
            (xi[None, :], yi[:, None]),
            method='nearest'
        )
        zi[mask_nans] = zi_nearest[mask_nans]
    
    # Apply a mask that strictly follows the head circle
    dr = xi[1] - xi[0]
    for i in range(N):
        for j in range(N):
            r = np.sqrt((xi[i] - xy_center[0])**2 + (yi[j] - xy_center[1])**2)
            if r > radius:
                zi[j, i] = np.nan
    
    dist = ax.contourf(xi, yi, zi, 60, cmap = plt.get_cmap('coolwarm'), zorder = 1)
    ax.contour(xi, yi, zi, 15, linewidths = 0.5,colors = "grey", zorder = 2)
    
    if draw_cbar:
        cbar = fig.colorbar(dist, ax=ax, format='%.1e')
        cbar.ax.tick_params(labelsize=8)

    # Plot electrode positions
    ax.scatter(x, y, marker = 'o', c = 'b', s = 15, zorder = 3)
    
    # Add channel labels next to each electrode
    for i, label in enumerate(filtered_labels):
        # Calculate optimal label offset based on position to avoid overlaps
        # Move labels slightly away from the center
        dx = x[i] * 0.05  # Small offset in x direction based on position
        dy = y[i] * 0.05  # Small offset in y direction based on position
        ax.text(x[i] + dx, y[i] + dy, label, fontsize=9, 
                ha='center', va='center', color='black', fontweight='bold',
                zorder=5)  # Higher zorder to ensure labels are on top
    
    circle = patches.Circle(xy = xy_center, radius = radius, edgecolor = "k", facecolor = "none", zorder=4)
    ax.add_patch(circle)

    for loc, spine in ax.spines.items():
        spine.set_linewidth(0)
    
    ax.set_xticks([])
    ax.set_yticks([])

    # Add anatomical markers (ears, nose)
    head_height = 2.1  # Slightly taller than wide
    
    # Ears on the sides
    ear_width, ear_height = 0.15, 0.4
    left_ear = patches.Ellipse((-1.0, 0), width=ear_width, height=ear_height, 
                               angle=0, edgecolor='black', facecolor='lightgray', zorder=0)
    right_ear = patches.Ellipse((1.0, 0), width=ear_width, height=ear_height, 
                                angle=0, edgecolor='black', facecolor='lightgray', zorder=0)
    ax.add_patch(left_ear)
    ax.add_patch(right_ear)
    
    # Nose at the top (triangle shape)
    nose_tip_y = 1.05
    nose_triangle = patches.Polygon(
        [[0, nose_tip_y + 0.15], [-0.1, nose_tip_y], [0.1, nose_tip_y]],
        closed=True, 
        edgecolor='black',
        facecolor='lightgray',
        lw=1.5,
        zorder=0
    )
    ax.add_patch(nose_triangle)
    
    # Set axis limits
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    
    # Add title if provided
    if title:
        ax.set_title(title)

    return ax


if __name__ == "__main__":
    import mne
    import matplotlib.pyplot as plt
    import sys
    import os
    import argparse
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Generate EEG topographic map from EDF file')
    parser.add_argument('edf_file', nargs='?', default='1.edf', 
                        help='Path to EDF file (default: 1.edf)')
    parser.add_argument('--indices', type=str, help='Comma-separated list of channel indices to use')
    parser.add_argument('--channels', type=str, help='Comma-separated list of channel names to use')
    parser.add_argument('--prefix', action='store_true',
                        help='Auto-detect channels starting with common EEG prefixes')
    parser.add_argument('--output', '-o', type=str, 
                        help='Output filename for the topographic map image')
    
    args = parser.parse_args()
    
    # Validate the EDF file exists
    edf_file = args.edf_file
    if not os.path.exists(edf_file):
        print(f"Error: File {edf_file} not found.")
        sys.exit(1)

    # Load EEG data file
    print(f"Loading EDF file: {edf_file}")
    data = mne.io.read_raw_edf(edf_file)
    raw_data = data.get_data()
    ch_names = data.ch_names
    
    print(f"File contains {len(ch_names)} channels: {ch_names}")
    
    # Determine which channels to use
    eeg_ch_indices = []
    eeg_ch_names = []
    
    if args.channels:
        # Use specified channel names
        requested_channels = args.channels.split(',')
        for name in requested_channels:
            if name in ch_names:
                idx = ch_names.index(name)
                eeg_ch_indices.append(idx)
                eeg_ch_names.append(name)
            else:
                print(f"Warning: Channel '{name}' not found in the data file.")
    
    elif args.indices:
        # Use specified channel indices
        try:
            requested_indices = [int(i) for i in args.indices.split(',')]
            for idx in requested_indices:
                if 0 <= idx < len(ch_names):
                    eeg_ch_indices.append(idx)
                    eeg_ch_names.append(ch_names[idx])
                else:
                    print(f"Warning: Channel index {idx} out of range (0-{len(ch_names)-1}).")
        except ValueError:
            print("Error: Invalid channel indices. Please provide comma-separated integers.")
            sys.exit(1)
    
    elif args.prefix or len(eeg_ch_indices) == 0:
        # Auto-detect channels based on common EEG name prefixes
        standard_prefixes = ['F', 'C', 'P', 'O', 'T', 'A', 'FP', 'AF', 'FC', 'CP', 'TP', 'PO']
        
        # Avoid these prefixes as they're typically not EEG channels
        excluded_prefixes = ['CQ_', 'RAW_', 'TIME_', 'COUNTER', 'INTERPOLATED', 'MARKER', 'SYNC', 'GYRO']
        
        for i, name in enumerate(ch_names):
            name_upper = name.upper()
            # Check if this is an EEG channel and not a non-EEG channel
            if (any(name_upper.startswith(prefix) for prefix in standard_prefixes) or 'EEG' in name_upper) and \
               not any(name_upper.startswith(ex) for ex in excluded_prefixes):
                eeg_ch_indices.append(i)
                eeg_ch_names.append(name)
    
    # If still no channels identified, use Emotiv channels as fallback (indices 2:16)
    if not eeg_ch_indices and len(raw_data) > 16:
        print("No EEG channels identified. Using Emotiv channel indices (2-16) as fallback.")
        eeg_ch_indices = list(range(2, 16))
        eeg_ch_names = [ch_names[i] for i in eeg_ch_indices]
    
    if not eeg_ch_indices:
        print("Error: No valid EEG channels identified.")
        sys.exit(1)
    
    # Extract EEG data and calculate power spectral density
    ch_data = raw_data[eeg_ch_indices, :]
    pwrs, _ = get_psds(ch_data)
    
    print(f"Processing {len(eeg_ch_indices)} EEG channels: {eeg_ch_names}")
    
    # Create topographic plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create the title string
    title = f'EEG Topography - {os.path.basename(edf_file)}'
    
    # Generate the plot
    plot_topomap(pwrs, ax, fig, channel_labels=eeg_ch_names, draw_cbar=True, title=title)
    
    plt.tight_layout()
    
    # Determine output filename
    if args.output:
        output_file = args.output
    else:
        output_file = f"topograph_{os.path.splitext(os.path.basename(edf_file))[0]}.png"
    
    # Display and save figure
    plt.show()
    fig.savefig(output_file, bbox_inches='tight', dpi=300)
    print(f"Saved topographic map to {output_file}")
