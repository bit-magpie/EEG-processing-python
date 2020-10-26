import mne
import numpy as np
import matplotlib.pyplot as plt 
from topograph import get_psds, plot_topomap
import time

data = mne.io.read_raw_edf('1.edf')
raw_data = data.get_data()
ch_data = raw_data[2:16,:]

plt.ion()
fig, ax = plt.subplots(figsize=(8,8))

chunk_data = np.array_split(ch_data, 50, axis=1)

for chunk in chunk_data:   
    pwrs, _ = get_psds(chunk)
    ax.clear()     
    plot_topomap(pwrs, ax, fig, draw_cbar=False)
    fig.canvas.draw()
    fig.canvas.flush_events()

    time.sleep(0.1)