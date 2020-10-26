# EEG-processing-python

## EEG Topograph 

### Dependencies
1. numpy (`pip install numpy`)
2. scipy (`pip install scipy`)
3. matplotlib (`pip install matplotlib`)
4. mne (optional) (`pip install mne`): Only for read EDF file 

### Usage
Input: data- 1D array 14 power values 
       ax- Matplotlib subplot object to be plotted every thing
       fig- Matplot lib figure object to draw colormap
       draw_cbar- Visualize color bar in the plot (boolean)

Output: matplotlib axis

#### Static visualization

Code (static_visualization_test.py):
```python
import mne
import matplotlib.pyplot as plt 
from topograph import get_psds, plot_topomap


data = mne.io.read_raw_edf('1.edf')
raw_data = data.get_data()
ch_data = raw_data[2:16,:]
pwrs, _ = get_psds(ch_data)

fig, ax = plt.subplots(figsize=(10,8))
plot_topomap(pwrs, ax, fig)
plt.show()
fig.savefig("topograph.png", bbox_inches='tight')
```

Output image:

![Topograph](topograph.png)

#### Animation

Code (animation_test.py):
```python
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
```

Output:

![Topograph](topograph_animation.gif)

To download sample EDF dataset please refer [Person identification from EEG using various machine learning techniques with inter-hemispheric amplitude ratio](https://doi.org/10.1371/journal.pone.0238872)
