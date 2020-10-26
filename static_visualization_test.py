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
fig.savefig("topograph2.png", bbox_inches='tight')