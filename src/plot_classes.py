import numpy as np 
import matplotlib.pyplot as plt 
import h5py 
import dataprep 

csvfile = "/home/timclements/CS249FINAL/data/merge.csv"
h5path = "/home/timclements/CS249FINAL/data/merge.hdf5"
df = pd.read_csv(csvfile)
EQdf = df[df['trace_category'] == "earthquake_local"]
EQdf["snr_db"] = np.nanmedian(dataprep.extract_snr(EQdf["snr_db"].values),axis=0)
EQdf = EQdf.sort_values("snr_db",ascending=True)
fl = h5py.File(h5path, 'r')

# get location of highest snr 
ID = EQdf.iloc[900900]["trace_name"]
dataset = fl.get('data/'+str(ID))
data = np.array(dataset).transpose()              
p_start = int(dataset.attrs['p_arrival_sample'])
s_start = int(dataset.attrs['s_arrival_sample'])
snr = dataset.attrs['snr_db']
data = dataprep.highpass(data,2.,100.)
fl.close()

# plot three classes together 
startat = 143
numsamples = 1100
fs = 100.
freq = 2. 
t = np.linspace(0,numsamples/fs,numsamples)
fig, ax = plt.subplots()
ax.plot(t,data[0,startat:startat+numsamples],c="k",alpha=0.85)
ax.axvline([(p_start - startat) / fs],c="red")
ax.axvline([(s_start - startat) / fs],c="blue")
ax.axvspan((startat+50) / fs,(startat+250) / fs ,alpha=0.5, color="grey",label="Noise") 
ax.axvspan((p_start-startat-100) / fs,(p_start - startat+100) / fs ,alpha=0.5, color="red",label="P-wave") 
ax.axvspan((s_start-startat-100) / fs,(s_start - startat+100) / fs ,alpha=0.5, color="blue",label="S-wave")
ax.set_xlabel("Second",fontsize=18)
ax.set_ylabel("Amplitude Counts",fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
fig.savefig("/home/timclements/CS249FINAL/FIGURES/all3.pdf")
plt.close()

pwave = data[:,p_start-100:p_start+100]
swave = data[:,s_start-100:s_start+100]
noise = data[:,startat+50:startat+250]
pwave = dataprep.normalize(pwave)
swave = dataprep.normalize(swave)
noise = dataprep.normalize(noise)

# plot all three classes normalized to same amplitude 
zoom_t = np.linspace(0,1.99,200)
yticks = [-1,0,1]
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3)
ax1.plot(zoom_t,noise[0,:],c="grey",alpha=0.5,linewidth = 2)
ax1.set_title("Noise               ",fontsize=20,color="Grey")
ax1.set_xlim(xmin=0)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
# Only show ticks on the left and bottom spines
ax1.yaxis.set_ticks_position('left')
ax1.xaxis.set_ticks_position('bottom')
ax1.set_xticks([], [])
ax1.tick_params(labelsize=12)
ax1.yaxis.set_ticks(yticks)
ax1.tick_params(direction='in')
ax1.spines['bottom'].set_bounds(min(zoom_t), max(zoom_t))
ax1.set_xlim([-0.05, 2.])
ax2.plot(zoom_t,pwave[0,:],c="red",alpha=0.5,linewidth = 2)
ax2.set_title("P-wave               ",fontsize=20,color="red")
ax2.set_ylabel("Normalized Amplitude",fontsize=14)
ax2.set_xlim(xmin=0)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
# Only show ticks on the left and bottom spines
ax2.yaxis.set_ticks_position('left')
ax2.xaxis.set_ticks_position('bottom')
ax2.set_xticks([], [])
ax2.yaxis.set_ticks(yticks)
ax2.tick_params(labelsize=12)
ax2.tick_params(direction='in')
ax2.spines['bottom'].set_bounds(min(zoom_t), max(zoom_t))
ax2.set_xlim([-0.05, 2.])
ax3.plot(zoom_t,swave[0,:],c="blue",alpha=0.5,linewidth = 2)
ax3.set_title("S-wave               ",fontsize=20,color="blue")
ax3.set_xlabel("Second",fontsize=18)
ax3.set_xlim(xmin=0)
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
# Only show ticks on the left and bottom spines
ax3.yaxis.set_ticks_position('left')
ax3.xaxis.set_ticks_position('bottom')
ax3.tick_params(labelsize=12)
ax3.yaxis.set_ticks(yticks)
ax3.set_ylim([-1.25, 1])
ax3.spines['left'].set_bounds(-1, 1)
ax3.tick_params(direction='in')
ax3.spines['bottom'].set_bounds(0,2.1)
ax3.set_xlim([-0.05, 2.])
plt.tight_layout()
plt.savefig("/home/timclements/CS249FINAL/FIGURES/classes.pdf")
plt.close()
