import numpy as np 
import scipy 
import matplotlib.pyplot as plt 
import pickle 

# DNNlatency = [1,1,1,2,2,2,4,4,4,5,8,8,8,9,11,15,15,16,18,22,29]
with open('/home/timclements/CS249FINAL/DNN_models.pkl', 'rb') as f:
    DNNmodels = pickle.load(f)
Nmodels = len(DNNmodels)
DNNacc = np.zeros(Nmodels)
DNNlat = np.zeros(Nmodels)
DNNparams = np.zeros(Nmodels)

for ii in range(Nmodels): 
    DNNacc[ii] = DNNmodels[ii]["report"]["accuracy"]
    DNNlat[ii] = DNNmodels[ii]["latency"]
    DNNparams[ii] = DNNmodels[ii]["params"]

# CNNlatency = [10,13,19,23,34,54,62,100,175,190,330,612,643,1185,2274]
with open('/home/timclements/CS249FINAL/CNN_models.pkl', 'rb') as f:
    CNNmodels = pickle.load(f)
Nmodels = len(CNNmodels)
CNNacc = np.zeros(Nmodels)
CNNlat = np.zeros(Nmodels)
CNNparams = np.zeros(Nmodels)

for ii in range(Nmodels): 
    CNNacc[ii] = CNNmodels[ii]["report"]["accuracy"]
    CNNlat[ii] = CNNmodels[ii]["latency"]
    CNNparams[ii] = CNNmodels[ii]["params"]

# DSCNNlatency = [8,11,15,23,16,21,29,45,32,40,57,90,65,81,113]
with open('/home/timclements/CS249FINAL/DSCNN_models.pkl', 'rb') as f:
    DSCNNmodels = pickle.load(f)
Nmodels = len(CNNmodels)
DSCNNacc = np.zeros(Nmodels)
DSCNNlat = np.zeros(Nmodels)
DSCNNparams = np.zeros(Nmodels)
for ii in range(Nmodels): 
    DSCNNacc[ii] = DSCNNmodels[ii]["report"]["accuracy"]
    DSCNNlat[ii] = DSCNNmodels[ii]["latency"]
    DSCNNparams[ii] = DSCNNmodels[ii]["params"]

# plot latency vs accuracy 
fig, ax = plt.subplots(figsize=(6,6))
ax.axvline([0.1],c="grey",linestyle="--",linewidth=2,alpha=0.75)
ax.scatter(DNNlat / 1000,DNNacc,100,alpha=0.85,label="DNN",c="dodgerblue",edgecolor="k")
ax.scatter(DSCNNlat / 1000,DSCNNacc,80,alpha=0.85,label="DS-CNN",c="chartreuse",edgecolor="k",marker="D")
ax.scatter(CNNlat / 1000,CNNacc,100,alpha=0.85,label="CNN",c="crimson",edgecolor="k",marker="^")
ax.set_xscale("log")
ax.set_xlabel("Latency [s]",fontsize=18)
ax.set_ylabel("Accuracy",fontsize=18)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylim([0.695, 0.93])
ax.spines['left'].set_bounds(0.70, 0.93)
ax.tick_params(direction='in')
ax.spines['bottom'].set_bounds(8e-4,3)
ax.legend(loc="lower right",fontsize=14)
ax.tick_params(labelsize=12)
plt.tight_layout()
plt.savefig("/home/timclements/CS249FINAL/FIGURES/accuracy-vs-latency.pdf")
plt.close()



