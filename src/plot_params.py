import numpy as np 
import scipy 
import matplotlib.pyplot as plt 
import pickle 

with open('/home/timclements/CS249FINAL/DNN_models.pkl', 'rb') as f:
    DNNmodels = pickle.load(f)
Nmodels = len(DNNmodels)
DNNacc = np.zeros(Nmodels)
DNNsize = np.zeros(Nmodels)
DNNparams = np.zeros(Nmodels)

for ii in range(Nmodels): 
    DNNacc[ii] = DNNmodels[ii]["report"]["accuracy"]
    DNNsize[ii] = DNNmodels[ii]["model_header_size"]
    DNNparams[ii] = DNNmodels[ii]["params"]

with open('/home/timclements/CS249FINAL/CNN_models.pkl', 'rb') as f:
    CNNmodels = pickle.load(f)
Nmodels = len(CNNmodels)
CNNacc = np.zeros(Nmodels)
CNNsize = np.zeros(Nmodels)
CNNparams = np.zeros(Nmodels)

for ii in range(Nmodels): 
    CNNacc[ii] = CNNmodels[ii]["report"]["accuracy"]
    CNNsize[ii] = CNNmodels[ii]["model_header_size"]
    CNNparams[ii] = CNNmodels[ii]["params"]

with open('/home/timclements/CS249FINAL/DSCNN_models.pkl', 'rb') as f:
    DSCNNmodels = pickle.load(f)
Nmodels = len(CNNmodels)
DSCNNacc = np.zeros(Nmodels)
DSCNNsize = np.zeros(Nmodels)
DSCNNparams = np.zeros(Nmodels)
for ii in range(Nmodels): 
    DSCNNacc[ii] = DSCNNmodels[ii]["report"]["accuracy"]
    DSCNNsize[ii] = DSCNNmodels[ii]["model_header_size"]
    DSCNNparams[ii] = DSCNNmodels[ii]["params"]

# plot accuracy vs trainable parameters 
fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(DNNparams,DNNacc,100,alpha=0.85,label="DNN",c="dodgerblue",edgecolor="k")
ax.scatter(DSCNNparams,DSCNNacc,80,alpha=0.85,label="DS-CNN",c="chartreuse",edgecolor="k",marker="D")
ax.scatter(CNNparams,CNNacc,100,alpha=0.85,label="CNN",c="crimson",edgecolor="k",marker="^")
ax.set_xscale("log")
ax.set_xlabel("Trainable Parameters",fontsize=18)
ax.set_ylabel("Accuracy",fontsize=18)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylim([0.695, 0.93])
ax.spines['left'].set_bounds(0.70, 0.93)
ax.tick_params(direction='in')
ax.spines['bottom'].set_bounds(1e3,2e5)
ax.legend(loc="lower right",fontsize=14)
ax.tick_params(labelsize=12)
plt.tight_layout()
plt.savefig("/home/timclements/CS249FINAL/FIGURES/accuracy-vs-params.pdf")
plt.close()


