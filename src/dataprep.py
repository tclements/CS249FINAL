# Data preparation script 
# some of the preprocessing functions are borrow from EQTransformer 
# https://github.com/smousavi05/EQTransformer/blob/master/EQTransformer/core/EqT_utils.py

import os, re 
import numpy as np 
import scipy 
import scipy.signal as signal
import matplotlib.pyplot as plt 
import tensorflow as tf 
import tensorflow.keras as keras 
import pandas as pd 
import h5py 
from tqdm import tqdm

def quantize(A):
    'quantize float data in range [-127,127]'
    A -= np.mean(A,axis=-1,keepdims=True)
    m = np.max(np.abs(A))
    factors = 127 / m
    A *= factors 
    return np.round(A)

def fp32uint8(A): 
    'quantize float data in range [0,255]'
    m = np.min(np.min(A),0)
    factor = 255 / (np.max(A) -m) 
    return np.round((A - m) * factor).astype(np.uint8)

def vel2acc(A,freqs):
    'convert velocity waveforms to acceleration using FFT'
    Xw = np.fft.fft(A,axis=-1)
    Xw *= freqs * 1j * 2 * np.pi 
    return np.real(np.fft.ifft(Xw,axis=-1))

def vel2accdiff(A):
    'convert velocity waveforms to accerlation using finite difference'
    return np.hstack([np.diff(A,axis=-1),np.array([0,0,0])[:,np.newaxis]])

def taper(A,alpha):
    'taper signal'
    window = signal.tukey(A.shape[-1],alpha)
    A *= window
    return A

def wav2spec(A,n,noverlap,fs): 
    'convert time domain data to time-frequency domain'
    f,t,sxx = signal.spectrogram(data,nperseg=n,noverlap=noverlap,fs=fs,axis=0)
    sxx = sxx.swapaxes(1,2)
    # make smallest value across channels equal to 1
    sxx += (1 - np.min(sxx))
    return np.log10(sxx)

def ricker(f,n,fs,t0):
    'create ricker wavelet' 
    tau = np.arange(0,n/fs,1/fs) - t0 
    return (1 - tau * tau * f**2 * np.pi**2) * np.exp(-tau**2 * np.pi**2 * f**2)

def butter_highpass(freq, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = freq / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def highpass(data, freq, fs, order=4):
    'Highpass filtering'
    b, a = butter_highpass(freq, fs, order=order)
    y = signal.lfilter(b, a, data,axis=-1)
    return y

def detrend(X):
    'Remove mean and trend from data'
    N = X.shape[-1]
    # create linear trend matrix 
    A = np.zeros((2,N),dtype=X.dtype)
    A[1,:] = 1
    A[0,:] = np.linspace(0,1,N)
    R = A @ np.transpose(A)
    Rinv = np.linalg.inv(R)
    factor = np.transpose(A) @ Rinv
    X -= (X @ factor) @ A
    return X

def drop_channel(data, rate):
    'Randomly replace values of one or two components to zeros in earthquake data'
        
    if np.random.uniform(0, 1) < rate: 
        c1 = np.random.choice([0, 1])
        c2 = np.random.choice([0, 1])
        c3 = np.random.choice([0, 1])
        if c1 + c2 + c3 > 0:
            data[np.array([c1, c2, c3])== 0,...] = 0
    return data

def add_gaps(data, rate): 
    'Randomly add gaps (zeros) of different sizes into waveforms'
    
    N = data.shape[-1]
    gap_start = np.random.randint(0, N * .5)
    gap_end = np.random.randint(gap_start, N)
    if np.random.uniform(0, 1) < rate: 
        data[gap_start:gap_end,:] = 0           
    return data  

def add_noise(data, rate):
    'Randomly add Gaussian noie with a random SNR into waveforms'
    
    data_noisy = np.empty((data.shape))
    if np.random.uniform(0, 1) < rate: 
        data_noisy = np.empty((data.shape))
        data_noisy[0,:] = data[0,:] + np.random.normal(0, np.random.uniform(0.01, 0.15)*np.max(np.abs(data[0,:])), data.shape[-1])
        data_noisy[1,:] = data[1,:] + np.random.normal(0, np.random.uniform(0.01, 0.15)*np.max(np.abs(data[1,:])), data.shape[-1])
        data_noisy[2,:] = data[2,:] + np.random.normal(0, np.random.uniform(0.01, 0.15)*np.max(np.abs(data[2,:])), data.shape[-1])   
    else:
        data_noisy = data
    return data_noisy    

def scale_amplitude( data, rate):
    'Scale amplitude or waveforms'
    
    tmp = np.random.uniform(0, 1)
    if tmp < rate:
        data *= np.random.uniform(1, 3)
    elif tmp < 2*rate:
        data /= np.random.uniform(1, 3)
    return data

def normalize( data, mode = 'max'):
    'Normalize waveforms in each batch'
        
    data -= np.mean(data, axis=-1, keepdims=True)
    if mode == 'max':
        max_data = np.max(data, axis=-1, keepdims=True)
        assert(max_data.shape[0] == data.shape[0])
        max_data[max_data == 0] = 1
        data /= max_data              

    elif mode == 'std':               
        std_data = np.std(data, axis=-1, keepdims=True)
        assert(std_data.shape[0] == data.shape[0])
        std_data[std_data == 0] = 1
        data /= std_data
    return data

def adjust_amplitude_for_multichannels(data):
    'Adjust the amplitude of multichaneel data'
    
    tmp = np.max(np.abs(data), axis=-1, keepdims=True)
    assert(tmp.shape[0] == data.shape[0])
    if np.count_nonzero(tmp) > 0:
        data *= data.shape[0] / np.count_nonzero(tmp)
    return data

def shift_event(data, maxshift, rate,start,halfdim): 
    'Randomly rotate the array to shift the event location'
    
    if np.random.uniform(0, 1) < rate:
        start += int(np.random.uniform(-maxshift, maxshift))             
    return data[:,start-halfdim:start+halfdim]

def test_train_split(df,split=0.8,):
    'split dataframe into testing and training'
    N = df.shape[0]
    NOISEdf = df[df['trace_category'] == "noise"]
    EQdf = df[df['trace_category'] == "earthquake_local"]
    EQdf = clean(EQdf)
    EQdf = EQdf.sort_values("trace_start_time")

    # do EQ test train split by date 
    Nsplit = int(np.round(EQdf.shape[0]) * split)
    EQtrain = EQdf.iloc[:Nsplit,:]
    EQtest = EQdf.iloc[Nsplit:,:]
    lasttime = pd.to_datetime(EQtrain.iloc[-1]["trace_start_time"])

    # do NOISE test train split by date 
    NOISEtrain = NOISEdf[pd.to_datetime(NOISEdf["trace_start_time"]) < lasttime]
    NOISEtest = NOISEdf[pd.to_datetime(NOISEdf["trace_start_time"]) > lasttime]

    # take random subset of noise (2 windows per file)
    NOISEtrain = NOISEtrain.iloc[np.random.choice(NOISEtrain.shape[0],EQtrain.shape[0])]
    NOISEtest = NOISEtest.iloc[np.random.choice(NOISEtest.shape[0],EQtest.shape[0])]
    
    # create training / testing dfs
    TRAIN = pd.concat([EQtrain,NOISEtrain])
    TEST = pd.concat([EQtest,NOISEtest])
    return TRAIN, TEST

def extract_snr(snr):
    'convert snr string to float'
    N = snr.shape[0]
    snr3 = np.zeros((3,N))
    for ii in range(N):
        snr[ii] = snr[ii].replace("nan","0.0")
        snr3[:,ii] = np.float32(re.findall(r"[-+]?\d*\.\d+|\d+", snr[ii]))
    snr3[snr3 == 0] = np.nan
    return snr3

def clean(df):
    'clean dataset based on snr and pick confidence'
    # round p and s arrivals to int
    df["p_arrival_sample"] = df["p_arrival_sample"].round().astype(int)
    df["s_arrival_sample"] = df["s_arrival_sample"].round().astype(int)

    # remove where p_wave starts too early in the sample 
    df = df[df["p_arrival_sample"] >= 200]

    # remove earthquakes > 100 km away 
    df = df[df["source_distance_km"] < 100]

    # update snr with median value 
    df["snr_db"] = np.nanmedian(extract_snr(df["snr_db"].values),axis=0)

    # get snr > 40 
    df = df[df["snr_db"] > 40] 
    
    # check p and s arrival samples
    df = df[df["s_arrival_sample"] - df["p_arrival_sample"] > 20]

    # find where p and s weight greater than 0.5 
    df = df[df["p_weight"] > 0.5]
    df = df[df["s_weight"] > 0.5]
    return df 

def data_reader(list_IDs, 
                 file_name, 
                 wlen=2., 
                 n_channels=3, 
                 fsin=100.,
                 fsout=40.,
                 freq=2.,
                 alpha=0.05,
                 maxshift=80,
                 norm_mode='max',
                 augmentation=True,  
                 max_shift=80,
                 shift_event_r=0.995,                                  
                 add_noise_r=0.5, 
                 drop_channel_r=0.3, 
                 scale_amplitude_r=0.3, ):   
    
    """ 
    
    For pre-processing and loading of data into memory. 
    
    Parameters
    ----------
    list_IDsx: str
        List of trace names.
            
    file_name: str
        Path to the input hdf5 datasets.
            
    wlen: float, default=2.
        Length of output traces, in seconds. 
           
    n_channels: int, default=3
        Number of channels.
            
    norm_mode: str, default=max
        The mode of normalization, 'max' or 'std'.
            
    augmentation: bool, default=True
        If True, batch will be augmented.
            
    add_event_r: {float, None}, default=None
        Chance for randomly adding a second event into the waveform.
            
    shift_event_r: {float, None}, default=0.99
        Rate of augmentation for randomly shifting the event within a trace. 
            
    add_noise_r: {float, None}, default=None
        Chance for randomly adding Gaussian noise into the waveform.
            
    drop_channe_r: {float, None}, default=None
        Chance for randomly dropping some of the channels.
            
    scale_amplitude_r: {float, None}, default=None
        Chance for randomly amplifying the waveform amplitude.
            
    Returns
    --------        
    X: pre-processed waveform as input.
    y: outputs labels for P, S and noise.
    """  
    
    fl = h5py.File(file_name, 'r')
    dim = int(wlen * fsout)
    X = np.zeros((2*len(list_IDs),dim,n_channels,1))
    y = np.zeros((2*len(list_IDs), 1)) 
    indim = int(fsin * wlen) // 2 
    freqs = np.fft.fftfreq(int(60 * fs),1/fs)

    # Generate data
    pbar = tqdm(total=len(list_IDs)) 
    for i, ID in enumerate(list_IDs):
        pbar.update()
        dataset = fl.get('data/'+str(ID))
        
        if ID.split('_')[-1] == 'EV':            
            data = np.array(dataset).transpose()              
            p_start = int(dataset.attrs['p_arrival_sample'])
            s_start = int(dataset.attrs['s_arrival_sample'])
            snr = dataset.attrs['snr_db']
                    
        elif ID.split('_')[-1] == 'NO':
            data = np.array(dataset).transpose()

        data = vel2acc(data,freqs)
        data = detrend(data)
        data = taper(data,alpha)
        data = highpass(data,freq,fs)

        if ID.split('_')[-1] == 'EV':            
            if shift_event_r:
                data1 = shift_event(data, maxshift, shift_event_r,p_start,indim)
                data2 = shift_event(data, maxshift, shift_event_r,s_start,indim)
            else:
                data1 = shift_event(data, maxshift, 0,p_start,indim)
                data2 = shift_event(data, maxshift, 0,s_start,indim)
        
        elif ID.split('_')[-1] == 'NO':
            data1 = shift_event(data, maxshift, 0,int(np.random.uniform(data.shape[-1] * alpha, data.shape[-1] * (1-alpha))),indim)
            data2 = shift_event(data, maxshift, 0,int(np.random.uniform(data.shape[-1] * alpha, data.shape[-1] * (1-alpha))),indim)

        if fsin != fsout:
            data1 = signal.resample(data1,dim,axis=-1)
            data2 = signal.resample(data2,dim,axis=-1)
           
        if augmentation:                 
            if norm_mode: 
                data1 = normalize(data1, norm_mode)   
                data2 = normalize(data2, norm_mode)
                          
            if dataset.attrs['trace_category'] == 'earthquake_local':
                if drop_channel_r:    
                    data1 = drop_channel(data1, drop_channel_r)
                    data1 = adjust_amplitude_for_multichannels(data1)
                    data2 = drop_channel(data2, drop_channel_r)
                    data2 = adjust_amplitude_for_multichannels(data2)
                          
                if scale_amplitude_r:
                    data1 = scale_amplitude(data1, scale_amplitude_r) 
                    data2 = scale_amplitude(data2, scale_amplitude_r) 

                if add_noise_r:
                    data1 = add_noise(data1, add_noise_r)
                    data2 = add_noise(data2, add_noise_r)
                    
                if norm_mode:    
                    data1 = normalize(data1, norm_mode)
                    data2 = normalize(data2, norm_mode)
                     
                            
            if dataset.attrs['trace_category'] == 'noise':
                if drop_channel_r:    
                    data1 = drop_channel(data1, drop_channel_r)
                    data1 = adjust_amplitude_for_multichannels(data1)
                    data2 = drop_channel(data2, drop_channel_r)
                    data2 = adjust_amplitude_for_multichannels(data2)   

                if norm_mode:               
                    data1 = normalize(data1, norm_mode)     
                    data2 = normalize(data2, norm_mode) 

        data1 = data1.transpose()
        data2 = data2.transpose()
                    
        X[i,:,:,0] = data1 
        X[len(list_IDs)+i,:,:,0] = data2 
        if ID.split('_')[-1] == 'EV':
            y[i] = 0 
            y[len(list_IDs)+i] = 1
        else:
            y[i] = 2
            y[len(list_IDs)+i] = 2                                    
    fl.close()  
    y = keras.utils.to_categorical(y)                     
    return X.astype('float32'), y.astype('float32')

if __name__ == "__main__":

    csvfile = "/home/timclements/CS249FINAL/data/merge.csv"
    h5path = "/home/timclements/CS249FINAL/data/merge.hdf5"
    df = pd.read_csv(csvfile)
    batch_size = 256 

    # split into test train 
    TRAINdf, TESTdf = test_train_split(df)
    Xtrain, Ytrain = data_reader(TRAINdf["trace_name"].values,h5path)
    np.savez("/home/timclements/CS249FINAL/data/train",Xtrain=Xtrain,Ytrain=Ytrain)
    dataset = tf.data.Dataset.from_tensor_slices((Xtrain,Ytrain)).batch(batch_size)
    tf.data.experimental.save(dataset, "/home/timclements/CS249FINAL/data/waves/train")
    Xtest, Ytest = data_reader(TESTdf["trace_name"].values,h5path)
    np.savez("/home/timclements/CS249FINAL/data/test",Xtest=Xtest,Ytest=Ytest)
    dataset = tf.data.Dataset.from_tensor_slices((Xtest,Ytest)).batch(batch_size)
    tf.data.experimental.save(dataset, "/home/timclements/CS249FINAL/data/waves/test")