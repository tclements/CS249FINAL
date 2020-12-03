import os, re 
import numpy as np 
import scipy 
import scipy.signal as signal
import matplotlib.pyplot as plt 
import tensorflow as tf 
import tensorflow.keras as keras 
import pandas as pd 
import h5py 

def fp32int8(A):
    A -= np.mean(A,axis=-1)
    m = np.max(np.abs(A))
    factors = 127 / m
    A *= factors 
    return np.round(A).astype(np.int8)

def fp32uint8(A): 
    m = np.min(np.min(A),0)
    factor = 255 / (np.max(A) -m) 
    return np.round((A - m) * factor).astype(np.uint8)

def vel2acc(A,freqs):
    Xw = np.fft.fft(A,axis=-1)
    Xw *= freqs * 1j * 2 * np.pi 
    return np.real(np.fft.ifft(Xw,axis=-1))

def vel2accdiff(A):
    return np.hstack([np.diff(A,axis=-1),np.array([0,0,0])[:,np.newaxis]])

def taper(A,alpha):
    window = signal.tukey(A.shape[-1],alpha)
    A *= window[:,np.newaxis]
    return None 

def wav2spec(A,n,noverlap,fs): 
    f,t,sxx = signal.spectrogram(data,nperseg=n,noverlap=noverlap,fs=fs,axis=-1)
    sxx = sxx.swapaxes(1,2)
    # make smallest value across channels equal to 1
    sxx += (1 - np.min(sxx))
    return np.log10(sxx)

def ricker(f,n,fs,t0):
    # create ricker wavelet 
    tau = np.arange(0,n/fs,1/fs) - t0 
    return (1 - tau * tau * f**2 * np.pi**2) * np.exp(-tau**2 * np.pi**2 * f**2)

def butter_highpass(freq, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = freq / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def highpass(data, freq, fs, order=4):
    b, a = butter_highpass(freq, fs, order=order)
    y = signal.lfilter(b, a, data,axis=-1)
    return y

def detrend(X):
    N = X.shape[-1]
    # create linear trend matrix 
    A = np.zeros((2,N),dtype=X.dtype)
    A[1,:] = 1
    A[0,:] = np.linspace(0,1,N)
    R = A @ np.transpose(A)
    Rinv = np.linalg.inv(R)
    factor = np.transpose(A) @ Rinv
    X -= (X @ factor) @ A
    return None

def test_train_split(df,split=0.8,):
    N = df.shape[0]
    NOISEdf = df[df['trace_category'] == "noise"]
    EQdf = df[df['trace_category'] == "earthquake_local"]
    EQdf = clean(EQdf)
    EQdf = EQdf.sort_values("trace_start_time")

    # do EQ test train split 
    Nsplit = int(np.round(EQdf.shape[0]) * split)
    EQtrain = EQdf.iloc[:Nsplit,:]
    EQtest = EQdf.iloc[Nsplit:,:]

    # do NOISE test train split 
    NOISEtrain = NOISEdf[pd.to_datetime(NOISEdf["trace_start_time"]) < lasttime]
    NOISEtest = NOISEdf[pd.to_datetime(NOISEdf["trace_start_time"]) > lasttime]
    return EQtrain, EQtest, NOISEtrain, NOISEtest

def extract_snr(snr):
    N = snr.shape[0]
    snr3 = np.zeros((3,N))
    for ii in range(N):
        snr[ii] = snr[ii].replace("nan","0.0")
        snr3[:,ii] = np.float32(re.findall(r"[-+]?\d*\.\d+|\d+", snr[ii]))
    snr3[snr3 == 0] = np.nan
    return snr3

def clean(df):
    # round p and s arrivals to int
    df["p_arrival_sample"] = df["p_arrival_sample"].round().astype(int)
    df["s_arrival_sample"] = df["s_arrival_sample"].round().astype(int)

    # remove where p_wave starts too early in the sample 
    df = df[df["p_arrival_sample"] >= 400]

    # remove earthquakes > 100 km away 
    df = df[df["source_distance_km"] < 100]

    # update snr with median value 
    df["snr_db"] = np.nanmedian(extract_snr(df["snr_db"].values),axis=0)

    # get snr > 40 
    df = df[df["snr_db"] > 40] 
    
    # check p and s arrival samples
    df = df[df["p_arrival_sample"] > 200]
    df = df[df["s_arrival_sample"] - df["p_arrival_sample"] > 20]

    # find where p and s weight greater than 0.5 
    df = df[df["p_weight"] > 0.5]
    df = df[df["s_weight"] > 0.5]

    return df 
