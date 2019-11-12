# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import numpy.linalg as npl
import plotly.graph_objects as go
import pywt
import scipy.fftpack as fftpack
import scipy.signal as signal
import math

# %%
res = np.loadtxt("../data/2019-04-0619.56.55.csv.txt", delimiter=",", skiprows=1)


# %%

def average_duplicate_times():
    unique, idx, count = np.unique(res[:,0], return_index=True, return_counts=True)
    ret = [np.concatenate(((u,), np.mean(res[i:i+c,1:], axis=0))) for u, i, c in zip(unique, idx, count)]
    return np.array(ret)
cleaned_res = average_duplicate_times()
time = cleaned_res[:,0]
ax = cleaned_res[:,1]
ay = cleaned_res[:,2]
az = cleaned_res[:,3]
aT = cleaned_res[:,4]

time_o = res[:,0]
ax_o = res[:,1]
ay_o = res[:,2]
az_o = res[:,3]
aT_o = res[:,4]

# %%
fig = go.Figure()
fig.add_trace(go.Scatter(x=time, y=ax))
fig.add_trace(go.Scatter(x=time, y=ay))
fig.add_trace(go.Scatter(x=time, y=az))
fig.add_trace(go.Scatter(x=time, y=aT))
#fig.add_trace(go.Scatter(x=time_o, y=ax_o))
#fig.add_trace(go.Scatter(x=time_o, y=ay_o))
#fig.add_trace(go.Scatter(x=time_o, y=az_o))
#fig.add_trace(go.Scatter(x=time_o, y=aT_o))
fig.show()


# %%
def extract_time(start, end):
    idx = np.where((time > start) & (time < end))
    ret_time = time[idx] - time[idx][0]
    return ret_time, ax[idx], ay[idx], az[idx], aT[idx]


# %%
time_fa, ax_fa, ay_fa,  az_fa, aT_fa = extract_time(63, 64.5)

# %%
fig = go.Figure()
fig.add_trace(go.Scatter(x=time_fa, y=ax_fa))
fig.add_trace(go.Scatter(x=time_fa, y=ay_fa))
fig.add_trace(go.Scatter(x=time_fa, y=az_fa))
fig.add_trace(go.Scatter(x=time_fa, y=aT_fa))
fig.show()


# %%
def plot_fft(fft):
    fig = go.Figure()
    freqs = fftpack.fftfreq(fft.shape[0], 1.0/1000)
    freqs = fftpack.fftshift(freqs)
    fft = fftpack.fftshift(fft)
    fig.add_trace(go.Scatter(x=freqs, y=np.abs(fft)))
    fig.show()


# %%
def cleanup_signal(time, signal):
    
    delta = np.max(np.abs(signal))
    
    time = np.array(np.round(time * 1000), dtype=np.int)
    nogap = np.arange(time[0], time[-1] + 1, 1)
    existing_times = np.isin(nogap, time, assume_unique=True)
    nonexisting_times = ~existing_times
    num_nonexisting = np.sum(nonexisting_times)
    full_signal = np.zeros(nogap.shape[0])
    full_signal[existing_times] = signal
    
    print(full_signal.shape)
    print(signal.shape)
    print(signal.shape[0] / full_signal.shape[0])
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=signal))
    fig.add_trace(go.Scatter(x=nogap, y=full_signal))
    fig.show()
    if False:
        #signal = full_signal
        #time = nogap 
        N = full_signal.shape[0]
        y = full_signal

        #D = np.zeros((N, N))
        #np.fill_diagonal(D, 1)
        #D = fftpack.fft(D, axis=1)
        #ones = np.ones(N)

        test = np.array([[1, 2],[3, 4]])
        print(np.sum(test, axis=1))
        print(test[0])

        for _ in range(5):
            #scaled_delta = delta*D[nonexisting_times]
            #d2 = delta*np.ones(N)
            diagonal_delta = np.zeros((N, N))
            np.fill_diagonal(diagonal_delta, delta)
            for _ in range(20):
                filled_y = np.tile(y, (N, 1))
                #Y = fftpack.fft(y)
                f1=filled_y[nonexisting_times] + diagonal_delta[nonexisting_times]
                f2=filled_y[nonexisting_times] - diagonal_delta[nonexisting_times]
                F1 = fftpack.fft(f1, axis=1)
                F2 = fftpack.fft(f2, axis=1)
                #g = np.sum(np.abs(Y + scaled_delta), axis=1)
                #g -= np.sum(np.abs(Y - scaled_delta), axis=1)
                g = np.sum(np.abs(F1), axis=1)
                g -= np.sum(np.abs(F2), axis=1)
                g /= N
                y[nonexisting_times] -= g
            delta = delta / np.sqrt(10)
            #delta /= 2.0

    N = full_signal.shape[0]
    y = full_signal
    
    D = np.zeros((N, N))
    np.fill_diagonal(D, 1)
    D = fftpack.fft(D, axis=1)
    
    original_delta = delta
    
    if False:
        for i in range(30):
            scaled_delta = delta*D[nonexisting_times]
            for _ in range(30):
                Y = fftpack.fft(y)
                g = np.sum(np.abs(Y + scaled_delta)[:, 100:800], axis=1)
                g -= np.sum(np.abs(Y - scaled_delta)[:,100:800], axis=1)
                g /= N
                y[nonexisting_times] -= 2 * delta * g
            delta = delta / np.sqrt(10)
            #delta /= 2.0

        delta = original_delta / 10
        
    prev_g = None
    alpha_limit = math.radians(170)
    delta = delta / 3
    for i in range(10):
        scaled_delta = delta*D[nonexisting_times]
        Y = fftpack.fft(y)
        plot_fft(Y)
        prev_g = None
        for j in range(10000):
            g = np.sum(np.abs(Y + scaled_delta), axis=1)
            g -= np.sum(np.abs(Y - scaled_delta), axis=1)
            g /= N
            y[nonexisting_times] -= g
            Y = fftpack.fft(y)
            if prev_g is not None:
                a = np.sum(g * prev_g)
                b = np.sqrt(np.sum(prev_g**2))
                b *= np.sqrt(np.sum(g**2))
                try:
                    alpha = math.acos(a/b)
                    if abs(alpha) > alpha_limit:
                        break
                except:
                    break
            prev_g = g
        delta = delta / np.sqrt(10)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=signal))
    fig.add_trace(go.Scatter(x=nogap, y=y))
    fig.show()
    plot_fft(Y)
    
t, _, y,  _, _fa = extract_time(63.9, 64.5)
cleanup_signal(t, y - 0.4)

# %%
print(fftpack.fftfreq(70, 1.0/1000))

# %%
