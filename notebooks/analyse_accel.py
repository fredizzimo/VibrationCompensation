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
idx = np.where((time > 63) & (time < 64.5))
print(time[idx][0])
print(time[idx][1])
time_fa = time[idx] - time[idx][0]
ax_fa = ax[idx]
ay_fa = ay[idx]
az_fa = az[idx]
aT_fa = aT[idx]

# %%
fig = go.Figure()
fig.add_trace(go.Scatter(x=time_fa, y=ax_fa))
fig.add_trace(go.Scatter(x=time_fa, y=ay_fa))
fig.add_trace(go.Scatter(x=time_fa, y=az_fa))
fig.add_trace(go.Scatter(x=time_fa, y=aT_fa))
fig.show()


# %%
def generate_equally_sampled_signal(time, signal):
    time = np.array(np.round(time * 1000), dtype=np.int)
    nogap = np.arange(time[0], time[-1] + 1, 1)
    existing_times = np.isin(nogap, time, assume_unique=True)
    nonexisting_times = ~existing_times
    full_signal = np.zeros(nogap.shape[0])
    full_signal[existing_times] = signal
    signal = full_signal
    time = nogap 
    coeffs = pywt.wavedec(signal, 'db1', level=8, mode="zero")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=nogap, y=full_signal))
    for c in coeffs[1:]:
        fig.add_trace(go.Scatter(x=nogap, y=c))
    fig.show()
generate_equally_sampled_signal(time_fa, ay_fa)
