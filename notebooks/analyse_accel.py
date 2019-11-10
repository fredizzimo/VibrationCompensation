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
dt = time[1:] - time[:-1]
print(dt)
fig = go.Figure()
fig.add_trace(go.Scatter(x=np.array(range(dt.shape[0])), y=dt))
fig.show()

# %%
T = 0.001
n = (time / T).astype(np.int)
print(n[:10])

def reconstruct(time, values, step):
    print(time[0])
    print(time[-1])
    sample_points = np.arange(time[0], time[-1], step=step)
    print(sample_points)

    A = np.tile(-sample_points[1:], (time.shape[0], 1))
    A = A + time[:,np.newaxis]
    print(A.shape)
    with np.printoptions(precision=3, suppress=True):
        print(A)
    A = A / step
    A = np.sinc(A)
    #res = np.zeros(time.shape[0])
    res = npl.lstsq(A, values, rcond=None)[0]
    print(res.shape)
    
    resampled_times = np.arange(time[0], time[-1], step=0.0001)
    def eval(t):
        sum = 0
        n = 0
        for i in res:
            sum += i*np.sinc((t-step*n)/step)
            n+=1
        return sum
    resampled_values = [eval(t) for t in resampled_times]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sample_points, y=res))
    fig.add_trace(go.Scatter(x=time, y=values))
    fig.add_trace(go.Scatter(x=resampled_times, y=resampled_values))
    fig.show()

reconstruct(time[:100], ay[:100], 0.003)
#A = np.add(ay, sample_points)
#print(a.shape)

