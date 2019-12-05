"""
Example of spatial graph
"""
import geoplotlib
from geoplotlib.utils import read_csv
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fftpack import fft,fftfreq

#audio signal processing 
samplerate, data = wavfile.read("ImperialMarch60.wav")
samplerate
data.shape
samples = data.shape[0]
samples
plt.plot(data[:200])
datafft = fft(data)
#Get the absolute value of real and complex component:
fftabs = abs(datafft)
freqs = fftfreq(samples,1/samplerate)
plt.plot(freqs,fftabs)
plt.xlim( [10, samplerate/2] )
plt.xscale( 'log' )
plt.grid( True )
plt.xlabel( 'Frequency (Hz)' )
plt.plot(freqs[:int(freqs.size/2)],fftabs[:int(freqs.size/2)])
samplerate, data = wavfile.read("ImperialMarch60.wav")
data.shape
samples = data.shape[0]
samples
plt.plot(data[:4*samplerate]) #plot first 4 seconds
data = data[:]
data.shape
plt.plot(data[:4*samplerate]) #plot first 4 seconds
datafft = fft(data)

fftabs = abs(datafft)
freqs = fftfreq(samples,1/samplerate)
plt.xlim( [10, samplerate/2] )
plt.xscale( 'log' )
plt.grid( True )
plt.xlabel( 'Frequency (Hz)' )
plt.plot(freqs[:int(freqs.size/2)],fftabs[:int(freqs.size/2)])


#Scipy example of curve fitting noisy data
np.random.seed(0)

x = np.linspace(-1, 1, 2000)
y = np.cos(x) + 0.3*np.random.rand(2000)
p = np.polynomial.Chebyshev.fit(x, y, 90)

t = np.linspace(-1, 1, 200)
plt.plot(x, y, 'r.')
plt.plot(t, p(t), 'k-', lw=3)
plt.show()

#geoplot example of mapping patterns according to flight paths
data = read_csv('flights.csv')
geoplotlib.graph(data,
                 src_lat='lat_departure',
                 src_lon='lon_departure',
                 dest_lat='lat_arrival',
                 dest_lon='lon_arrival',
                 color='hot_r',
                 alpha=16,
                 linewidth=2)
geoplotlib.show()

# 3D plotting and visualization of signals using numpy
np.random.seed(1)

N = 70

fig = go.Figure(data=[go.Mesh3d(x=(70*np.random.randn(N)),
                   y=(55*np.random.randn(N)),
                   z=(40*np.random.randn(N)),
                   opacity=0.5,
                   color='rgba(244,22,100,0.6)'
                  )])

fig.update_layout(scene = dict(
        xaxis = dict(nticks=4, range=[-100,100],),
                     yaxis = dict(nticks=4, range=[-50,100],),
                     zaxis = dict(nticks=4, range=[-100,100],),),
                     width=700,
                     margin=dict(r=20, l=10, b=10, t=10))

fig.show()

