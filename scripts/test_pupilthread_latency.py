import freehead as fh
import time
import numpy as np
import matplotlib.pyplot as plt

pthread = fh.PupilThread()
pthread.start()
pthread.started_running.wait()

for i in range(15):
    time.sleep(1)
    print(i + 1)
    
data = pthread.get_shortened_data()

pthread.should_stop.set()
pthread.join()

latencies = data[:, 1] - data[:, 0]
mean_latency = latencies.mean()

#%%
plt.scatter(np.arange(latencies.size), latencies, label='latencies', linewidth=1, facecolor=(0, 0, 0, 0), edgecolor=(0.3, 0.4, 0.7, 0.4))
plt.axhline(mean_latency, 0, 1, label='mean latency', color='k')
plt.title(f'Mean latency for sample receipt {mean_latency * 1000:.2f}ms')
plt.legend()
plt.show()

#%%
plt.figure()
plt.plot(np.diff(data[:, 1]))
plt.plot(np.diff(data[:, 0]))
plt.show()
