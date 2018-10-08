import numpy as np
import freehead as fh
import pickle
from scipy.optimize import minimize
from scipy.optimize import basinhopping

othread = fh.OptotrakThread()
othread.start()
othread.started_running.wait()

probe = fh.FourMarkerProbe()

corners = ['upper left', 'upper right', 'lower left', 'lower right']
n_measurements = 5

results = {}
for c in corners:
    clist = []
    for i in range(n_measurements):
        print(c, 'measurement', i)
        
        measurement = {}
        measurement['dist'] = float(input('measured distance: '))
        
        print('measure position')
        while True:
            input('press enter')
            r_probe, t_probe = probe.solve(othread.current_sample[15:27].reshape((4, 3)))
            if fh.anynan(r_probe):
                print('try again')
                continue
            measurement['pos'] = t_probe
            break
        clist.append(measurement)
    results[c] = (clist)
        
print('done')
othread.should_stop.set()
othread.join()

#%%
with open('screen_corners_triangulation_data.pickle', 'wb') as file:
    pickle.dump(results, file)
    
#%%
with open('screen_corners_triangulation_data.pickle', 'rb') as file:
    results_read = pickle.load(file)
    
#%%
solved_corners = {}
for (corner, result) in results.items():
    distances = np.array([1000.0 * r['dist'] for r in result])
    positions = np.vstack([r['pos'] for r in result])
    
    def err_func(corner_pos):
        difference_vectors = positions - corner_pos
        differences = np.sqrt(np.sum(difference_vectors ** 2, axis=1))
        errors = differences - distances
        error = np.sum(np.abs(errors))
#        print(difference_vectors, '\n')
#        print(differences, '\n')
#        print(distances, '\n')
#        print(errors, '\n')
#        print(error, '\n')
#        input()
        return error
    
    optim_result = basinhopping(err_func, np.array([0.0, 5000.0, 1350.0]), niter=500)
    print('error', corner, optim_result.fun, 'mm')
    solved_corners[corner] = optim_result.x
    
with open('screen_corners_triangulation_result.pickle', 'wb') as file:
    pickle.dump(solved_corners, file)
    
with open('screen_corners_triangulation_result.pickle', 'rb') as file:
    solved_corners_read = pickle.load(file)