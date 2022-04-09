#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('dark_background')
import planetary_data as pd
from OrbitPropagator import OrbitPropagator as OP

cb = pd.earth

if __name__ == "__main__":
    #Initial condition
    r_mag = cb["radius"]+1500.0
    v_mag = np.sqrt(cb["mu"]/r_mag)

    #initial position and velocity vectors
    r0 = [r_mag,r_mag*0.01,r_mag*-0.1]#it's a python list, that is because we want to add the lists together
    v0 = [0,v_mag,v_mag*0.3]

    #timespan
    tspan = 100*60.0

    #timestep
    dt = 100.0

    op = OP(r0,v0,tspan,dt,cb)
    op.propagate_orbit()
    op.plot_3d(show_plot=True)



# %%
