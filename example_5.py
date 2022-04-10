#%%
from tkinter import Widget


#%matplotlib Widget

import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import planetary_data as pd
import tools as t
from OrbitPropagator import OrbitPropagator as OP

print('program jalan')

tspan = 3600*24*1.0
dt = 10.0

cb = pd.earth

if __name__ == "__main__":
    r_mag = cb['radius']+400
    v_mag = np.sqrt(cb['mu']/r_mag)

    r0 = np.array([r_mag,0,0])
    v0 = np.array([0,v_mag,0])

    r_mag = cb['radius']+1000
    v_mag = np.sqrt(cb['mu']/r_mag)*1.3

    r00 = np.array([r_mag,0,0])
    v00 = np.array([0,v_mag,0.3])

    op0 = OP(r0,v0,tspan,dt,cb = cb)
    op00 = OP(r00,v00,tspan,dt,cb = cb)

    op0.propagate_orbit()
    op00.propagate_orbit()

    t.plot_n_orbits([op0.rs,op00.rs],labels = ['planet 0','planet 1'],cb = cb, show_plot=True)












# %%
plt.style.available



