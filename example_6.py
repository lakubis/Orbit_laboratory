#%%

import matplotlib
from matplotlib import widgets


%matplotlib Widget

import planetary_data as pd
import tools as t
from OrbitPropagator import OrbitPropagator as OP

tspan = 3600*24*1.0
dt = 10.0

#central body
cb = pd.earth

if __name__ == '__main__':
    #ISS
    c0 = [cb['radius']+ 417.0,0.0004481,51.6448,0.0,7.0055,313.0706] 

    #GEO
    c1 = [cb['radius'] + 35800.0,0.0,0.0,0.0,0.0,0.0]

    #random
    c2 = [cb['radius'] + 3000, 0.3,20.0,0.0,15.0,40.0]

    op0 = OP(c0,tspan,dt,True,cb = cb)
    op1 = OP(c1,tspan,dt,True,cb = cb)
    op2 = OP(c2,tspan,dt,True,cb = cb)


    op0.propagate_orbit()
    op1.propagate_orbit()
    op2.propagate_orbit()

    t.plot_n_orbits([op0.rs,op1.rs,op2.rs],labels = ['ISS', 'GEO', 'Random'],cb = cb, show_plot= True)

# %%
