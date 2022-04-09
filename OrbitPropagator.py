

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode

import planetary_data as pd

#Ini adalah class pertama
class OrbitPropagator:
    #If you don't pass anything in as the cb, it will take earth as the default value
    def __init__(self,r0,v0,tspan,dt,cb=pd.earth):
        self.r0 = r0
        self.v0 = v0
        self.tspan = tspan
        self.dt = dt
        self.cb = cb

    def propagate_orbit(self):
        #total number of steps:
        self.n_steps = int(np.ceil(self.tspan/self.dt))

        #initialize arrays
        self.ys = np.zeros((self.n_steps,6))
        self.ts = np.zeros((self.n_steps,1))

        #initial conditions
        self.y0 = self.r0 + self.v0 #because both of them are lists, they will combine instead of adding up the components
        self.ys[0] = np.array(self.y0)
        self.step = 1 #because we want the next state to fill in 1

        #initiate solver
        self.solver = ode(self.diffy_q)
        self.solver.set_integrator('lsoda')
        self.solver.set_initial_value(self.y0,0)

        #propagate orbit
        while self.solver.successful() and self.step<self.n_steps:
            self.solver.integrate(self.solver.t + self.dt)
            self.ts[self.step] = self.solver.t
            self.ys[self.step] = self.solver.y
            self.step+=1

        self.rs = self.ys[:,:3]
        self.vs = self.ys[:,3:]
    
    def diffy_q(self,t,y):
        #unpack the states that we need
        rx,ry,rz,vx,vy,vz=y
        r = np.array([rx,ry,rz])

        #norm of the radius vector
        norm_r = np.linalg.norm(r)

        #two body acceleration
        ax,ay,az= -r*self.cb["mu"]/ norm_r**3 #Ini pangkat 3 karena  kita punya bentuk mu/|r|^2 * r/|r|

        return [vx,vy,vz,ax,ay,az]

    def plot_3d(self,show_plot = False,save_plot = False,title = 'Test Title'):
        #3D plot
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111,projection = '3d')

        #plot trajectory and starting point
        ax.plot(self.r[:,0],self.r[:,1],self.r[:,2],'b',label = 'Trajectory',zorder = 1)
        ax.plot([self.r[0,0]],[self.r[0,1]],[self.r[0,2]],'bo',label = 'Initial Position', zorder = 1)

        #plot earth
        _u,_v = np.mgrid[0:2*np.pi:20j,0:np.pi:10j]
        _x = self.cb['radius']*np.cos(_u)*np.sin(_v)
        _y = self.cb['radius']*np.sin(_u)*np.sin(_v)
        _z = self.cb['radius']*np.cos(_v)
        ax.plot_surface(_x,_y,_z,cmap ="gist_earth",zorder = 0)

        

        l = self.cb['radius']*2.0
        x,y,z = [[0,0,0],[0,0,0],[0,0,0]]
        u,v,w = [[l,0,0],[0,l,0],[0,0,l]]
        ax.quiver(x,y,z,u,v,w,color = 'k')

        max_val = np.max(np.abs(self.r))
        ax.set_xlim([-max_val,max_val])
        ax.set_ylim([-max_val,max_val])
        ax.set_zlim([-max_val,max_val])
        ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)'); ax.set_zlabel('Z  (km)')
        ax.set_title(title)
        ax.set_box_aspect((1,1,1))

        if show_plot:
            plt.show()
        
        if save_plot:
            plt.savefig(title + '.png',dpi = 300)










