import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-dark')
import planetary_data as pd

from mpl_interactions import ioff, panhandler, zoom_factory

d2r = np.pi/100.0

def plot_n_orbits(rs, labels,cb, show_plot = False,save_plot = False,title = 'Many Orbits'):
        #3D plot
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111,projection = '3d')

        #plot trajectory and starting point
        n = 0
        for r in rs:
            ax.plot(r[:,0],r[:,1],r[:,2],label = labels[n] )
            ax.plot([r[0,0]],[r[0,1]],[r[0,2]])
            n+=1

        #plot earth
        _u,_v = np.mgrid[0:2*np.pi:20j,0:np.pi:10j]
        _x = cb['radius']*np.cos(_u)*np.sin(_v)
        _y = cb['radius']*np.sin(_u)*np.sin(_v)
        _z = cb['radius']*np.cos(_v)
        #ax.plot_surface(_x,_y,_z,cmap ="Blues",zorder = 0)
        ax.plot_wireframe(_x,_y,_z,color = "k", linewidth = 0.5)

        

        l = cb['radius']*2.0
        x,y,z = [[0,0,0],[0,0,0],[0,0,0]]
        u,v,w = [[l,0,0],[0,l,0],[0,0,l]]
        ax.quiver(x,y,z,u,v,w,color = 'k')

        max_val = np.max(np.abs(rs))
        ax.set_xlim([-max_val,max_val])
        ax.set_ylim([-max_val,max_val])
        ax.set_zlim([-max_val,max_val])
        ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)'); ax.set_zlabel('Z  (km)')
        ax.set_title(title)
        ax.legend()
        ax.set_box_aspect((1,1,1))

        if show_plot:
            #panhandler = panhandler(fig)
            #display(fig.canvas)
            plt.show()
        
        if save_plot:
            plt.savefig(title + '.png',dpi = 300)