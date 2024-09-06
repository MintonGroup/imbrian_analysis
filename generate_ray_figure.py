import numpy as np
import matplotlib.pyplot as plt
import ctem
import matplotlib.patches as patches
import matplotlib.colors as colors
from ctem import craterproduction
import matplotlib
import os

def find_melt_at_pixel(x,y,depth):
    nlayers, bothickness = traverse(y,x,depth)
    meltarray = np.zeros((nlayers, nlist))
    nm = np.zeros((nlayers, nlist))
    rt = np.zeros(nlayers)
    rt[0] = bothickness
    if nlayers > 1:
        rt[1:] = rego[y,x][-nlayers+1:]
    for i in range(1,nlayers+1):
        if i == 1:
            meltarray[i-1] = meltdist[y,x][-nlist:]
        else:
            meltarray[i-1] = meltdist[y,x][-(i * nlist):-(nlist * (i-1))]
        nm[i-1] = meltarray[i-1] * rt[-i]
    meltarray[nlayers-1] = meltarray[nlayers-1] * (rt[0]/rego[y,x][-nlayers])
    totmelt = np.zeros(nlist)
    for k in range(nlist):
        totmelt[k] = np.sum(meltarray[:,k])
    return totmelt, np.sum(rt)

if __name__ == '__main__':

	path = os.path.join(os.getcwd(),'ray_generation_data')
	gridsize = 1000
	pix = 12.32e3
	pixkm = pix / 1000

	rego = ctem.util.read_linked_list_binary(os.path.join(path,'rego_000001.dat'), gridsize)
	melt = ctem.util.read_linked_list_binary(os.path.join(path,'melt_000001.dat'), gridsize)
	stack = ctem.util.read_unformatted_binary(os.path.join(path,'/stack_000001.dat'), gridsize, kind='I4B')
	ejm = ctem.util.read_linked_list_binary(os.path.join(path,'/ejm_000001.dat'), gridsize)

	surfmelt = np.zeros((gridsize,gridsize))
	for i in np.arange(gridsize):
	    for j in np.arange(gridsize):
	        surfmelt[i,j] = ejm[i,j][-1] / (pix*pix)

	#Plot Figure 5

	f1, a1 = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

	cmap = plt.cm.viridis
	cmap.set_bad(color='#3C044A')

	# Plot the first data
	md = a1[0].imshow(surfmelt, origin='lower',cmap=cmap,vmin=0,vmax=100)
	cbar = f1.colorbar(md, label='Ejected melt thickness (m)', ax=a1[0],shrink=0.75)
	a1[0].yaxis.set_ticklabels([])  # Hide y-axis tick labels
	a1[0].xaxis.set_ticklabels([])  # Hide x-axis tick labels
	a1[0].set_xticks([])
	a1[0].set_yticks([])
	# Plot the second data
	md2 = a1[1].imshow(surfmelt, origin='lower', norm=colors.LogNorm(),cmap=cmap)
	cbar2 = f1.colorbar(md2, label='Ejected melt thickness (m)', ax=a1[1],shrink=0.75)
	a1[1].yaxis.set_ticklabels([])  # Hide y-axis tick labels
	a1[1].xaxis.set_ticklabels([])  # Hide x-axis tick labels
	a1[1].set_xticks([])
	a1[1].set_yticks([])
	# Make both plots square
	a1[0].set_aspect('equal')
	a1[1].set_aspect('equal')

	# Adjust the layout to fit everything nicely
	plt.tight_layout()
	matplotlib.rcParams.update({'font.size': 16})
	plt.savefig('figure_5.pdf')

