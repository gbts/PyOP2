# This file is part of PyOP2
#
# PyOP2 is Copyright (c) 2012, Imperial College London and
# others. Please see the AUTHORS file in the main source directory for
# a full list of copyright holders.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * The name of Imperial College London or that of other
#       contributors may not be used to endorse or promote products
#       derived from this software without specific prior written
#       permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTERS
# ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.

"""
This demo verfieis that the integral of a unit cube is 1.

The cube will be unstructured in the 2D plane and structured vertically.
"""

from pyop2 import op2, utils
from pyop2.ffc_interface import compile_form
from triangle_reader import read_triangle
from ufl import *
from computeind import compute_ind_extr
#from computeind import swap_ind_entries
import sys

import numpy as np
import time

parser = utils.parser(group=True, description="PyOP2 2D mass equation demo")
parser.add_argument('-m', '--mesh',
                    action='store',
                    type=str,
                    required=True,
                    help='Base name of triangle mesh (excluding the .ele or .node extension)')

parser.add_argument('-l', '--layers',
                    action='store',
                    type=str,
                    required=True,
                    help='Base name of triangle mesh (excluding the .ele or .node extension)')
parser.add_argument('-p', '--partsize',
                    action='store',
                    type=str,
                    required=False,
                    help='Base name of triangle mesh (excluding the .ele or .node extension)')
opt = vars(parser.parse_args())
op2.init(**opt)
mesh_name = opt['mesh']
layers = int(opt['layers'])

# Generate code for kernel

mass = op2.Kernel("""
void comp_vol(double A[1], double *x[], int j)
{
    double abs = x[0][0]*(x[2][1]-x[4][1])+x[2][0]*(x[4][1]-x[0][1])+x[4][0]*(x[0][1]-x[2][1]);
    if (abs < 0)
      abs = abs * (-1.0);
    A[0]+=0.5*abs*0.1;
}""","comp_vol");

#mass = op2.Kernel("""
#void comp_vol(double A[1], double *x, int j)
#{
#    double abs = x[0]*(x[5]-x[9])+x[4]*(x[9]-x[1])+x[8]*(x[1]-x[5]);
#    if (abs < 0)
#      abs = abs * (-1.0);
#    A[0]+=0.5*abs*0.1;
#}""","comp_vol");


data_comp = op2.Kernel("""
void comp_dat(double *x[], double *y[], int j)
{
    for(int i=0; i<6; i++){
        for (int k=0; k<2; k++){
            x[i][k] = y[i][k];
        }
    }
    }""","comp_dat");



# Set up simulation data structures

valuetype=np.float64

nodes, coords, elements, elem_node = read_triangle(mesh_name) #<----------------------------------------------------<<

#mesh data
mesh2d = np.array([3,3,1])
mesh1d = np.array([2,1])
A = np.array([[0,1],[0]])

#the array of dof values for each element type
dofs = np.array([[2,0],[0,0],[0,0]])

#ALL the nodes, edges amd cells of the 2D mesh
nums = np.array([nodes.size,0,elements.size])

#compute the various numbers of dofs
dofss = dofs.transpose().ravel()

#number of dofs
noDofs = 0 #number of dofs

noDofs = np.dot(mesh2d,dofs)
#print "number of dofs per 2D element = %d" % noDofs[0]

noDofs = len(A[0])*noDofs[0] + noDofs[1]
#print "total number of dofs = %d" % noDofs



### Number of elements in the map only counts the first reference to the dofs related to a mesh element
map_dofs = 0
for d in range(0,2):
  for i in range(0,len(mesh2d)):
    for j in range(0,mesh2d[i]*len(A[d])):
      if dofs[i][d] != 0:
        map_dofs += 1
#print "The size of the dofs map is = %d" % map_dofs



### EXTRUSION DETAILS
wedges = layers - 1

### NEW MAP
# When building this map we need to make sure we leave space for the maps that
# might be missing. This is because when we construct the ind array we need to know which
# maps is associated with each dof. If the element to node is missing then
# we will  have the cell to edges in the first position which is bad
# RULE: if all the dofs in the line are ZERO then skip that mapping else add it

mappp = elem_node.values
mappp = mappp.reshape(-1,3)


lins,cols = mappp.shape
mapp=np.empty(shape=(lins,), dtype=object)

t0ind= time.clock()
### DERIVE THE MAP FOR THE EDGES
edg = np.empty(shape = (nums[0],),dtype=object)
for i in range(0, nums[0]):
  edg[i] = []

k = 0
count = 0
addNodes = dofs[0][0] != 0 or dofs[0][1] != 0
addEdges = dofs[1][0] != 0 or dofs[1][1] != 0
addCells = dofs[2][0] != 0 or dofs[2][1] != 0

for i in range(0,lins): #for each cell to node mapping
  ns = mappp[i] - 1
  ns.sort()
  pairs = [(x,y) for x in ns for y in ns if x<y]
  res = np.array([], dtype=np.int32)
  if addEdges:
    for x,y in pairs:
      ys = [kk for yy,kk in edg[x] if yy == y]
      if ys == []:
        edg[x].append((y,k))
        res = np.append(res,k)
        k += 1
      else:
        res = np.append(res,ys[0])
  if addCells:
    res = np.append(res, i) # add the map of the cell
  if addNodes:
    mapp[i] = np.append(mappp[i], res)
  else:
    mapp[i] = res

nums[1] = k #number of edges

### construct the initial indeces ONCE
### construct the offset array ONCE
off = np.zeros(map_dofs, dtype = np.int32)
### THE OFFSET array
#for 2D and 3D
count = 0
for d in range(0,2): #for 2D and then for 3D
  for i in range(0,len(mesh2d)): # over [3,3,1]
    for j in range(0,mesh2d[i]):
      for k in range(0,len(A[d])):
        if dofs[i][d]!=0:
            off[count] = dofs[i][d]
            count+=1

#assemble the dat
#compute total number of dofs in the 3D mesh
no_dofs = np.dot(nums,dofs.transpose()[0])*layers + wedges * np.dot(dofs.transpose()[1],nums)

###
#THE DAT
###
dat_size = np.dot(sum(np.dot(nums.reshape(1,3),dofs)),np.array([layers,layers-1]))
dat = np.zeros(dat_size)
#dat_mp = np.array(dat_size)

t0dat = time.clock()
count = 0
for d in range(0,2): #for 2D and then for 3D
  for i in range(0,len(mesh2d)): # over [3,3,1]
      for k in range(0, nums[i]):
        for k1 in range(0,layers-d):
          for l in range(0,dofs[i][d]):
            dat[count] = coords.data[k][l]
            count+=1
tdat = time.clock() - t0dat

### DECLARE OP2 STRUCTURES
#create the set of dofs, they will be our'virtual' mesh entity
dofsSet = op2.Set(no_dofs,"dofsSet")

#the dat has to be based on dofs not specific mesh elements
coords = op2.Dat(dofsSet, 1, dat, np.float64, "coords")
coords.setDatSize(dat_size)
#coords_mp = op2.Dat(dofsSet, 1, dat_mp, np.float64, "coords_mp")




### THE MAP from the ind
#create the map from element to dofs for each element in the 2D mesh
lsize = nums[2]*map_dofs
ind = compute_ind_extr(nums,map_dofs,lins,layers,mesh2d,dofs,A,wedges,mapp,lsize)

#SWAP ind entries for performance
#k = 10 #depth
#same = 2
#my_cache = map_dofs*np.arange(0,k,dtype = np.int32)
#pos = 0
#ahead = 100
#swaps = 0

#ind = swap_ind_entries(ind, k, map_dofs,lsize,ahead,my_cache,same)

#print "swaps = %d" % swaps


#ind = np.zeros(nums[2]*map_dofs, dtype=np.int32)
#count = 0
#for mm in range(0,lins):
# print mapp[mm]
#  offset = 0
#  for d in range(0,2):
#    c = 0
#    for i in range(0,len(mesh2d)):
#      if dofs[i][d] != 0:
#        for j in range(0, mesh2d[i]):
#          m = mapp[mm][c]
#          for k in range(0, len(A[d])):
#              ind[count] = m*dofs[i][d]*(layers - d) + A[d][k]*dofs[i][d] + offset
#              count+=1
#          c+=1
#      elif dofs[i][1-d] != 0:
#        c+= mesh2d[i]
#
#      offset += dofs[i][d]*nums[i]*(layers - d)
#
#tind = time.clock() - t0ind
#ppp = nums[2]*map_dofs
#print "size of ind = %d" % ind.size
#print "size of ind = %d" % ppp
# Create the map from elements to dofs
elem_dofs = op2.Map(elements,dofsSet,map_dofs,ind,"elem_dofs",off);


### THE RESULT ARRAY
g = op2.Global(1, data=0.0, name='g')

duration1 = time.clock() - t0ind

### ADD LAYERS INFO TO ITERATION SET
# the elements set must also contain the layers
elements.setLayers(layers)

##LOOP THAT COPIES THE DAT
#op2.par_loop(data_comp, elements,
#             coords_mp(elem_dofs, op2.INC),
#             coords(elem_dofs, op2.READ)
#            )


#t0loop= time.clock()
### CALL PAR LOOP
# Compute volume
tloop = 0
#for j in range(0,10):
t0loop= time.clock()
t0loop2 = time.time()
for i in range(0,100):
        op2.par_loop(mass, elements,
             g(op2.INC),
             coords(elem_dofs, op2.INC)
            )
tloop += time.clock() - t0loop # t is CPU seconds elapsed (floating point)
tloop2 = time.time() - t0loop2

ttloop = tloop / 10
print nums[0], nums[1], nums[2], layers, duration1, tloop, tloop2, g.data

#print g.data