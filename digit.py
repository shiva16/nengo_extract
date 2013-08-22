N=3                   # number of spiking neurons per RBM node for layers 1-3
N2=10                 # number of spiking neurons per RBM node for layer 4 
pstc=0.006            # post-synaptic time constant      
seed=2                # random number seed for neuron creation

import nef
import random
import numeric
import hrr

# utility function for reading data from a csv file
def read(fn):
    data=[]
    for line in file(fn).readlines():
        row=[float(x) for x in line.strip().split(',')]
        data.append(row)
    return data


net=nef.Network('RBM Digit Recognition',fixed_seed=seed)

# the sigmoid function used by the RBM model
def transform(x):
    return 1.0/(1+math.exp(-x[0]))

w1=read('mat_1_w.csv')   # weights for layer 1 (computed using standard Matlab learning model)
b1=read('mat_1_b.csv')   # bias for layer 1 (computed using standard Matlab learning model)

layer1=net.make_array('layer1',N,len(w1[0]),encoders=[[1]],intercept=(0,0.8))
bias1=net.make_input('bias1',b1[0])
net.connect(bias1,layer1)


w2=read('mat_2_w.csv')   # weights for layer 2 (computed using standard Matlab learning model)
b2=read('mat_2_b.csv')   # bias for layer 2 (computed using standard Matlab learning model)

layer2=net.make_array('layer2',N,len(w2[0]),encoders=[[1]],intercept=(0,0.8))
bias2=net.make_input('bias2',b2[0])
net.connect(bias2,layer2)
net.connect(layer1,layer2,func=transform,transform=numeric.array(w2).T,pstc=pstc)

w3=read('mat_3_w.csv')   # weights for layer 3 (computed using standard Matlab learning model)
b3=read('mat_3_b.csv')   # bias for layer 3 (computed using standard Matlab learning model)

layer3=net.make_array('layer3',N,len(w3[0]),encoders=[[1]],intercept=(0,0.8))
bias3=net.make_input('bias3',b3[0])
net.connect(bias3,layer3)
net.connect(layer2,layer3,func=transform,transform=numeric.array(w3).T,pstc=pstc)

w4=read('mat_4_w.csv')   # weights for layer 4 (computed using standard Matlab learning model)
b4=read('mat_4_b.csv')   # bias for layer 4 (computed using standard Matlab learning model)

layer4=net.make_array('layer4',N2,len(w4[0]))
bias4=net.make_input('bias4',b4[0])
net.connect(bias4,layer4)
net.connect(layer3,layer4,func=transform,transform=numeric.array(w4).T,pstc=pstc)


net.add_to_nengo()
        
        

