import nef
import spa2

class Rules:
    def rule1(state='A'):
        set(state='B')
    def rule2(state='B'):
        set(state='C')
    def rule3(state='C'):
        set(state='D')
    def rule4(state='D'):
        set(state='E')
    def rule5(state='E'):
        set(state='A')

class Model(spa2.SPA):
    dimensions = 16
    
    state = spa2.Buffer(dimensions=16)
    
    bg = spa2.BasalGanglia(Rules)
    thal = spa2.Thalamus(bg)

net = nef.Network('SPA', fixed_seed=1)
model = Model(net)
    
import extract

extract.extract(net, filename='spa.txt')            
