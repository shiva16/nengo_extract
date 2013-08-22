import nef
import spa2

class Rules:
    def rule1(state1='A'):
        set(state2=state1)

class Model(spa2.SPA):
    state1 = spa2.Buffer(dimensions=256)
    state2 = spa2.Buffer(dimensions=256)
    
    bg = spa2.BasalGanglia(Rules)
    thal = spa2.Thalamus(bg)

net = nef.Network('SPA', fixed_seed=1)
model = Model(net)
    
import extract

extract.extract(net, filename='spa_routing.txt')            
