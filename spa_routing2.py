import nef
import spa2

class Rules:
    def rule1(state1='A'):
        set(state3=state1*state2)

class Model(spa2.SPA):
    state1 = spa2.Buffer(dimensions=32)
    state2 = spa2.Buffer(dimensions=32)
    state3 = spa2.Buffer(dimensions=32)

    bg = spa2.BasalGanglia(Rules)
    thal = spa2.Thalamus(bg)

net = nef.Network('SPA', fixed_seed=1)
model = Model(net)

import extract

extract.extract(net, filename='spa_routing2.txt')

