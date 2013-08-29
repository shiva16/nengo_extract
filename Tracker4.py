import nef
import spa2
import math
import numeric as np
import hrr

from ca.nengo.model.impl import NetworkArrayImpl
def make_inhib_gate(net,name='Gate', gated='visual', neurons=40 ,pstc=0.01):
    gate=net.make(name, neurons, 1, intercept=(0.7, 1), encoders=[[1]])
    def addOne(x):
        return [x[0]+1]            
    net.connect(gate, None, func=addOne, origin_name='xBiased', create_projection=False)
    output=net.network.getNode(gated)
    if isinstance(output,NetworkArrayImpl):
        weights=[[-10]]*(output.nodes[0].neurons*len(output.nodes))
    else:
        weights=[[-10]]*output.neurons
    
    count=0
    while 'gate_%02d'%count in [t.name for t in output.terminations]:
        count=count+1
    oname = str('gate_%02d'%count)
    output.addTermination(oname, weights, pstc, False)
    
    orig = gate.getOrigin('xBiased')
    term = output.getTermination(oname)
    net.network.addProjection(orig, term)

########## Global params ##########
D = 80 #Number of dimensions in semantic pointers
aN = 30 #Number of neurons per dimension in arrays
cN = 70 #Number of neurons per dim in convolution 
#grp = 20 #Number of dimensions grouped in arrays

#for position counter
gain = 1.0
pstc_inhib = 0.005
pstc_in = 0.05
pstc_fb = 0.1
in_scale = 2.0      # Input scale to the integrators (default = 1.0)
l_intercept = 0.1   # intercepts lower value (default = 0.1): for cleanup purposes

########## Behaviour rules ##########

class Rules:
    def ready(vision='2*READY'): #If vision is READY
        set(state='5*LOAD') #Go to LOAD state
        set(position_gen='P0') #Reset position
        set(memory_ResetMem='UPDATE') #Reset memory
        set(motor_ResetTimer='UPDATE') #Reset motor timer
        set(motor_InhibitMotor='UPDATE')
            
    def load(state='LOAD', vision='-READY-GO'):
        set(memory_Load='UPDATE')
        set(position_gen_Update='UPDATE') #Start increment position  
    
    def next_input(vision='ZIP-READY-GO', state='.8*LOAD'): #If vision is 
            #empty and LOAD state
        set(state='LOAD') #Finish increment position and update load state for fun
   
    def start_move(vision='2*GO-READY', state='.5*LOAD'): #If vision is GO and 
            #in LOAD state
        set(state='5*MOVE') #Go to MOVE state
        set(position_gen='P0') #Reset position
        set(motor_ResetTimer='UPDATE') #Reset timer

    def move(state='MOVE'): #If motor is 
            #done a step and in MOVE state
        set(position_gen_Update='UPDATE') #Increment position
        set(motor_StartTimer='UPDATE') #Start the motor timer
        set(motor_DisinhibitMotor='UPDATE')
    
    def next_move(motor='UPDATE', state='.8*MOVE'):
        #set(position_gen_Update='-UPDATE')
        set(motor_DisinhibitMotor='UPDATE')
        set(motor_ResetTimer='UPDATE') #Reset timer
    
     
########## Additional modules ##########

class Vision(spa2.Module):
    def init(self, dimensions=16, aN=30):
        self.net.make('Threshold', 2, 2, intercept=(0.3, 1.0))
        self.net.make('Vision Integrator', 2, 2)
        self.net.make_array('Cleanup', aN, 4, 1, encoders=[[1]])
        self.net.make_array('Visual SP', aN, D, radius=1.0/math.sqrt(D))

        self.spa.add_source(self, 'Visual SP')
        self.spa.add_sink(self, 'Visual SP')

    def connect(self):
        self.net.connect('Vision Integrator', 'Threshold')
        self.net.connect('Vision Integrator', 'Vision Integrator')
        
        vocab = self.spa.sources[self.name]
        vocab.parse('RIGHT+LEFT+FWD+BACK')  #add items to vocab

        pd = [] # list of preferred direction vectors
        for item in ['RIGHT','LEFT','FWD','BACK']:
            pd.append(vocab[item].v.tolist())
              
        transform = np.array(pd).T
    
        self.net.connect('Threshold', 'Cleanup', transform=[[0,1],[-1,0],[1,0],[-1,0]])
        self.net.connect('Cleanup', 'Visual SP', transform=transform)

class State(spa2.Module):
    def init(self, dimensions=16, aN=30):
        self.net.make_array('Reset',1,D, mode='direct')
        self.net.make_array('State', aN, D, intercept=(l_intercept,1), 
            radius=1.0/math.sqrt(D)) #The current state
        self.net.make_array('Passthrough', aN, D, radius=1.0/math.sqrt(D)) 
        
        self.spa.add_sink(self, 'Reset') #Pos to bind
        self.spa.add_source(self, 'State') #What is currently in memory

    def connect(self):

        self.net.connect('State', 'Passthrough', pstc=pstc_fb)
        self.net.connect('Passthrough', 'State', weight = in_scale, pstc = pstc_in)
        self.net.connect('Reset','Passthrough')

class Memory(spa2.Module):
    def init(self, dimensions=16, aN=30):
        self.net.make_array('Difference', aN, D, radius=1.0/math.sqrt(D))
        self.net.make_array('Serial memory', aN, D, radius=1.0/math.sqrt(D))
        self.net.make_array('Vision input', 1, D, mode='direct') #Direct mode pop
        self.net.make_array('Position input', 1, D, mode='direct') #Direct mode pop
        self.net.make('ResetSP', 1, D, mode='direct')
        self.net.make('LoadSP', 1, D, mode='direct')

        nef.templates.gate.make(self.net, name='Load', gated='Difference', 
            neurons=20, pstc=0.005)
        make_inhib_gate(self.net, name='Reset', gated='Serial memory', 
            neurons=20, pstc=0.005)

        self.spa.add_sink(self, 'ResetSP', 'ResetMem') #Empty the memory if set to UPDATE
        self.spa.add_sink(self, 'LoadSP', 'Load') #Load something into memory
        self.spa.add_sink(self, 'Vision input', 'VisionIn')  #Elements to be bound
        self.spa.add_sink(self, 'Position input', 'PositionIn') #Pos to bind
        self.spa.add_source(self, 'Serial memory') #What is currently in memory
        
    def connect(self):
        vocab = self.spa.sources[self.name]
        self.net.connect('ResetSP','Reset', transform=vocab.parse("UPDATE").v)
        self.net.connect('LoadSP','Load', transform=vocab.parse("UPDATE").v)

        nef.convolution.make_convolution(self.net, 'Bind position', 
            'Vision input', 'Position input', 'Difference', cN, 1)

        self.net.connect('Serial memory', 'Serial memory', pstc=0.1)
        self.net.connect('Serial memory', 'Difference', weight=-1)
        self.net.connect('Difference', 'Serial memory')

class Position(spa2.Module):
    def init(self, dimensions=16, aN=30):   
    
        self.net.make_array('Current position', aN, D, 
            intercept=(l_intercept,1), radius=1.0/math.sqrt(D)) 
        self.net.make_array('Store', aN, D, 
            intercept=(l_intercept,1), radius=1.0/math.sqrt(D)) #Store next target
        self.net.make_array('Update', aN, D, 
            radius=1.0/math.sqrt(D)) #Compute the update of current position
        self.net.make_array('Passthrough', aN, D, 
            radius=1.0/math.sqrt(D)) #Passthrough of state
        self.net.make('UpdateSP', 1, D, mode='direct')
        self.net.make('Pass repn', aN, 1)

        nef.templates.gate.make(self.net, name='Update gate', gated='Update', 
            neurons=20, pstc=pstc_inhib)
        nef.templates.gate.make(self.net, name='Pass gate', gated='Passthrough', 
            neurons=20, pstc=pstc_inhib)

        self.spa.add_sink(self, 'Passthrough')  #To set the position on reset
        self.spa.add_sink(self, 'UpdateSP', 'Update') #Control to do one step
        self.spa.add_source(self, 'Current position') #To report the cur position
        
    def connect(self):
        vocab = self.spa.sources[self.name]
        self.net.connect('UpdateSP','Pass repn', transform=vocab.parse("UPDATE").v)
        def addOne(x):
            return -x[0]+1
        self.net.connect('Pass repn', 'Pass gate', func=addOne)
        self.net.connect('Pass repn', 'Update gate')
        
        self.net.connect('Update', 'Store', 
            transform=vocab.parse("ADD1").get_transform_matrix(), pstc=pstc_in)

        self.net.connect('Current position','Update', pstc=pstc_fb)
        self.net.connect('Update','Current position', pstc=0.005)

        self.net.connect('Store', 'Passthrough', pstc=pstc_fb)
        self.net.connect('Passthrough', 'Current position', 
            weight = in_scale, pstc = pstc_in)
        self.net.connect('Passthrough', 'Store', pstc=0.005)

class Motor(spa2.Module):    # Motor
    def init(self, dimensions=16, aN=30):
        self.net.make_array('Memory input', 1, D, mode='direct') #Direct mode pop
        self.net.make_array('Position input', 1, D, mode='direct') #Direct mode pop   
        self.net.make_array('Move direction', aN, D, radius=1.0/math.sqrt(D)) 
        self.net.make('Motor command', 200, 2)
        
        self.net.make('Motor timer', 100, 1, encoders=[[1]], intercept=[.1,1])
        self.net.make('Timer thresh', 20, 1, encoders=[[1]], intercept=[.8,1])
        self.net.make('Reset timer', 1, D, mode='direct')
        self.net.make('Start timer', 1, D, mode='direct')
        self.net.make('Timer done', 1, D, mode='direct')
        self.net.make('Disinhibit motor', 1, D, mode='direct')
                
        nef.templates.gate.make(self.net, name='Disinhibit', gated='Move direction', 
            neurons=20, pstc=0.005)
        
        self.spa.add_sink(self, 'Memory input', 'MemoryIn')
        self.spa.add_sink(self, 'Disinhibit motor', 'DisinhibitMotor')
        self.spa.add_sink(self, 'Position input', 'PositionIn')
        self.spa.add_sink(self, 'Reset timer', 'ResetTimer')
        self.spa.add_sink(self, 'Start timer', 'StartTimer')
        self.spa.add_source(self, 'Timer done')
    
    def connect(self):
        vocab = self.spa.sinks[self.name+'_MemoryIn']

        self.net.connect('Disinhibit motor','Disinhibit', 
            transform=vocab.parse("UPDATE").v)        

        nef.convolution.make_convolution(self.net, 'Unbind motor', 
            'Memory input', 'Position input', 'Move direction', cN, 1, invert_second=True)

        #Connect the motor timer elements
        self.net.connect('Motor timer', 'Motor timer', weight=1, pstc=.1)
        make_inhib_gate(self.net, name='Reset', gated='Motor timer', 
            neurons=20, pstc=0.005)
        self.net.connect('Reset timer','Reset', transform=vocab.parse("UPDATE").v)
        self.net.connect('Start timer', 'Motor timer', 
            transform=.2*vocab.parse("UPDATE").v)
        self.net.connect('Motor timer', 'Timer thresh')
        self.net.connect('Timer thresh', 'Timer done', transform=vocab.parse("UPDATE").v)
 
        #Connect the elements to compute the motor command from an SP       
        pd=[]
        pd.append(vocab["LEFT"].v.tolist())
        pd.append(vocab["RIGHT"].v.tolist())
        pd.append(vocab["FWD"].v.tolist())
        pd.append(vocab["BACK"].v.tolist())
        self.net.make_array('Cleanup', aN, 4, encoders=[[1]], intercept = [.1, 1])

        self.net.connect('Move direction', 'Cleanup', transform=pd)
        self.net.connect('Cleanup', 'Motor command', 
            transform=[[-1, 1, 0, 0],[0,0,1,-1]])

#Define the main model class
class Tracker(spa2.SPA):
    dimensions=D
    vision = Vision()
    state = State()
    position_gen = Position()
    memory = Memory()
    motor = Motor()
        
    bg = spa2.BasalGanglia(Rules)
    thal = spa2.Thalamus(bg)
    
    input = spa2.Input(0.01, vision='0*READY')
    input.next(0.1, vision='READY')
    input.next(0.15, vision='RIGHT')
    input.next(0.15, vision='ZIP')
    input.next(0.14, vision='BACK')
    input.next(0.15, vision='GO')
    input.next(10, vision='0*ZIP')
    
########### Main Network ##########

# Construct the network
net = nef.Network('Tracker', fixed_seed=3)

vocab = hrr.Vocabulary(D, max_similarity = 0.1, include_pairs = False, 
        unitary = ["ADD1", "P0"])
vocab.add('P1', vocab.parse('P0*ADD1'))
vocab.add('P2', vocab.parse('P1*ADD1'))

tracker=Tracker(net, vocab=vocab)

net.connect('memory.Serial memory', 'motor.Memory input')
net.connect('vision.Visual SP', 'memory.Vision input')
net.connect('position_gen.Current position', 'memory.Position input')
net.connect('position_gen.Current position', 'motor.Position input')

##Test inputs
net.make_array('Vision input', 1, D, mode='direct')
net.connect('Vision input', 'vision.Visual SP')


net.set_layout({'state': 0, 'height': 521, 'width': 950, 'x': 1920, 'y': 0},
 [(u'vision.Visual SP', 'semantic pointer', {'label': False, 'normalize': False, 'sel_dim': [0, 1, 2, 3, 4], 'last_maxy': 1.0, 'sel_all': False, 'height': 61, 'width': 195, 'x': 32, 'smooth_normalize': False, 'fixed_y': None, 'show_graph': False, 'show_pairs': False, 'autozoom': False, 'y': 149}),
  (u'position_gen.Current position', 'semantic pointer', {'label': True, 'normalize': False, 'sel_dim': [0, 1, 2, 3, 4], 'last_maxy': 1.0, 'sel_all': False, 'height': 103, 'width': 208, 'x': 487, 'smooth_normalize': False, 'fixed_y': None, 'show_graph': False, 'show_pairs': False, 'autozoom': False, 'y': 13}),
  (u'bg', None, {'label': False, 'height': 33, 'width': 36, 'x': 269, 'y': 342}),
  (u'vision', None, {'label': False, 'height': 33, 'width': 68, 'x': 194, 'y': 209}),
  (u'memory', None, {'label': False, 'height': 33, 'width': 93, 'x': 280, 'y': 117}),
  (u'motor', None, {'label': False, 'height': 33, 'width': 70, 'x': 518, 'y': 220}),
  (u'position_gen', None, {'label': False, 'height': 33, 'width': 138, 'x': 411, 'y': 121}),
  (u'state', None, {'label': False, 'height': 33, 'width': 60, 'x': 305, 'y': 231}),
  (u'thal', None, {'label': False, 'height': 33, 'width': 49, 'x': 405, 'y': 345}),
  (u'motor.Motor command', 'XY plot|X', {'label': True, 'sel_dim': [0, 1], 'last_maxy': 1.0, 'height': 200, 'width': 200, 'x': 714, 'autohide': True, 'autozoom': True, 'y': 178}),
  (u'motor.Motor timer', 'value|X', {'label': True, 'sel_dim': [0], 'last_maxy': 1.0, 'sel_all': True, 'height': 117, 'width': 176, 'x': 531, 'fixed_y': None, 'autozoom': False, 'y': 267}),
  (u'memory.Serial memory', 'semantic pointer', {'label': True, 'normalize': False, 'sel_dim': [], 'last_maxy': 1.0, 'sel_all': False, 'height': 95, 'width': 264, 'x': 188, 'smooth_normalize': False, 'fixed_y': None, 'show_graph': False, 'show_pairs': True, 'autozoom': False, 'y': 9}),
  (u'motor.Move direction', 'semantic pointer', {'label': True, 'normalize': False, 'sel_dim': [0, 1, 2, 3, 4], 'last_maxy': 1.0, 'sel_all': False, 'height': 78, 'width': 181, 'x': 721, 'smooth_normalize': False, 'fixed_y': None, 'show_graph': False, 'show_pairs': False, 'autozoom': False, 'y': 63})],
 {'dt': 0, 'show_time': 0.19999999999999998, 'sim_spd': 0, 'update_freq': 30.303030303030305, 'filter': 0.03, 'rcd_time': 4.0}) 

net.add_to_nengo()
#net.view()

import extract
extract.extract(net, 'Tracker4.txt')
# End network construction


