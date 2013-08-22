import numeric

def str_array(matrix):
    m = [str_vector(row) for row in matrix]
    return '[%s]'%(','.join(m))    
def str_vector(vector):
    return '[%s]'%(','.join(['%1.3f'%x for x in vector]))

def is_zero(vector):
    for x in vector:
        if x!=0: return False
    return True    
    
    
class Population:
    def __init__(self, name, tau_rc, tau_ref, neurons, dimensions):
        self.name = name
        self.tau_rc = tau_rc
        self.tau_ref = tau_ref
        self.neurons = neurons
        self.dimensions = dimensions
        self.projections = {}
    def add_projection(self, origin, dim, target, transform, tau, weights=False):
        for i in range(dim):
            name = '%s.%d'%(origin, i)
            if name not in self.projections:
                self.projections[name]=[]
            trans = numeric.array(transform)[:,i]  

            if not is_zero(trans):
                if weights: 
                    assert max(trans)==min(trans)
                    assert max(trans)<0
                    target=target+'*'
                self.projections[name].append(('%s'%target, trans[:1], tau))    
    def create_text(self):
        r=['%s, %g, %g, %d, %d'%(self.name, self.tau_rc, self.tau_ref, self.neurons, self.dimensions)]
        for name, proj in sorted(self.projections.items()):
            projs = ['(%s, %s, %g)'%(p[0], str_vector(p[1]), p[2]) for p in proj]
        
            r.append('    %s, [%s]'%(name, ','.join(projs)))
        
        return '\n'.join(r)        

class Data:
    def __init__(self):
        self.population = {}
        self.ensembles = {}
        self.inputs = {}
        self.arrays = {}
        self.networks = {}
    def add(self, name, pop):
        assert name not in self.population
        self.population[name]=pop
    def create_text(self):
        r=[]
        for k in sorted(self.population.keys()):
            r.append(self.population[k].create_text())
        return '\n'.join(r)    

def process_input(node, data, prefix):
    name = node.name
    if prefix is not None: name = '%s.%s'%(prefix,name)

    data.inputs[node]=name
    
def process_ensemble(node, data, prefix):
    name = node.name
    if prefix is not None: name = '%s.%s'%(prefix,name)
    
    data.add(name, Population(name, node.ensembleFactory.nodeFactory.tauRC, node.ensembleFactory.nodeFactory.tauRef, node.neurons, node.dimension))
    
    data.ensembles[node]=name

def process_array(node, data, prefix):
    name = node.name
    if prefix is not None: name = '%s.%s'%(prefix,name)
    
    for n in node.getNodes():
        process_ensemble(n, data, prefix=name)
    
    data.arrays[node]=name
    
        
def process_network(network, data, prefix=None):
    for n in network.getNodes():
        klass = n.__class__.__name__
        if klass in ['FunctionInput']:
            process_input(n, data, prefix=prefix)
        elif klass in ['NEFEnsembleImpl']:
            process_ensemble(n, data, prefix=prefix)
        elif klass in ['NetworkArrayImpl']:
            process_array(n, data, prefix=prefix)
        elif klass in ['NetworkImpl']:
            name = n.name
            if prefix is not None: name = '%s.%s'%(prefix,name)
            data.networks[n] = name
            process_network(n, data, prefix=name)
        else:
            print 'Unknown node',prefix,n.name, n.__class__.__name__
    
    for p in network.projections:
        handle_projection(p.origin, p.termination, data, prefix)

def handle_projection(origin, termination, data, prefix, transform_start=None, transform_end=None):
    node1 = origin.node
    node2 = termination.node
    
    while node1 in data.networks:
        origin = origin.getWrappedOrigin()
        node1 = origin.node
    while node2 in data.networks:
        termination = termination.getWrappedTermination()
        node2 = termination.node
    
    if node1 in data.inputs:
        pass
    elif node1 in data.arrays:
        start=0
        for o in origin.getWrappedOrigin().getNodeOrigins():
            end = start + o.dimensions
            handle_projection(o, termination, data, prefix, transform_start=start, transform_end=end)
            start += o.dimensions
    elif node2 in data.arrays:
        for t in termination.getWrappedTermination().getNodeTerminations():
            handle_projection(origin, t, data, prefix, transform_start=transform_start, transform_end=transform_end)
    elif node1 in data.ensembles and node2 in data.ensembles:
        if termination.__class__.__name__ in ['DecodedTermination']:
            trans = numeric.array(termination.transform)
            if transform_start is not None:
                trans = trans[:,transform_start:transform_end]
            data.population[data.ensembles[node1]].add_projection(origin.name, origin.dimensions, data.ensembles[node2], trans, termination.tau)
        elif termination.__class__.__name__ in ['EnsembleTermination']:    
            weights = numeric.array([t.weights for t in termination.getNodeTerminations()])
            data.population[data.ensembles[node1]].add_projection(origin.name, origin.dimensions, data.ensembles[node2], weights, termination.tau, weights=True)
            
        else:
            print 'Unknown projection', prefix, node1, node2
               
    else:
        print 'Unknown projection', prefix, node1, node2
    
            
def extract(network, filename=None):
    if hasattr(network, 'network'):
        network = network.network
        
    data = Data()
    process_network(network, data)
    
    text = data.create_text()
    if filename is not None:
        f = open(filename, 'w')
        f.write(text)
        f.close()
    return text
        
