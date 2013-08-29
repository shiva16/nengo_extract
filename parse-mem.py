if True:
    import nef
    import spa2

    net = nef.Network('ParseMem',fixed_seed=1)

    class Rules:
        def verb(vision='WRITE+REMEMBER'):
            set(verb=vision)
        def noun(vision='ONE+TWO+NUMBER'):
            set(noun=vision)
        
        def write(vision='0.5*(NONE)-WRITE-ONE-TWO-NUMBER', phrase='0.7*WRITE*VERB-NOUN*NUMBER'):
            set(motor=phrase*'~NOUN')

        def write_memory(vision='0.4*(NONE)-WRITE-REMEMBER-ONE-TWO-NUMBER', phrase='0.3*(WRITE*VERB+NOUN*NUMBER)'):
            set(motor=memory)
            
        def remember(vision='0.4*(NONE)-WRITE-REMEMBER-ONE-TWO-NUMBER', phrase='0.5*REMEMBER*VERB'):
            set(memory=phrase*'~NOUN')
            

    class Model(spa2.SPA):
        dimensions = 512
        verbose = True

        vision = spa2.Buffer(feedback=0)
        
        noun = spa2.Buffer()
        verb = spa2.Buffer()
        phrase = spa2.Buffer(feedback=0)
        motor = spa2.Buffer(feedback=0)

        memory = spa2.Buffer()
        
        bg = spa2.BasalGanglia(Rules)
        thal = spa2.Thalamus(bg)
        
        

    model = Model(net)

    # connect noun and verb to phrase such that they compute the convolution
    #  with NOUN and VERB, respectively
    vocab = model.sources['phrase']
    net.connect('source_noun', 'sink_phrase', transform=vocab.parse('NOUN').get_transform_matrix())
    net.connect('source_verb', 'sink_phrase', transform=vocab.parse('VERB').get_transform_matrix())

    net.add_to_nengo()
        
    
    import extract
    extract.extract(net, filename='parse-mem.txt') 
