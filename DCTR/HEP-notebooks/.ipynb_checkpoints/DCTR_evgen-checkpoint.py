import os
import sys
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.append('/Users/matangrinberg/Library/CloudStorage/GoogleDrive-matan.grinberg@gmail.com/My Drive/(21-24) University of California, Berkeley/ML HEP/parametrized-classifiers/data')

import pythia8
import fastjet
import numpy as np
import numpy.random as random

def generate_event(event_generator, jet_definition, alphaS, aLund, probStoUD):
    boolisphoton = True
    while (boolisphoton):
        event_generator.next()
        boolisphoton = False
        particlesForJets = []
        for i in range(event_generator.event.size()):
            p = fastjet.PseudoJet(event_generator.event[i].px(), event_generator.event[i].py(), event_generator.event[i].pz(), event_generator.event[i].e())
            p.set_user_index(i)
            if (event_generator.event[i].isFinal()==False):
                continue;
            if (abs(event_generator.event[i].id())==12):
                continue;
            if (abs(event_generator.event[i].id())==14):
                continue;
            if (abs(event_generator.event[i].id())==13):
                continue;
            if (abs(event_generator.event[i].id())==16):
                continue;
            particlesForJets += [p]
            pass

        cs = fastjet.ClusterSequence(particlesForJets, jet_definition)
        myJets = fastjet.sorted_by_pt(cs.inclusive_jets(25.0));

        if (len(myJets) > 0):
            if (len(myJets[0].constituents())==1):
                myid = myJets[0].constituents()[0].user_index()
                origin = 0
                while (event_generator.event[myid].id()==22):
                    myid = event_generator.event[myid].mother1()
                    origin = event_generator.event[myid].id()
                    pass
                if (abs(origin)==11):
                    boolisphoton = True
                    pass
                pass
            pass

        if (len(myJets)==0):
            boolisphoton = True
        else:
            outstring = np.zeros((51,7))
            for i in range(len(myJets[0].constituents())):
                outstring[i] = myJets[0].constituents()[i].px(), myJets[0].constituents()[i].py(), myJets[0].constituents()[i].pz(), myJets[0].constituents()[i].user_index(), alphaS, aLund, probStoUD
                pass
    return outstring


def parameter_distribution(nx, ny=None, nz=None):
    alphaS0, aLund0, probStoUD0 = np.array([0.16]), np.array([0.7]), np.array([0.27])
    rad_alphaS, rad_aLund, rad_probStoUD = 0.032, 0.014, 0.052
    
    if ny==None: ny = nx
    if nz==None: nz = nx
    
    alphaS_low, aLund_low, probStoUD_low = alphaS0 - rad_alphaS, aLund0 - rad_aLund, probStoUD0 - rad_probStoUD
    alphaS_high, aLund_high, probStoUD_high = alphaS0 + rad_alphaS, aLund0 + rad_aLund, probStoUD0 + rad_probStoUD
    
    if nx == 1: alphaS = alphaS0
    else: alphaS = np.linspace(alphaS_low[0], alphaS_high[0], nx)
    
    if ny == 1: aLund = aLund0
    else: aLund = np.linspace(aLund_low[0], aLund_high[0], ny)
    
    if nz == 1: probStoUD = probStoUD0
    else: probStoUD = np.linspace(probStoUD_low[0], probStoUD_high[0], nz)
    
    return np.transpose(alphaS), np.transpose(aLund), np.transpose(probStoUD)


def generate_dataset(n_points, n_mult, nx, ny=None, nz=None):
    alphaS, aLund, probStoUD = parameter_distribution(nx, ny, nz)
    n = n_points * n_mult
    dataset = np.zeros((n, 51, 7))
    
    for i in range(n_points):
        alphaS_i = random.choice(alphaS)
        aLund_i = random.choice(aLund)
        probStoUD_i = random.choice(probStoUD)
        evgen = pythia8.Pythia()
        jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.8)

        evgen.readString("WeakSingleBoson:ffbar2gmZ  = on");
        evgen.readString("23:onMode = off");
        evgen.readString("23:onIfAny =  1 2 3");
        evgen.readString("Print:quiet = on");
        evgen.readString("Beams:idA = 11");
        evgen.readString("Beams:idB = -11");
        evgen.readString("Beams:eCM = 92");

        evgen.readString("TimeShower:alphaSvalue = " + str(alphaS_i))
        evgen.readString("StringZ:aLund = " + str(aLund_i))
        evgen.readString("StringFlav:probStoUD = " + str(probStoUD_i))
        evgen.init()
        
        for j in range(n_mult):
            dataset[i * n_mult + j] = generate_event(evgen, jetdef, alphaS_i, aLund_i, probStoUD_i)
            
    return dataset


def generate_dataset_ref(n_points, n_mult, nx, ny=None, nz=None):
    n = n_points * n_mult
    dataset = np.zeros((n, 51, 7))
    
    evgen = pythia8.Pythia()
    jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.8)

    evgen.readString("WeakSingleBoson:ffbar2gmZ  = on");
    evgen.readString("23:onMode = off");
    evgen.readString("23:onIfAny =  1 2 3");
    evgen.readString("Print:quiet = on");
    evgen.readString("Beams:idA = 11");
    evgen.readString("Beams:idB = -11");
    evgen.readString("Beams:eCM = 92");

    evgen.readString("TimeShower:alphaSvalue = 0.16")
    evgen.readString("StringZ:aLund = 0.7")
    evgen.readString("StringFlav:probStoUD = 0.27")
    evgen.init()

    alphaS_fake, aLund_fake, probStoUD_fake = parameter_distribution(nx, ny, nz)
    
    for i in range(n):
        alphaS_fake_i = random.choice(alphaS_fake)
        aLund_fake_i = random.choice(aLund_fake)
        probStoUD_fake_i = random.choice(probStoUD_fake)
        dataset[i] = generate_event(evgen, jetdef, alphaS_fake_i, aLund_fake_i, probStoUD_fake_i)
            
    return dataset
