import numpy as np
import pythia8 #conda install -c conda-forge pythia8
from pyjet import cluster,DTYPE_PTEPM #pip install pyjet
from numpy.lib.recfunctions import append_fields

pythia = pythia8.Pythia("", False)

pythia.readString("Random:setSeed = on")
pythia.readString("Random:seed = 0") #change to 0 to be tied to time of day.

pythia.readString("WeakSingleBoson:ffbar2gmZ  = on");
pythia.readString("23:onMode = off"); #23 = Z boson
pythia.readString("23:onIfAny =  1 2 3");
pythia.readString("Print:quiet = on");

pythia.readString("Beams:idA = 11"); #11 = electron
pythia.readString("Beams:idB = -11");
pythia.readString("Beams:eCM = 92"); #units are GeV (this is LEP 1 energy)

#Parameters to fit
strange = 0.275;
alphas = 0.16;
lundval = 0.8; 

pythia.readString("TimeShower:alphaSvalue = "+str(alphas))
pythia.readString("StringZ:aLund = "+str(lundval))
pythia.readString("StringFlav:probStoUD = "+str(strange))

pythia.init();

for k in range(1):

    vetoevent = False

    while(vetoevent == False):
        print("HERE")
        pythia.next()
    
        jet_inputs_ids = []
        jet_input_pt = []
        jet_input_eta = []
        jet_input_phi = []
        for ip in range(pythia.event.size()):
            #cut out quarks and gluons (we only observe hadrons!)
            if (pythia.event[ip].isFinal()==False):
                continue;
            #cut out leptons
            if (abs(pythia.event[ip].id())==12):
                continue;
            if (abs(pythia.event[ip].id())==14):
                continue;
            if (abs(pythia.event[ip].id())==13):
                continue;
            if (abs(pythia.event[ip].id())==16):
                continue;
            jet_inputs_ids += [ip]
            jet_input_pt += [pythia.event[ip].pT()]
            jet_input_eta += [pythia.event[ip].eta()]
            jet_input_phi += [pythia.event[ip].phi()]
            pass
        
        pseudojets_input = np.zeros(len(jet_inputs_ids), dtype=DTYPE_PTEPM)
        for i in range(len(pseudojets_input)):
            pseudojets_input[i]['pT'] = jet_input_pt[i]
            pseudojets_input[i]['eta'] = jet_input_eta[i]
            pseudojets_input[i]['phi'] = jet_input_phi[i]
            pass
        
        pseudojets_input = append_fields(pseudojets_input, 'id', data=np.array(jet_inputs_ids))
        
        sequence = cluster(pseudojets_input, R=0.8, p=-1)
        jets = sequence.inclusive_jets(ptmin=25)
        if (len(jets)>0):

            #veto ISR
            if (len(jets[0])==1):
                vetoevent = False
                continue
            
            output_string = str(strange)+" "+str(alphas)+" "+str(lundval)+" "
            for constit in jets[0]:
                output_string += str(constit.pt)+" "+str(constit.eta)+" "+str(constit.phi)+" "+str(pythia.event[int(constit.id)].id()) + " "
                pass
            vetoevent = True
            print(output_string)
