#conda install -c conda-forge pythia8
import pythia8

#python -m pip install fastjet
import fastjet

evgen = pythia8.Pythia()
jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.8)

evgen.readString("WeakSingleBoson:ffbar2gmZ  = on");
evgen.readString("23:onMode = off");
evgen.readString("23:onIfAny =  1 2 3");
evgen.readString("Print:quiet = on");
evgen.readString("Beams:idA = 11");
evgen.readString("Beams:idB = -11");
evgen.readString("Beams:eCM = 92");

evgen.readString("TimeShower:alphaSvalue = 0.1365")
evgen.readString("StringFlav:probStoUD = 0.275")
evgen.readString("StringZ:aLund = 0.68")
evgen.init()

boolisphoton = True
while (boolisphoton):
    evgen.next()
    boolisphoton = False
    particlesForJets = []
    for i in range(evgen.event.size()):
        p = fastjet.PseudoJet(evgen.event[i].px(), evgen.event[i].py(), evgen.event[i].pz(), evgen.event[i].e())
        p.set_user_index(evgen.event[i].id())
        if (evgen.event[i].isFinal()==False):
            continue;
        if (abs(evgen.event[i].id())==12):
            continue;
        if (abs(evgen.event[i].id())==14):
            continue;
        if (abs(evgen.event[i].id())==13):
            continue;
        if (abs(evgen.event[i].id())==16):
            continue;
        particlesForJets += [p]
        pass

    cs = fastjet.ClusterSequence(particlesForJets, jetdef)
    myJets = fastjet.sorted_by_pt(cs.inclusive_jets(25.0));

    if (len(myJets) > 0):
        if (len(myJets[0].constituents())==1):
            myid = myJets[0].constituents()[0].user_index()
            origin = 0
            while (evgen.event[myid]==22):
                myid = evgen.event[myid].mother1()
                origin = evgen.event[myid].id()
                pass
            if (abs(origin)==11):
                boolisphoton = True
                pass
            pass
        pass

    if (len(myJets)==0):
        boolisphoton = True
    else:
        outstring = ""
        for i in range(len(myJets[0].constituents())):
            outstring+=str(myJets[0].constituents()[i].px())+" "+str(myJets[0].constituents()[i].py()) + " "+str(myJets[0].constituents()[i].pz()) + str(myJets[0].constituents()[i].user_index())
            pass
        print(outstring)
        
