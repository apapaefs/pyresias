import pyhepmc

# set the four momentum order: px, py, pz, E
fourmomentum_order = [1, 2, 3, 0]

# Create the run info object
RunInfo = pyhepmc.GenRunInfo()
RunInfo.tools = [("Pyresias", "0.1", "Pyresias Toy Parton Shower")]
RunInfo.weight_names = ["0"]

# Function that writes HepMC: 
def WriteHepMC(writer, events):
    # loop over events:
    for eventno, event in enumerate(events):
        # initialize event
        evt = pyhepmc.GenEvent(pyhepmc.Units.GEV, pyhepmc.Units.CM)
        # set the cross section
        cs = pyhepmc.GenCrossSection()
        cs.set_cross_section(1.2, 0.2, 3, 10)
        cs.xsec_err(0) == 0.2
        evt.cross_section = cs
        # the event weights
        evt.weights = [1]
        # the event number
        evt.event_number = eventno
        v1 = pyhepmc.GenVertex((1.0, 1.0, 1.0, 1.0))
        v1.status = 1
        # loop over particles
        for p in event:
            if p[1] == -1: # incoming
                particle = pyhepmc.GenParticle((p[2],p[3], p[4],p[5]), p[0], 4)
                v1.add_particle_in(particle)
            elif p[1] == 1: # outgoing
                particle = pyhepmc.GenParticle((p[2],p[3], p[4],p[5]), p[0], 1)
                v1.add_particle_out(particle)
            particle.generated_mass = 0
            evt.add_particle(particle)
        # set the in-out vertex:
        evt.add_vertex(v1)
        # set the runinfo
        evt.run_info = RunInfo
        writer.write(evt)
        

