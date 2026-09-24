"""Optional HepMC output. No cross section is invented when none is supplied."""
from itertools import repeat


def WriteHepMC(writer, events, weights=None, cross_section=None):
    import pyhepmc
    info = pyhepmc.GenRunInfo()
    info.tools = [("Pyresias", "0.2", "Educational parton shower")]
    info.weight_names = ["nominal"]
    weights = repeat(1.) if weights is None else iter(weights)
    for eventno, particles in enumerate(events):
        event = pyhepmc.GenEvent(pyhepmc.Units.GEV, pyhepmc.Units.MM)
        event.run_info = info
        event.weights = [next(weights)]
        event.event_number = eventno
        if cross_section is not None:
            section = pyhepmc.GenCrossSection()
            section.set_cross_section(*cross_section)
            event.cross_section = section
        vertex = pyhepmc.GenVertex()
        for p in particles:
            if p[1] not in (-1, 1):
                continue
            particle = pyhepmc.GenParticle(tuple(p[2:6]), int(p[0]), 4 if p[1] == -1 else 1)
            particle.generated_mass = float(p[6])
            if p[1] == -1:
                vertex.add_particle_in(particle)
            else:
                vertex.add_particle_out(particle)
        event.add_vertex(vertex)
        writer.write(event)
