"""Compatibility writer; prefer lhe_io to retain an existing LHE run header."""
import math
from lhe_io import LHEEvent, write_event


def init_lhe(filename, sigma, stddev, ECM):
    stream = open(filename, 'w')
    stream.write('<LesHouchesEvents version="1.0">\n<init>\n')
    stream.write(f'11 -11 {ECM/2:.16e} {ECM/2:.16e} 0 0 0 0 3 1\n')
    stream.write(f'{sigma:.16e} {stddev:.16e} 1.0 9999\n</init>\n')
    return stream


def write_lhe(stream, events, shat, debug=False, weights=None):
    """Write explicit event weights; unknown couplings use the LHE sentinel -1.

    Colored particles must carry the color/anticolor fields produced by the
    shower. This avoids inventing singlet gluons or an unrelated color flow.
    """
    for i, particles in enumerate(events):
        if any((1 <= abs(p[0]) <= 6 or p[0] == 21) and len(p) < 9 for p in particles):
            raise ValueError('Colored particles need color tags; use the shower event runner')
        weight = 1. if weights is None else weights[i]
        record = LHEEvent(particles, [str(len(particles)), '9999', str(weight),
                                      str(math.sqrt(shat)), '-1', '-1'])
        write_event(stream, particles, record)
        if debug:
            print(f'Wrote event {i} with weight {weight}')


def finalize_lhe(stream):
    stream.write('</LesHouchesEvents>\n')
    stream.close()
