"""Small streaming LHE reader/writer preserving hard-process metadata and weights."""
from dataclasses import dataclass
from contextlib import contextmanager
import gzip
import os
from pathlib import Path
import re
import tempfile
import xml.etree.ElementTree as ET


def number(value):
    return float(value.replace('D', 'E').replace('d', 'e'))


def read_momenta(line):
    f = line.split()
    if len(f) != 13:
        raise ValueError('An LHE particle must have 13 fields')
    return [int(f[0]), int(f[1]), *map(number, f[6:11]), int(f[4]), int(f[5])]


@dataclass
class LHEEvent:
    particles: list
    header: list
    extra: str = ''

    @property
    def weight(self):
        return number(self.header[2])

    @property
    def multiweights(self):
        root = ET.fromstring('<extra>' + self.extra + '</extra>')
        return {w.attrib['id']: number(w.text.strip()) for w in root.iter('wgt')}


class LHEFile:
    def __init__(self, path):
        self.path = str(path)

    def __enter__(self):
        opener = gzip.open if self.path.endswith('.gz') else open
        self.stream = opener(self.path, 'rt')
        lines = []
        self.first = None
        for line in self.stream:
            if re.match(r'\s*<event(?:\s|>)', line):
                self.first = line
                break
            if '</LesHouchesEvents>' in line:
                self.first = line
                break
            lines.append(line)
        self.preamble = ''.join(lines)
        if '<LesHouchesEvents' not in self.preamble or '</init>' not in self.preamble:
            self.stream.close()
            raise ValueError('Missing LHE header/init block (check for an unexpanded Git LFS pointer)')
        return self

    def __exit__(self, *args):
        self.stream.close()

    def __iter__(self):
        line = self.first
        self.first = None
        while line is not None:
            if '</LesHouchesEvents>' in line:
                return
            if re.match(r'\s*<event(?:\s|>)', line):
                block = [line]
                for line in self.stream:
                    block.append(line)
                    if '</event>' in line:
                        break
                else:
                    raise ValueError('Truncated LHE event')
                try:
                    root = ET.fromstring(''.join(block))
                except ET.ParseError as exc:
                    raise ValueError(f'Invalid LHE event XML in {self.path}: {exc}') from exc
                rows = [row.strip() for row in (root.text or '').splitlines()
                        if row.strip() and not row.lstrip().startswith('#')]
                if not rows or len(rows[0].split()) != 6:
                    raise ValueError('Invalid LHE event header')
                header = rows[0].split()
                n = int(header[0])
                if len(rows) != n + 1:
                    raise ValueError('LHE particle count does not match its event header')
                extra = ''.join(ET.tostring(child, encoding='unicode') for child in root)
                yield LHEEvent([read_momenta(row) for row in rows[1:]], header, extra)
            line = next(self.stream, None)
        raise ValueError('Missing closing LesHouchesEvents tag')


def readlhefile(path, limit=None):
    """Compatibility API returning particles, nominal weights and reweights."""
    from itertools import islice
    with LHEFile(path) as source:
        records = list(islice(source, limit))
    return ([event.particles for event in records], [event.weight for event in records],
            [event.multiweights for event in records])


def write_event(stream, particles, original):
    """Write a flat shower record; preserve the original process/weight/scale fields."""
    header = [str(len(particles)), *original.header[1:]]
    stream.write('<event>\n' + ' '.join(header) + '\n')
    incoming = [i + 1 for i, p in enumerate(particles) if p[1] == -1]
    for p in particles:
        mothers = [0, 0] if p[1] == -1 else [min(incoming), max(incoming)]
        colors = p[7:9] if len(p) >= 9 else [0, 0]
        fields = [int(p[0]), int(p[1]), *mothers, *map(int, colors)]
        row = ' '.join(map(str, fields)) + ' ' + ' '.join(f'{x:.16e}' for x in p[2:7])
        # Unknown spin is 9 in LHE; do not fabricate polarized final states.
        stream.write(row + ' 0.0 9.0\n')
    stream.write(original.extra)
    stream.write('</event>\n')


@contextmanager
def lhe_output(path, preamble):
    """Publish complete files only; a failed shower leaves existing output intact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix=path.name + '.',
                                         suffix='.part', delete=False) as out:
            temporary = out.name
            out.write(preamble)
            yield out
            out.write('</LesHouchesEvents>\n')
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def assign_quark_colors(jets, start=501):
    """Continue the input color line through successive q->qg emissions."""
    next_tag = max([start, *[int(c) for p, _ in jets for c in p[7:9]]]) + 1
    for parent, daughters in jets:
        quark = parent[0] > 0
        current = int(parent[7 if quark else 8])
        for p in daughters:
            if p[0] == 21:
                p[7:9] = [current, next_tag] if quark else [next_tag, current]
                current = next_tag
                next_tag += 1
            else:
                p[7:9] = [current, 0] if quark else [0, current]
