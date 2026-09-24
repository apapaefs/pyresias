# Pyresias: How to write a toy parton shower

Pyresias is a tutorial on building a simple parton shower in Python. The aim
is to connect the theoretical description of QCD radiation to code that can
be read, run and modified. We start with the Sudakov veto algorithm, then
use it to generate radiation in $e^+e^-\to q\bar q$ events and reconstruct
the final-state momenta.

The accompanying [tutorial paper](https://arxiv.org/abs/2406.03528) explains the
physics. The [Jupyter notebook](pyresias_nb.ipynb) walks through the first steps
of the implementation. Some familiarity with Python and basic QCD will be
useful.

## Getting started

Use Python 3.11 or 3.12. From the repository root, create an environment and
install the tutorial dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-tutorials.txt
```

Run the commands below from the same directory, with this environment active.
You can also open the notebook through Binder:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/apapaefs/pyresias/HEAD?labpath=pyresias_nb.ipynb)

## 1. Generate branchings with the Sudakov veto algorithm

Start with the notebook:

```bash
jupyter lab pyresias_nb.ipynb
```

Run the cells in order. They introduce the $q\to qg$ splitting function,
construct an overestimate, generate trial branchings and apply the vetoes.
This first exercise evolves a single quark line and records its branching
variables.

[pyresias_test.py](pyresias_test.py) contains the same algorithm as a standalone
script:

```bash
python pyresias_test.py -n 1000 -Q 1000 -c 1 --seed 12345 -o plots/tutorial
```

Here `-n` is the number of quark-line evolutions, `-Q` is the starting scale
and `-c` is the cutoff, both in GeV. The script writes plots of the branching
variables to `plots/tutorial/`. It also checks the splitting function by
sampling $z$ on a fixed interval; this is separate from the distribution of
all emissions in the shower.

Try increasing `-n`, changing the cutoff, or adding `--coupling pt` to replace
the fixed coupling with a running coupling. The option `--method Numerical`
provides an alternative way to sample the next trial scale. In the notebook,
these choices are controlled by `Nevolve`, `Qc`, `scaleoption` and `tMethod`.

## 2. Shower a hard event and reconstruct its momenta

We then apply the evolution to the quark and antiquark in an
$e^+e^-\to q\bar q$ event. The supplied input
[data/eejj_ECM206.lhe.gz](data/eejj_ECM206.lhe.gz) contains 10,000 hard events at
a centre-of-mass energy of 206 GeV, generated with MadGraph5_aMC@NLO in Les
Houches Event (LHE) format.

[pyresias.py](pyresias.py) introduces a simple momentum reconstruction and a
global recoil step to conserve four-momentum:

```bash
python pyresias.py data/eejj_ECM206.lhe.gz \
  -n 1000 --seed 12345 -o output/introductory.lhe
```

Following this example, [pyresias_qtilde.py](pyresias_qtilde.py) implements the
angular-ordered evolution and Sudakov-basis reconstruction discussed in the
paper:

```bash
python pyresias_qtilde.py data/eejj_ECM206.lhe.gz \
  -n 1000 --seed 12345 -o output/qtilde.lhe
```

In these programs, `-n` limits the number of hard events to shower. Omitting it
processes the whole input. Both examples use a fixed random seed so that a run
can be reproduced. A one-million-event input, `data/eejj_ECM206_1E6.lhe.gz`, is
also included for larger studies.

The two implementations use different starting scales, coupling settings and
momentum reconstructions. Comparing them is an exercise in understanding how
these choices affect the shower.

## 3. Inspect the results

Plot the distributions from the two samples generated above:

```bash
MPLBACKEND=Agg python analysis/lhe_analyzer.py \
  output/introductory.lhe output/qtilde.lhe \
  --labels Introductory Qtilde -o plots/comparison
```

The analyzer writes PDF plots and a `histograms.json` file. Look first at the
gluon multiplicity and the quark and gluon energy distributions. The ratio
panels use the first sample as the reference, and the main panels show
statistical uncertainties evaluated over events. You can supply a single file
to inspect one sample, or add `--jets` to cluster the final-state partons with
FastJet.

## 4. Let the emitted gluons branch

[pyresias_qtilde_full.py](pyresias_qtilde_full.py) extends the angular-ordered
example with $g\to gg$ and $g\to q\bar q$. It generates a candidate for each
allowed channel, selects the one at the highest scale, and evolves both
daughters. Secondary quarks and gluons can therefore radiate as well.

```bash
python pyresias_qtilde_full.py data/eejj_ECM206.lhe.gz \
  -n 1000 --seed 12345 -o output/qtilde-full.lhe

MPLBACKEND=Agg python analysis/lhe_analyzer.py \
  output/qtilde.lhe output/qtilde-full.lhe \
  --labels QuarkLines AllChannels -o plots/gluon-branching
```

Compare the quark and gluon multiplicities with the previous example. By
default, gluons can produce any of five massless quark flavours. Try
`--flavours 0` to retain $g\to gg$ while turning off quark-pair production.
The [full-shower notes](docs/full-shower.md) give the kernels, validation and
commands for a matching Herwig 7.3.0 comparison.

Read the full script alongside `pyresias_qtilde.py`: it follows the same
function names and sequence, with comments marking the additions. Use
`--quark-only` to recover the original quark-line shower and compare the
two implementations with the same seed.

With the matched settings in the notes, one million events per generator
at 206 GeV show good statistical agreement with Herwig 7.3.0. The
[validation results](VALIDATION.md) include the tutorial tests and the
same-seed check of the quark-only limit.

## Scope and further details

The first examples follow massless quark lines through $q\to qg$; the final
extension also showers the emitted gluons and their daughters. The hard events must be
$e^+e^-\to q\bar q$ in the centre-of-mass frame. Hadronization, spin
correlations and matrix-element corrections are outside the tutorial.

Each script provides `--help` for its command-line options. Numerical checks
and the scope of the validation are described in [VALIDATION.md](VALIDATION.md).
To run the tests, including the notebook:

```bash
python -m pip install pytest
MPLBACKEND=Agg python -m pytest -q
```

On macOS, run the Python analysis outside a sourced Herwig environment to
avoid loading its FastJet libraries. If needed, prefix the analysis command
with `env -u DYLD_LIBRARY_PATH -u PYTHONPATH`.

Pyresias is written by Andreas Papaefstathiou. See the
[tutorial paper](https://arxiv.org/abs/2406.03528) and its
[manuscript repository](https://github.com/apapaefs/PyresiasDoc).
