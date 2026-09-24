# Pyresias

A toy parton shower for educational purposes, with a Sudakov-veto tutorial and
two showers for massless $e^+e^-\to q\bar q$ events.

## Installation

Use Python 3.11 or 3.12. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-tutorials.txt
```

This installs the shower, plotting, Jupyter, FastJet and optional HepMC
dependencies. For the shower scripts and basic LHE plots alone, install
`requirements.txt` instead. LHAPDF is not required. A conda environment is also
provided in `environment.yml`.

On macOS, a sourced Herwig environment can load a different FastJet library
from the one used by the Python wheel. Run Python analysis outside that
environment, or prefix the command with `env -u DYLD_LIBRARY_PATH -u PYTHONPATH`.
For batch plotting, set `MPLBACKEND=Agg`.

## Sudakov-veto tutorial

The script evolves individual quark lines without reconstructing a full event:

```bash
python pyresias_test.py -n 1000 -Q 1000 -c 1 --seed 12345 -o plots/tutorial
```

Scales are in GeV. The default uses a fixed coupling at `Q/2` and an analytic
overestimate for the emission scale. Use `--coupling pt` for the running
coupling at the emission transverse momentum, or `--method Numerical` to
invert the Sudakov with scale-dependent momentum-fraction limits. `--no-plots`
runs the evolution only; `-d` prints the trial emissions.

The momentum-fraction plots check the splitting kernel on a fixed interval,
$0.01<z<0.99$. One shows $P_{qq}(z)$ and the other uses weights $1-z$. The
distribution of all emissions in the shower also depends on the evolving
phase-space limits and is therefore a different quantity.

The step-by-step notebook contains the same algorithm:

```bash
jupyter lab pyresias_nb.ipynb
```

Select the environment's Python kernel and run all cells in order. The default
is 1,000 evolutions; increase `Nevolve` for smoother distributions.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/apapaefs/pyresias/HEAD?labpath=pyresias_nb.ipynb)

## Showering hard events

Two MadGraph5_aMC@NLO inputs at a centre-of-mass energy of 206 GeV are included
in `data/`: `eejj_ECM206.lhe.gz` contains 10,000 events and
`eejj_ECM206_1E6.lhe.gz` contains one million. Both are tracked directly in Git.

The introductory shower uses a simple reconstruction and the coupling in
`alphaS.py`, with $\alpha_s(M_Z)=0.118$:

```bash
python pyresias.py data/eejj_ECM206.lhe.gz \
  -n 1000 --seed 12345 -o output/introductory.lhe
```

The angular-ordered shower in `pyresias_qtilde.py` reconstructs momenta in a
Sudakov basis with transverse momenta fixed by the generated branchings. Use
this version for the restricted Herwig comparison:

```bash
python pyresias_qtilde.py data/eejj_ECM206.lhe.gz \
  -n 1000 --seed 12345 -o output/qtilde.lhe

# Omit -n to process all one million hard events.
python pyresias_qtilde.py data/eejj_ECM206_1E6.lhe.gz \
  --seed 12345 -o output/qtilde-1M.lhe
```

Both programs support `--help`. Options may appear before or after the input
filename. `-n N` writes up to exactly N events, stopping earlier only at the
end of the input; `-n 0` writes a valid empty sample. The seed defaults to
12345. If `-o` is omitted, the output is written beside the input with suffix
`_pyr.lhe`. Compressed inputs are read as a stream, so the million-event input
does not have to fit in memory.

Only massless light quarks with $q\to qg$ branchings are implemented. The input
must contain a colour-connected quark-antiquark pair in the centre-of-mass
frame, with electron and positron beams. Gluons do not branch; hadronization,
matrix-element corrections and spin correlations are absent. The introductory
and Sudakov-basis showers have different starting scales and reconstruction
prescriptions and are not expected to give identical distributions.

The global recoil step checks the reconstructed four-momentum. If a shower
cannot be reconstructed at the available energy, it is regenerated on the
same hard event, up to 100 attempts. The run reports the number of retries.
An unrecoverable error aborts the run without replacing an existing output.

The input run header, cross sections, nominal event weights, process IDs,
hard-process scales and additional weight blocks are retained. The output
is a flat partonic LHE record with consistent quark/gluon colour connections;
it does not encode the full branching history. No cross section is inferred
from the number of events. Explicit HepMC export is available through
`HEPMCWriter.WriteHepMC`.

## Coupling for the Herwig comparison

The default `pyresias_qtilde.py` configuration uses two-loop running with
$\alpha_s(91.1876\,\mathrm{GeV})=0.1074$ and charm/bottom/top thresholds of
1.6/5.0/172.69 GeV. `alphaS_HW.py` numerically inverts the running formula and
matches continuously at each threshold. These thresholds affect the flavour
count in the coupling; the shower quarks remain massless.

The coupling is frozen below 0.935 GeV (`-c` or `--coupling-freeze`). The
independent emission transverse-momentum cutoff is 0.900 GeV (`--ptmin`). The
veto and its overestimate use the same frozen coupling, and an invalid veto
probability raises an error. The default has no CMW conversion. Alternative
CMW settings in the source are outside the saved Herwig comparison.

The numerical reference and its provenance are recorded in
`tests/fixtures/herwig-7.3.0-coupling.json`. `Herwig/LHE-LEP.in` is a historical
card, not a portable installation or a record of every historical run. A
comparison also needs matching input events, branching channels, masses,
cutoffs, recoil, corrections and analysis definitions.

## Analysis and comparison plots

From the repository root, supply one or more LHE files; the first is the ratio
reference. The filenames below are examples of separately generated samples:

```bash
MPLBACKEND=Agg python analysis/lhe_analyzer.py \
  output/herwig.lhe output/qtilde.lhe \
  --labels Herwig Pyresias -o plots/comparison
```

Use `--jets` to add anti-$k_T$ jets with $R=0.4$, `-n 1000` to limit the number
of events read from each input, or `--observables ng Eg ptg` to select plots.
The default bins cover the supplied 206 GeV samples; edit `BINS` for other
energies or ranges. One-input analysis and comparisons of more than two
samples use the same interface.

Only particles with final-state status 1 enter the distributions. Nominal LHE
weights, common bin edges and bin widths are used throughout. The default
`--normalization shape` divides by the weighted number of entries, including
underflow and overflow; `--normalization per-event` divides by the sum of event
weights. The latter retains the inclusive particle multiplicity. The analyzer
does not convert either choice to an absolute cross section.

Each sample has event-level standard errors, including the correlation between
particles from the same event and the normalization. Ratios are direct
divisions with no extra rescaling; bins with a zero reference are left empty.
Ratio error bands are omitted because correlations between samples are not
estimated. `histograms.json` records the bin contents, errors, event counts and
underflow/overflow counts alongside the PDF plots. With fewer than two events,
the sample uncertainty is unavailable, as marked in that file.

## Tests

With the tutorial dependencies installed:

```bash
python -m pip install pytest flake8
MPLBACKEND=Agg python -m pytest -q
python -m flake8 . --select=E9,F --show-source --statistics
```

The tests cover the saved Herwig coupling, both shower entry points, recoil
and colour conservation, LHE/HepMC metadata, histogram normalization, both
Sudakov samplers and execution of the notebook in a fresh kernel. CI runs this
suite on Python 3.11 and 3.12. Numerical regression tests and finite samples do
not establish agreement with a complete Herwig shower.

The local checks and sample runs are recorded in [VALIDATION.md](VALIDATION.md).

## Author and paper

Andreas Papaefstathiou. See the [paper](https://arxiv.org/abs/2406.03528) and
the [manuscript repository](https://github.com/apapaefs/PyresiasDoc).
