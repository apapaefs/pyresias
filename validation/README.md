# Reproducing the paper's validation figures

The scripts and data in this directory reproduce the validation figures of the
Pyresias tutorial paper ([arXiv:2406.03528](https://arxiv.org/abs/2406.03528)):
the Sudakov closure test of the veto algorithm (figure 4) and the comparison with
the angular-ordered shower of Herwig 7.3.0 (figures 5 and 6, table 1). Run the
commands below from the repository root, with the tutorial dependencies installed
(see the main [README](../README.md)). Outputs are written to `validation/output/`,
which is ignored by git; use `-o` to choose another directory.

## Sudakov closure test (figure 4)

`make_sudakov_plot.py` tests the single-line veto sampler of `pyresias_test.py`. It
runs the tutorial's own `Evolve` function for 5,000,000 independent quark lines,
with Python's random seed set to 12345, `t_max = (206 GeV)^2`, `Q_c = 1 GeV` and the
running coupling (`scaleoption = "pt"`, `mu = max(pT, Q_c)`), and records the first
accepted emission of each line.

The reference integrates `t Gamma(t)` numerically over the physical region
`z(1-z) sqrt(t) >= Q_c`, using the tutorial's own coupling and kernel functions. The
bin probabilities are differences of the resulting Sudakov factor, and the
no-emission probability is `Delta(16 Q_c^2, t_max)`. The comparison is a
multinomial Pearson test over the 40 bins plus the no-emission category.

```bash
MPLBACKEND=Agg python validation/make_sudakov_plot.py
```

The run takes about 100 s. It writes `sudakov-closure.pdf` and
`sudakov-closure.json` (settings, Pyresias commit, counts, reference probabilities
and test result). The committed `validation/sudakov-closure.json` records the run
used in the paper: no-emission fraction `0.11783 +- 0.00014` versus `0.11809`, and
`chi2 = 51.1` for 40 degrees of freedom (`p = 0.11`). The run is deterministic for
a given seed, so a rerun reproduces these numbers.

Independent-seed check: with 1,000,000 lines each, seeds 12345, 1, 2, 3 and 4 gave
`chi2/ndf` = 63.6, 47.8, 55.9, 43.5 and 43.2 (/40). The two largest pulls in the
seed-12345 run (adjacent bins near 8.8 GeV, -3.4 and +4.1) did not recur in the
other seeds, and the combined 5,000,000 lines gave `chi2 = 47.5/40` (`p = 0.19`)
with a no-emission pull of -0.2. For the bins between 6.5 and 13 GeV, a reference
cross-check with the z integral split at the coupling thresholds and a 2001-point
Simpson rule in `ln t` agreed with the adaptive quadrature to better than 1e-7.
Sequence, for transparency: the first run used 1,000,000 lines (seed 12345,
`chi2 = 63.6/40`, `p = 0.01`); after the checks above, the sample was increased to
5,000,000 lines with the same seed, a choice made before that larger run's result
was known.

## Herwig comparison (figures 5 and 6, table 1)

`reference-results.json` contains the four plotted histogram summaries, the event
means, the settings and the provenance hashes of the million-event comparison. The
event samples themselves (about 2.5 GB of LHE files) are not part of this
repository.

The setup is massless `e+e- -> qqbar` at 206 GeV, without primary bottom quarks,
with one million events in each sample and seed 12345 in each generator. The shower
includes `q -> qg`, `g -> gg` and `g -> qqbar` for five secondary flavours. Both
programs use the coupling, cut-off and reconstruction settings recorded in the JSON
and described in the paper; the Herwig input card is
[`Herwig/LHE-LEP-full.in`](../Herwig/LHE-LEP-full.in). Herwig is version 7.3.0 with
ThePEG 2.3.0.

The Pyresias samples were produced with commit
`7995f8267efe9ca6b4981f7cd12c0278e9b95df7`. The all-channel script's Python abstract
syntax tree equals that of the saved production snapshot; their byte hashes differ
only because trailing whitespace was removed after production. The coupling,
quark-line shower, shared reconstruction, event I/O, analysis and full-shower test
sources match the production hashes.

To regenerate the four panels with NumPy and Matplotlib:

```bash
MPLBACKEND=Agg python validation/make_comparison_plots.py
```

The spectra retain the campaign binning and event-based errors. Rapidity
normalization includes particles outside the displayed range. For multiplicities,
each event contributes to one category. The final categories collect every event
with at least 16 gluons or at least six quarks plus antiquarks; no tail events are
removed. Their binomial errors follow from the grouped category counts. The script
asserts normalization and zero flow losses before grouping.

In the lower panels, Herwig's own uncertainty is divided by its central value and
shown around one. The Pyresias central value and its own uncertainty are also
divided by the Herwig central value. These are separate sample errors, not an
uncertainty on the ratio that includes denominator fluctuations or a between-sample
covariance. The two generators use the same hard input.

For the event means of table 1, the shared hard input induces no covariance: at
fixed collision energy the massless, flavour-blind shower is independent of the
hard event's orientation and light flavour, and each generator uses its own random
numbers. The mean gluon and quark-plus-antiquark counts differ by 0.45 and 1.22
combined standard errors. The gluon transverse momentum, rapidity, longitudinal
momentum and polar angle are measured with respect to the beam axis, so they do
share the event orientation. For these, the per-bin pulls (largest 2.55 in the
displayed panels: 2.55 for gluon pT, 2.34 for rapidity, 0.94 and 1.09 for the
grouped multiplicities, errors added in quadrature) are diagnostics, not a global
test.

The compact JSON supports the figures and the table; it is not a substitute for the
full event samples.
