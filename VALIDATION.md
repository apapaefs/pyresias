# Cleanup validation — 24 September 2026

The cleanup was checked in a fresh virtual environment on macOS arm64 with
Python 3.11.15, NumPy 2.4.6, SciPy 1.17.1, Matplotlib 3.11.2, FastJet 3.5.1.5,
pyhepmc 2.16.1 and nbclient 0.11.0. Herwig's `DYLD_LIBRARY_PATH` and `PYTHONPATH`
were removed from the test environment.

## Automated checks

`MPLBACKEND=Agg python -m pytest -q` passes 30 tests, including 36 subtests.
`python -m flake8 . --select=E9,F` and `git diff --check` also pass.

The suite checks:

- the coupling against the saved Herwig 7.3.0 reference, threshold continuity
  and the veto bound;
- parallel and antiparallel rotations, axis-aligned and signed boosts, recoil,
  on-shell final particles and colour connections in both showers;
- deterministic output, event limits, argument ordering, empty samples,
  malformed inputs and preservation of existing output on failure;
- nominal and additional LHE weights, run metadata and LHE/HepMC round trips;
- common histogram bins, bin-width normalization, event-level uncertainties,
  final-state selection, FastJet momenta and one-, two- and three-file analysis;
- both Sudakov samplers, fixed and running coupling modes, empty tutorial plots,
  and fixed-interval sampling of the splitting kernel;
- all notebook cells in a fresh Jupyter kernel, with generated emissions
  identical to the script for the same configuration and seed.

## Sample runs

Both runs used `data/eejj_ECM206.lhe.gz`, massless quark-antiquark hard events
at 206 GeV, seed 12345 and each shower's default parameters.

| Check | Introductory shower | Sudakov-basis shower |
|---|---:|---:|
| Written events | 1,000 | 10,000 |
| Reconstruction retries | 0 | 0 |
| Maximum absolute four-momentum residual [GeV] | 5.69e-13 | 5.97e-13 |
| Maximum absolute final-particle mass squared [GeV²] | 1.64e-11 | 1.82e-11 |

All final particles had finite, positive energies and valid colour connections.
For the 10,000-event run, the input and output event headers, weights and
additional weight blocks were compared event by event. The run header was
preserved. The mean final-state gluon multiplicity was 3.4577.

The standalone tutorial completed 1,000 evolutions and generated 1,391
emissions. Its five PDF plots were produced. The LHE analyzer generated all 17
plots, including jets, from 1,000 events per shower. Representative tutorial
and comparison plots were rendered and inspected.

The hard-input SHA-256 checksums are:

```text
b49c38320a16a760592440c199f5ac2557d4203824c30e87f37cdd73bc538e47  eejj_ECM206.lhe.gz
1dfd7902c4f8419b67fb6ec747c5652e347d44cf9a6585d0fbef5d16bb2248ce  eejj_ECM206_1E6.lhe.gz
```

## Scope

The TTN scripts were removed. The saved Herwig coupling fixture and hard-event
inputs were retained. This cleanup changes reconstruction, event records and
analysis normalization, so old generated files do not exercise all of these
fixes. A new million-event Herwig comparison was not run as part of this check.

CI is configured for Python 3.11 and 3.12; the local checks used Python 3.11.
The obsolete LHAPDF startup hook and its `.binder` directory were removed, so
Binder can discover the root `environment.yml`, following the
[repo2docker configuration rules](https://repo2docker.readthedocs.io/en/latest/use/repository/).
A remote Binder build and the conda installation were not exercised locally.

## Full shower extension — 24 September 2026

The new `pyresias_qtilde_full.py` adds `g -> gg` and `g -> qqbar` for five
massless flavours. Both daughters evolve to the cutoff, including secondary
quarks and gluons. The initial extension passed **39 tests and 36 subtests**,
including the existing notebook. Static checks (`flake8 --select=E9,F` and
`git diff --check`) also pass.

The additional checks cover the symmetry-normalized gluon kernel, each
overestimate and its inverse primitive, and the competition sampler against
an independent integration of the physical fixed-coupling rates. They test
the no-branching probability, first-branching scale distribution and
per-flavour channel probabilities. Seeded shower tests check daughter angular
ordering, secondary radiation, trees with more than 100 leaves, conservation
of four-momentum, colour and flavour, and preservation of LHE weights and
event limits. Failed reconstruction is retried on the same hard event.

### Million-event Herwig comparison

Both generators processed `data/eejj_ECM206_1E6.lhe.gz`, with the checksum
listed above, at 206 GeV and seed 12345. Herwig 7.3.0 / ThePEG 2.3.0 used
`Herwig/LHE-LEP-full.in` and the existing custom LHEWriter plugin. The
[full-shower notes](docs/full-shower.md) give the physics settings, inspected
Herwig and Julia sources, and reproducible commands.

| Quantity | Herwig | Pyresias full |
|---|---:|---:|
| Complete output events | 1,000,000 | 1,000,000 |
| Mean gluon multiplicity | 4.316635 +/- 0.002433 | 4.319020 +/- 0.002431 |
| Mean quark multiplicity | 2.243988 +/- 0.000697 | 2.243974 +/- 0.000699 |
| Fraction with secondary quark pairs | 0.115098 +/- 0.000319 | 0.114816 +/- 0.000319 |
| Mean summed gluon energy [GeV] | 59.7220 +/- 0.0392 | 59.7592 +/- 0.0392 |
| Maximum absolute four-momentum residual [GeV] | 1.05e-12 | 6.54e-13 |
| Maximum absolute final mass squared [GeV^2] | 2.04e-10 | 2.73e-11 |

The errors are standard errors of event means. Every event has finite,
positive final energies and balanced colour and net flavour. Pyresias made
51 reconstruction retries without losing hard events. Its accepted histories
contain 3,463,877 quark emissions, 977,130 gluon emissions and 121,987
quark-pair splittings.

The gluon multiplicity test gives chi-squared 14.38 for 16 degrees of freedom
(p=0.57); the secondary-pair multiplicity test gives 5.39 for two degrees of
freedom (p=0.067). Rare tails are combined into the final category of each
test. The 15 physical distributions, including anti-kt jets with R=0.4,
show good agreement. Populated-bin pulls have RMS values from 0.84 to 1.36;
the largest absolute pull is 3.71. These pulls use event-level uncertainties
and are diagnostics, not a global test with all bin correlations included.
No physical-distribution bin with at least 25 entries in both samples
exceeds five standard errors.

Two further plots diagnose numerical conservation. The signed residual
mass-squared distribution has different populations immediately to either
side of zero, reflecting floating-point reconstruction and output precision;
both absolute mass-squared bounds are given above. This should not be
interpreted as a physics discrepancy. Rapidity plots retain their existing
range of -3 to 3 and record out-of-range particles. Three Herwig and four
Pyresias events overflow the jet-count plot's range of 0 to 19 jets.

The local campaign is `ShowerML/HWcomparisons/full-1M/`, with 17 comparison
plots, machine-readable histograms, event audits, `comparison.json`, a
manifest and a snapshot of the source used for production. The LHE samples
and generated campaign files are kept outside this code repository.

## Full-shower tutorial refactor — 24 September 2026

`pyresias_qtilde_full.py` now follows the procedural structure, helper names
and function order of `pyresias_qtilde.py`. The original script is unchanged.
The additions are marked in the source and mapped in the
[full-shower notes](docs/full-shower.md#reading-alongside-the-quark-line-example).
Both scripts use the shared command-line runner and reconstruction checks.

The full suite passes **40 tests and 36 subtests**, including the notebook.
The added regression checks 100 hard events against the original script
with identical seeds and `--quark-only`: momenta agree to floating-point
precision, colour tags agree and reconstruction retry counts match.
The independent fixed-coupling Sudakov and branching-tree checks still pass.

The physics parameters, kernels, competition probabilities and Herwig card
are unchanged. The refactor uses the original script's four random draws
per trial, so the full sample has a different random sequence from the
initial extension at the same seed. Its distributions therefore require a
fresh statistical comparison. The new campaign is kept separately in
`ShowerML/HWcomparisons/full-refactor-1M/`, with a direct
`qtilde-to-full.diff`, production source snapshots, event audits and logs.

After a 1,000-event smoke run of each generator, both produced exactly
1,000,000 events from the same million-event input with seed 12345. Herwig's
new LHE checksum and all its histograms exactly reproduce the preceding
campaign. Both output files pass the complete event audit, with finite
positive final energies, balanced colour and conserved net flavour.

| Quantity | Herwig | Refactored Pyresias full |
|---|---:|---:|
| Mean gluon multiplicity | 4.316635 +/- 0.002433 | 4.318199 +/- 0.002434 |
| Mean quark multiplicity | 2.243988 +/- 0.000697 | 2.245190 +/- 0.000699 |
| Fraction with secondary quark pairs | 0.115098 +/- 0.000319 | 0.115592 +/- 0.000320 |
| Mean summed gluon energy [GeV] | 59.7220 +/- 0.0392 | 59.7473 +/- 0.0393 |
| Maximum absolute four-momentum residual [GeV] | 1.05e-12 | 6.54e-13 |
| Maximum absolute final mass squared [GeV^2] | 2.04e-10 | 2.73e-11 |

Pyresias made 59 reconstruction retries on the same hard events, with no
lost events. Accepted histories contain 3,462,328 quark emissions, 978,466
gluon emissions and 122,595 quark-pair splittings. The mean gluon and quark
multiplicities differ by 0.45 and 1.22 combined standard errors. With the
same tail grouping as above, the multiplicity tests give chi-squared 6.289
for 16 degrees of freedom (p=0.985) for gluons, and 1.289 for two degrees
of freedom (p=0.525) for secondary pairs.

All 17 comparison plots were regenerated with the same analysis settings.
The 15 physical distributions show good agreement: bins with at least
25 entries per sample have pull RMS values of 0.63 to 1.26, with a maximum
absolute pull of 2.67. None exceeds three combined standard errors. These
are event-level bin diagnostics, not a global test including all
correlations. The numerical mass-squared plot retains the roundoff-related
difference described above. Three Herwig and two Pyresias events overflow
the jet-count range. Representative multiplicity, parton and jet PDFs were
rendered and inspected. Static checks and `git diff --check` pass.
