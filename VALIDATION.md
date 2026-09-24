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
