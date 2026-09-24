# Pyresias

<h1>A toy parton shower for educational purposes.</h1>

<h2>Pre-requisites:</h2>
<ol>
<li>python 3.x</li>
<li>pyhepmc</li>
<li>matplotlib, numpy, scipy.</li>
</ol>

<h2>Usage:</h2>

There are three main files in this tutorial: 

1. The "test" code, providing a basic demonstration of the parton shower sudakov veto algorithm: 

```
python3 pyresias_test.py -n [Number of Branches] -Q [Starting scale] -c
[Cutoff Scale] -o [outputdirectory] -d [enable debugging output]
```

2. The JupyterLab notebook ```jupyter_nb.ipynb```, which includes a step-by-step guide of the above "test" code: 

Click below to launch a Binder repository to use the JupyterLab notebook ```jupyter_nb.ipynb``` directly!

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/apapaefs/pyresias/HEAD)

3. The parton shower on $e^+ e^- \rightarrow q\bar{q}$ LHE files: 

```
python3 pyresias.py data/LHE_FILE.lhe.gz
```

An example LHE file (```eejj_ECM206.lhe.gz```) is provided in the "data" directory, generated through MadGraph5_aMC@NLO (10k events).

A larger example file, with 1 million events (```eejj_ECM206_1E6.lhe.gz```) is available through git Large File Storage (LFS). To obtain this file following cloning of the repository, making sure that git LFS is installed (https://git-lfs.com), type ```git lfs pull```.  

## Angular-ordered Herwig comparison

For the massless quark-only comparison at 206 GeV, use `pyresias_qtilde.py`:

```bash
python3 pyresias_qtilde.py data/eejj_ECM206_1E6.lhe.gz -o pyresias-coupling-fixed.lhe
```

The default coupling uses two-loop running with alpha_s(91.1876 GeV) = 0.1074,
charm/bottom/top thresholds of 1.6/5.0/172.69 GeV, and no CMW conversion.
`alphaS_HW.py` numerically inverts its running formula and matches continuously
at each threshold. These thresholds affect the coupling's flavour count;
the shower quarks are still massless. The shared coupling class also changes
the running used by `pyresias_qtilde_ttn.py`; its shower has not been validated
as part of this comparison.

The coupling is frozen below 0.935 GeV. Set this with `-c` or
`--coupling-freeze`. The independent emission pT cutoff is 0.900 GeV, adjustable
with `--ptmin`. Previously `-c` changed the overestimate while the acceptance
used the emission cutoff as its freeze scale. Both now use the same frozen
coupling, and an out-of-range coupling veto probability raises an error.

Checks against the saved Herwig 7.3.0 configuration, including its provenance,
are in `tests/fixtures/herwig-7.3.0-coupling.json`. Run the regression checks with:

```bash
python3 -m unittest discover -s tests -v
```

This fixes the coupling mismatch; it does not establish full shower agreement.
Reconstruction edge cases and rare unphysical two-jet configurations remain to
be corrected. The experimental CMW kernels are outside this validation.
The existing CLI also still requires the input filename before options, and
its `-n` limit has an off-by-two error; omit `-n` to shower the whole input file.

## About the Author

[Andreas Papaefstathiou](https://facultyweb.kennesaw.edu/apapaefs/) is Assistant Professor of Physics at Kennesaw State University. This website was originally created in January 2024 and is updated on a best-effort basis.

## References

[arxiv reference](https://arxiv.org/abs/2406.03528) 
