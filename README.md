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

An example LHE file is provided in the "data" directory, generated through MadGraph5_aMC@NLO.

## About the Author

[Andreas Papaefstathiou](https://facultyweb.kennesaw.edu/apapaefs/) is Assistant Professor of Physics at Kennesaw State University. This website was originally created in January 2024 and is updated on a best-effort basis.

## References

Main reference coming soon!
