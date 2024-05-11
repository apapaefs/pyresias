# Pyresias

<h1>A toy parton shower for educational purposes.</h1>

<h2>Pre-requisites:</h2>
<ol>
<li>LHAPDF 6.5.x with the python interface built in.</li>
<li>python 3.x</li>
<li>pyhepmc</li>
<li>matplotlib, numpy, scipy.</li>
</ol>

<h2>Usage:</h2>
python3 pyresias_test.py -n [Number of Branches] -Q [Starting scale] -c
[Cutoff Scale] -o [outputdirectory] -d [enable debugging output]

python3 pyresias.py LHE_FILE.lhe.gz

An example LHE file is provided in the "data" directory, generated through MadGraph5_aMC@NLO.
