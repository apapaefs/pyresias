# Adding gluon branchings

`pyresias_qtilde_full.py` extends the quark-line tutorial with `g -> gg` and
`g -> qqbar`. The original scripts and notebook retain their simpler scope.
The extension uses massless kinematics, five quark flavours by default,
angular ordering and transverse-momentum-preserving reconstruction. It still
requires a massless `e+e- -> qqbar` hard event in its centre-of-mass frame.

## Reading alongside the quark-line example

The full script follows `pyresias_qtilde.py` in function names, order and
conventions. It uses the same procedural structure and shared command-line
runner. Comments marked `ADDED` and `EXTENDED` identify the main changes.

| Original function or stage | Extension in the full script |
|---|---|
| `Pqq`, `Pqq_over` and coupling helpers | Add `Pgg`, `Pgq` and their overestimates; keep the coupling helpers |
| `tGamma`, `inversetGamma`, `zp_over`, `zm_over` | Add a `channel` argument, defaulting to the original `qg` channel |
| `Get_tEmission_direct`, `Get_zEmission`, `Generate_Emission` | Apply the same sampling and veto steps to the selected channel |
| Evolution before `EvolveParticle` | Add `Next_Emission` to finish each channel's veto loop, then `Choose_Emission` to select the largest accepted scale |
| `EvolveParticle` | Record both daughters and continue evolving every pending parton |
| `Shower`, `find_color_partner`, `reconstructSudakov` | Keep the event/reconstruction sequence; propagate the Sudakov coefficients along both daughter branches |
| `main` | Use `run_shower`, with the full-shower options enabled |

An emission retains the original first five entries,
`[tEm, zEm, pT, MsqEm, phi]`, followed by the channel and quark flavour.
The history uses dictionaries with an emission and two children, since a
single list of emissions along the quark line is no longer sufficient.
Momentum reconstruction remains a separate step after evolution.

To inspect the additions directly from the repository root:

```bash
git diff --no-index -- pyresias_qtilde.py pyresias_qtilde_full.py
```

`git diff --no-index` returns status 1 when the files differ. For a direct
check of the original quark-line limit, add `--quark-only` to the full
script's command. With the same seed, the test suite checks that its events,
momenta and colour tags agree with `pyresias_qtilde.py` to floating-point
precision. This differs from `--flavours 0`, which retains `g -> gg`.

The refactor follows the original script's four random draws per trial.
Consequently, a full-shower sample differs event by event from the previous
class-based version for the same seed; the physics probabilities and
comparison settings are unchanged.

## Kernels and competition

With the branching probability written as

$$d\mathcal P_i=\frac{d\tilde q^2}{\tilde q^2}\,dz\,
\frac{\alpha_s(p_T)}{2\pi}P_i(z),\qquad p_T=z(1-z)\tilde q,$$

the massless kernels and their overestimates are:

| Channel | Kernel | Overestimate |
|---|---|---|
| $q\to qg$ | $C_F(1+z^2)/(1-z)$ | $2C_F/(1-z)$ |
| $g\to gg$ | $C_A[1-z(1-z)]^2/[z(1-z)]$ | $C_A/[z(1-z)]$ |
| $g\to q\bar q$, one flavour | $T_R[z^2+(1-z)^2]$ | $T_R$ |

Here $C_F=4/3$, $C_A=3$ and $T_R=1/2$. The gluon kernel includes the
identical-daughter factor of one half when sampling the full $0<z<1$
interval. Multiplying it by two would double the branching rate. Each
allowed quark flavour contributes a separate quark-pair channel.

The overestimate primitives are $-2C_F\ln(1-z)$,
$C_A\ln[z/(1-z)]$ and $T_Rz$, respectively. For a trial starting scale
$Q$, the quark proposal uses $p_{T,\min}/Q<z<1-p_{T,\min}/Q$;
the gluon channels use
$z_\pm=[1\pm\sqrt{1-4p_{T,\min}/Q}]/2$. The proposal bounds are held
fixed while sampling a trial scale, then the physical bounds and
$p_T\geq p_{T,\min}$ are checked at that new scale. A rejected trial lowers
the upper evolution scale within that channel.

For each channel we generate an accepted candidate with the Sudakov veto
algorithm, or no branching above the cutoff. The candidate with the largest
$\tilde q$ wins. This samples the combined no-emission probability
$\Delta_{\mathrm{tot}}=\prod_i\Delta_i$. Both daughters then evolve from
$z\tilde q$ and $(1-z)\tilde q$. There is no fixed particle-count limit;
secondary gluons and quarks are treated in the same way as earlier partons.

The reconstruction propagates each daughter's light-cone fraction and
transverse momentum through the branching tree. Final particles are put on
shell, followed by the same global recoil step as the quark-line shower.
An impossible reconstruction resamples the shower of the same hard event;
it does not discard its weight or substitute an unshowered event.

## Matched comparison settings

The supplied [Herwig card](../Herwig/LHE-LEP-full.in) was checked with Herwig
7.3.0 / ThePEG 2.3.0. It retains `q -> qg`, `g -> gg` and `g -> qqbar` for
`d,u,s,c,b`, with zero kinematic masses. Top-pair production is disabled.
All three splitting functions are explicitly angular ordered, including
`g -> qqbar`, and use $p_T$ as the coupling scale.

Both showers use $\alpha_s(M_Z)=0.1074$ at $M_Z=91.1876$ GeV, two-loop running
matched continuously at 1.6, 5 and 172.69 GeV, frozen below 0.935 GeV. These
thresholds affect the coupling, independently of the massless shower
kinematics. The emission cutoff is $p_{T,\min}=0.900$ GeV. The veto
overestimate is the coupling at the freeze scale. No additional CMW
rescaling is applied.

The Herwig card selects `EvolutionScheme pT`, global recoil and
`FinalFinalWeight No`. It disables ISR, QED radiation, soft/spin
correlations, hard-emission corrections, hadronization and decays.
The name "full" refers to the three branching channels in this tutorial;
it does not imply all of Herwig's physics features.

## Run and compare

From the repository root, with the tutorial Python environment active:

```bash
mkdir -p output/full
python pyresias_qtilde_full.py data/eejj_ECM206_1E6.lhe.gz \
  -n 1000000 --seed 12345 -o output/full/pyresias-full.lhe
```

In a separate shell with Herwig 7.3.0 activated, prepare its working directory:

```bash
mkdir -p output/full
cd output/full
ln -s ../../data/eejj_ECM206_1E6.lhe.gz eejj_ECM206_1E6.lhe.gz
Herwig read ../../Herwig/LHE-LEP-full.in -L /path/to/lhewriter-library
Herwig run HW-full-1M.run -L /path/to/lhewriter-library -N 1000000 -s 12345
```

The `-L` directory must contain the custom `LHEWriter.so` from
[herwiglhewriter](https://gitlab.com/apapaefs/herwiglhewriter), compiled against
the active Herwig installation. The card reads the million-event input from
the working directory and forbids reopening it. Start with `-n 1000` and
`-N 1000` in a separate smoke-test directory when changing settings.

Finally, from the repository root in the Python environment:

```bash
env -u DYLD_LIBRARY_PATH -u PYTHONPATH MPLBACKEND=Agg \
  python analysis/lhe_analyzer.py \
  output/full/HW-full-1M-S12345.lhe output/full/pyresias-full.lhe \
  --labels Herwig Pyresias --jets -o plots/full
```

The environment prefix avoids the macOS FastJet library collision when a
Herwig environment has been sourced. Histograms use common bins and
event-level statistical errors. Compare quark/gluon multiplicities as well
as energy, transverse momentum, rapidity and jet distributions. The custom
Herwig writer emits unit weights for this constant-weight input, whereas
Pyresias retains the input weight of 15.0075; shape normalization cancels
this common factor. Do not compare their raw weight sums as cross sections.

## Sources and checks

The implementation was checked against the local Herwig 7.3.0 sources:

- `Shower/QTilde/SplittingFunctions/{HalfHalfOneSplitFn,OneOneOneSplitFn,OneHalfHalfSplitFn}.cc`:
  massless kernels, overestimates and inverse primitives;
- `SplittingGenerator.cc:chooseForwardBranching` and
  `SudakovFormFactor.cc:{guesstz,generateNextTimeBranching,computeTimeLikeLimits}`
  in the same directory: competition, vetoes and phase-space bounds;
- `SplittingFunction.cc:evaluateFinalStateScales`: daughter starting scales;
- `Shower/QTilde/Kinematics/FS_QTildeShowerKinematics1to2.cc`: Sudakov reconstruction.

Herwig generates two half-rate `g -> gg` candidates, one per colour line.
For this pure-QCD `qqbar` setup their upper scales remain equal, so a single
candidate with their summed rate is equivalent. The emitting colour line
is then chosen with equal probability.

The student's [PartonShower.jl](https://github.com/Zykerin/PartonShower.jl)
implementation by Caitlyn G was also consulted, in particular
`src/SplittingFunctions.jl`, `src/Shower.jl`, `src/ShowerHelpers.jl` and
`src/Kinematics.jl`. The local working copy contained uncommitted changes;
its exact source hashes are retained with the local comparison campaign.
The new script uses independent Python code and the corrected shared
kinematics and LHE utilities. It does not retain the Julia working copy's
fixed particle-loop limit.

`tests/test_full_shower.py` checks the kernels, overestimate bounds,
independently integrated first-branching probabilities, flavour competition,
secondary radiation, angular ordering, large branching trees, four-momentum,
on-shell masses, colour/flavour conservation, deterministic event limits and
LHE weight preservation. See [VALIDATION.md](../VALIDATION.md) for results.
