# HDCProjection-empirical

An empirical study of **projection and encoding matrices in HDC pipelines**
(feature extractor -> projection -> prototype classification) on synthetic
workloads, testing the hypothesis:

> The projection matrix is not a neutral implementation detail. Which
> similarities a pipeline can express is set by the projection family, its
> output code, and its interaction with the feature extractor — and the
> choices that maximize in-distribution accuracy are routinely *not* the
> choices that maximize robustness, domain-shift accuracy, and novel-class
> structure.

The study crosses the axes the HDC literature usually treats separately:

* **Projection / encoding** (29 configurations): 16 random-matrix
  constructions (Gaussian, Rademacher, uniform, Laplace, Student-t, Cauchy,
  Achlioptas, ternary, very sparse, Count-Sketch, SRHT, deterministic
  Hadamard, orthogonal, partially orthogonalized, Sobol, unipolar binary),
  random Fourier features (RFF/SignRFF), fractional power encoding (phasor
  and phase-quantized / cyclic-group variants), Nystrom kernel features
  (Gaussian and Laplacian kernels), a PIONEER-style learned projection,
  best-of-K data-informed selection, input-modulated projection, and
  sequence encoders with binding/position codes.
* **Output coding / quantization**: fp32, int8/4/2 (MSE-optimal and
  max-scaled), bipolar, unipolar binary, ternary, block-sparse,
  thermometer (TQHD-style), k-means vector quantization (HDVQ-VAE-style),
  plus quantization-aware prototype retraining (QuantHD-style).
* **Feature extractor**: raw features, frozen random MLP, CE-, SupCon-,
  proxy-anchor- and angular-margin-trained MLPs, k-means weight clustering,
  and co-training with the HDC objective through a straight-through
  estimator.
* **Processing**: cosine/dot/Hamming/L1/RBF similarity, single-pass vs
  OnlineHD retraining, margin/confidence-weighted bundling,
  multi-prototype classes, normalization/standardization/PCA whitening,
  and symbolic binding/position encodings.

All methods run through the same pipeline and the same evaluation on the
same synthetic tasks, 3 seeds, mean reported.

## Layout

```
modules/
  rp.py         # 16 projection-matrix constructions + storage/ops metadata
  quant.py      # MSE-PTQ, max-scaled, sign, ternary, block-sparse,
                # thermometer, k-means VQ codecs
  encoders.py   # LinearRP, RFF, FPE, Nystrom, learned (STE), data-informed,
                # input-modulated, sequence encoders; output codes; preproc
  extractors.py # MLP extractors: random/CE/SupCon/proxy/arc/clustered,
                # co-training with the HDC objective (STE)
  data.py       # synthetic tasks: linear, fine, xor, shells, heavy_tail,
                # shift (3 severities), novel (held-out classes), symbolic
  hd.py         # OnlineHD-style prototype classifier + variants
  metrics.py    # accuracy, OOD, clustering, margins, JL distortion,
                # geometry fidelity, kernel MSE
  bench.py      # Spec -> result-row harness (extractor -> encoder -> clf)
  resources.py  # memory / ops accounting
tests/test_sanity.py    # 15 tests: RP conventions, quantizers, kernel
                        # approximation, symmetry limits, task structure, ...
experiments/
  specs.py, run_all.sh  # shared Spec library; resumable sequential runner
  exp1_projection_sweep.py   # 29 methods x 7 tasks x 3 seeds
  exp2_dimension_sweep.py    # D = 32..4096 (compression vs expansion)
  exp3_robustness.py         # noise / dropout / model bit-flip curves
  exp4_quantization.py       # output codecs, projection PTQ, QAT
  exp5_extractor.py          # extractor architectures/losses/clustering
  exp6_processing.py         # similarity / bundling / multi-proto / preproc
  exp6b_sequence.py          # binding and position encodings (symbolic task)
  exp7_learned.py            # learned vs data-informed vs random projections
  exp8_geometry.py           # JL, L1/L2, kernel approx, bandwidth, margins
  exp9_interactions.py       # projection x codec x dimension
  make_summaries.py          # results/summary_*.md tables
results/        # CSVs, PNG figures, summary_*.md
```

## Usage

```bash
pip install -r requirements.txt
python -m tests.test_sanity
bash experiments/run_all.sh            # all experiments, skips finished ones
python experiments/make_summaries.py   # regenerate results/summary_*.md
```

`run_all.sh` runs one experiment at a time and skips any whose CSV already
exists (`--force` re-runs). Set `HDCPROJ_DEVICE=cpu` to run without the GPU.
Total runtime on an RTX 5080: **~10.5 minutes** (exp1 95 s, exp2 59 s,
exp3 18 s, exp4 182 s, exp5 94 s, exp6 51 s, exp6b 3 s, exp7 49 s,
exp8 6 s, exp9 77 s).

> **GPU stability note.** The first full run hung the shared display GPU
> (`NVRM: RC watchdog: GPU is probably locked!` + `Xid 8`, then a Hyprland
> abort) while the Nystrom encoder was running cuSOLVER `eigh` on a
> 4096 x 4096 landmark kernel. No CPU/GPU memory OOM occurred (peak 1.7 GB /
> 16 GB). Nystrom now uses a **CPU** Cholesky whitening instead (numerically
> equivalent kernel approximation, no multi-second GPU kernel), and the
> runner is resumable. See "Deviations" for details.

## Experimental protocol

Tasks (F=64 features, 150 train / 40 val / 50 test samples per class):

| task | construction | what it probes |
|---|---|---|
| `linear` | 30 Gaussian classes, shared power-law covariance | RP-friendly baseline |
| `fine` | same, inter-class spread halved (0.35) | high-similarity / fine-grained |
| `xor` | 4 classes = parity of latent sign bits | even-symmetric, non-linear label |
| `shells` | 6 radial quantiles of a latent Gaussian | scale-dependent label |
| `heavy_tail` | Student-t(3) noise + 5% per-feature outliers | outlier robustness |
| `shift` | 3 covariate-shift targets (norms 6/12/18, noise x1.15-1.45) | OOD generalization |
| `novel` | 10 held-out classes from the same family | discovery + OOD detection |
| `symbolic` | class = presence of an ordered bigram | binding / position encoding |

Encoder: `h = codec(P x)` at D=4096 (unless stated). Classifier: OnlineHD
prototype bundling + 30 error-driven retraining epochs, cosine similarity.
Metrics: ID accuracy, OOD accuracy, novel-class NMI/ARI (k-means),
OOD-detection AUROC (max similarity), geometry fidelity (Spearman rho of
pairwise cosine similarities before/after encoding), margins (correct minus
best-wrong similarity), robustness to feature noise (sigma), feature dropout
(p) and prototype sign flips (p), and the memory/ops proxies in
`modules/resources.py`.

## Theoretical limits of the pipeline (validated)

Two properties of `h(x) = sign(P x)` bound every *linear* projection, random
or learned, and explain three of the headline results below:

1. **Scale invariance.** `sign(P(cx)) = sign(Px)` for `c > 0`: the encoder
   cannot see the norm of its input. Tasks whose classes differ only in
   radius (`shells`) are at chance for *every* sign-linear encoder,
   including trained ones: `learned` reaches 0.40 vs chance 0.21 (6
   classes), and no amount of training fixes it (exp2 shows the same
   plateau for all D). Kernel approximations are not scale-invariant — the
   phase `omega.x` scales — and reach 0.85-0.90 on `shells`.
2. **Odd symmetry.** `sign(P(-x)) = -sign(Px)`: every feature is odd, so a
   linear decoder over them is odd. Labels invariant under `x -> -x` (XOR/
   parity) have equal class-conditional feature means and are at chance for
   every linear projection. Measured: gaussian/rademacher/orthogonal/sobol/
   learned all 0.21-0.28 on 4-class `xor` (chance 0.25) even after 120 STE
   training epochs; only features with an even component do better
   (FPE 1-bit 0.52, SignRFF 0.31-0.45, co-trained ReLU extractor 0.51).

`tests/test_sanity.py` asserts both limits.

## Hypotheses and verdicts

| # | hypothesis (from the experimental brief) | verdict |
|---|---|---|
| H1 | RFF beats linear RP on non-linear data; they tie on linear data | **supported** (shells 0.88 vs 0.22; linear/fine tie within 1 pt) |
| H2 | SignRFF dominates SignRP in high-similarity regimes | **refuted** here (fine: 0.706 vs 0.722; xor: better only at D>=1024) |
| H3 | Group-VSA phase quantization recovers the binary-HDC gap | **supported** (FPE b3 = phasor on shells 0.878 vs 0.873; b1 costs 2-20 pts) |
| H4 | Learned projections close the gap to random ones | **supported where a linear map suffices** (+15-24 pts at D=128; chance on XOR/shells) |
| H5 | Quantization-aware training beats post-hoc quantization | **weakly supported** (2-bit prototypes: +0.8 pt on fine, +0.1-0.2 elsewhere) |
| H6 | Nystrom supports kernels RFF cannot (Laplacian) | **supported** (Laplacian > Gaussian kernel: linear 0.987 vs 0.918, heavy-tail 0.991 vs 0.845) |
| H7 | Deterministic projections (Sobol) cost accuracy vs random | **refuted** (identical to Gaussian; 6x lower JL distortion) |
| H8 | Structured/sketching projections win when D < F and lose when D > F | **partially supported** (Count-Sketch strong at D<=F; SRHT/Hadamard lose 4-6 pts for D > F) |
| H9 | Heavy-tailed data favors heavy-tailed projections and L1 similarity | **supported** (Cauchy 0.997 clean / 0.929 dropout on heavy-tail, best of all RPs) |
| H10 | Kernel methods lose OOD robustness under covariate shift | **supported and bandwidth-dependent** (shift OOD 0.40 at bw=0.25 vs 0.68 for dense RPs; 0.65 at bw=1) |
| H11 | Learned projections hurt novel-class discovery | **supported** (NMI 0.666 vs 0.961 Gaussian; the best ID model on novel data) |
| H12 | Margins predict robustness better than ID accuracy does | **supported** (margin vs noise rho=+0.75; margin vs ID rho=+0.77; geometry rho=+0.79 with OOD) |
| H13 | Frozen random extractors are enough; training them does not help HDC | **mixed** (trained MLPs *lower* OOD here; co-training recovers +2.6 pts OOD and unlocks XOR) |

## Exp1: projection-family sweep (`results/exp1_projection_sweep.csv`)

29 methods x 7 tasks x 3 seeds at D=4096. ID accuracy (means; OOD and
robustness in the tables after):

| spec | linear | fine | xor | shells | heavy_tail | shift | novel |
|---|---|---|---|---|---|---|---|
| **gaussian** | 0.997 | 0.722 | 0.275 | 0.216 | 0.948 | 0.997 | 0.977 |
| rademacher | 0.996 | 0.719 | 0.244 | 0.247 | 0.892 | 0.996 | 0.976 |
| uniform | 0.995 | 0.710 | 0.264 | 0.211 | 0.927 | 0.995 | 0.981 |
| laplace | 0.997 | 0.715 | 0.239 | 0.266 | 0.967 | 0.997 | 0.979 |
| student_t(3) | 0.996 | 0.716 | 0.261 | 0.230 | 0.972 | 0.996 | 0.978 |
| **cauchy** | 0.995 | 0.682 | 0.283 | 0.219 | **0.997** | 0.995 | 0.974 |
| achlioptas | 0.995 | 0.721 | 0.285 | 0.213 | 0.968 | 0.995 | 0.979 |
| ternary(2/3) | 0.997 | 0.718 | 0.242 | 0.205 | 0.927 | 0.997 | 0.980 |
| **very_sparse** | 0.996 | 0.714 | 0.288 | 0.247 | 0.995 | 0.996 | 0.979 |
| count_sketch | 0.959 | 0.430 | 0.260 | 0.485 | 0.993 | 0.959 | 0.892 |
| srht | 0.952 | 0.422 | 0.279 | 0.152 | 0.739 | 0.952 | 0.887 |
| hadamard_det | 0.951 | 0.420 | 0.260 | 0.105 | 0.744 | 0.951 | 0.889 |
| orthogonal | 0.997 | 0.721 | 0.244 | 0.197 | 0.950 | 0.997 | 0.980 |
| orth_blend | 0.997 | 0.711 | 0.263 | 0.205 | 0.948 | 0.997 | 0.978 |
| **sobol** | 0.997 | 0.715 | 0.232 | 0.208 | 0.948 | 0.997 | 0.980 |
| binary01 | 0.866 | 0.427 | 0.242 | 0.167 | 0.592 | 0.866 | 0.853 |
| SignRFF bw=0.25 | 0.995 | 0.710 | 0.269 | **0.881** | 0.705 | 0.995 | 0.979 |
| SignRFF bw=1.0 | 0.997 | 0.706 | 0.301 | 0.504 | 0.931 | 0.997 | 0.983 |
| SignRFF val-tuned | 0.996 | 0.706 | 0.447 | 0.848 | 0.931 | 0.996 | 0.980 |
| RFF real (bw=1.0) | 0.997 | 0.706 | 0.297 | 0.504 | 0.931 | 0.997 | 0.983 |
| FPE phasor | 0.997 | 0.722 | 0.304 | 0.873 | 0.750 | 0.997 | 0.984 |
| FPE 3-bit phase | 0.997 | 0.720 | 0.269 | 0.878 | 0.742 | 0.997 | 0.984 |
| FPE 1-bit phase | 0.987 | 0.501 | **0.522** | 0.898 | 0.651 | 0.987 | 0.946 |
| Nystrom Gaussian | 0.918 | 0.612 | 0.376 | 0.701 | 0.845 | 0.918 | 0.887 |
| Nystrom Laplacian | 0.987 | 0.671 | 0.303 | 0.789 | **0.991** | 0.987 | 0.947 |
| **learned** | 0.997 | **0.734** | 0.211 | 0.396 | **0.979** | 0.997 | **0.985** |
| data-informed K=16 | 0.996 | 0.720 | 0.263 | 0.227 | 0.948 | 0.996 | 0.979 |
| input-modulated | 0.997 | 0.723 | 0.263 | 0.260 | 0.946 | 0.997 | 0.978 |

### Distribution shape barely matters; sparsity is free

Rademacher, uniform, Laplace, Student-t, ternary, very-sparse, orthogonal
and Sobol all land within ~1 pt of Gaussian on every task (linear 0.995-0.997,
fine 0.710-0.721, heavy-tail 0.892-0.995). This is the sub-Gaussian result
made practical: on this pipeline the moment structure, not the distribution
shape, sets accuracy. Rademacher/SRHT/Hadamard matrices store at 1 bit and
very-sparse at 2 bits/nonzero, so the same accuracy is available at **2.9x
smaller model memory** (0.50 MiB vs 1.47 MiB at D=4096, F=64) and 2.4x
fewer ops (very sparse, 0.126 density). Uncentered `binary01` is the
cautionary exception: its nonzero mean biases the encoder (linear 0.866).

### Heavier tails help on heavy-tailed data

Cauchy (no finite variance, L1-stable) and very-sparse (1/8 density) are
the best clean-accuracy models on `heavy_tail` (0.997, 0.995 vs 0.948
Gaussian). Cauchy is also the *best* heavy-tail model under every
corruption in exp3 (dropout-0.5: 0.929; flips-0.40: 0.953). The mechanism
is exactly the 1-stability the construction is chosen for — exp8 measures
it: Cauchy preserves the L1 order of pairwise distances at rho=0.978
(L2: 0.318), Gaussian the reverse (L2 0.993, L1 0.920).

### Kernel methods win on non-linear geometry and lose under shift

RFF/SignRFF/FPE are the only families that solve `shells` (0.85-0.90 vs
0.20-0.27 for every linear RP), and FPE's scale sensitivity is what breaks
the scale-invariance limit: the 1-bit-phase variant even reaches 0.52 on
`xor` (chance 0.25). But on `shift` the ordering inverts completely:

| spec | shift 6 | shift 12 | shift 18 | avg OOD |
|---|---|---|---|---|
| orthogonal | 0.946 | 0.695 | 0.426 | **0.689** |
| sobol | 0.947 | 0.694 | 0.424 | 0.688 |
| laplace | 0.945 | 0.690 | 0.423 | 0.686 |
| gaussian | 0.943 | 0.690 | 0.414 | 0.682 |
| very_sparse | 0.941 | 0.672 | 0.406 | 0.673 |
| cauchy | 0.922 | 0.637 | 0.365 | 0.641 |
| learned | 0.943 | 0.628 | 0.340 | 0.637 |
| Nystrom Laplacian | 0.876 | 0.591 | 0.364 | 0.610 |
| count_sketch | 0.807 | 0.522 | 0.334 | 0.554 |
| srht | 0.796 | 0.483 | 0.299 | 0.526 |
| Nystrom Gaussian | 0.719 | 0.426 | 0.235 | 0.460 |
| SignRFF (val-tuned) | 0.920 | 0.364 | 0.092 | 0.458 |
| FPE phasor | 0.928 | 0.298 | 0.045 | 0.424 |
| SignRFF bw=0.25 | 0.908 | 0.231 | 0.047 | 0.395 |
| FPE 1-bit | 0.838 | 0.136 | 0.035 | 0.336 |

A stationary kernel has a bandwidth-limited receptive field: once the
covariate shift moves test points outside it, the kernel value against
*every* training point decays toward the noise floor and the class evidence
disappears together. Sign-linear RPs degrade gracefully because `sign()`
discards magnitude and shifted points keep their alignment with the
discriminant directions. Exp8 shows how sharply this depends on bandwidth:
ID accuracy is flat for bw >= 0.25 (0.995-0.996) while shift OOD moves
0.354 -> 0.550 -> 0.646 as bw goes 0.25 -> 0.5 -> 1.0 — and the
validation-tuned SignRFF picks the ID-optimal small bandwidth and inherits
the worst OOD region (0.458). **ID/validation accuracy is blind to this
knob.**

### Novel-class discovery: the learned-projection blind spot

| spec | novel ID | NMI | ARI | AUROC |
|---|---|---|---|---|
| orthogonal | 0.980 | **0.968** | 0.967 | 0.983 |
| very_sparse | 0.979 | **0.968** | 0.967 | 0.981 |
| orth_blend | 0.978 | 0.966 | 0.965 | 0.982 |
| laplace | 0.979 | 0.964 | 0.962 | 0.982 |
| gaussian | 0.977 | 0.961 | 0.958 | 0.982 |
| Nystrom Laplacian | 0.947 | 0.946 | 0.944 | 0.893 |
| SignRFF val-tuned | 0.980 | 0.942 | 0.911 | 0.871 |
| FPE phasor | 0.984 | 0.890 | 0.843 | 0.874 |
| SignRFF bw=0.25 | 0.979 | 0.872 | 0.810 | 0.870 |
| srht | 0.887 | 0.839 | 0.817 | 0.911 |
| count_sketch | 0.892 | 0.812 | 0.789 | 0.906 |
| **learned** | **0.985** | **0.666** | **0.610** | 0.972 |
| FPE 1-bit | 0.946 | 0.260 | 0.102 | 0.661 |
| binary01 | 0.853 | 0.341 | 0.141 | 0.813 |

The trained projection has the *best* known-class accuracy (0.985) and the
worst novel-class cluster structure (NMI 0.666 vs 0.961 Gaussian): it
optimizes exactly the known-class directions it was trained on and does not
preserve the isotropic geometry that keeps unseen classes separable. Its
OOD-detection AUROC is still 0.972 because max-similarity detection only
needs the known classes to look different from novel points. Dense random
RPs — orthogonal, very sparse, Laplace, Gaussian — keep both. This is the
projection analogue of the MicroHD finding that ID validation cannot see
what generalization needs.

### Robustness and margins

Mean over `linear`+`fine`+`heavy_tail`+`shift` (detail in
`results/summary_exp1.md`):

| spec | noise s=0.5 | dropout p=0.2 | flips p=0.05 | margin ID | margin OOD | geom rho |
|---|---|---|---|---|---|---|
| learned | **0.895** | 0.888 | **0.927** | **0.314** | **0.248** | 0.734 |
| very_sparse | 0.891 | **0.890** | 0.923 | 0.206 | 0.154 | 0.965 |
| cauchy | 0.886 | 0.881 | 0.917 | **0.209** | 0.147 | 0.904 |
| gaussian | 0.884 | 0.874 | 0.915 | 0.189 | 0.156 | 0.982 |
| sobol | 0.884 | 0.872 | 0.914 | 0.189 | 0.158 | 0.988 |
| orthogonal | 0.884 | 0.874 | 0.913 | 0.190 | 0.158 | **0.989** |
| rademacher | 0.869 | 0.854 | 0.898 | 0.176 | 0.156 | 0.977 |
| signrff_tuned | 0.877 | 0.865 | 0.872 | 0.095 | 0.056 | 0.549 |
| Nystrom Gaussian | 0.769 | 0.740 | 0.786 | 0.027 | 0.014 | 0.470 |
| fpe_b1 | 0.730 | 0.714 | 0.713 | 0.046 | 0.024 | 0.025 |
| count_sketch | 0.790 | 0.721 | **0.086** | 0.001 | 0.001 | 0.616 |
| srht | 0.723 | 0.697 | 0.764 | 0.112 | 0.100 | 0.650 |
| binary01 | 0.657 | 0.637 | 0.687 | 0.044 | 0.070 | 0.275 |

Three mechanisms show up here:

* **Margins are the mechanism** (exp8, n=348-609 seed-level rows):
  `margin_id` vs ID accuracy rho=+0.77, vs noise robustness rho=+0.75;
  `margin_ood` vs shift-12 rho=+0.80; geometry fidelity vs avg OOD
  rho=+0.79 (but vs ID accuracy only rho=+0.28). Low-margin families
  (FPE 1-bit 0.046, Nystrom 0.027, SignRFF 0.095) are consistently the
  most fragile under noise and flips.
* **Count-Sketch codes are structurally fragile to model noise**: 5% sign
  flips on its prototypes drop accuracy from 0.959 to 0.086 on `linear`,
  because 98% of its coordinates are constant +1 and its discriminant
  information lives in only F=64 coordinates — flipping 5% of D hits a
  large fraction of the informative ones. The same projection is one of the
  *strongest* under feature noise and dropout (0.790/0.721), because those
  corruptions act on the 64 informative input features while the
  representation stays redundant. Robustness is corruption-model-specific.
* **Geometry fidelity predicts discovery, not ID accuracy**: SRHT keeps
  rho=0.65 and NMI 0.84 while count-sketch keeps rho=0.62/NMI 0.81; the
  learned projection has the lowest rho among dense methods (0.73) and the
  lowest NMI.

### Resources (D=4096, F=64, C=30, natural storage encoding)

| spec | projection memory | total model | ops/query |
|---|---|---|---|
| gaussian (fp32) | 1.00 MiB | 1.47 MiB | 389 k |
| rademacher / SRHT / Hadamard (1-bit) | 32 KiB | 0.50 MiB | 389 k |
| achlioptas (2-bit, 33% dense) | 21 KiB | 0.49 MiB | 214 k |
| ternary (2-bit, 67% dense) | 43 KiB | 0.51 MiB | 302 k |
| very_sparse (2-bit, 12.6% dense) | 8 KiB | 0.48 MiB | 160 k |
| count_sketch (1 bit + index) | 104 B | 0.47 MiB | 127 k |
| RFF / SignRFF | 1.02 MiB | 1.48 MiB | 389 k |
| FPE phasor (32-bit phases, 2D codes) | 1.03 MiB | 1.95 MiB | 516 k |
| learned | 1.00 MiB | 1.47 MiB | 389 k |
| Nystrom (L=D=4096 landmarks) | 65.0 MiB | 65.5 MiB | 17.2 M |

The classifier prototypes dominate the model for D >> F (480 KiB of the
1.47 MiB), so projection-side savings matter mostly through ops: very sparse
cuts ops 2.4x, Rademacher-style 1-bit storage 2.9x at identical accuracy.
Nystrom's inference cost is inherently quadratic in D (landmark kernel
expansion), which is why it is not the efficiency story even though its
Laplacian variant is competitive in accuracy.

## Exp2: dimension sweep (`results/exp2_dimension_sweep.csv`)

ID accuracy across D (F=64; the dotted line in the figure marks D=F):

| spec | D=32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|---|---|---|---|
| gaussian, `linear` | 0.589 | 0.859 | 0.950 | 0.978 | 0.988 | 0.994 | 0.996 | 0.997 |
| gaussian, `fine` | 0.164 | 0.298 | 0.438 | 0.571 | 0.665 | 0.698 | 0.712 | 0.722 |
| count_sketch, `linear` | **0.576** | **0.810** | 0.902 | 0.942 | 0.954 | 0.954 | 0.958 | 0.959 |
| count_sketch, `fine` | 0.145 | 0.227 | 0.333 | 0.377 | 0.386 | 0.414 | 0.422 | 0.430 |
| srht, `linear` | 0.525 | 0.795 | 0.909 | 0.922 | 0.936 | 0.952 | 0.954 | 0.952 |
| srht, `fine` | 0.151 | 0.258 | 0.386 | 0.419 | 0.428 | 0.421 | 0.402 | 0.422 |
| sobol, `linear` | 0.533 | 0.842 | 0.961 | 0.986 | 0.994 | 0.997 | 0.996 | 0.997 |
| learned, `fine` | **0.506** | **0.612** | **0.676** | **0.708** | **0.721** | **0.728** | **0.736** | 0.734 |
| FPE 1-bit, `linear` | 0.123 | 0.235 | 0.456 | 0.711 | 0.897 | 0.962 | 0.980 | 0.987 |
| FPE 1-bit, `shells` | 0.632 | 0.709 | 0.776 | 0.814 | 0.867 | 0.867 | 0.900 | **0.898** |
| SignRFF, `shells` | 0.504 | 0.593 | 0.612 | 0.712 | 0.778 | 0.742 | 0.801 | 0.848 |

* **Expansion (D > F) is where HDC lives, but structured/sketching
  constructions were designed for compression.** Count-Sketch is
  competitive for D <= F (0.576 vs Gaussian 0.589 at D=32) and then
  saturates ~0.96 because it only ever exposes F=64 informative
  coordinates. SRHT/Hadamard *peak* at D=512-1024 and then lose 5 pts:
  when D > F their rows are repeats of the F Hadamard rows, so extra
  coordinates add redundancy rather than independent random directions.
  Gaussian/Rademacher/Sobol keep improving because every row is a fresh
  direction in R^F. Exp8 confirms the geometry: SRHT/Hadamard have
  `max off-diagonal |cos| = 1.0` (repeated rows) at D=4096.
* **Learned projections dominate in the low-D regime** (fine: 0.506 vs
  0.164 gaussian at D=32; +0.10 at D=128, +0.02 by D=4096) — learning pays
  off exactly where random projections cannot afford enough directions.
* **Kernel/frequency codes need D**: FPE-1bit is useless below D=128 and
  keeps improving through D=4096; SignRFF on `shells` improves from 0.50 to
  0.85. Linear RPs saturate on separation tasks by D~256-512.
* **XOR stays at chance for every linear family at every D** (all rows
  0.21-0.29); the only movement with D is for even-component encoders
  (FPE-1bit 0.28 -> 0.52).

## Exp3: robustness curves (`results/exp3_robustness.csv`, summary)

Corruption level at which accuracy falls to 90% of its clean value
(higher = more robust; caps: noise 1.5, dropout 0.5, flips 0.4):

| spec | noise (fine/heavy/linear) | dropout (heavy/linear) | flips (fine/linear) |
|---|---|---|---|
| gaussian (cosine) | 0.349 / 1.288 / 1.363 | 0.419 / 0.333 | 0.363 / 0.400 |
| cauchy + L1 rule | 0.359 / 1.316 / 1.442 | **0.500** / 0.384 | 0.370 / 0.400 |
| gaussian + L1 rule | 0.328 / 1.284 / 1.370 | 0.420 / 0.337 | 0.362 / 0.400 |
| learned | 0.339 / 1.286 / 1.394 | 0.358 / 0.358 | 0.283 / 0.400 |
| very_sparse | 0.364 / 1.264 / 1.371 | **0.500** / 0.344 | 0.364 / 0.400 |
| Sobol | 0.326 / 1.244 / 1.352 | 0.409 / 0.319 | 0.359 / 0.400 |
| SignRFF (val-tuned) | 0.332 / 1.317 / **0.968** | 0.447 / 0.392 | **0.032** / 0.330 |
| FPE 1-bit | 0.333 / 0.708 / 0.946 | 0.163 / 0.188 | **0.004** / 0.393 |
| Nystrom Gaussian | 0.382 / 0.787 / 0.569 | 0.204 / 0.180 | 0.062 / 0.182 |
| count_sketch | 0.414 / 0.877 / 0.679 | 0.140 / 0.261 | **0.009** / **0.009** |

* Dense random RPs share an almost identical corruption profile; the L1
  decision rule and Cauchy lift the heavy-tail/dropout corner (0.5 cap).
* RFF degrades with input noise at less than half the noise budget of
  dense RPs on `linear` (0.97 vs 1.36), a second face of the
  bandwidth-limited receptive field.
* FPE-1bit and count-sketch collapse under tiny flip probabilities
  (0.004-0.009) — both have extremely low effective coordinate diversity
  (FPE's phases are near-constant at the default bandwidth; count-sketch's
  codes are constant except F coordinates). Their clean accuracy is fine;
  their stored models are not robust.

## Exp4: quantization (`results/exp4_quantization.csv`)

**Output codecs are free; vector quantization is not.** ID accuracy for a
Gaussian RP (D=4096), and avg shift OOD:

| codec | fine | heavy_tail | novel | shift | avg OOD |
|---|---|---|---|---|---|
| fp32 / int8 / int4 / int2 | 0.722 | 0.948 | 0.977 | 0.997 | 0.682 |
| bipolar / ternary | 0.722 | 0.948 | 0.977 | 0.997 | 0.682 |
| unipolar binary | 0.725 | 0.949 | 0.980 | 0.997 | 0.680 |
| thermometer (4 bits/coord) | 0.722 | 0.949 | 0.980 | 0.997 | 0.680 |
| block-sparse (1/8 kept) | 0.657 | 0.929 | 0.965 | 0.992 | 0.655 |
| k-means VQ (K=256) | **0.130** | **0.471** | 0.841 | 0.986 | 0.659 |

For sign-like codes (SignRFF, FPE-3bit) the same holds within 1 pt. The
finding is the *opposite* of the usual quantization intuition: 2-bit
scalar codes cost nothing because `sign()`-dominated codes carry their
information in directions, not magnitudes, while an aggressive **k-means
codebook (256 codewords for a 4096-dim cloud) destroys accuracy** — the
codebook is too coarse to represent the class-conditional geometry.

**Projection-matrix PTQ** (Gaussian and SRHT, exp4 `ptq`):

| config | fine | heavy_tail | novel |
|---|---|---|---|
| fp32 | 0.722 | 0.948 | 0.977 |
| int8 MSE / max | 0.718 / 0.721 | 0.947 / 0.948 | 0.977 / 0.976 |
| int4 MSE / max | **0.727** / 0.722 | 0.947 / 0.948 | 0.978 / 0.978 |
| int2 MSE / max | 0.718 / **0.603** | 0.942 / **0.997** | 0.977 / 0.956 |
| ternary | 0.713 | 0.944 | 0.978 |
| sign (1-bit) | 0.714 | 0.890 | 0.980 |
| SRHT fp32 -> sign | 0.422 (all PTQ variants identical: SRHT is already +/-1) | 0.739 | 0.887 |

* MSE-optimal per-tensor PTQ is **free down to 2 bits** (fine 0.718 vs
  0.722 fp32; heavy-tail 0.942 vs 0.948).
* Max-scaled 2-bit PTQ (`gaussian_p2_max`) *helps* heavy-tail (0.997, the
  best single number in the study) and *hurts* fine (0.603): the coarse
  max-scaled step clips the weight tail, which acts as an outlier
  regularizer when labels are noise-corrupted and as information loss when
  classes are close. Same codec, opposite sign of effect.
* A 1-bit (sign) projection costs 3-6 pts on heavy-tail but is otherwise
  close; ternary is nearly free.
* Quantization-aware prototype retraining (2/3/4 bits): +0.8/+0.5/+0.1 pt
  on fine, +0.1-0.2 elsewhere — real but small, because post-hoc MSE-PTQ is
  already near-lossless here. QAT matters when the codec is the bottleneck,
  not when it is free.

## Exp5: feature extractors (`results/exp5_extractor.csv`)

D=4096, 30 epochs. The table reports the `+gauss` projection; the
conclusions for `+signrff`/`+learned` differ only on `fine`/`heavy_tail`
(e.g. random MLP: 0.369 -> 0.579 fine with a learned projection):

| extractor | fine | heavy_tail | shift ID | shift OOD |
|---|---|---|---|---|
| raw (no extractor) | 0.722 | 0.948 | 0.997 | **0.682** |
| random MLP (frozen) | 0.369 | 0.826 | 0.923 | 0.502 |
| CE MLP | 0.690 | 0.928 | 0.992 | 0.607 |
| CE MLP, 256-wide | 0.698 | 0.936 | 0.993 | 0.623 |
| CE MLP, 2 hidden layers | 0.660 | 0.909 | 0.978 | 0.551 |
| SupCon (CE + contrastive) | 0.706 | 0.944 | 0.996 | 0.655 |
| proxy-anchor metric | 0.689 | 0.935 | 0.994 | 0.618 |
| angular-margin (arc) | 0.682 | 0.941 | 0.995 | 0.633 |
| CE + weight clustering (8 levels) | 0.691 | 0.926 | 0.992 | 0.602 |
| co-trained with fixed random RP | 0.694 | 0.942 | 0.996 | 0.633 |
| co-trained, learned RP | 0.679 | 0.944 | 0.995 | 0.613 |

* **A trained extractor is not automatically better for HDC.** Every MLP
  variant lowers shift OOD relative to raw features (best: SupCon 0.655,
  co-train 0.633, CE 0.607 vs raw 0.682) while helping at most 1 pt on
  fine. The extractor is optimized for a CE classifier and compresses
  away directions that the prototype classifier was exploiting.
* **The projection can rescue random features**: a frozen random MLP with
  a learned projection reaches fine 0.579 vs 0.369 with a random
  projection (+21 pts) and heavy-tail 0.898 vs 0.826.
* **Weight clustering is nearly free** (4 levels: -1.9 pts fine, +0.7 OOD
  relative to CE; 8 levels: +0.1/-0.5) — a 4x weight-memory reduction for
  ~1 pt.
* **Co-training through the HDC objective is the only variant that helps
  robustness and unlocks XOR** (exp7: XOR 0.51 vs 0.25 chance for any
  linear projection) at the cost of ~0.5 pt fine accuracy.

## Exp6: processing and classifier variants (`results/exp6_processing.csv`)

Gaussian RP, D=4096:

| variant | fine | heavy_tail | novel | shift |
|---|---|---|---|---|
| cosine (base) | 0.722 | 0.948 | 0.977 | 0.997 |
| dot product | 0.686 | 0.943 | 0.974 | 0.994 |
| L1 (L2-normalized vectors) | 0.671 | 0.942 | 0.979 | 0.996 |
| RBF (normalized) | 0.676 | 0.942 | 0.978 | 0.997 |
| Hamming on {0,1} codes | 0.700 | 0.938 | 0.972 | 0.994 |
| unipolar binary + cosine | 0.725 | 0.949 | 0.980 | 0.997 |
| margin-weighted bundling | 0.723 | 0.945 | 0.974 | 0.995 |
| confidence-weighted bundling | 0.720 | 0.947 | 0.977 | 0.997 |
| multi-prototype k=2 | 0.652 | 0.934 | 0.975 | 0.994 |
| multi-prototype k=4 | 0.616 | 0.897 | 0.971 | 0.994 |
| multi-prototype k=8 | 0.581 | 0.840 | 0.971 | 0.993 |
| single pass (no retraining) | 0.708 | 0.947 | 0.976 | 0.996 |
| prototypes @ 2 bits | 0.718 | 0.947 | 0.977 | 0.996 |
| prototypes @ sign (1 bit) | 0.709 | 0.948 | 0.978 | 0.996 |
| L2 / standardize / PCA-whiten inputs | 0.722 (all identical) | 0.948 | 0.977 | 0.997 |

* Similarity metric is a 3-5 pt knob at most; cosine is a safe default.
  **L1/RBF must be applied to L2-normalized vectors** — with raw
  accumulated prototypes the ranking is dominated by prototype norm and
  accuracy collapses to chance (0.03-0.04). This is a trap worth stating:
  all four metrics are implemented on normalized vectors in `hd.py`.
* **Multi-prototype classes hurt** (k=8: -14 pts fine, -11 heavy-tail):
  cosine k-means fragments the class-conditional mode that a single
  bundled prototype represents well. 30 balanced Gaussian-ish classes do
  not need mixture prototypes.
* Preprocessing is a no-op for sign codes (as expected — `sign` sees only
  the direction), and retraining buys ~1.4 pts on fine.
* Prototype storage at 2 bits or even 1 sign bit is free for Gaussian RP
  (total classifier memory /16 or /32).

## Exp6b: symbolic sequence processing (`results/exp6b_sequence.csv`)

Class = presence of an ordered bigram; bag-of-symbols is at chance (1/6),
so ID accuracy measures the encoding, not the classifier (D=2048):

| mode | mul (bipolar) | xor (binary) | circ (HRR) |
|---|---|---|---|
| bag (symbols only) | 0.301 | 0.301 | 0.312 |
| bag + position code | 0.348 | 0.348 | 0.372 |
| bigram + position code | 0.601 | 0.601 | 0.523 |
| **bigram, position-invariant** | **0.692** | **0.692** | 0.519 |

* The conjunction matters most: binding adjacent symbols raises accuracy
  from 0.30 to 0.69, and **adding a position code hurts** (0.69 -> 0.60)
  when the label is position-invariant — the same bigram at different
  positions lands in different codes, diluting the prototype. Position
  should be encoded only when order matters.
* Bipolar `mul` binding and binary `xor` binding are exactly equivalent
  (as expected for +/-1 codes); HRR `circ` binding is 5-17 pts worse here
  at matched D and dimension parity.

## Exp7: learned, data-informed and co-trained projections (`results/exp7_learned.csv`)

D=1024 (where random RPs still leave room):

| spec | fine | heavy_tail | linear | shift | xor |
|---|---|---|---|---|---|
| gaussian | 0.698 | 0.939 | 0.994 | 0.994 | 0.264 |
| SignRFF (val-tuned) | 0.698 | 0.906 | 0.995 | 0.995 | 0.390 |
| learned, 10 epochs | 0.726 | 0.964 | 0.996 | 0.996 | 0.251 |
| learned, 40 epochs | 0.730 | 0.974 | 0.996 | 0.996 | 0.240 |
| learned, 120 epochs | 0.728 | 0.984 | 0.996 | 0.996 | 0.238 |
| learned, Rademacher init | 0.729 | 0.970 | 0.996 | 0.996 | 0.233 |
| learned, lr=3e-2 | 0.728 | 0.976 | 0.997 | 0.997 | 0.211 |
| data-informed K=1 | 0.698 | 0.939 | 0.994 | 0.994 | 0.264 |
| data-informed K=16 | 0.692 | 0.939 | 0.994 | 0.994 | 0.278 |
| data-informed K=256 | 0.695 | 0.939 | 0.994 | 0.994 | 0.265 |
| input-modulated (gain 0.5) | 0.693 | 0.936 | 0.994 | 0.994 | 0.283 |
| co-trained, fixed random RP | 0.698 | 0.944 | 0.996 | 0.996 | **0.510** |
| co-trained, learned RP | 0.676 | 0.943 | 0.995 | 0.995 | 0.475 |

* Learning costs +3.2 pts on fine and +4.5 on heavy-tail at D=1024 (the
  gain at D=128 is +15-24 pts, exp2); epochs beyond ~40 do not matter and
  init/lr barely do.
* **Best-of-K selection is not a substitute**: K from 1 to 256 changes
  nothing at D=1024 (0.692-0.698 fine). Selection can only pick a good
  *random* sample of the same distribution; it cannot shape it.
* Input modulation does not help either.
* **Co-training is the only method that solves XOR** (0.51 vs chance 0.25,
  +0.25 over the next-best projection): a ReLU extractor is not an odd
  function, so the symmetry obstruction of pure linear projections does not
  apply. Learned *linear* projections remain at chance even at 120 epochs
  (0.24) — a direct empirical confirmation of the odd-symmetry limit.

## Exp8: geometry and theory validation (`results/exp8_*.csv`)

**(a) Johnson-Lindenstrauss distortion.** Std of the relative squared-L2
distance error (variance-preserving rescaling) at D=4096:

| construction | measured std | sqrt(2/D) prediction | ratio |
|---|---|---|---|
| gaussian | 0.0226 | 0.0221 | 1.02 |
| rademacher / SRHT / very-sparse | 0.0219 / 0.0219 / 0.0231 | 0.0221 | 0.99-1.05 |
| ternary / achlioptas / uniform | 0.021-0.022 | 0.0221 | 0.93-1.01 |
| student_t(3) | 0.031 | 0.0221 | 1.42 |
| **sobol** | **0.0037** | 0.0221 | **0.17** |
| orthogonal / hadamard_det / count_sketch | ~1e-7 | 0.0221 | 0 (exact rescaled isometry) |
| binary01 | 0.729 | 0.0221 | 33 |
| cauchy | ~1e5 | 0.0221 | n/a (no finite variance) |

The sub-Gaussian JL guarantees hold to within a few percent for every
zero-mean bounded-variance construction. **The low-discrepancy Sobol
projection is 6x below the random JL rate** — on this metric determinism
buys accuracy, not costs it. Cauchy's distance estimator is useless for
L2 (as theory says) and excellent for L1 (next table).

**(b) L1 vs L2 order preservation** (Spearman rho of pairwise distance
estimates, 3000 pairs, D=4096):

| construction | preserves L2 (rho) | preserves L1 (rho) |
|---|---|---|
| gaussian | **0.993** | 0.920 |
| laplace | 0.992 | 0.928 |
| rademacher | 0.993 | 0.914 |
| student_t(3) | 0.985 | 0.940 |
| **cauchy** | 0.318 | **0.978** |

This is the cleanest confirmation that the projection family *is* a choice
of geometry: a Cauchy RP classifier should be paired with an L1 decision
rule (exp3/exp6), where it is the best heavy-tail model in the study.

**(c) Kernel approximation** (MSE against each encoder's own Gaussian
kernel):

| method | D=64 | 256 | 1024 | 4096 |
|---|---|---|---|---|
| RFF (real) | 0.0120 | 0.0028 | 0.0007 | **0.0002** |
| FPE phasor | 0.0042 | 0.0011 | 0.0003 | **0.0001** |
| SignRFF (1-bit) | 0.0155 | 0.0116 | 0.0098 | 0.0078 |
| Nystrom (random landmarks) | 0.207 | 0.321 | 0.295 | 0.167 |

RFF/FPE track the 1/D Monte-Carlo rate; the phasor (bias-free) variant is
4x better in absolute error. **SignRFF has an irreducible bias floor**:
64x more dimensions buys only 2x lower error, which is the price of the
1-bit code (exp1 still shows SignRFF matching linear RPs on accuracy, so
the floor matters for kernel fidelity, not for classification).
Random-landmark Nystrom does not improve with D here — landmark choice,
not dimension, is its bottleneck.

**(d) Bandwidth is the OOD knob ID cannot see** (shift task, D=2048):

| bw scale | 0.1 | 0.25 | 0.5 | 1.0 | 2.0 | 4.0 |
|---|---|---|---|---|---|---|
| ID accuracy | 0.718 | 0.995 | 0.996 | 0.996 | 0.996 | 0.996 |
| avg shift OOD | **0.054** | 0.354 | 0.550 | 0.643 | 0.646 | 0.640 |

**(e) Cross-links** (Spearman over seed-level rows,
`results/exp8_crosslinks.csv`): margin ID -> ID 0.773; margin ID ->
noise-0.5 0.751; margin OOD -> shift-12 0.800; geometry fidelity -> avg
OOD 0.786; geometry fidelity -> ID 0.279. Margins and geometry predict
generalization; ID accuracy does not predict OOD.

**(f) Row incoherence** (max off-diagonal |cos| between projection rows,
D=1024): cauchy 1.00, srht/hadamard_det 1.00 (repeated rows), student_t
0.84, binary01 0.80, sobol 0.77, very_sparse 0.73, rademacher 0.66,
gaussian 0.55, count_sketch 0.00 (disjoint supports). This is the
geometric reason SRHT/Hadamard underperform at D >> F.

## Exp9: projection x codec x dimension (`results/exp9_interactions.csv`)

The hypothesized projection x quantization interaction is essentially
**absent** in this pipeline for scalar codecs: at every D in {256, 1024,
4096} and for every projection in {Gaussian, SignRFF, FPE-1bit, learned},
the ID accuracy of fp32, int4, ternary and bipolar codes is *identical to
three decimals* on all three tasks (e.g. Gaussian fine 0.571 / 0.698 /
0.722 at the three dimensions for all four codecs). The only interaction
found: at small D, SignRFF loses 5-15 pts when its real-valued features are
sign-quantized (D=256 fine: fp32 0.524 vs bipolar 0.374 vs ternary 0.439)
because a small number of directions is carrying the signal before the
sign. Scalar codecs quantize *magnitudes* that the cosine classifier
ignores; the projection's information lives in directions, so the two axes
do not interact.

## Decision guide

| goal | recommendation |
|---|---|
| smooth/linear-like data | any zero-mean dense RP; use Rademacher/SRHT 1-bit storage or very-sparse (2.4x fewer ops) for free |
| fastest projection | Count-Sketch (D ops/query) — but expect ~4 pts lower accuracy when D > F and fragile stored prototypes |
| radial / non-linear structure | RFF, FPE or Nystrom with a **tuned bandwidth**; expect 0.85-0.90 vs 0.20-0.25 for linear RPs |
| covariate-shift OOD | dense sign-linear RPs (Gaussian/Sobol/orthogonal/Laplace/very sparse); do **not** tune bandwidth on ID accuracy (59-pt OOD swing) |
| novel-class discovery | dense zero-mean RPs (orthogonal, very sparse, Laplace); avoid learned projections (NMI 0.67 vs 0.96) |
| heavy-tailed / outlier-prone features | Cauchy or very-sparse RP, optionally with an L1 decision rule (best clean and dropout robustness) |
| close classes at small D | learned projection (STE), +10-24 pts over random at D=128 |
| parity/XOR-like labels | no linear projection can help; use a ReLU extractor co-trained with the HDC objective (0.51 vs 0.25 chance) |
| 1-2 bit prototype storage | MSE-PTQ (free at 2 bits); QAT adds <1 pt; skip k-means VQ (catastrophic) |
| sequence bigrams | bind adjacent symbols (`mul`/`xor`); add position codes only if the task needs them |
| reproducing without run-to-run variance | scrambled-Sobol RP: identical accuracy to Gaussian, 6x lower JL distortion |

## Deviations and limitations

* **Synthetic data only.** Gaussian/radial/parity/heavy-tail/symbolic tasks
  isolate mechanisms; no images, audio, strings or graph kernels. The
  Nystrom claim about non-Euclidean kernels is tested only through the
  Laplacian-vs-Gaussian comparison.
* **Group VSA** is represented by phase-quantized FPE (cyclic-group style),
  not the full finite-group construction of the literature. The 1-bit phase
  case is exactly `sign(cos)`, i.e. a bias-free SignRFF.
* **Nystrom** uses random landmarks and a CPU Cholesky whitening
  (`K^{-1/2}` up to a rotation); k-means landmarks and string kernels are
  not implemented. The landmark kernel (L x L) is used only during fit and
  excluded from the inference memory model.
* **Learned projections** implement a PIONEER/LeHDC-style surrogate: a
  straight-through binarized projection trained with a prototype-cosine
  head. The trained proxies are discarded at evaluation; prototypes are
  re-accumulated. Not the published network architecture.
* **QuantHD / HDVQ-VAE** are represented by prototype quantization-aware
  retraining and k-means hypervector codebooks respectively, not their full
  training procedures.
* **Feature extractors** are small MLPs (1-2 hidden layers, width 128-256)
  on 64-dimensional synthetic features. The extractor results may not
  transfer to large pretrained backbones where features are already
  highly structured.
* **Robustness models** are additive feature noise, feature dropout, and
  per-coordinate sign flips of stored prototypes; not hardware bit-error
  models over a quantized memory hierarchy.
* **Resource metrics** are analytical proxies (stored bits, multiply-adds);
  no wall-clock latency or accelerator measurements.
* **`HDCPROJ_DEVICE=cpu`** re-runs everything on CPU (slower but immune to
  the display-GPU watchdog issue).

## Reproducibility

* Python 3.14, PyTorch 2.13 (CUDA 13.3), NumPy 2.5, SciPy 1.18,
  scikit-learn 1.9, pandas 3, matplotlib 3.11; RTX 5080 (16 GB).
* 3 seeds per cell; data, projection and retraining seeds are derived from
  the cell seed; `Spec.meta()` records every configuration in the CSVs.
* Total suite ~10.5 minutes (per-experiment timings in `logs/`); peak VRAM
  1.7 GB; peak RSS ~5 GB.
* Per-experiment summaries: `results/summary_*.md`; figures
  `results/fig_*.png`; raw CSVs alongside.
