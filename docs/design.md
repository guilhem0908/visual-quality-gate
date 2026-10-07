# Design notes

Derivations and decisions behind the code, in the order a part travels through it. Result
numbers live in the README (generated from `results/summary.json`); this file only contains
quantities that follow from the design itself.

## 1. Problem and vocabulary

A quality station photographs every part and must answer OK or NOK. Defects are rare and not
known in advance, so the model is fitted on good parts only (one-class, or "unsupervised",
anomaly detection). Two error types matter to the line, and they are not symmetric:

| Term | Meaning | Cost |
|:--|:--|:--|
| false reject | a good part is refused | scrap or manual re-inspection |
| escape | a defective part passes | a defect reaches the customer |

A detector outputs a real-valued score per image; the gate is `NOK if score > t`. Choosing
`t` moves errors from one type to the other. AUROC summarises all thresholds at once, which
is why it cannot answer "how many defects escape if I accept 5 % false rejects".

## 2. Frozen features

Neither method trains a network. A classification backbone pretrained on ImageNet
(the original torchvision `IMAGENET1K_V1` weights) is used as a fixed feature extractor: its intermediate activations describe local texture and
shape well enough that "unusual activation" is a usable proxy for "unusual part".

- Input: resize the short side to 256 px, centre-crop 224 x 224, ImageNet normalisation
  (both papers).
- For a 224 px input, `layer1`, `layer2`, `layer3` of a ResNet have strides 4, 8, 16:
  grids of 56 x 56, 28 x 28 and 14 x 14. PatchCore (Sec. 3.1) argues that the last stage is
  too coarse and too biased towards ImageNet classes; `layer4` and the classifier are never
  run here.
- ResNet-18 gives 64 + 128 + 256 = 448 channels over the three layers, Wide-ResNet-50-2 gives
  256 + 512 + 1024 = 1792.

## 3. PaDiM

**Embedding.** The three maps are brought to the 56 x 56 grid by repeating the coarser
activations (nearest neighbour) and concatenated, so each of the P = 3136 positions carries
448 values. PaDiM keeps a random subset of d = 100 channels; the paper reports that this
random selection loses little accuracy and does better than PCA (its Table II).

**Model.** For each position (i, j) independently, the N good embeddings are modelled by a
Gaussian with the usual unbiased estimates and a ridge term:

    mu    = 1/N sum_k x_k
    Sigma = 1/(N-1) sum_k (x_k - mu)(x_k - mu)^T + eps I,     eps = 0.01

The ridge keeps Sigma invertible when fewer than d images are available and bounds the score
in directions that barely vary.

**Score.** The Mahalanobis distance M(x) = sqrt((x - mu)^T Sigma^-1 (x - mu)) is the
Euclidean distance after whitening. With the Cholesky factor Sigma = L L^T,

    (x - mu)^T Sigma^-1 (x - mu) = || L^-1 (x - mu) ||^2 ,

so the model stores W = L^-1 (lower triangular) and the score is one matrix-vector product
and a norm per position. If the good embeddings really were Gaussian, M^2 would follow a
chi-square law with d degrees of freedom (mean d, variance 2d) at every position: scores of
different positions are on a common scale, which is what makes "take the maximum over the
image" meaningful. `tests/test_padim.py` checks the estimates against NumPy, the distance
against SciPy, the chi-square moments on fresh samples and the invariance of M under an
invertible linear map of the features.

**One-pass fit.** Only sum x and sum x x^T are accumulated (in float64), so the fit never
holds all embeddings; the covariance is (sum x x^T - N mu mu^T) / (N - 1).

**Cost.** The model is P x d x d whitening matrices plus P x d means:
3136 x 100 x 100 x 4 bytes = 125.4 MB + 1.3 MB with ResNet-18 and d = 100, independent of the
number of training images. The paper's Wide-ResNet-50 variant uses d = 550 and reports 3.8 GB
(its Table VIII); that does not fit the "one CPU box" framing, so it is not run here.

## 4. PatchCore

**Locally aware patch features.** `layer2` and `layer3` are each averaged over a 3 x 3
neighbourhood (stride 1), `layer3` is upsampled bilinearly to the 28 x 28 grid, and the two
are concatenated: 784 patches per image with 384 (ResNet-18) or 1536 (Wide-ResNet-50)
channels. The official implementation additionally pools every patch vector to 1024
dimensions; this repository keeps the plain concatenation, which is simpler to explain and
is one documented difference with the published numbers.

**Memory bank and coreset.** All patch features of the good images form a set X; storing
all of them would make every inspection a search over N x 784 vectors. PatchCore keeps a
subset C that covers X: it minimises the coverage radius

    r(C) = max_{x in X} min_{c in C} || x - c || .

This is the k-centre problem (NP-hard). Farthest-first traversal adds, at each step, the
point farthest from the centres chosen so far.

*Why it is a 2-approximation.* Let the greedy centres be c_1..c_k with radius r, and let p be
a point at distance r from them. By construction every greedy centre was at distance >= r
from the earlier ones, so the k + 1 points c_1..c_k, p are pairwise >= r apart. Any k
optimal centres must put two of these k + 1 points in the same cluster (pigeonhole); both
are within r_opt of that cluster's centre, hence r <= 2 r_opt. The test suite checks this
bound against a brute-force optimum, and the pairwise-separation property used in the proof.

*Why a random projection.* Each greedy step needs the distance from one centre to all
points. In a 128-dimensional Gaussian random projection R with entries N(0, 1/128), squared
norms are preserved in expectation and a distance ratio has standard deviation about
1/sqrt(2 x 128) = 6 % (Johnson-Lindenstrauss). Selection runs in the projected space; the
bank stores the original features of the selected indices.

*Nesting.* The first m picks of a greedy run are exactly the m-centre greedy solution of the
same run. The benchmark therefore runs the selection once at 10 % and reads the 1 % bank as a
prefix.

*Cost.* One step is one matrix-vector product over all projected points
(|| x - c ||^2 = || x ||^2 + || c ||^2 - 2 x.c with the norms cached). For the screw category
that is 250,880 patches x 128 values, about 128 MB read per step, 25,088 steps at 10 %. The
fit is memory-bandwidth bound: this is the slowest part of the whole pipeline, and the reason
the benchmark uses the GPU when one is present.

**Score.** s(m) = distance from a test patch feature to its nearest neighbour in the bank;
the image score is the maximum over the 784 patches. The paper's optional re-weighting of the
image score (its Eq. 7) is not implemented. Distances use the same expansion as above in
float32; cancellation makes the score of a stored patch slightly positive instead of exactly
0, far below the scale of real scores (a test bounds it).

**Cost.** Bank size is ratio x N x 784 vectors: for screw (N = 320), 25,088 x 1536 x 4 bytes
= 154.1 MB at 10 % with Wide-ResNet-50, 15.4 MB at 1 %, 3.9 MB at 1 % with ResNet-18.
Unlike PaDiM it grows with the number of training images, and so does the search time.

## 5. From patch scores to a map

Patch scores are upsampled bilinearly to 224 x 224 and smoothed with a Gaussian of sigma = 4
px (both papers). PaDiM's image score is the maximum of the smoothed map; PatchCore's is the
maximum of the raw patch scores. Pixel-level AUROC pools every pixel of every test image of a
category (good images contribute only negatives).

## 6. Metrics

AUROC is computed from ranks: with average ranks R for ties,

    AUROC = ( sum of R over defective samples - n_pos (n_pos + 1) / 2 ) / ( n_pos n_neg ) ,

the Mann-Whitney U statistic divided by the number of (defective, good) pairs: the
probability that a random defective sample outscores a random good one. Tests compare it
with scikit-learn, including heavy ties.

## 7. The gate: a threshold from good parts only

Tuning `t` on defects is not possible on a line that has none on record. Split conformal
prediction needs only good parts that the detector did not see during the fit:

1. hold out n good parts, score them: s_(1) <= ... <= s_(n);
2. set t = s_(k) with k = ceil((n + 1)(1 - alpha)).

*Why it works.* If a new good part's score is exchangeable with the n held-out scores, its
rank among the n + 1 values is uniform on 1..n + 1, so P(score > s_(k)) = 1 - k/(n + 1)
<= alpha. Nothing is assumed about the score distribution or about the detector.

*What it costs.*
- k <= n requires n >= 1/alpha - 1: 19 held-out parts for a 5 % target, 99 for 1 %. With the
  42 to 64 parts held out here, 1 % cannot be certified at all.
- The guarantee is an average over held-out draws. For one draw, the realised false-reject
  rate follows a Beta(n + 1 - k, k) law: with n = 48 and alpha = 5 % its mean is 4.1 % and
  its standard deviation 2.8 points. A threshold from about 50 parts is a noisy threshold,
  which is why the README reports per-seed spreads.
- Parts used for calibration are not used for the fit (80 % / 20 % split here).

*What can break it.* Exchangeability. Scoring the training images themselves would be the
extreme case (PatchCore gives them a distance of zero), hence the hold-out. A milder version
remains: if the held-out parts resemble the fit parts more than future parts do, their
scores are too low and so is the threshold. The benchmark measures this directly: the
"shift" column of the README is the AUROC of test-good scores against held-out-good scores,
50 % when the two are indistinguishable. On a real line the held-out parts should come from
a different batch, day or shift than the fit parts, and the realised false-reject rate
should be monitored and the threshold recalibrated when it drifts.

## 8. Protocol decisions

- **Two fits per seed.** `full` (all good training images) reproduces the papers' setting for
  the AUROC comparison; `gate` (80 %) is the only honest way to obtain held-out scores. The
  README shows both so the price of holding parts out is visible.
- **Seeds.** A seed sets the fit/held-out split, PaDiM's channel subset and PatchCore's
  projection matrix and first centre. Three seeds; counts are summed and the per-seed range
  is shown.
- **Five categories.** bottle, grid, metal_nut, screw, zipper: small archives (156 to 195 MB
  each), at least 209 good training images, one texture and four objects, one hard case
  (screw). The 15-category averages of the papers are not comparable and are not quoted.
- **Raw counts.** Test sets hold 20 to 41 good parts per category; one false reject moves a
  category rate by 2.4 to 5 points. Rates are always printed next to their counts.

Images per category (MVTec AD, Bergmann et al., CVPR 2019). *Held out* is the part of the good
training images that is scored for the threshold and not used for the fit (a fifth of them):

<!-- results:data -->
| Category | Train good | of which held out | Test good | Test defective |
| :-- | --: | --: | --: | --: |
| bottle | 209 | 42 | 20 | 63 |
| grid | 264 | 53 | 21 | 57 |
| metal_nut | 220 | 44 | 22 | 93 |
| screw | 320 | 64 | 41 | 119 |
| zipper | 240 | 48 | 32 | 119 |
| **total** | 1253 | 251 | 136 | 451 |
<!-- /results:data -->

## 9. Engineering decisions

- **Scores are the source of truth.** The benchmark writes one row per scored image to
  `results/scores/*.csv`. Every image-level number (AUROC, thresholds, false rejects,
  escapes, trade-off curve) is recomputed from those files by `reproduce.py --stages report`
  without the dataset, in CI. Pixel-level AUROC and timings need the images and are stored in
  `results/benchmark.json` and `results/latency.json`.
- **README cannot drift.** The README tables, its verdict sentence and the data table above
  are generated from `results/summary.json` and compared by `scripts/check_readme.py`.
- **Features once.** Images are decoded once per category and backbone features extracted
  once per (backbone, category); seeds, protocols and coreset ratios reuse them.
- **Timing in separate processes.** Fit and inference run in two fresh processes per
  configuration so that peak memory at inference is not inflated by the training features.
  Latency is measured for one image at a time, from the preprocessed tensor to score and
  map, with 4 threads; PNG decoding is excluded and stated as such.
- **Tests without downloads.** Unit tests use synthetic features with known answers; the
  end-to-end test uses a randomly initialised ResNet-18 and synthetic parts with a planted
  defect, so CI needs neither the dataset nor the pretrained weights.
