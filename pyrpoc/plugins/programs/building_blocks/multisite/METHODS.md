# Multisite: methods, audit and changes

This is a technical account of the calibrate → survey → loop pipeline in this package. For each stage it covers:
- the methods the stage uses and where they come from;
- the flaws the audit found, with quantitative evidence from a ground-truth simulation;
- what was changed, and why;
- what remains limited.

`README.md` describes the code's structure; this document describes its science.

Each method is marked **[standard]** (taken from the literature, cited), **[adapted]** (a standard method applied in a specific way here) or **[specific]** (a design choice of this package, justified from first principles).

---

## Summary

The pipeline's building blocks were mostly sound and well chosen: masked NCC registration, drift-cancelling calibration, robust fits with prediction intervals, and global stitching. The serious problems were in how the stages fit together, and in what the loop does to the sample, which is where an experiment's conclusions are made.

| Severity | Finding | Change | Effect (simulation) |
|---|---|---|---|
| **High** (experimental design) | Focus checks landed on the units, and always on the same one | Checks made beside the units, on surveyed content clear of every unit | Dose across units: CV 0.39 → 0.00 |
| **High** (experimental design) | Full-field scans exposed, and treated, neighbouring units; no sham control | Per-unit sub-field scans; sham first scan for untreated units | A control unit took the treated dose (511 against 182); now controls take 184–196 and treated units 326–333, the difference being the treatment light |
| **High** (focus) | The loop's focus search could not grow with elapsed time and ignored a search that ended at its edge | Kalman filter on offset and drift rate; search sized by its uncertainty; widening on edge results; started from the survey's measured drift | At 0.3 µm/min with a 20 min gap, mean focus error −4.05 → −0.04 µm (max 6.21 → 0.62 µm) |
| **Medium** (focus) | Survey drift aliased into the plane's y-tilt (rows are visited in time order) | Revisits refocused; plane and drift fitted jointly | Plane error at 0.3 µm/min: max 1.24 → 0.22 µm; drift measured as 0.317 ± 0.018 against a true 0.300 µm/min |
| **Medium** (tracking) | Neighbours entering the template's margin captured it; ambiguous and lost matches were recorded as the unit's intensity | Context match, then outline-masked refinement; peak-ratio test; the monitor records only found units | p95 error 11.2 → 9.9 µm; scans within 2 µm 70.5 → 78.2 %; no measurement taken from an ambiguous match |
| **Medium** (detection) | Noise floor from the whole plane's median and MAD fails in dense cultures | Darkest-blocks start, then upper κσ clipping | At 50 % coverage, 6 % → 61 % of pixels usable for focus |
| **Medium** (calibration) | Precision limited by stage repeatability; moves taken as commanded; rotation of scan against stage never checked | Rounds to a pixel-based precision target; reported positions; off-axis angle measured | σ_landing 1 µm: 3/8 failures → 0/8; with encoders, scale SD 0.7 % → 0.02 % |
| Low | One-pixel offset for reversed scan axes | Raster offset for a reversed axis | 0.59 µm systematic offset removed |

The main open problem is re-identifying units after a long gap between survey and loop (§5.5).

---

## 1. How the methods were evaluated

Every number below comes from a headless harness that drives the stages, unmodified, against the simulated dish (`pyrpoc/utils/simulation`) on a virtual clock. Ground truth is known exactly:
- the scanner's true scale;
- the focal surface at every instant;
- every cell's position;
- every cell's accumulated light dose.

| Setting | Value |
|---|---|
| Field | 150 µm at 256 px (0.586 µm/px), 2 µs dwell |
| True scale | 102.0 µm/V (x), 97.5 µm/V (y); calibration starts from a nominal 100 µm/V |
| Timing | stage 2 mm/s; 0.5 s xy settle, 0.2 s z settle, 0.3 s per-scan overhead (a 36-tile survey then takes about 4.4 min, as on a rig) |
| Sample | cells moving at 1 µm/min (mean speed); focal plane tilted (0.004, −0.003) µm/µm; stage creep (0.05, −0.03) µm/min |
| Focus drift | 0.03 µm/min by default; 0.3 µm/min as a stress case (typical of a freshly mounted, settling dish) |
| Stage landing error | optional Gaussian σ per axis. *Open loop*: the stage reports its target. *Encoder*: it reports where it really is |
| Survey | 750 × 750 µm, 15 % overlap, focus search ±30 µm, 5 µm first step, 0.5 µm final step |
| Loop | 8 isolated cells, each drawn as a 20 µm square on the overview, monitored every 60 s for 30 min, focus check interval 300 s; alternate units "treated" in the treatment case. Tracking statistics are pooled over 2 dish seeds |

**What the simulator does not model.** It does not model:
- a non-planar focal surface;
- non-linear drift;
- realistic defocus blur (its blur grows linearly with |defocus|, so focus curves have a cusp where real ones are smooth);
- cells that deform, divide or exclude one another (here they pass through each other);
- backlash, vignetting, or rotation of the scan against the stage (rotation was injected in the harness).

The results show whether each method behaves correctly under known conditions. They are not predictions of performance on the rig.

---

## 2. Calibration (`stages/calibration/`)

### 2.1 Method

- **The stage as the ruler [adapted].** The sample is moved a known distance by the stage, and the image shift is measured. The slope of shift against move gives µm per volt.
- **Registration [standard].** Whole-frame normalised cross-correlation over the overlap, at every integer offset at once. The cross term is computed by zero-padded FFT, and the overlap's sums and variances from integral images (masked NCC: Padfield 2012; integral-image NCC: Lewis 1995). Correlating whole frames uses every structured pixel as evidence of the one shift. Before correlation each frame is band-passed by a difference of Gaussians: σ = 1 px removes shot noise, and σ = field/16 removes scanner-fixed shading that does not move with the stage. The peak is refined to sub-pixel precision by a three-point parabola (accuracy discussed in §5.10).
- **Prediction-limited search [specific].** Moves grow geometrically (1, 4, 16 µm …), and each is predicted from the scale measured so far, so the shift is only searched for near where it should be. The search radius is capped at half the distance at which the frame's own autocorrelation first reaches 0.5 away from the origin. A periodic sample (a bead lattice) therefore cannot be read one period off.
- **Drift-cancelling out-and-back [standard].** Each move goes out and comes back, and the shift is estimated as (out − back)/2. Any drift linear in time over the three frames cancels, the same principle as the symmetric A–B–A sequences of drift-eliminating comparison designs.
- **Unidirectional approach [standard].** Every stop is approached from the same side, 10 µm short. Backlash is then taken up identically at every stop (unidirectional positioning, ISO 230-2).
- **Robust line fit [standard].** The line is started from medians, points beyond 5 robust σ are rejected (MAD × 1.4826; Hampel 1974; Rousseeuw & Croux 1993), and the rest are fitted by least squares with an intercept. The calibration is accepted if the slope's standard error is under 1 % of the slope.

### 2.2 Flaws found

| # | Flaw | Evidence |
|---|---|---|
| C1 | **Precision was limited by stage repeatability, with no recourse.** The slope's relative standard error is ≈ σ_landing·√(3/2) / √(Σ m²); for the 8 moves of 15–61 µm that is ≈ 0.5 % per µm of landing σ. | σ_landing 0.5 µm → scale SD 0.28 % / 0.24 % (x/y). σ_landing 1 µm → SD 0.71 % / 0.58 %, and **3 of 8 calibrations rejected**. |
| C2 | **Moves were taken as commanded.** A stage with encoders could not contribute its better knowledge of where it went. | With encoder read-back the original still failed 3 of 8 at σ_landing = 1 µm. |
| C3 | **The shift across the axis was measured and discarded.** The whole pipeline assumes the scan axes run along the stage's (`FieldOfView.on_dish` only flips), yet that assumption was never checked. | A 1° rotation passed silently. |
| C4 | **Reversed axes were off by one pixel.** The raster samples pixel *i* at (*i* − *N*/2) pixels from the stage. After `on_dish` reverses an axis running against the stage, the stage sits at pixel *N*/2 − 1, not *N*/2. Every downstream step (tile corners, unit centres, crops) assumes *N*/2. | A systematic 0.59 µm offset on rigs with a negative µm/V. |
| — | (Not a flaw.) A small, consistent bias of +0.04 % ± 0.01 % with all motion and drift switched off. It is at the level of the registration's own sub-pixel error (static synthetic pairs register to ≤ 0.04 px). | Negligible: 0.06 µm across a field. |

### 2.3 Changes

1. **Rounds of moves to a precision target [adapted].** The round of 8 moves is repeated, up to 4 times, until the slope's standard error satisfies σ_s/s ≤ EDGE_PX / (COVERAGE·N/2), with EDGE_PX = 0.5 px and COVERAGE = 2. For N = 256 that is 0.20 %: the field's edge is then known to half a pixel at two standard errors. A scale known better than the image can resolve is not needed; one known worse shows as seams and misplaced units. This is fixed-precision sequential estimation (Anscombe 1953; Chow & Robbins 1965). The stopping rule depends only on precision, not on the estimate, so it does not bias the estimate. The hard acceptance limit stays at 1 %.
2. **Moves measured from the stage's reported positions.** Each move is taken as reported(there) − ½[reported(start) + reported(back)], mirroring the out-and-back shift estimator. On an open-loop stage this equals the commanded move (no loss); with encoders, landing error vanishes from the regression.
3. **Cross-axis component measured and reported.** The cross shifts are fitted against the moves over the same kept points. The stage's own reported sideways stray is removed first, using the other axis's measured scale, because that is landing, not angle. This gives each axis's angle off the stage, θ, and the misplacement at the field's edge, tan θ·N/2 px. Both are logged as a new `AxisPrecision` event, together with the scale's uncertainty and the number of rounds taken.
4. **`FieldOfView.scan` shifts a reversed axis's raster by one pixel**, so the stage sits at pixel N/2 on every axis.

### 2.4 Results (8 seeds each)

| Condition | Version | Failures | Scale SD (x / y) | Scans | Angle read (true) |
|---|---|---|---|---|---|
| No landing error | original | 0/8 | 0.03 % / 0.03 % | 46 | — |
| | new | 0/8 | 0.03 % / 0.03 % | 46 | ±0.01° (0°) |
| σ_landing 0.5 µm, open loop | original | 0/8 | 0.28 % / 0.24 % | 45 | — |
| | new | 0/8 | **0.16 % / 0.21 %** | 141 (4 rounds) | ±0.1° (0°) |
| σ_landing 1 µm, open loop | original | **3/8** | 0.71 % / 0.58 % | 45 | — |
| | new | **0/8** | 0.32 % / 0.35 % | 141 (4 rounds) | ±0.2° (0°) |
| σ_landing 1 µm, encoder | original | 3/8 | 0.71 % / 0.57 % | 45 | — |
| | new | 0/8 | **0.02 % / 0.04 %** | 44 (1 round) | ±0.01° (0°) |
| Scan rotated 1° to stage | original | 0/8 | 0.05 % / 0.03 % | 46 | not measured |
| | new | 0/8 | 0.05 % / 0.03 % | 46 | **+1.03° / −0.94°** (±1°) |

The opposite signs on x and y for a pure rotation are expected: each axis's angle is measured against its own direction.

---

## 3. Survey (`stages/survey/`)

### 3.1 Method

- **Boustrophedon tiling [standard]** with overlap, so that the stage never runs back.
- **Content detection [adapted].** The plane is smoothed at feature scale (σ = 1.5 px), and pixels standing more than 5 noise σ above background are counted. A tile has content if at least 25 px do. Five σ is the usual detection bar.
- **Focus metric [standard].** Tenengrad, the sum of the squared Sobel gradient (Tenenbaum 1970; Krotkov 1988), computed per channel over the dilated content mask. Tenengrad is among the best-performing metrics in comparative studies (Groen et al. 1985; Sun et al. 2004). Computing it per channel avoids cross-terms between detectors.
- **Focus search [adapted].** A coarse sweep across the range nearest-first, then a hill climb with step halving down to the final step.
- **Focus prediction [standard].** A robust plane fit is maintained: least squares, refitted after 3σ MAD rejection, with tilt fitted only along directions the points actually span. The next tile's search spans the least-squares prediction interval, t₀.₉₉₈₇,ν · s · √(1 + x₀ᵀ(XᵀX)⁻¹x₀) (Draper & Smith 1998; Montgomery et al. 2012). The spread is floored at step/√12, the deviation of uniform rounding to the final step. The hand-written Student-t quantile was checked against tables (2.5706 at p = 0.975, ν = 5; 3.1693 at p = 0.995, ν = 10).
- **Lost-focus recovery [specific].** The search widens to the full range when it ends at its own edge, or when an empty tile lies beside tiles that held content. Without the second rule, a survey that lost focus would see only empty tiles, which are never focused, and would never recover.
- **Motion [adapted].** The 6 earliest tiles with content are revisited. Their clear features (Otsu 1979 threshold, then 5σ) are found again by NCC within 20 % of a field, and speed is taken as displacement over time, summarised by median and MAD.
- **Stitching [standard + adapted].** Neighbouring tiles are registered pairwise by masked NCC (Padfield 2012), only near the stage-predicted offset, so a repeated pattern cannot pull a tile far. Each pair is weighted by its NCC score and by inverse variance, with registration precision and possible content motion between the two tiles added in quadrature. All positions are then solved together by least squares, held weakly to the stage's positions, so a tile with no confident neighbour stays where the stage put it (global optimisation as in Preibisch et al. 2009; a stage model as in Chalfoun et al. 2017). Tiles are blended with linear feathering (Szeliski 2006).

### 3.2 Flaws found

| # | Flaw | Evidence |
|---|---|---|
| S1 | **Focus drift was confounded with tilt.** In boustrophedon order, time is close to a linear function of row, and so of y. A plane z = a + bx + cy fitted to tiles taken over T minutes absorbs a drift rate r into c, by ≈ r·T/H. Within the walk this is harmless (the next tile is near in both y and t), but the plane is handed to the loop as if it held for the whole region at one time. The error is a spatial pattern that the loop's single offset cannot remove. | At 0.3 µm/min: fitted tilt_y −0.00113 against a true −0.0030; plane error at hand-off up to **1.24 µm** (SD 0.50 µm). |
| S2 | **The noise floor failed in dense samples.** Median and MAD over the whole plane assume at least half of it is background. In a dense culture they measure the cells: σ is overestimated, and the content mask used to judge focus collapses. | At 50 % coverage, 6 % of pixels clear against 61 % with a correct floor. At 82 %, σ is inflated 20× and 0.1 % of pixels are clear. Even at 19 % coverage, σ is inflated 1.75× (0.0091 against a true 0.0052). |
| S3 | **Revisits were imaged at their old z.** With drift, the second frame used for motion was out of focus. | (Folded into S1's fix.) |

### 3.3 Changes

1. **Revisits are refocused, and drift is fitted jointly with the plane [adapted].** A tile focused twice, minutes apart, measures drift independently of tilt; this is the standard way to break a confound, by adding observations in which the confounded factors vary independently. Each revisit's search starts from the tile's first focus, moved on by the drift the previous revisits measured, so revisits rarely need the full sweep. The final model is fitted robustly:

   z = z₀ + b·x + c·y + r·(t − t_ref)

   Here t_ref is the time the last tile was taken. The plane is reported as it stood at t_ref; its covariance at t_ref is the spatial block of (XᵀX)⁻¹, so the drift's uncertainty is included. The rate and its standard error are logged in a new `FocusDriftMeasured` event and stored in the new `SurveyFinding.drift` field, which the loop's focus filter starts from (§4.3.2). During the walk the plane stays spatial only, as before.
2. **Background floor from the darkest blocks, refined by upper κσ clipping [adapted].** The first estimate is the median and MAD of the darkest quarter of 16-px blocks. It is then refined five times as median and MAD over every pixel no more than 3σ above the level. Clipping only from above keeps all of the background's lower half and never counts signal. This is the κσ-clipping used for sky backgrounds in crowded astronomical fields (Bertin & Arnouts 1996). Selecting blocks by darkness rather than by low σ matters: with only about 9 independent samples per smoothed block, choosing the lowest-σ blocks biases σ down by 2×.

### 3.4 Results

**Focus plane handed to the loop** (error against the true focal surface at t_ref, 5 × 5 grid over the region, 2 seeds):

| Drift | Version | tilt_y (true −0.0030) | Plane error max / SD | Drift measured (true) | Survey scans |
|---|---|---|---|---|---|
| 0.03 µm/min | original | −0.0030 / −0.0028 | 0.19 / 0.10 µm; 0.23 / 0.12 µm | — | 372 |
| | new | −0.0031 / −0.0030 | 0.16 / 0.08 µm | 0.017 ± 0.020; 0.027 ± 0.020 (0.030) | 414 (+11 %) |
| 0.3 µm/min | original | **−0.00113** | **1.24 / 0.50 µm** | — | 375 |
| | new | −0.0027 | **0.22 / 0.09 µm** | 0.317 ± 0.018 (0.300) | 437 (+17 %) |

**Noise floor** (fraction of pixels clear of background, by coverage; channel 0 / channel 1):

| Coverage | Original | New |
|---|---|---|
| 19 % / 5 % | 0.23 / 0.09 | 0.25 / 0.09 |
| 50 % / 16 % | **0.06** / 0.23 | **0.61** / 0.27 |
| 82 % / 35 % | 0.001 / **0.18** | 0.001 / **0.54** |
| 96 % / 54 % | 0.001 / **0.002** | 0.001 / **0.68** |
| Pure noise (Poisson, Gaussian, correlated): frames called content | 0 / 40 each | 0 / 40 each |

Beyond about 80 % coverage no background remains for any estimator to find (§5.9).

---

## 4. Loop (`stages/loop/`, `routines/`)

### 4.1 Method

- Every (unit, routine) pair is a thread, and the queue runs the earliest-due step first.
- Each scan, the stage goes to the unit's last known centre and sets z from the focus model. The unit is found again by template matching against its image from the last scan (NCC), then its template is updated, keeping the sub-pixel crop offset so rounding never accumulates.
- The routine is given the unit's window and outline mask. The monitor routine records the mean intensity inside the outline.
- A loop cut short is replayed from its event log.

### 4.2 Flaws found

| # | Flaw | Evidence (original code) |
|---|---|---|
| L1 | **Focus checks were made on the units.** A check (about 10 scans) ran on whichever unit's scan was due when the check timer expired. With a steady period that is the same unit every time. Light dose drives bleaching and phototoxicity (Icha et al. 2017; Laissue et al. 2017), so for a longitudinal intensity measurement this is a systematic, unit-specific confound. | **7/7 checks on unit 1**, which took **2.4×** the dose of the others (429 against 182 dose units). |
| L2 | **Each unit was scanned with the whole 150 µm field.** Every cell within 75 µm was exposed, and a treatment line, driven over the whole scan, treated every cell in that field. | Units 7 and 8, closer than a field, took 2× dose (364). In the treatment case **control unit 8 took the treated dose** (511, against 182 for other controls). |
| L3 | **No sham control.** Only treated units got the first scan, which is an imaging exposure as well as the treatment, so the treated–control comparison confounded treatment with an extra imaging scan. Light-matched controls are the standard remedy (Owen et al. 2019 show that light and heat alone have effects). | Treated units took one more imaging scan than controls. |
| L4 | **The focus keeper could not follow drift.** Its search span was fixed (max(3·residual, 4·final step), about 2 µm whatever the time since the last check). A check ending at the edge of its span was taken as a measurement anyway. The offset was set to the latest check alone, with no prediction between checks. | At 0.3 µm/min with a 20 min gap before the loop: focus error mean **−4.0 µm**, max **6.2 µm**, and tracking collapsed (p95 65 µm). Without the gap: mean −0.69 µm, max 1.91 µm. |
| L5 | **The template was captured by neighbours.** The template was the unit's bounds plus a 5 µm margin. Neighbours drifting into the margin move independently and pull the match toward a compromise, a little more with every update: the template update problem (Matthews, Ishikawa & Baker 2004). Error grew at roughly the cell speed while `found` stayed true. | p95 tracking error 11.2 µm with no loss reported (pooled). |
| L6 | **Lost-unit measurements were recorded as real ones.** When the unit was not found, the routine was given the image at the last position, and the monitor recorded it as the unit's intensity. | All scans recorded. |
| L7 | **Matches were never tested for ambiguity.** After a long interval the search radius covers several look-alike cells, and the best one was taken however close the runner-up was. | 20 min gap: 28 % of scans within 2 µm of the right cell. |

### 4.3 Changes

#### 4.3.1 Light goes only where the experiment puts it [specific]

- **Focus checks beside the units.** Candidate check fields are centred whole fields away from the unit due (out to two fields, nearest first). A candidate qualifies if it overlaps no present unit's bounds, grown by a margin and by the tracker's motion radius for that unit (where it may now be). It must also lie within the survey's overview and show content there. Only if none qualifies is focus checked on the unit itself, and that fallback is logged (`FocusingAtUnit`); otherwise the site is logged (new `FocusingBeside` event). Focus found a field away is carried to the unit by the survey's plane.
- **Each unit is scanned over a sub-field**: side = window + 2·(motion radius) + 2 px, rounded up to a multiple of 32 px (so a rig builds few rasters) and capped at the field. The pixel size stays the survey's.
- **Sham scans.** With a treatment, every unit's first scan is the same scan, with the line driven only for the treated group, so the groups differ only in the treatment.

#### 4.3.2 Focus as a tracked, drifting state [standard + specific]

The offset of focus from the survey's plane is estimated by a Kalman filter (Kalman 1960). Its state is the offset and its rate, under the nearly-constant-velocity model with continuous white-noise acceleration (Bar-Shalom, Li & Kirubarajan 2001, ch. 6):

F = [[1, Δ], [0, 1]],  Q = q·[[Δ³/3, Δ²/2], [Δ²/2, Δ]],  q = σ_r²/τ

The parts of the filter:
- **Prior on the rate.** σ_r = 0.3 µm/min and τ = 30 min: a stated physical prior that a settling dish drifts at up to about that, and its rate changes by as much over half an hour.
- **Measurement noise.** R = the survey's residual², which covers both the search's own error and how far the dish departs from a plane between a check site and a unit.
- **Start.** The filter starts at zero offset at t_ref, with the survey's measured drift ± its standard error as the initial rate. A gap between survey and loop is then predicted rather than discovered.
- **Use between checks.** Each scan uses the predicted offset at its own time, so steady drift is followed between checks.
- **Search span.** Each check searches 3·√(P_oo + R) either side of the prediction, at least 4 final steps and at most the full range. It sweeps then climbs when the span is wider than a first step, and widens to the full range if the result lands on its edge. An edge result over the full range is a bound, not a measurement: it is not used, and is logged as `FocusNotFound`.
- **When a check is due [specific].** At the user's interval, or sooner whenever √P_oo exceeds the final step, the precision the user asked for. Early on, before the drift is known, checks come faster.
- **Jumps.** A check whose normalised innovation squared exceeds χ²₁(0.999) = 10.83 means focus jumped (the dish was knocked), and the filter is re-initialised from it (the NIS consistency test, Bar-Shalom et al. 2001).
- **Resume.** The filter is deterministic in its checks, so a resumed loop replays every `FocusChecked` event and reconstructs its exact state. This was verified in the harness: the replayed state matched the cut-short run's exactly.

The loop now takes the same focus search range and first step as the survey.

#### 4.3.3 Tracking that knows when it does not know [adapted]

A unit is found in three steps:
1. **Choose.** The whole window is matched within the motion radius. The neighbourhood is what tells a unit from look-alikes, which matters most when the radius is wide.
2. **Refine.** The match is refined on the unit's outline alone (outline dilated by an edge's blur, 1 µm), within the margin, using masked NCC. Neighbours moving independently can no longer drag the template.
3. **Test.** Each match must pass a **peak-ratio test**: the best correlation must reach 1.5× the best outside its own peak (the region joined to it above half its height). This is the detectability criterion of correlation velocimetry (Keane & Adrian 1992). A match that fails is *not found*: the unit stays where it was, and its search radius grows with time until a distinctive match returns.

The survey's motion measurement does not apply the test, deliberately. There each match is one of many speeds summarised by a median and MAD, which tolerate occasional mismatches; requiring distinct matches left only 7–8 features (§5.4).

`Seen` now carries whether the unit was found, and the monitor records intensity only from scans that found it.

### 4.4 Results

**Dose per unit** (dose units from the simulator; 8 units, 30 min):

| Case | Original | New |
|---|---|---|
| No treatment | 429, 182 × 5, 364, 364 (CV 0.39) | **178 × 8 (CV 0.00)** |
| Treatment, treated units (1, 3, 5, 7) | 575, 329, 329, 511 | 327, 326, 327, 333 |
| Treatment, control units (2, 4, 6, 8) | 182, 182, 182, **511** | **184, 184, 190, 196** |

With the change the groups differ by the treatment light (≈ 143 units), and the sham accounts for the controls' +6 over the untreated case. The remaining 6–12 units on two units come from scans that did not find them, whose search, and so whose sub-field, is wider.

**Focus error at each scan** (z used against the true focal surface; 8 units, 240 scans):

| Case | Original: mean / SD / max | New: mean / SD / max |
|---|---|---|
| Default (0.03 µm/min) | −0.08 / 0.16 / 0.42 µm | −0.05 / 0.21 / 0.57 µm |
| 0.3 µm/min | −0.69 / 0.53 / 1.91 µm | **0.10 / 0.07 / 0.23 µm** |
| 0.3 µm/min, 20 min gap | **−4.05 / 0.98 / 6.21 µm** | **−0.04 / 0.28 / 0.62 µm** |

With the gap, the new keeper holds focus within 0.62 µm even though every unit is lost; in the original, focus loss and tracking collapse fed each other. Under the default drift both are well inside a typical depth of field. The new keeper's slightly larger SD there comes from a measurement quantised to the 0.5 µm final step, which it trusts according to the survey's residual (§5.6).

**Tracking error** (tracked centre against the true cell; 16 drawn units, 2 seeds, pooled):

| Case | Version | Median | p90 | p95 | Within 2 µm | Scans reported not found |
|---|---|---|---|---|---|---|
| Default | original | 0.67 µm | 7.9 µm | 11.2 µm | 70.5 % | 0 |
| | new | 0.48 µm | 5.8 µm | 9.9 µm | 78.2 % | 4 of 450 |
| 20 min gap | original | 18.0 µm | 64.3 µm | 69.8 µm | 28.3 % | 0 |
| | new | (not found) | | | 0.0 % | **450 of 450** |

**The 20 min gap case is a trade-off, not a win.** The original recorded every scan; 28 % came from the right cell and the rest, recorded as the unit's intensity, from look-alikes. The new tracker finds no match distinct enough in any scan, so the monitor records nothing for any unit. That is the defensible outcome, since no wrong cell is measured as the unit, but no data is collected either. And because a lost unit's possible position keeps growing, focus checks find no clear site and fall back onto the units (6 of 7 checks), and their sub-fields widen to the full field, so the dose equality of §4.3.1 is lost too. §5.5 gives what a real fix needs. If missing data is judged worse than mixed data, `PEAK_RATIO` in `analysis/shifts.py` is the single setting that trades one for the other.

---

## 5. Limitations that remain

These were examined and left unchanged. Each gives the reason, and what a fix would involve.

1. **Stitching corrects tile positions, not tile contents, for a scale error.** `axis_scale` rescales registered offsets to "stage pixels", so the mosaic's geometry follows the stage. The tiles' contents keep the calibration's scale error, so seams can mismatch by up to ε·N px. At the new calibration target that is under 0.5 px; a full fix is an affine stitching model that resamples tiles.
2. **The overview is composited at whole pixels**, and without flat-field correction. Placement error is up to 0.5 px. Vignetting and seams show in the overview and in templates cut from it; retrospective shading correction (BaSiC, Peng et al. 2017; CIDRE, Smith et al. 2015) would remove them.
3. **Rotation is measured, not corrected.** The pipeline remains translation-only. A rotation θ misplaces a field's edge by tan θ·N/2 px, now reported at calibration. Correcting it needs a rotated raster or resampled frames.
4. **The motion model is ballistic.** Speeds come from displacement over the survey's revisit interval (minutes) and are extrapolated linearly to the loop's intervals. Cells move as persistent random walks (Gail & Boone 1970; Dunn 1983; Wu et al. 2014), diffusive beyond their persistence time, so displacement over a short interval can exceed this extrapolation. Two further caveats:
   - "Median + 2 MAD" of a Rayleigh-like speed distribution is about its 96th percentile, not a stated quantile.
   - The revisit displacement includes the stage's return error, and the 30 µm feature search admits look-alike neighbours.

   Against the true speeds of the cells in the revisited tiles (3 seeds; true median 0.91–0.99, spread 0.41–0.48 µm/min), the survey reports a median 12–15 % high and a spread 50–65 % high, from 66–95 features. The tracker's radius is therefore conservative by about 30–45 %. Requiring distinct matches removes the mismatches but leaves 7–8 features, whose estimate was 38 % low on one seed, so it was not adopted.

   The principled upgrade is to learn the displacement distribution at the loop's own intervals from its own tracking.
5. **Re-identifying units after a long gap is unsolved: the main open problem.** If cells move a large fraction of their spacing between the survey and the loop's first scan, appearance alone cannot say which cell a unit was.
   - In the 20 min gap case the search radius (77 µm) holds about 20 look-alike cells.
   - The original took the best match regardless, and its 28 % of scans within 2 µm were partly chance; the rest followed the wrong cell and were recorded as the unit.
   - The new tracker refuses all of those matches as ambiguous (450 of 450 scans not found), so it records nothing rather than the wrong cell, but it does not recover the right one either.

   Twenty minutes is realistic: a large survey plus the time to draw units. A principled solution would combine:
   - a motion prior at the relevant interval, learned from the loop's own tracking (item 4);
   - joint assignment of all units to candidate cells, each cell taken by at most one unit (the assignment problem, Kuhn 1955; standard in multi-target tracking, Bar-Shalom et al. 2001);
   - re-imaging the units' neighbourhoods just before the loop starts.

   Until then: start the loop promptly after the survey, or redraw units on a fresh overview.
6. **Focus measurement.**
   - Tenengrad is not normalised for intensity, so bleaching during a search lowers later probes and biases toward its start. A normalised metric (cf. normalised variance, Groen et al. 1985) would remove a multiplicative effect.
   - The final z is quantised to the climb's last step (between 1 and 2 final steps). Peak interpolation from the samples already taken would refine it at no cost in scans. It was not added because the simulator's cusp-shaped focus curve cannot validate it.
7. **The Kalman priors σ_r and τ are stated, not measured.** The survey's drift estimate replaces σ_r when there is one.
8. **Scheduling is fixed-delay, and units are served in join order when due together.** Each period starts at the end of the scan before, and scan times are systematically ordered by unit. Timestamps are recorded, so this can be analysed, but group assignment and drawing order should be randomised (Fisher 1935; Montgomery 2017) so that group is not confounded with time of scan.
9. **Content detection fails beyond about 80 % coverage.** No background remains for any estimator, and focus is then judged on few pixels.
10. **Sub-pixel conventions.**
    - The three-point parabolic peak has a small systematic bias that depends on the sub-pixel position ("peak locking", Westerweel 1997). It measured ≤ 0.04 px here, well below every other error.
    - The raster samples pixel *i* at (*i* − *N*/2) px from the stage (pixel corners), while crops treat coordinates as pixel spans: a consistent offset of ≤ 0.5 px.
11. **Residual exposure.**
    - The treatment line gates a unit's whole sub-field rather than its outline; per-pixel gating, as RPOC does elsewhere, would confine it further.
    - Units closer than a sub-field still expose each other.
    - The survey exposes tiles unequally: content tiles take about 10 focusing scans, and revisited tiles more.

---

## 6. Compatibility

- **Persisted data.**
  - No existing persisted field was removed or changed.
  - New event kinds: `FocusingBeside`, `FocusNotFound` (loop); `FocusDriftMeasured` (survey); `AxisPrecision` (calibration). Older logs simply do not contain them.
  - One new field, `SurveyFinding.drift`. Surveys saved before it load with `None`, and the loop then starts its filter with the rate unknown.
- **Inputs.** Loop programs gain Search Range and Initial Step, the same inputs the survey has.
- **Resuming.** A loop cut short under the previous version resumes: its `FocusChecked` events replay into the filter.

---

## References

- Anscombe, F. J. (1953). Sequential estimation. *J. R. Stat. Soc. B* 15(1), 1–29.
- Bar-Shalom, Y., Li, X. R. & Kirubarajan, T. (2001). *Estimation with Applications to Tracking and Navigation*. Wiley.
- Bertin, E. & Arnouts, S. (1996). SExtractor: software for source extraction. *Astron. Astrophys. Suppl. Ser.* 117, 393–404.
- Chalfoun, J., Majurski, M., Blattner, T., Bhadriraju, K., Keyrouz, W., Bajcsy, P. & Brady, M. (2017). MIST: accurate and scalable microscopy image stitching tool with stage modeling and error minimization. *Sci. Rep.* 7, 4988.
- Chow, Y. S. & Robbins, H. (1965). On the asymptotic theory of fixed-width sequential confidence intervals for the mean. *Ann. Math. Stat.* 36(2), 457–462.
- Draper, N. R. & Smith, H. (1998). *Applied Regression Analysis*, 3rd ed. Wiley.
- Dunn, G. A. (1983). Characterising a kinesis response: time averaged measures of cell speed and directional persistence. *Agents Actions Suppl.* 12, 14–33.
- Fisher, R. A. (1935). *The Design of Experiments*. Oliver & Boyd.
- Gail, M. H. & Boone, C. W. (1970). The locomotion of mouse fibroblasts in tissue culture. *Biophys. J.* 10(10), 980–993.
- Groen, F. C. A., Young, I. T. & Ligthart, G. (1985). A comparison of different focus functions for use in autofocus algorithms. *Cytometry* 6(2), 81–91.
- Hampel, F. R. (1974). The influence curve and its role in robust estimation. *J. Am. Stat. Assoc.* 69(346), 383–393.
- Icha, J., Weber, M., Waters, J. C. & Norden, C. (2017). Phototoxicity in live fluorescence microscopy, and how to avoid it. *BioEssays* 39(8), 1700003.
- ISO 230-2:2014. *Test code for machine tools — Part 2: Determination of accuracy and repeatability of positioning of numerically controlled axes.*
- Kalman, R. E. (1960). A new approach to linear filtering and prediction problems. *J. Basic Eng.* 82(1), 35–45.
- Keane, R. D. & Adrian, R. J. (1992). Theory of cross-correlation analysis of PIV images. *Appl. Sci. Res.* 49(3), 191–215.
- Krotkov, E. (1988). Focusing. *Int. J. Comput. Vis.* 1(3), 223–237.
- Kuhn, H. W. (1955). The Hungarian method for the assignment problem. *Nav. Res. Logist. Q.* 2(1–2), 83–97.
- Laissue, P. P., Alghamdi, R. A., Tomancak, P., Reynaud, E. G. & Shroff, H. (2017). Assessing phototoxicity in live fluorescence imaging. *Nat. Methods* 14(7), 657–661.
- Lewis, J. P. (1995). Fast normalized cross-correlation. *Vision Interface* 95, 120–123.
- Matthews, I., Ishikawa, T. & Baker, S. (2004). The template update problem. *IEEE Trans. Pattern Anal. Mach. Intell.* 26(6), 810–815.
- Montgomery, D. C. (2017). *Design and Analysis of Experiments*, 9th ed. Wiley.
- Montgomery, D. C., Peck, E. A. & Vining, G. G. (2012). *Introduction to Linear Regression Analysis*, 5th ed. Wiley.
- Otsu, N. (1979). A threshold selection method from gray-level histograms. *IEEE Trans. Syst. Man Cybern.* 9(1), 62–66.
- Owen, S. F., Liu, M. H. & Kreitzer, A. C. (2019). Thermal constraints on in vivo optogenetic manipulations. *Nat. Neurosci.* 22(7), 1061–1065.
- Padfield, D. (2012). Masked object registration in the Fourier domain. *IEEE Trans. Image Process.* 21(5), 2706–2718.
- Peng, T., Thorn, K., Schroeder, T., Wang, L., Theis, F. J., Marr, C. & Navab, N. (2017). A BaSiC tool for background and shading correction of optical microscopy images. *Nat. Commun.* 8, 14836.
- Preibisch, S., Saalfeld, S. & Tomancak, P. (2009). Globally optimal stitching of tiled 3D microscopic image acquisitions. *Bioinformatics* 25(11), 1463–1465.
- Rousseeuw, P. J. & Croux, C. (1993). Alternatives to the median absolute deviation. *J. Am. Stat. Assoc.* 88(424), 1273–1283.
- Smith, K., Li, Y., Piccinini, F., Csucs, G., Balazs, C., Bevilacqua, A. & Horvath, P. (2015). CIDRE: an illumination-correction method for optical microscopy. *Nat. Methods* 12(5), 404–406.
- Sun, Y., Duthaler, S. & Nelson, B. J. (2004). Autofocusing in computer microscopy: selecting the optimal focus algorithm. *Microsc. Res. Tech.* 65(3), 139–149.
- Szeliski, R. (2006). Image alignment and stitching: a tutorial. *Found. Trends Comput. Graph. Vis.* 2(1), 1–104.
- Tenenbaum, J. M. (1970). *Accommodation in Computer Vision*. PhD thesis, Stanford University.
- Westerweel, J. (1997). Fundamentals of digital particle image velocimetry. *Meas. Sci. Technol.* 8(12), 1379–1392.
- Wu, P.-H., Giri, A., Sun, S. X. & Wirtz, D. (2014). Three-dimensional cell migration does not follow a random walk. *Proc. Natl. Acad. Sci. USA* 111(11), 3949–3954.
