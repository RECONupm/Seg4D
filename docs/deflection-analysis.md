# Deflection Analysis

The tool measures a beam's geometric deviation from its end chord. It does not
infer displacement since construction or an earlier scan. Select a longitudinal
surface/profile for each beam, with the Z axis vertical and consistent length
units. Selecting one cloud, several clouds, or a folder is supported.

## Inclined beams and direction

- **Vertical** reports the difference in elevation between the profile and its
  chord at the same horizontal longitudinal coordinate. This remains the default.
- **Perpendicular** reports signed distance normal to the chord in the vertical
  longitudinal plane. For a chord inclined by theta, this equals the vertical
  difference times cos(theta). It does not include lateral bending.

A straight inclined beam yields zero deflection in both Data and Fit. Exactly
vertical beams, clouds without a distinct longitudinal axis, and sections with
no useful horizontal span are rejected rather than assigned a spurious result.
The longitudinal axis is estimated in 3D and projected into XY; no ordering of
input points or shared minimum-bounding-rectangle routine is required.

## Data and Fit

**Data** measures the maximum absolute deviation of the observed slice. Each
support elevation is estimated by a robust local linear trend over the selected
end region, evaluated at the actual endpoint. This replaces the incorrect use of
an end-region median height as the height at its boundary, which introduced a
false deflection solely from beam inclination.

The end-region percentage remains an estimation parameter. A finite local linear
window approximates a curved/noisy support region; changing its width can change
the inferred support elevation. Data also retains actual point noise and isolated
profile defects. It is not an automatic reconstruction of a beam's neutral axis.

**Fit** uses a scaled polynomial fit and its exact endpoint elevations. The maximum
is found at the endpoints and real interior roots of the derivative of
`polynomial - chord`, rather than on a plot sampling grid. Polynomial degree must
be lower than the number of distinct longitudinal coordinates. If Fit cannot be
computed, Data can still complete and the fit warning is recorded.

Each mode uses **its own chord length** for L/deflection and the user-entered L/n
limit. Maximum-location distances are orthogonal projections onto that chord,
measured from its first endpoint. The plot's longitudinal/elevation coordinates
are relative to a recorded local origin to avoid conditioning errors on large
survey coordinates. CloudCompare global scale is removed for measurements;
clones and extracted points retain the original coordinates and shift/scale.

## Outputs and checks

Each run creates a separate `deflection_YYYYMMDD_HHMMSS_ffffff` directory under
the selected output directory. It contains plots, a tab-separated numerical
report and a per-beam error/warning report. Numbers are saved at full precision.
Each plot includes the profile and a separate signed-deflection panel with the
inclination removed, so a small sag remains visible on a sloping beam.
The numerical report records the direction, selected mode, both chord lengths,
limit, local origin, axis and parameters. Plot failures are reported without
discarding a valid calculation or its scalar fields.
The original twelve report columns retain their names and order; additional
context is appended after them.

Original clouds and their attributes are preserved. A clone receives the selected
result in `Maximum deflection`, `Relative_deflection`, and the historical
`Is deflected` field. The latter means **exceeds the configured L/n limit**;
it is not a structural safety assessment. Existing result fields in the clone
are replaced using the supported scalar-field API. Unused attributes containing
NaN do not remove geometry from the calculation.

For zero deflection, L/deflection is infinite in the report. CloudCompare stores
the corresponding scalar ratio as undefined (NaN), while maximum deflection and
the exceedance flag remain zero.

Regression tests cover straight/inclined profiles, known sag, rotation, survey
offsets, robust support estimates, polynomial extrema, steep beams, invalid
inputs, global scale, metadata, scalar fields and report consistency:

```powershell
python -B -m unittest discover -s tests -v
```

The CloudCompare adapter tests use an in-memory host double. They do not replace
testing the actual embedded Python plugin and its window inside CloudCompare.
