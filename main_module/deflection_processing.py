"""Beam profiles and chord deflection, independent of CloudCompare and Tk.

Coordinates must share length units and Z must be vertical. A profile describes
geometry relative to its end chord, not displacement from an earlier survey.
"""

import warnings

import numpy as np


DEFLECTION_DIRECTIONS = ("Vertical", "Perpendicular")


def _direction(value):
    for name in DEFLECTION_DIRECTIONS:
        if str(value).strip().lower() == name.lower():
            return name
    raise ValueError("Deflection direction must be Vertical or Perpendicular")


def _profile(x, z):
    x = np.asarray(x, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    if x.ndim != 1 or z.ndim != 1 or x.shape != z.shape:
        raise ValueError("Longitudinal coordinates and elevations must be matching 1D arrays")
    finite = np.isfinite(x) & np.isfinite(z)
    x, z = x[finite], z[finite]
    if x.size < 2 or np.ptp(x) <= 0:
        raise ValueError("At least two finite points with different longitudinal coordinates are required")
    return x, z


def _end_elevation(x, z, endpoint):
    """Robust local linear trend evaluated at the actual end, not its median X.

    Local centering/scaling avoids coordinate-offset conditioning. Huber
    reweighting limits the effect of isolated support-region outliers.
    """
    width = np.ptp(x)
    if width == 0:
        return float(np.median(z))
    t = (x - endpoint) / width
    center_z = np.median(z)
    design = np.column_stack((np.ones(len(x)), t))
    values = z - center_z
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    floor = 64 * np.finfo(float).eps * max(1.0, float(np.ptp(z)))
    for _ in range(12):
        residual = values - design @ coefficients
        median = np.median(residual)
        scale = 1.4826 * np.median(np.abs(residual - median))
        if scale <= floor:
            coefficients[0] += median
            break
        weights = np.sqrt(np.minimum(1.0, 1.5 * scale / np.maximum(np.abs(residual), floor)))
        updated = np.linalg.lstsq(design * weights[:, None], values * weights, rcond=None)[0]
        if np.max(np.abs(updated - coefficients)) <= floor:
            coefficients = updated
            break
        coefficients = updated
    return float(center_z + coefficients[0])


def _relative_to_chord(x, z, x_left, x_right, z_left, z_right, direction):
    direction = _direction(direction)
    span = float(x_right - x_left)
    rise = float(z_right - z_left)
    length = float(np.hypot(span, rise))
    if not np.isfinite(length) or length <= 0:
        raise ValueError("The beam chord must have a finite positive length")
    slope = rise / span
    reference = z_left + rise * ((x - x_left) / span)
    vertical = (z - z_left) - rise * ((x - x_left) / span)
    # Suppress only round-off relative to the profile's length/elevation scale.
    epsilon = 64 * np.finfo(float).eps * max(1.0, length, float(np.ptp(z)))
    vertical[np.abs(vertical) <= epsilon] = 0.0
    signed = vertical if direction == "Vertical" else vertical * span / length
    maximum_index = int(np.argmax(np.abs(signed)))
    stations = ((x - x_left) * span + (z - z_left) * rise) / length
    maximum = float(np.abs(signed[maximum_index]))
    return {
        "direction": direction,
        "span": span,
        "length": length,
        "slope": slope,
        "reference": reference,
        "signed_vertical_deflection": vertical,
        "signed_deflection": signed,
        "maximum": maximum,
        "relative": np.inf if maximum == 0.0 else length / maximum,
        "x_maximum": float(x[maximum_index]),
        "z_maximum": float(z[maximum_index]),
        "maximum_station": float(stations[maximum_index]),
        "x_ends": np.array([x_left, x_right]),
        "z_ends": np.array([z_left, z_right]),
    }


def chord_deflection(x, z, endpoint_fraction=0.05, direction="Vertical"):
    """Measure vertical or chord-normal deflection of an observed profile.

    The support-region trend is extrapolated to each true end. In particular,
    a straight inclined profile has zero deflection, for every support width.
    With fraction zero, repeated endpoint heights are reduced by their median.
    """
    if not np.isfinite(endpoint_fraction) or not 0 <= endpoint_fraction <= 0.5:
        raise ValueError("The support-region fraction must be between 0 and 0.5")
    x, z = _profile(x, z)
    left, right = float(x.min()), float(x.max())
    width = endpoint_fraction * (right - left)
    left_mask, right_mask = x <= left + width, x >= right - width
    z_left = _end_elevation(x[left_mask], z[left_mask], left)
    z_right = _end_elevation(x[right_mask], z[right_mask], right)
    return _relative_to_chord(x, z, left, right, z_left, z_right, direction)


def polynomial_chord_deflection(curve, x_left, x_right, direction="Vertical"):
    """Find the continuous polynomial maximum, including both boundaries."""
    if not np.isfinite([x_left, x_right]).all() or x_right <= x_left:
        raise ValueError("The fitted profile must have a finite positive span")
    if not np.isfinite(curve.coef).all():
        raise ValueError("The fitted curve contains non-finite coefficients")
    left_z, right_z = float(curve(x_left)), float(curve(x_right))
    slope = (right_z - left_z) / (x_right - x_left)
    roots = (curve.deriv() - slope).roots()
    roots = roots[np.isclose(roots.imag, 0.0, atol=1e-10, rtol=0)].real
    roots = roots[np.isfinite(roots) & (roots > x_left) & (roots < x_right)]
    candidates = np.concatenate(([x_left, x_right], roots))
    return _relative_to_chord(candidates, curve(candidates), x_left, x_right,
                              left_z, right_z, direction)


def extract_beam_profile(points, tolerance):
    """Extract a vertical longitudinal slice, retaining original coordinates/IDs.

    The dominant 3D axis is projected onto XY. Unlike a bounding rectangle of
    unordered XY points, this also identifies a steep beam whose width exceeds
    its horizontal span. Exactly vertical beams have no Z(X) profile.
    """
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("The thickness threshold must be finite and positive")
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Beam coordinates must be an N x 3 array")
    original_indices = np.flatnonzero(np.isfinite(points).all(axis=1))
    finite_points = points[original_indices]
    if len(finite_points) < 3:
        raise ValueError("The beam must contain at least three finite points")
    origin = np.median(finite_points, axis=0)
    centered = finite_points - origin
    _, singular, axes = np.linalg.svd(centered - centered.mean(axis=0), full_matrices=False)
    if singular[0] <= 0 or (singular[1] > 0 and singular[0] <= 1.05 * singular[1]):
        raise ValueError("The cloud has no distinct longitudinal beam axis")
    axis = axes[0, :2].copy()
    horizontal_norm = np.linalg.norm(axis)
    if horizontal_norm <= 1e-8:
        raise ValueError("A vertical beam cannot be analysed as an elevation profile")
    axis /= horizontal_norm
    if axis[np.argmax(np.abs(axis))] < 0:
        axis *= -1
    perpendicular = np.array([-axis[1], axis[0]])
    across = centered[:, :2] @ perpendicular
    across -= np.median(across)
    selected = np.abs(across) <= tolerance
    if np.count_nonzero(selected) < 2:
        raise ValueError("Fewer than two points remain in the longitudinal slice")
    local = np.column_stack((centered[selected, :2] @ axis,
                             across[selected], centered[selected, 2]))
    if np.ptp(local[:, 0]) <= 1e-10 * max(1.0, np.ptp(local[:, 2])):
        raise ValueError("The beam has insufficient horizontal span for an elevation profile")
    return {"points": local, "skeleton": finite_points[selected].copy(),
            "indices": original_indices[selected], "origin": origin,
            "horizontal_axis": axis}


def analyze_beam(points, tolerance=0.02, degree=4, endpoint_fraction=0.05,
                 direction="Vertical", require_fit=True):
    """Analyse a beam without relying on CloudCompare selection or attributes."""
    if not isinstance(degree, (int, np.integer)) or degree < 1:
        raise ValueError("The polynomial degree must be a positive integer")
    section = extract_beam_profile(points, tolerance)
    x, z = section["points"][:, 0], section["points"][:, 2]
    observed = chord_deflection(x, z, endpoint_fraction, direction)
    curve, fitted, fit_error = None, None, ""
    try:
        if len(np.unique(x)) <= degree:
            raise ValueError("The polynomial degree must be lower than the number of unique longitudinal coordinates")
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            curve = np.polynomial.Polynomial.fit(x, z, degree)
        if any(item.category.__name__ == "RankWarning" for item in captured):
            raise ValueError("The polynomial fit is poorly conditioned")
        fitted = polynomial_chord_deflection(curve, float(x.min()), float(x.max()), direction)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
        curve = None
        fit_error = str(error)
        if require_fit:
            raise ValueError(f"Polynomial fit unavailable: {error}") from error
    return {**section, "data": observed, "fit": fitted, "curve": curve,
            "fit_error": fit_error}
