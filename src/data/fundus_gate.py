"""Fundus image gating — rejects non-retinal images before inference.

Three-layer validation:
1. Structural: aspect ratio, resolution, color profile
2. Statistical: channel histograms match fundus image distribution
3. Confidence: post-inference OOD detection (all predictions near zero = non-retinal)

Retinal fundus images have distinctive properties:
- Roughly circular field on a dark background
- Warm color palette (red/orange channel dominant from retinal vasculature)
- Specific luminance distribution (dark periphery, bright optic disc region)
- Typical aspect ratios near 1:1 (4:3 also common from camera sensors)
"""
import logging
from dataclasses import dataclass

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Fundus image statistical profiles (derived from RFMiD training set)
# These are the expected ranges for normalized channel means
FUNDUS_RED_MEAN_RANGE = (0.20, 0.75)    # fundus images are red-dominant
FUNDUS_GREEN_MEAN_RANGE = (0.08, 0.55)
FUNDUS_BLUE_MEAN_RANGE = (0.02, 0.40)
FUNDUS_RED_GREEN_RATIO_MIN = 1.05        # red channel always > green in fundus
FUNDUS_DARK_PIXEL_RATIO_MIN = 0.05       # at least 5% dark pixels (circular mask border)
FUNDUS_DARK_PIXEL_RATIO_MAX = 0.80       # not mostly dark (would be blank/corrupt)
FUNDUS_MIN_DIMENSION = 100               # fundus cameras produce at least 100px
FUNDUS_MAX_ASPECT_DEVIATION = 0.65       # max deviation from square (4:3 = 0.33, 16:9 = 0.78)
FUNDUS_SATURATION_MIN = 0.03             # fundus images have some color saturation

# Post-inference OOD thresholds
OOD_MAX_CONFIDENCE = 0.15   # if no disease has >15% confidence, likely non-retinal
OOD_MEAN_CONFIDENCE = 0.03  # if mean across all 45 diseases is <3%, likely non-retinal


@dataclass
class GateResult:
    """Result of fundus image gating."""
    passed: bool
    confidence: float          # 0-1, how likely this is a fundus image
    reason: str                # human-readable explanation
    checks: dict               # individual check results
    layer: str                 # which layer made the decision: structural | statistical | ood


def check_structural(image: Image.Image) -> tuple[bool, dict]:
    """Layer 1: Fast structural checks (no pixel analysis)."""
    checks = {}

    # Resolution
    w, h = image.size
    checks["resolution"] = {"width": w, "height": h}
    if w < FUNDUS_MIN_DIMENSION or h < FUNDUS_MIN_DIMENSION:
        checks["resolution"]["passed"] = False
        return False, checks
    checks["resolution"]["passed"] = True

    # Aspect ratio (fundus images are roughly square or 4:3)
    aspect = max(w, h) / max(min(w, h), 1)
    deviation = abs(aspect - 1.0)
    checks["aspect_ratio"] = {"ratio": round(aspect, 2), "deviation": round(deviation, 2)}
    checks["aspect_ratio"]["passed"] = deviation <= FUNDUS_MAX_ASPECT_DEVIATION

    # Mode must be RGB (not grayscale, RGBA, palette)
    checks["color_mode"] = {"mode": image.mode}
    checks["color_mode"]["passed"] = image.mode in ("RGB", "RGBA")

    all_passed = all(c.get("passed", True) for c in checks.values())
    return all_passed, checks


def check_statistical(image: Image.Image) -> tuple[bool, float, dict]:
    """Layer 2: Color distribution analysis.

    Returns: (passed, confidence_score, checks_dict)
    """
    checks = {}
    score = 0.0
    total_checks = 0

    # Convert to RGB numpy array
    rgb = image.convert("RGB")
    pixels = np.array(rgb, dtype=np.float32) / 255.0

    # Channel means
    r_mean = float(pixels[:, :, 0].mean())
    g_mean = float(pixels[:, :, 1].mean())
    b_mean = float(pixels[:, :, 2].mean())

    checks["channel_means"] = {"red": round(r_mean, 3), "green": round(g_mean, 3), "blue": round(b_mean, 3)}

    # Red channel in expected range
    r_ok = FUNDUS_RED_MEAN_RANGE[0] <= r_mean <= FUNDUS_RED_MEAN_RANGE[1]
    checks["red_range"] = {"passed": r_ok, "expected": FUNDUS_RED_MEAN_RANGE}
    if r_ok:
        score += 1
    total_checks += 1

    # Green channel in expected range
    g_ok = FUNDUS_GREEN_MEAN_RANGE[0] <= g_mean <= FUNDUS_GREEN_MEAN_RANGE[1]
    checks["green_range"] = {"passed": g_ok, "expected": FUNDUS_GREEN_MEAN_RANGE}
    if g_ok:
        score += 1
    total_checks += 1

    # Blue channel in expected range
    b_ok = FUNDUS_BLUE_MEAN_RANGE[0] <= b_mean <= FUNDUS_BLUE_MEAN_RANGE[1]
    checks["blue_range"] = {"passed": b_ok, "expected": FUNDUS_BLUE_MEAN_RANGE}
    if b_ok:
        score += 1
    total_checks += 1

    # Red-dominant (retinal vasculature makes red channel strongest)
    r_g_ratio = r_mean / max(g_mean, 0.001)
    r_dominant = r_g_ratio >= FUNDUS_RED_GREEN_RATIO_MIN
    checks["red_dominant"] = {"passed": r_dominant, "ratio": round(r_g_ratio, 2)}
    if r_dominant:
        score += 1.5  # weighted higher — very distinctive
    total_checks += 1.5

    # Dark pixel ratio (fundus has dark border from circular aperture)
    luminance = 0.299 * pixels[:, :, 0] + 0.587 * pixels[:, :, 1] + 0.114 * pixels[:, :, 2]
    dark_ratio = float((luminance < 0.1).mean())
    dark_ok = FUNDUS_DARK_PIXEL_RATIO_MIN <= dark_ratio <= FUNDUS_DARK_PIXEL_RATIO_MAX
    checks["dark_border"] = {"passed": dark_ok, "ratio": round(dark_ratio, 3)}
    if dark_ok:
        score += 1
    total_checks += 1

    # Saturation check (fundus images have warm color, not grayscale)
    channel_range = max(r_mean, g_mean, b_mean) - min(r_mean, g_mean, b_mean)
    sat_ok = channel_range >= FUNDUS_SATURATION_MIN
    checks["saturation"] = {"passed": sat_ok, "range": round(channel_range, 3)}
    if sat_ok:
        score += 0.5
    total_checks += 0.5

    # Center brightness (optic disc region tends to be brighter than edges)
    h, w = pixels.shape[:2]
    center = luminance[h//4:3*h//4, w//4:3*w//4]
    edge = np.concatenate([luminance[:h//4].flatten(), luminance[3*h//4:].flatten()])
    center_brighter = float(center.mean()) > float(edge.mean()) * 0.8
    checks["center_bright"] = {"passed": center_brighter, "center_mean": round(float(center.mean()), 3), "edge_mean": round(float(edge.mean()), 3)}
    if center_brighter:
        score += 1
    total_checks += 1

    # Circular dark border pattern — fundus cameras produce a circular field
    # Check if corners are significantly darker than center (circular aperture)
    corner_size = max(h // 8, 1)
    corners = np.concatenate([
        luminance[:corner_size, :corner_size].flatten(),
        luminance[:corner_size, -corner_size:].flatten(),
        luminance[-corner_size:, :corner_size].flatten(),
        luminance[-corner_size:, -corner_size:].flatten(),
    ])
    center_region = luminance[h//3:2*h//3, w//3:2*w//3]
    corner_mean = float(corners.mean())
    center_mean = float(center_region.mean())
    # Corners should be at least 40% darker than center for circular aperture
    circular_ok = center_mean > 0.05 and corner_mean < center_mean * 0.6
    checks["circular_aperture"] = {
        "passed": circular_ok,
        "corner_mean": round(corner_mean, 3),
        "center_mean": round(center_mean, 3),
    }
    if circular_ok:
        score += 2.0  # heavily weighted — most distinctive fundus feature
    total_checks += 2.0

    # Luminance variance — fundus images have complex texture (vessels, disc, macula)
    # Uniform images (solid colors, skin close-ups) have very low variance
    lum_std = float(luminance.std())
    texture_ok = lum_std >= 0.06
    checks["texture_variance"] = {"passed": texture_ok, "std": round(lum_std, 4)}
    if texture_ok:
        score += 1.0
    total_checks += 1.0

    # -------------------------------------------------------------------
    # Radial boundary sharpness — fundus cameras produce a SHARP circular
    # mask edge.  Cinematic vignettes fade gradually.  Sample luminance
    # along radial lines and measure the ratio of the single largest step
    # to the average step.  A sharp aperture gives one dominant step.
    # -------------------------------------------------------------------
    h_img, w_img = pixels.shape[:2]
    cy, cx = h_img / 2.0, w_img / 2.0
    max_r = min(h_img, w_img) / 2.0
    n_angles, n_bins = 36, 20
    profile = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    for a in range(n_angles):
        angle = 2 * np.pi * a / n_angles
        for ri in range(n_bins):
            r = max_r * (ri + 0.5) / n_bins
            iy = int(cy + r * np.sin(angle))
            ix = int(cx + r * np.cos(angle))
            if 0 <= iy < h_img and 0 <= ix < w_img:
                profile[ri] += luminance[iy, ix]
                counts[ri] += 1
    profile = profile / np.maximum(counts, 1)
    radial_diffs = np.abs(np.diff(profile))
    max_step = float(radial_diffs.max()) if len(radial_diffs) else 0
    mean_step = float(radial_diffs.mean()) if len(radial_diffs) else 1
    sharpness_ratio = max_step / max(mean_step, 0.001)
    sharp_boundary = sharpness_ratio > 2.5 and max_step > 0.08
    checks["radial_sharpness"] = {
        "passed": sharp_boundary,
        "ratio": round(sharpness_ratio, 2),
        "max_step": round(max_step, 4),
    }
    if sharp_boundary:
        score += 2.5
    total_checks += 2.5

    # -------------------------------------------------------------------
    # Hue concentration — fundus images are overwhelmingly red/orange due
    # to retinal vasculature and illumination.  Natural images, movie
    # screenshots, portraits etc. have broadly distributed hues.  Measure
    # fraction of saturated pixels whose hue falls in the red-orange band
    # (H < 40° or H > 340° in 360° space, i.e. H < 28 or H > 238 in
    # OpenCV 0-179 range).
    # -------------------------------------------------------------------
    hsv = rgb.convert("HSV")
    hsv_arr = np.array(hsv, dtype=np.float32)
    hue = hsv_arr[:, :, 0]          # 0-255 in PIL (maps to 0-360°)
    sat = hsv_arr[:, :, 1] / 255.0
    # Only consider pixels with meaningful saturation (>0.10)
    sat_mask = sat > 0.10
    n_saturated = int(sat_mask.sum())
    if n_saturated > 100:
        hue_sat = hue[sat_mask]
        # Red-orange band: hue < 40° or hue > 340° (in 0-255 PIL scale:
        # 40° = 28, 340° = 241)
        red_orange = ((hue_sat < 28) | (hue_sat > 241)).sum()
        hue_concentration = float(red_orange) / n_saturated
    else:
        hue_concentration = 0.0
    hue_ok = hue_concentration >= 0.50
    checks["hue_concentration"] = {
        "passed": hue_ok,
        "red_orange_fraction": round(hue_concentration, 3),
        "saturated_pixels": n_saturated,
    }
    if hue_ok:
        score += 2.0   # heavily weighted — very discriminative
    total_checks += 2.0

    # -------------------------------------------------------------------
    # Green-channel micro-structure — retinal blood vessels appear as fine
    # dark branching lines in the green channel.  Compute Laplacian
    # variance inside the bright field; vessels create high variance.
    # -------------------------------------------------------------------
    green = pixels[:, :, 1]
    lap = (
        -4 * green[1:-1, 1:-1]
        + green[:-2, 1:-1] + green[2:, 1:-1]
        + green[1:-1, :-2] + green[1:-1, 2:]
    )
    bright_mask = luminance[1:-1, 1:-1] > 0.15
    if bright_mask.sum() > 100:
        lap_var = float(lap[bright_mask].var())
    else:
        lap_var = 0.0
    # Fundus vessel structures: moderate variance (0.0001–0.002).
    # Random noise / complex scenes have much higher variance (>0.005).
    vessel_ok = 0.0001 < lap_var < 0.003
    checks["green_microstructure"] = {
        "passed": vessel_ok,
        "laplacian_var": round(lap_var, 6),
    }
    if vessel_ok:
        score += 1.5
    total_checks += 1.5

    confidence = score / total_checks if total_checks > 0 else 0

    # Hard requirement: must have (dark border or circular aperture) AND
    # sharp radial boundary AND red-orange hue concentration.
    has_fundus_spatial = (
        (checks.get("dark_border", {}).get("passed", False)
         or checks.get("circular_aperture", {}).get("passed", False))
        and checks.get("radial_sharpness", {}).get("passed", False)
        and checks.get("hue_concentration", {}).get("passed", False)
    )

    passed = confidence >= 0.55 and has_fundus_spatial

    return passed, confidence, checks


def check_ood_postinference(predictions: list[dict], all_probabilities: dict) -> tuple[bool, dict]:
    """Layer 3: Post-inference out-of-distribution detection.

    If the model isn't confident about ANY disease (all near 0), the image
    is likely not a retinal fundus image at all.
    """
    checks = {}

    if not all_probabilities:
        return True, {"note": "no predictions to check"}

    probs = [
        v["probability"] if isinstance(v, dict) else float(v)
        for v in all_probabilities.values()
    ]

    max_conf = max(probs) if probs else 0
    mean_conf = float(np.mean(probs)) if probs else 0

    checks["max_confidence"] = {"value": round(max_conf, 4), "threshold": OOD_MAX_CONFIDENCE}
    checks["mean_confidence"] = {"value": round(mean_conf, 4), "threshold": OOD_MEAN_CONFIDENCE}

    # Both conditions must be met to flag as OOD
    is_ood = max_conf < OOD_MAX_CONFIDENCE and mean_conf < OOD_MEAN_CONFIDENCE
    checks["is_ood"] = is_ood

    return not is_ood, checks


def gate_image(image: Image.Image) -> GateResult:
    """Run the full pre-inference fundus gating pipeline.

    Returns GateResult with pass/fail and detailed checks.
    """
    # Layer 1: Structural
    struct_passed, struct_checks = check_structural(image)
    if not struct_passed:
        failed_checks = [k for k, v in struct_checks.items() if not v.get("passed", True)]
        return GateResult(
            passed=False,
            confidence=0.0,
            reason=f"Image failed structural checks: {', '.join(failed_checks)}. "
                   "Please upload a retinal fundus photograph.",
            checks={"structural": struct_checks},
            layer="structural",
        )

    # Layer 2: Statistical
    stat_passed, stat_confidence, stat_checks = check_statistical(image)
    if not stat_passed:
        return GateResult(
            passed=False,
            confidence=stat_confidence,
            reason="Image does not match retinal fundus color profile. "
                   f"Fundus confidence: {stat_confidence:.0%}. "
                   "Please upload a color retinal fundus photograph from a fundus camera.",
            checks={"structural": struct_checks, "statistical": stat_checks},
            layer="statistical",
        )

    return GateResult(
        passed=True,
        confidence=stat_confidence,
        reason="Image passes fundus validation",
        checks={"structural": struct_checks, "statistical": stat_checks},
        layer="statistical",
    )


def gate_predictions(predictions: list[dict], all_probabilities: dict) -> GateResult | None:
    """Run post-inference OOD check. Returns GateResult only if OOD detected, else None."""
    ood_passed, ood_checks = check_ood_postinference(predictions, all_probabilities)
    if not ood_passed:
        return GateResult(
            passed=False,
            confidence=0.0,
            reason="Model confidence is extremely low across all 45 diseases, "
                   "indicating this may not be a retinal fundus image. "
                   "Results should be interpreted with caution.",
            checks={"ood": ood_checks},
            layer="ood",
        )
    return None
