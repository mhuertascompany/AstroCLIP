"""Smooth surrogate star-formation histories for temporal-order ablations."""

from __future__ import annotations

import numpy as np


def iaaft_surrogate(values, rng, max_iterations=1000, tolerance=1e-7):
    """Return an IAAFT surrogate with the same values and similar spectrum.

    The iterative amplitude-adjusted Fourier transform preserves the exact
    one-point distribution (and therefore the sum) of ``values`` while
    approximately preserving its Fourier amplitudes. Fourier phases, which
    encode the temporal placement of features, are randomized.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 4 or not np.all(np.isfinite(values)):
        raise ValueError("values must be a finite one-dimensional array of length >= 4")
    if max_iterations < 1 or tolerance <= 0:
        raise ValueError("max_iterations and tolerance must be positive")

    sorted_values = np.sort(values)
    centered = values - values.mean()
    target_amplitudes = np.abs(np.fft.rfft(centered))
    denominator = max(float(np.linalg.norm(target_amplitudes[1:])), np.finfo(float).eps)
    surrogate = rng.permutation(values)
    previous_error = np.inf
    iterations = 0

    for iterations in range(1, max_iterations + 1):
        transform = np.fft.rfft(surrogate - surrogate.mean())
        phases = np.exp(1j * np.angle(transform))
        adjusted = np.fft.irfft(target_amplitudes * phases, n=len(values))

        order = np.argsort(adjusted, kind="mergesort")
        updated = np.empty_like(adjusted)
        updated[order] = sorted_values
        surrogate = updated

        amplitudes = np.abs(np.fft.rfft(surrogate - surrogate.mean()))
        error = float(np.linalg.norm(amplitudes[1:] - target_amplitudes[1:]) / denominator)
        if np.isfinite(previous_error) and (
            previous_error - error <= tolerance * max(previous_error, 1.0)
        ):
            break
        previous_error = error

    return surrogate, {"iterations": iterations, "spectral_error": error}


def _randomize_bounded_segment(
    values, rng, transition_bins, max_iterations, blend_left=False, blend_right=False,
):
    """IAAFT-randomize one segment while approaching fixed neighbours smoothly."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 4:
        raise ValueError("Each randomized SFH segment must contain at least four bins.")
    surrogate, diagnostics = iaaft_surrogate(
        values, rng, max_iterations=max_iterations,
    )
    target = float(values.sum())
    sides = int(blend_left) + int(blend_right)
    maximum_width = (
        (len(values) - 1) // 2 if sides == 2 else len(values) - 1
    )
    width = min(int(transition_bins), maximum_width)
    if width < 1:
        raise ValueError("transition_bins must be positive")

    while True:
        output = surrogate.copy()
        fixed = np.zeros(len(values), dtype=bool)
        alpha = np.sin(np.linspace(0.0, np.pi / 2.0, width)) ** 2
        if blend_left:
            output[:width] = (1.0 - alpha) * values[:width] + alpha * surrogate[:width]
            fixed[:width] = True
        if blend_right:
            right_alpha = np.ones(1) if width == 1 else alpha
            output[-width:] = (
                (1.0 - right_alpha) * surrogate[-width:]
                + right_alpha * values[-width:]
            )
            fixed[-width:] = True
        free = ~fixed
        remaining = target - float(output[fixed].sum())
        if remaining >= 0 or width == 1:
            break
        width -= 1

    free_sum = float(output[free].sum())
    if remaining > 0 and free_sum > 0:
        output[free] *= remaining / free_sum
    elif remaining > 0:
        output[free] = values[free]
        output[free] *= remaining / max(float(output[free].sum()), np.finfo(float).eps)
    else:
        output[free] = 0.0
    correction_index = int(np.flatnonzero(free)[0])
    output[correction_index] += target - float(output.sum())
    output[correction_index] = max(output[correction_index], 0.0)
    diagnostics["transition_bins"] = width
    diagnostics["integral_error"] = float(output.sum() - target)
    return output, diagnostics


def window_preserved_iaaft(
    weights,
    time,
    rng,
    window_start=0.1,
    window_end=0.2,
    transition_bins=5,
    max_iterations=1000,
):
    """Preserve one fractional-time window and randomize both sides with IAAFT.

    The half-open interval ``window_start <= time < window_end`` is copied
    exactly (including the final time bin when ``window_end == 1``). The two
    outside segments are randomized independently, crossfaded toward the fixed
    window, and normalized to their original integrals separately.
    """
    weights = np.asarray(weights, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)
    if (
        weights.ndim != 1
        or time.shape != weights.shape
        or not np.all(np.isfinite(weights))
        or np.any(weights < 0)
        or not np.all(np.isfinite(time))
        or np.any(np.diff(time) <= 0)
        or not 0 <= window_start < window_end <= 1
    ):
        raise ValueError("expected non-negative SFH weights and a valid time window")
    first = int(np.searchsorted(time, window_start, side="left"))
    last_side = "right" if np.isclose(window_end, 1.0) else "left"
    last = int(np.searchsorted(time, window_end, side=last_side))
    if last <= first:
        raise ValueError("The preserved window contains no SFH bins.")
    if first and first < 4:
        raise ValueError("The recent randomized segment contains fewer than four bins.")
    if last < len(weights) and len(weights) - last < 4:
        raise ValueError("The old randomized segment contains fewer than four bins.")

    output = weights.copy()
    component_info = []
    if first:
        output[:first], info = _randomize_bounded_segment(
            weights[:first], rng, transition_bins, max_iterations,
            blend_right=True,
        )
        component_info.append(info)
    if last < len(weights):
        output[last:], info = _randomize_bounded_segment(
            weights[last:], rng, transition_bins, max_iterations,
            blend_left=True,
        )
        component_info.append(info)

    outside = np.r_[0:first, last:len(weights)]
    correlation = np.corrcoef(weights[outside], output[outside])[0, 1]
    if not np.isfinite(correlation):
        correlation = 1.0 if np.allclose(weights[outside], output[outside]) else 0.0
    return output, {
        "window_start": float(window_start),
        "window_end": float(window_end),
        "window_start_index": first,
        "window_end_index": last,
        "n_preserved_bins": last - first,
        "n_randomized_bins": len(outside),
        "preserved_sum": float(weights[first:last].sum()),
        "outside_sum": float(weights[outside].sum()),
        "outside_integral_error": float(output[outside].sum() - weights[outside].sum()),
        "outside_correlation": float(correlation),
        "spectral_error": float(np.mean([x["spectral_error"] for x in component_info])),
        "iterations": int(max(x["iterations"] for x in component_info)),
        "transition_bins": int(max(x["transition_bins"] for x in component_info)),
    }


def recent_preserved_iaaft(
    weights,
    time,
    rng,
    recent_fraction=0.1,
    transition_bins=5,
    max_iterations=1000,
):
    """Randomize the older SFH while preserving the recent segment and sums.

    The recent interval ``time <= recent_fraction`` is copied exactly. The
    older portion is replaced by an IAAFT surrogate made from that galaxy's
    own values. A short crossfade begins with the original first old bin and
    ends with the surrogate. The remaining old bins are rescaled so that the
    older and total integrals are restored exactly.
    """
    weights = np.asarray(weights, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)
    if (
        weights.ndim != 1
        or time.shape != weights.shape
        or not np.all(np.isfinite(weights))
        or np.any(weights < 0)
        or not np.all(np.isfinite(time))
        or np.any(np.diff(time) <= 0)
        or not 0 < recent_fraction < 1
    ):
        raise ValueError("expected non-negative SFH weights on an increasing time grid")

    old_start = int(np.searchsorted(time, recent_fraction, side="right"))
    if old_start < 2 or len(weights) - old_start < 4:
        raise ValueError("recent_fraction leaves too few recent or old SFH bins")

    old = weights[old_start:]
    surrogate, diagnostics = iaaft_surrogate(
        old, rng, max_iterations=max_iterations,
    )
    target_old_sum = float(old.sum())

    # Crossfade only in the old segment. Reduce its width if a rare, very
    # concentrated boundary would leave a negative target for the tail.
    width = min(int(transition_bins), len(old) - 1)
    if width < 1:
        raise ValueError("transition_bins must be positive")
    while width > 1:
        alpha = np.sin(np.linspace(0.0, np.pi / 2.0, width)) ** 2
        fixed = (1.0 - alpha) * old[:width] + alpha * surrogate[:width]
        if fixed.sum() <= target_old_sum:
            break
        width -= 1
    alpha = np.sin(np.linspace(0.0, np.pi / 2.0, width)) ** 2
    fixed = (1.0 - alpha) * old[:width] + alpha * surrogate[:width]
    remaining = target_old_sum - float(fixed.sum())
    tail = surrogate[width:].copy()
    tail_sum = float(tail.sum())
    if remaining > 0 and tail_sum > 0:
        tail *= remaining / tail_sum
    elif remaining > 0:
        tail = old[width:].copy()
        tail *= remaining / max(float(tail.sum()), np.finfo(float).eps)
    else:
        tail.fill(0.0)

    output = weights.copy()
    output[old_start:old_start + width] = fixed
    output[old_start + width:] = tail
    # Correct floating-point accumulation without altering the recent bins.
    output[-1] += target_old_sum - float(output[old_start:].sum())
    output[-1] = max(output[-1], 0.0)

    old_correlation = np.corrcoef(old, output[old_start:])[0, 1]
    if not np.isfinite(old_correlation):
        # A constant old segment has no temporal ordering to randomize.
        old_correlation = 1.0 if np.allclose(old, output[old_start:]) else 0.0
    diagnostics.update({
        "old_start": old_start,
        "n_recent_bins": old_start,
        "n_old_bins": len(old),
        "transition_bins": width,
        "recent_sum": float(weights[:old_start].sum()),
        "old_sum": target_old_sum,
        "old_integral_error": float(output[old_start:].sum() - target_old_sum),
        "old_correlation": float(old_correlation),
    })
    return output, diagnostics


def past_preserved_iaaft(
    weights,
    time,
    rng,
    recent_fraction=0.1,
    transition_bins=5,
    max_iterations=1000,
):
    """Randomize the recent SFH while preserving the older segment and sums.

    The older interval ``time > recent_fraction`` is copied exactly. The most
    recent portion is replaced by an IAAFT surrogate made from that galaxy's
    own recent values. A short crossfade at the old/recent boundary approaches
    the original SFH, and the randomized segment is rescaled to preserve its
    integral exactly.
    """
    weights = np.asarray(weights, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)
    if (
        weights.ndim != 1
        or time.shape != weights.shape
        or not np.all(np.isfinite(weights))
        or np.any(weights < 0)
        or not np.all(np.isfinite(time))
        or np.any(np.diff(time) <= 0)
        or not 0 < recent_fraction < 1
    ):
        raise ValueError("expected non-negative SFH weights on an increasing time grid")

    old_start = int(np.searchsorted(time, recent_fraction, side="right"))
    if old_start < 4 or len(weights) - old_start < 2:
        raise ValueError("recent_fraction leaves too few recent or old SFH bins")

    recent = weights[:old_start]
    surrogate, diagnostics = iaaft_surrogate(
        recent, rng, max_iterations=max_iterations,
    )
    target_recent_sum = float(recent.sum())

    # The final recent bins touch the preserved past. Blend them progressively
    # back to the original values, then normalize only the freely randomized
    # bins so that the recent and total integrals remain unchanged.
    width = min(int(transition_bins), len(recent) - 1)
    if width < 1:
        raise ValueError("transition_bins must be positive")
    while width > 1:
        alpha = np.sin(np.linspace(0.0, np.pi / 2.0, width)) ** 2
        fixed = (1.0 - alpha) * surrogate[-width:] + alpha * recent[-width:]
        if fixed.sum() <= target_recent_sum:
            break
        width -= 1
    alpha = np.sin(np.linspace(0.0, np.pi / 2.0, width)) ** 2
    fixed = (1.0 - alpha) * surrogate[-width:] + alpha * recent[-width:]
    remaining = target_recent_sum - float(fixed.sum())
    head = surrogate[:-width].copy()
    head_sum = float(head.sum())
    if remaining > 0 and head_sum > 0:
        head *= remaining / head_sum
    elif remaining > 0:
        head = recent[:-width].copy()
        head *= remaining / max(float(head.sum()), np.finfo(float).eps)
    else:
        head.fill(0.0)

    output = weights.copy()
    output[:old_start - width] = head
    output[old_start - width:old_start] = fixed
    # Correct floating-point accumulation within the randomized interval only.
    output[0] += target_recent_sum - float(output[:old_start].sum())
    output[0] = max(output[0], 0.0)

    recent_correlation = np.corrcoef(recent, output[:old_start])[0, 1]
    if not np.isfinite(recent_correlation):
        recent_correlation = 1.0 if np.allclose(recent, output[:old_start]) else 0.0
    diagnostics.update({
        "old_start": old_start,
        "n_recent_bins": old_start,
        "n_old_bins": len(weights) - old_start,
        "transition_bins": width,
        "recent_sum": target_recent_sum,
        "old_sum": float(weights[old_start:].sum()),
        "recent_integral_error": float(output[:old_start].sum() - target_recent_sum),
        "recent_correlation": float(recent_correlation),
    })
    return output, diagnostics
