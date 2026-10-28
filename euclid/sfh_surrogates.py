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


def recent_preserved_independent_past(
    target_weights,
    donor_weights,
    time,
    recent_fraction=0.1,
    transition_bins=8,
):
    """Replace the past shape with an independent donor without a hard join.

    The target's recent interval and its mass fraction are copied exactly.  The
    donor's older interval is normalized to the target's complementary past
    fraction.  A short smoothstep bridge, confined to the replaced interval,
    blends a positive extrapolation of the target boundary into the rescaled
    donor.  Its scale is solved analytically after including the bridge, so the
    recent fraction and total unit integral remain exact.

    This deliberately does *not* match the donor on any galaxy property.  The
    only dependence between the two intervals is the mass-composition
    constraint and the local value/slope information needed for continuity.
    """
    target = np.asarray(target_weights, dtype=np.float64)
    donor = np.asarray(donor_weights, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)
    if (
        target.ndim != 1
        or donor.shape != target.shape
        or time.shape != target.shape
        or not np.all(np.isfinite(target))
        or not np.all(np.isfinite(donor))
        or np.any(target < 0)
        or np.any(donor < 0)
        or not np.all(np.isfinite(time))
        or np.any(np.diff(time) <= 0)
        or not 0 < recent_fraction < 1
        or int(transition_bins) < 1
    ):
        raise ValueError(
            "expected finite, non-negative SFHs on an increasing time grid"
        )

    old_start = int(np.searchsorted(time, recent_fraction, side="right"))
    if old_start < 2 or len(target) - old_start < 2:
        raise ValueError("recent_fraction leaves too few recent or old SFH bins")
    target_total = float(target.sum())
    donor_total = float(donor.sum())
    if target_total <= 0 or donor_total <= 0:
        raise ValueError("target and donor SFHs must have positive integrals")
    target = (
        target.copy() if np.isclose(target_total, 1.0, atol=1e-12, rtol=0)
        else target / target_total
    )
    donor = (
        donor.copy() if np.isclose(donor_total, 1.0, atol=1e-12, rtol=0)
        else donor / donor_total
    )

    target_old_sum = float(target[old_start:].sum())
    donor_old = donor[old_start:]
    donor_old_sum = float(donor_old.sum())
    if donor_old_sum <= np.finfo(float).eps:
        raise ValueError("the donor has no positive mass in the replaced interval")
    donor_shape = donor_old / donor_old_sum
    raw_old = target_old_sum * donor_shape

    # Continue the target boundary in log space.  Clipping only prevents a
    # two-bin numerical slope from exploding; over the short bridge it still
    # permits roughly an order-of-two change in either direction.
    epsilon = max(np.finfo(float).eps, 1e-12 * float(target.max()))
    boundary = max(float(target[old_start - 1]), epsilon)
    previous = max(float(target[old_start - 2]), epsilon)
    dt = float(time[old_start - 1] - time[old_start - 2])
    log_slope = (np.log(boundary) - np.log(previous)) / dt
    log_slope = float(np.clip(log_slope, -25.0, 25.0))
    old_time = time[old_start:]
    extension = boundary * np.exp(
        log_slope * (old_time - float(time[old_start - 1]))
    )

    width = min(int(transition_bins), len(donor_shape))
    while width >= 1:
        phase = np.arange(1, width + 1, dtype=np.float64) / width
        alpha = phase * phase * (3.0 - 2.0 * phase)
        extension_mass = float(np.sum((1.0 - alpha) * extension[:width]))
        donor_coefficient = float(
            np.sum(alpha * donor_shape[:width]) + donor_shape[width:].sum()
        )
        scale = (
            (target_old_sum - extension_mass) / donor_coefficient
            if donor_coefficient > 0 else -1.0
        )
        if scale >= 0:
            break
        width -= 1
    if width < 1:
        raise RuntimeError("could not construct a non-negative boundary bridge")

    smooth_old = scale * donor_shape
    smooth_old[:width] = (
        (1.0 - alpha) * extension[:width]
        + alpha * scale * donor_shape[:width]
    )
    # Correct accumulation only in the replaced interval.
    smooth_old[-1] += target_old_sum - float(smooth_old.sum())
    if np.any(smooth_old < -1e-14):
        raise RuntimeError("the boundary bridge produced negative SFH weights")
    smooth_old = np.maximum(smooth_old, 0.0)

    output = target.copy()
    output[old_start:] = smooth_old
    output[-1] += 1.0 - float(output.sum())
    output[-1] = max(output[-1], 0.0)
    raw = target.copy()
    raw[old_start:] = raw_old
    raw[-1] += 1.0 - float(raw.sum())

    old_correlation = np.corrcoef(
        target[old_start:], output[old_start:]
    )[0, 1]
    if not np.isfinite(old_correlation):
        old_correlation = 0.0
    diagnostics = {
        "old_start": old_start,
        "n_recent_bins": old_start,
        "n_old_bins": len(target) - old_start,
        "transition_bins": width,
        "recent_sum": float(target[:old_start].sum()),
        "old_sum": target_old_sum,
        "integral_error": float(output.sum() - 1.0),
        "recent_max_error": float(
            np.max(np.abs(output[:old_start] - target[:old_start]))
        ),
        "old_correlation": float(old_correlation),
        "raw_boundary_jump": float(raw[old_start] - target[old_start - 1]),
        "smooth_boundary_jump": float(
            output[old_start] - target[old_start - 1]
        ),
        "donor_scale": float(scale),
    }
    return output, raw, diagnostics


def factorized_independent_sfh(
    fraction_weights,
    recent_shape_weights,
    past_shape_weights,
    time,
    recent_fraction=0.1,
    transition_bins=8,
):
    """Assemble an SFH from three independent donor components.

    The recent mass fraction is taken from ``fraction_weights``.  The
    normalized shapes inside and outside the recent interval are taken from
    ``recent_shape_weights`` and ``past_shape_weights``, respectively.  The
    resulting curve therefore contains no component from the target galaxy to
    which it may later be paired.  The same past-side boundary bridge used by
    :func:`recent_preserved_independent_past` makes the join smooth while
    preserving both the recent fraction and the unit integral exactly.
    """
    fraction_source = np.asarray(fraction_weights, dtype=np.float64)
    recent_source = np.asarray(recent_shape_weights, dtype=np.float64)
    past_source = np.asarray(past_shape_weights, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)
    if (
        fraction_source.ndim != 1
        or recent_source.shape != fraction_source.shape
        or past_source.shape != fraction_source.shape
        or time.shape != fraction_source.shape
        or not np.all(np.isfinite(fraction_source))
        or not np.all(np.isfinite(recent_source))
        or not np.all(np.isfinite(past_source))
        or np.any(fraction_source < 0)
        or np.any(recent_source < 0)
        or np.any(past_source < 0)
    ):
        raise ValueError("donor SFHs must be finite, non-negative, equal-length arrays")

    split = int(np.searchsorted(time, recent_fraction, side="right"))
    if split < 2 or len(time) - split < 2:
        raise ValueError("recent_fraction leaves too few recent or past SFH bins")

    totals = [float(x.sum()) for x in (fraction_source, recent_source, past_source)]
    if any(total <= 0 for total in totals):
        raise ValueError("all donor SFHs must have positive integrals")
    fraction_source = fraction_source / totals[0]
    recent_source = recent_source / totals[1]
    past_source = past_source / totals[2]

    recent_fraction_value = float(fraction_source[:split].sum())
    recent_mass = float(recent_source[:split].sum())
    past_mass = float(past_source[split:].sum())
    if recent_mass <= np.finfo(float).eps:
        raise ValueError("recent-shape donor has no mass in the recent interval")
    if past_mass <= np.finfo(float).eps:
        raise ValueError("past-shape donor has no mass in the past interval")

    raw = np.empty_like(fraction_source)
    raw[:split] = recent_fraction_value * recent_source[:split] / recent_mass
    raw[split:] = (1.0 - recent_fraction_value) * past_source[split:] / past_mass
    # Put the round-off correction in the largest past bin.  Correcting the
    # final bin can make an exactly-zero endpoint negative by machine epsilon.
    correction_index = split + int(np.argmax(raw[split:]))
    raw[correction_index] += 1.0 - float(raw.sum())

    # A self-donor call applies precisely the bridge used by experiment 2,
    # while keeping the already factorized recent segment unchanged.
    output, _, bridge = recent_preserved_independent_past(
        raw,
        raw,
        time,
        recent_fraction=recent_fraction,
        transition_bins=transition_bins,
    )
    diagnostics = dict(bridge)
    diagnostics.update({
        "fraction_donor_recent_sum": recent_fraction_value,
        "recent_shape_donor_mass": recent_mass,
        "past_shape_donor_mass": past_mass,
        "raw_integral_error": float(raw.sum() - 1.0),
    })
    return output, raw, diagnostics


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
