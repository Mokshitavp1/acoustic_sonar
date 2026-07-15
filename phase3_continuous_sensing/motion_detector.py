"""
motion_detector.py

Phase 3: compares two cross-correlation profiles to determine whether
enough has changed between them to flag motion in the room.
"""

import numpy as np


def compare_profiles(profile_a: np.ndarray, profile_b: np.ndarray, threshold: float):
    """
    Compares two cross-correlation profiles using normalized
    cross-correlation similarity, and flags motion if they differ
    beyond `threshold`.

    Normalized cross-correlation gives a similarity score in [-1, 1],
    where 1 means the profiles are identical in shape (perfectly
    correlated), 0 means unrelated, and negative values mean inversely
    related. We convert that into a "difference" score in [0, 1] by
    taking 1 - similarity, so a bigger number means a bigger change.

    Args:
        profile_a: 1D numpy array, e.g. the previous cycle's profile
            (or a rolling buffer's smoothed average).
        profile_b: 1D numpy array, the current cycle's profile. Must be
            the same length as profile_a.
        threshold: Difference score above which motion is flagged.
            Typical starting point: try values around 0.1-0.3 and tune
            based on false positive/negative rate in a quiet room.

    Returns:
        Tuple of (motion_detected: bool, difference_score: float).

    Raises:
        ValueError: if the two profiles are different lengths.
    """
    profile_a = np.asarray(profile_a, dtype=np.float64)
    profile_b = np.asarray(profile_b, dtype=np.float64)

    if profile_a.shape != profile_b.shape:
        raise ValueError(
            f"Profiles must be the same shape to compare; got {profile_a.shape} and {profile_b.shape}"
        )

    # Normalize each profile (zero mean, unit norm) so similarity isn't
    # skewed by differences in overall signal amplitude between cycles.
    a_centered = profile_a - np.mean(profile_a)
    b_centered = profile_b - np.mean(profile_b)

    a_norm = np.linalg.norm(a_centered)
    b_norm = np.linalg.norm(b_centered)

    if a_norm == 0 or b_norm == 0:
        # One of the profiles is completely flat (e.g. silence) — treat
        # as maximally different, since there's no real signal to compare.
        similarity = 0.0
    else:
        similarity = float(np.dot(a_centered, b_centered) / (a_norm * b_norm))

    difference_score = 1.0 - similarity
    motion_detected = difference_score > threshold

    return motion_detected, difference_score


if __name__ == "__main__":
    # Quick self-test: identical profiles should show ~0 difference,
    # and a clearly different profile should show a large difference.
    rng = np.random.default_rng(42)
    base_profile = rng.normal(size=200)

    same_motion, same_score = compare_profiles(base_profile, base_profile, threshold=0.1)
    print(f"Identical profiles -> motion={same_motion}, difference_score={same_score:.4f}")

    different_profile = rng.normal(size=200)  # unrelated random profile
    diff_motion, diff_score = compare_profiles(base_profile, different_profile, threshold=0.1)
    print(f"Unrelated profiles -> motion={diff_motion}, difference_score={diff_score:.4f}")
