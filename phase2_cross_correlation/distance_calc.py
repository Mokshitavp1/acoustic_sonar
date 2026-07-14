"""
distance_calc.py

Phase 2: converts an echo's round-trip time delay into a one-way
distance estimate, using the speed of sound.
"""


def delay_to_distance(time_delay_sec: float, speed_of_sound: float = 343) -> float:
    """
    Converts an echo round-trip time delay into a one-way distance.

    The sound travels from the speaker to the object and back to the
    mic, so the measured delay covers the round trip. Distance to the
    object is therefore:

        distance = (time_delay_sec * speed_of_sound) / 2

    Args:
        time_delay_sec: Round-trip time delay of the echo, in seconds
            (as returned by find_echo_delay in cross_correlate.py).
        speed_of_sound: Speed of sound in air, in meters/second.
            Defaults to 343 m/s (approx. speed of sound at room
            temperature, ~20°C). Adjust if you want more precision for
            a different ambient temperature.

    Returns:
        Estimated one-way distance to the reflecting object, in meters.
    """
    return (time_delay_sec * speed_of_sound) / 2


if __name__ == "__main__":
    # Example: a round-trip delay of 3ms.
    example_delay_sec = 0.003
    distance_m = delay_to_distance(example_delay_sec)
    print(
        f"A round-trip delay of {example_delay_sec*1000:.1f}ms "
        f"corresponds to a distance of {distance_m:.3f}m "
        f"({distance_m*100:.1f}cm)."
    )
