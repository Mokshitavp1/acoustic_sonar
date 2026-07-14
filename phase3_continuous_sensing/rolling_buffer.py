"""
rolling_buffer.py

Phase 3: a small rolling buffer that stores the last N cross-correlation
profiles from consecutive sensing cycles, and can average them to smooth
out noise before comparing frame-to-frame for motion detection.
"""

from collections import deque

import numpy as np


class RollingBuffer:
    """
    Stores the last `max_size` cross-correlation profiles (numpy arrays)
    and can compute their average for noise smoothing.

    All profiles added must be the same length/shape — they're expected
    to come from find_echo_delay's underlying correlation array for a
    fixed chirp/recording configuration.
    """

    def __init__(self, max_size: int):
        """
        Args:
            max_size: Maximum number of profiles to retain. Once full,
                adding a new profile drops the oldest one.
        """
        if max_size < 1:
            raise ValueError("max_size must be at least 1")
        self.max_size = max_size
        self._buffer = deque(maxlen=max_size)

    def add(self, profile: np.ndarray) -> None:
        """
        Adds a new profile to the buffer. If the buffer is full, the
        oldest profile is dropped automatically.

        Args:
            profile: 1D numpy array (e.g. a cross-correlation magnitude
                array) for one sensing cycle.
        """
        self._buffer.append(np.asarray(profile))

    def average(self) -> np.ndarray:
        """
        Computes the average of all profiles currently in the buffer.

        Returns:
            1D numpy array, the elementwise mean of stored profiles.

        Raises:
            ValueError: if the buffer is empty.
        """
        if len(self._buffer) == 0:
            raise ValueError("Cannot compute average of an empty buffer")
        return np.mean(np.stack(list(self._buffer)), axis=0)

    def __len__(self) -> int:
        return len(self._buffer)

    def is_full(self) -> bool:
        return len(self._buffer) == self.max_size


if __name__ == "__main__":
    # Quick self-test: add a few profiles and confirm the average is correct.
    buf = RollingBuffer(max_size=3)
    buf.add(np.array([1.0, 1.0, 1.0]))
    buf.add(np.array([3.0, 3.0, 3.0]))
    buf.add(np.array([5.0, 5.0, 5.0]))
    print(f"Buffer length: {len(buf)}, is_full: {buf.is_full()}")
    print(f"Average: {buf.average()}")  # expect [3.0, 3.0, 3.0]

    # Adding a 4th profile should drop the oldest (the first one).
    buf.add(np.array([9.0, 9.0, 9.0]))
    print(f"After overflow, average: {buf.average()}")  # expect [5.667, 5.667, 5.667]
