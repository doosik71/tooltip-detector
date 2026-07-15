"""Asymmetric Kalman filter for the "which way should the camera move" signal.

What *is* well suited to Kalman filtering is the derived signal actually
sent to the robot: "which direction should the camera move right now".
Every frame, the caller feeds this filter a measurement -- the raw
center-to-tip vector when a tip is detected, or the zero vector when it is
not -- and the filter's ordinary predict/update cycle does the smoothing.
This makes the arrow ease toward newly detected tips instead of snapping to
them, and decay smoothly to zero (both length and, implicitly, direction)
when the tip is lost, all through the same mechanism.

The filter's process noise is asymmetric, and which rate applies is decided
by projecting the new measurement onto the *current* direction, not just by
whether a tip was found this frame:

  - The projection extends further than the current length (a still-valid
    detection genuinely further along the same heading) -> small, slow
    "grow" rate.
  - The projection falls short of the current length -- no detection, OR a
    valid detection whose tip moved toward center, past it, sideways, or
    reversed relative to the current vector -> large, fast "decay" rate.

This distinction matters because a surgeon signals "stop" by moving the
tool tip *back* toward, or past, center -- i.e. opposite the arrow -- while
the tip is still perfectly detected. Gating fast/slow purely on detection
validity would miss that intent entirely. A robot that keeps drifting after
the operator has clearly signalled a stop is a safety problem, so "stop"
must always win the race against "start moving" -- growing into a motion
may be gentle, but stopping may not be leisurely.
"""

import numpy as np

DEFAULT_PROCESS_VAR_GROW = 0.05      # process noise while a tip is tracked
DECAY_PROCESS_VAR = 150.0            # process noise once a tip is lost -- fixed
# and independent of the grow rate, so a
# "stop" always outraces a "start" toward
# a new detection regardless of tuning
DEFAULT_MEASUREMENT_VAR = 400.0      # ~20px measurement std

FRESH_START_LEN = 5.0                # px; state length below which a new
# detection counts as "starting from
# zero" (decay asymptotically nears
# zero but rarely hits it exactly) --
# matches the GUI's minimum draw length


class CameraMotionVector:
    """Random-walk Kalman filter over the center->tip vector.

    State is the (dx, dy) offset from a reference point (e.g. frame center)
    that the arrow should point to. Feeding it the zero vector on missed
    frames makes length decay to 0 through the normal update step instead
    of a hand-rolled decay rule.

    Process noise is asymmetric, and the choice is based on the *projection*
    of the new measurement onto the current direction, not merely whether a
    tip was detected:

      - No detection, OR a detection whose projection onto the current
        direction is smaller than the current length (the tip moved toward
        center, past it, sideways, or reversed) -> large decay rate.
      - A detection that extends further along the current direction ->
        small, user-tunable grow rate.

    This matters because a surgeon signals "stop" by moving the tool tip
    back toward (or past) center -- i.e. *opposite* the arrow -- while the
    tip is still very much detected. Gating the fast/slow choice on raw
    detection validity alone would miss that: the tip is found, but the
    correct response is still an immediate deceleration, exactly like a
    missed detection.
    """

    def __init__(self, process_var_grow: float, measurement_var: float):
        self.q_grow = process_var_grow
        self.r = measurement_var
        self.x = np.zeros(2, dtype=np.float64)
        # large initial uncertainty
        self.p = np.full(2, 1.0e4, dtype=np.float64)

    def reset(self):
        self.x[:] = 0.0
        self.p[:] = 1.0e4

    def set_noise(self, process_var_grow: float, measurement_var: float | None = None):
        self.q_grow = process_var_grow
        if measurement_var is not None:
            self.r = measurement_var

    def _is_growing(self, measurement: tuple[float, float], valid: bool) -> bool:
        if not valid:
            return False

        prev_len = float(np.hypot(self.x[0], self.x[1]))
        if prev_len < FRESH_START_LEN:
            return True  # no established direction yet -- any detection starts fresh

        z = np.asarray(measurement, dtype=np.float64)
        # Signed length of z's projection onto the current direction.
        projected = float(np.dot(z, self.x) / prev_len)
        return projected >= prev_len

    def _grow_steady_state_variance(self) -> float:
        """Posterior variance the filter settles into during ordinary,
        sustained tracking at the current q_grow/r. Used to reset self.p
        when starting fresh from zero length (see step())."""
        q = self.q_grow
        return float((-q + np.sqrt(q * q + 4.0 * q * self.r)) / 2.0)

    def step(self, measurement: tuple[float, float], valid: bool) -> tuple[float, float]:
        prev_len = float(np.hypot(self.x[0], self.x[1]))
        if valid and prev_len < FRESH_START_LEN:
            # Coming back from a full stop (or this is the very first
            # detection ever): discard whatever covariance built up while
            # decaying -- or the large startup default -- so the first frame
            # after a stop grows in at the same slow, steady rate as ongoing
            # tracking, instead of jumping because P was left inflated.
            self.p[:] = self._grow_steady_state_variance()

        growing = self._is_growing(measurement, valid)
        q = self.q_grow if growing else DECAY_PROCESS_VAR
        self.p = self.p + q
        z = np.asarray(measurement, dtype=np.float64)
        k = self.p / (self.p + self.r)
        self.x = self.x + k * (z - self.x)
        self.p = (1.0 - k) * self.p

        # Decay only asymptotically approaches zero and would otherwise leave
        # a permanent, never-quite-zero residual -- snap it to exactly zero
        # once it's below the same threshold that defines "starting fresh",
        # so the state and the fresh-start check above always agree. Gated
        # on "not growing" so this never clips the legitimately tiny first
        # steps of a slow grow-in from zero (those are just as small).
        if not growing and np.hypot(self.x[0], self.x[1]) < FRESH_START_LEN:
            self.x[:] = 0.0

        return float(self.x[0]), float(self.x[1])
