import numpy as np

W1 = 0.05             # slow blend weight -- "keep going"
W2 = 0.5              # fast blend weight -- "stop" (lost tip, or reversal)

DEFAULT_PROCESS_VAR_GROW = 0.5       # process noise while a tip is tracked
DECAY_PROCESS_VAR = 150.0            # process noise once a tip is lost -- fixed
# and independent of the grow rate, so a
# "stop" always outraces a "start" toward
# a new detection regardless of tuning
DEFAULT_MEASUREMENT_VAR = 400.0      # ~20px measurement std

FRESH_START_LEN = 5.0                # px; vector length below which it has no
# established direction to project onto, and
# below which a decaying result is snapped to
# an exact (0, 0) instead of lingering forever


class CameraMotionVectorMagnitudeBlend:
    """Direction-vector smoothing for the "which way should the camera move" signal.

    Only the *length* of the vector is smoothed here; the *direction*
    snaps to this frame's raw measurement immediately (or holds the
    previous direction when nothing was measured). This is a deliberate
    difference from CameraMotionVectorBlend, which smooths direction too.

    Let x be the current vector and m this frame's raw measurement (the
    center-to-tip vector, or the zero vector when nothing valid was
    detected this frame). p is the signed length of m's projection onto
    x's own direction:

        p = dot(m, x) / |x|

    The new length s blends the current length |x| with p:

        s = (1 - w) * |x| + w * p        (clamped to >= 0)

    using a weight w chosen from just two fixed values:

    - W1 (slow) when m points the same way x already does (p >= 0) and
      this frame's detection is valid -- "keep going".
    - W2 (fast) when there is no valid detection this frame, OR m points
      back toward/past the origin relative to x (p < 0) -- "stop". A
      surgeon signals "stop" by moving the tool tip back toward, or
      past, center -- i.e. opposite the current vector -- while the tip
      may still be perfectly detected, so this is judged by direction,
      not by detection validity alone.

    The new direction is simply unit(m) -- or, when m is the zero vector
    (nothing detected), the previous direction is kept so the vector
    shrinks in place instead of losing its heading.

    When x has no established direction yet (its length is below
    FRESH_START_LEN), the projection above is undefined, so s is instead
    just w * |m| -- W1 if this frame's detection is valid (a fresh start
    still ramps in slowly), W2 otherwise (|m| is 0 in that case anyway).
    Decay only asymptotically approaches zero, so once a non-growing
    step's length falls back under FRESH_START_LEN it is snapped to an
    exact (0, 0) instead of lingering indefinitely.
    """

    def __init__(self):
        self.x = np.zeros(2, dtype=np.float64)

    def reset(self):
        self.x[:] = 0.0

    def step(self, measurement: tuple[float, float], valid: bool) -> tuple[float, float]:
        m = np.asarray(measurement, dtype=np.float64)
        prev_len = float(np.hypot(self.x[0], self.x[1]))
        m_len = float(np.hypot(m[0], m[1]))

        if prev_len < FRESH_START_LEN:
            # No established direction to project onto -- treat the whole
            # measurement as the "aligned" target length.
            growing = valid
            w = W1 if valid else W2
            s = w * m_len
        else:
            p = float(np.dot(m, self.x)) / prev_len
            growing = valid and p >= 0.0
            w = W1 if growing else W2
            s = max(0.0, (1.0 - w) * prev_len + w * p)

        # Direction always follows the raw measurement directly (no
        # smoothing) -- except when nothing was measured this frame, in
        # which case there is no new direction to snap to, so the vector
        # just shrinks in place along its previous heading.
        if m_len > 0.0:
            direction = m / m_len
        elif prev_len > 0.0:
            direction = self.x / prev_len
        else:
            direction = np.zeros(2, dtype=np.float64)

        new_x = s * direction

        # Decay only asymptotically approaches zero and would otherwise
        # leave a permanent, never-quite-zero residual -- snap it to
        # exactly zero once it's below the same threshold that defines
        # "no established direction". Gated on "not growing" so this
        # never clips the legitimately tiny first steps of a slow
        # grow-in from zero.
        if not growing and s < FRESH_START_LEN:
            new_x = np.zeros(2, dtype=np.float64)

        self.x = new_x
        return float(self.x[0]), float(self.x[1])


class CameraMotionVectorBlend:
    """Direction-vector smoothing for the "which way should the camera move" signal.

    Why not track the tip position directly
    ------------------------------------------
    A surgical tool tip does not move with a consistent velocity or
    acceleration between frames -- in practice its motion is closer to
    random. Tracking the raw tip position itself is therefore the wrong
    problem. What matters is the derived signal actually sent to the robot:
    "which direction should the camera move right now" -- and that signal
    must only ever change gradually, since a sudden jump would translate
    into a sudden, unsafe robot motion.

    CameraMotionVectorBlend: direct geometric blend
    ---------------------------------------------
    An earlier version of this smoothing used a Kalman filter (preserved
    below as CameraMotionVectorKalman for reference/comparison). In
    practice its behavior did not match what was wanted -- most notably,
    its inherited uncertainty could make the vector jump when growth
    resumed after even a brief decay. CameraMotionVector replaces it with
    an explicit weighted blend that carries no hidden filter state.

    Let x be the current vector and m this frame's raw measurement (the
    center-to-tip vector, or the zero vector when nothing valid was
    detected this frame). m decomposes into its component p along x's own
    direction and its component q perpendicular to x (p + q == m always):

        dx = (1 - w) * x + w * p     # blended value along x's direction
        dy = w * q                   # perpendicular value
        x_new = dx + dy

    The weight w for this frame is chosen from just two fixed values:

    - W1 (slow) when m points the same way x already does (dot(m, x) >=
        0) and this frame's detection is valid -- "keep going".
    - W2 (fast) when there is no valid detection this frame, OR m points
        back toward/past the origin relative to x (dot(m, x) < 0) -- "stop".
        A surgeon signals "stop" by moving the tool tip back toward, or
        past, center -- i.e. opposite the current vector -- while the tip
        may still be perfectly detected, so this is judged by direction,
        not by detection validity alone.

    When x has no established direction yet (its length is below
    FRESH_START_LEN), the projection above is undefined, so x eases
    straight toward w * m instead -- W1 if this frame's detection is valid
    (a fresh start still ramps in slowly), W2 otherwise (stays at zero,
    since m is the zero vector in that case anyway). Decay only
    asymptotically approaches zero, so once a non-growing step's result
    falls back under FRESH_START_LEN it is snapped to an exact (0, 0)
    instead of lingering indefinitely.
    """

    def __init__(self):
        self.x = np.zeros(2, dtype=np.float64)

    def reset(self):
        self.x[:] = 0.0

    def step(self, measurement: tuple[float, float], valid: bool) -> tuple[float, float]:
        m = np.asarray(measurement, dtype=np.float64)
        prev_len = float(np.hypot(self.x[0], self.x[1]))

        if prev_len < FRESH_START_LEN:
            # No established direction to project onto -- ease straight
            # toward m instead. A valid detection still ramps in slowly
            # (W1); otherwise m is (0, 0) anyway so the weight is moot.
            growing = valid
            w = W1 if valid else W2
            new_x = w * m
        else:
            growing = valid and float(np.dot(m, self.x)) >= 0.0
            w = W1 if growing else W2

            p = (float(np.dot(m, self.x)) / (prev_len * prev_len)) * self.x
            q = m - p
            dx = (1.0 - w) * self.x + w * p
            dy = w * q
            new_x = dx + dy

        # Decay only asymptotically approaches zero and would otherwise
        # leave a permanent, never-quite-zero residual -- snap it to
        # exactly zero once it's below the same threshold that defines
        # "no established direction", so the two checks always agree.
        # Gated on "not growing" so this never clips the legitimately
        # tiny first steps of a slow grow-in from zero.
        if not growing and float(np.hypot(new_x[0], new_x[1])) < FRESH_START_LEN:
            new_x = np.zeros(2, dtype=np.float64)

        self.x = new_x
        return float(self.x[0]), float(self.x[1])


class CameraMotionVectorKalman:
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
        growing = self._is_growing(measurement, valid)

        if growing:
            # Pin P to the value ordinary, sustained tracking settles into
            # before every growing step -- not just ones starting from zero.
            # Any preceding decay run, however brief, inflates P (decay uses
            # a much larger process noise); left alone, that inflated P
            # would carry straight into the next growing step and produce a
            # single-frame jump instead of the usual slow climb. During an
            # uninterrupted growth run P is already at this value, so this
            # is a no-op there -- it only matters right after a decay.
            self.p[:] = self._grow_steady_state_variance()

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


CameraMotionVector = CameraMotionVectorMagnitudeBlend
