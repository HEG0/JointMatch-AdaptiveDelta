"""EMA-adaptive disagreement weight for JointMatch.

In the original JointMatch the weight on "disagreement" samples (delta, `disagree_weight`) is the constant 0.9:

    weight = delta * 1[the two nets disagree on confidence] + (1 - delta) * 1[the two nets agree on confidence]

Here delta is computed from an exponential moving average (EMA) of how often the two networks disagree
on which unlabeled samples are confident:

    A_t = m * A_{t-1} + (1 - m) * (#samples where both nets are confident)      (agree)
    D_t = m * D_{t-1} + (1 - m) * (#samples where exactly one net is confident) (disagree)
    r_t = D_t / (A_t + D_t)                    EMA disagreement rate among confident samples, in [0, 1]
    delta_t = delta_min + (delta_max - delta_min) * (1 - r_t)

Reasoning (following the paper's motivation for the disagreement weight):
  * few disagreements (r_t -> 0): the two networks have become alike, which is the collapse JointMatch tries to
    prevent, so the disagreement samples get the full weight delta_max (= the paper's 0.9);
  * many disagreements (r_t large): the networks are already diverse, and agreement samples are the more reliable
    pseudo-labels, so the weight shifts toward them (delta decreases toward delta_min).
Until the EMA has seen enough confident samples (`warmup_mass`), delta = delta_max, i.e. training starts exactly
like the paper (delta = 0.9). Everything else (constant lr, thresholds, losses, early stopping) is unchanged.

This rule is a design choice, not something taken from the paper.
"""


class DisagreementEMA:
    def __init__(self,
                 disagree_mode='fixed',    # 'fixed' (paper: constant delta) or 'adaptive'
                 fixed_weight=0.9,         # delta when disagree_mode == 'fixed'
                 delta_min=0.6,            # lowest delta (adaptive); the paper requires delta in (0.5, 1)
                 delta_max=0.9,            # highest delta (adaptive) = the paper's value
                 momentum=0.99,            # EMA momentum per unlabeled batch (time constant ~100 batches = 10 steps)
                 warmup_mass=20.0):        # min. EMA mass of confident samples before delta starts to adapt
        assert disagree_mode in ('fixed', 'adaptive'), disagree_mode
        assert 0.5 <= delta_min <= delta_max < 1.0 or disagree_mode == 'fixed'
        self.disagree_mode = disagree_mode
        self.fixed_weight = fixed_weight
        self.delta_min = delta_min
        self.delta_max = delta_max
        self.momentum = momentum
        self.warmup_mass = warmup_mass
        self.agree = 0.0       # A_t
        self.disagree = 0.0    # D_t

    def update(self, n_agree, n_disagree):
        """Feed the counts of one unlabeled batch (confident-by-both, confident-by-exactly-one)."""
        m = self.momentum
        self.agree = m * self.agree + (1 - m) * n_agree
        self.disagree = m * self.disagree + (1 - m) * n_disagree

    def rate(self):
        """EMA disagreement rate r_t among confident samples, or None before the warm-up mass is reached."""
        mass = self.agree + self.disagree
        if mass < self.warmup_mass * (1 - self.momentum) or mass <= 0:
            return None
        return self.disagree / mass

    def delta(self):
        if self.disagree_mode == 'fixed':
            return self.fixed_weight
        r = self.rate()
        if r is None:
            return self.delta_max
        return self.delta_min + (self.delta_max - self.delta_min) * (1.0 - min(1.0, max(0.0, r)))


# Named experiment variants used by panel_main.py / make_figures.py
VARIANTS = {
    'paper_fixed': dict(disagree_mode='fixed'),      # the paper: constant delta = 0.9
    'adaptive_ema': dict(disagree_mode='adaptive'),  # delta from the EMA disagreement rate
}


if __name__ == '__main__':
    # self-check
    f = DisagreementEMA()
    assert f.delta() == 0.9 and f.rate() is None
    a = DisagreementEMA(disagree_mode='adaptive')
    assert a.delta() == 0.9                                  # before warm-up: paper value
    for _ in range(500):
        a.update(7, 0)                                       # nets always agree
    assert abs(a.rate()) < 1e-9 and abs(a.delta() - 0.9) < 1e-9
    for _ in range(2000):
        a.update(2, 6)                                       # 75% disagreement
    print('rate %.3f  delta %.3f' % (a.rate(), a.delta()))
    assert abs(a.rate() - 0.75) < 0.01 and abs(a.delta() - (0.6 + 0.3 * 0.25)) < 0.01
    b = DisagreementEMA(disagree_mode='adaptive')
    b.update(1, 0)                                           # too little evidence -> still delta_max
    assert b.rate() is None and b.delta() == 0.9
    print('ok')
