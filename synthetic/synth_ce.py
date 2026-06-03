#!/usr/bin/env python3
"""
Synthetic heteroscedastic, heavy-tailed black-box optimization experiment.

Goal: show, in an environment unrelated to Tetris, that ranking CE candidates
by a finite-budget training score is biased (winner's curse), and that an
independent validation re-rank with a downside penalty corrects it.

Design (mirrors the paper's Tetris-CE setting):
  - True objective f(theta) is KNOWN (so we can measure true selection error).
  - We never observe f directly; we observe noisy evaluations
        y = f(theta) + eps(theta),
    where eps is heavy-tailed (log-normal) AND heteroscedastic: its scale
    GROWS for candidates that are far from the optimum. This makes some weak
    candidates occasionally post very high lucky scores -> winner's curse.
  - Standard CE selects the elite set by the training-score mean.
  - Robust-validation CE re-ranks a candidate pool on an INDEPENDENT noise
    stream by  R = val_mean - lambda * val_std.
  - We report the TRUE objective f of the finally selected point vs evaluation
    budget M, averaged over many independent runs.

Pure CPU, no external deps beyond numpy.
"""
import numpy as np
import json
import argparse


def true_objective(theta):
    # Concave (maximize): peak at origin, value 0 at optimum, negative elsewhere.
    # Smooth so CE can make progress; dimension d.
    return -np.sum(theta * theta, axis=-1)


def noise_scale(theta):
    # Heteroscedastic: noise scale grows with distance from optimum.
    # Far (weak) candidates have LARGER noise -> more chance of a lucky spike.
    r = np.sqrt(np.sum(theta * theta, axis=-1))
    return 0.5 + 2.0 * r


def noisy_eval(rng, theta, M):
    """Average of M heavy-tailed, heteroscedastic noisy samples of f(theta)."""
    f = true_objective(theta)          # (n,)
    s = noise_scale(theta)             # (n,)
    n = theta.shape[0]
    # Heavy-tailed zero-median multiplicative-ish noise: log-normal minus its
    # median, scaled by s. Right-skewed, heavy upper tail (lucky spikes).
    # shape (n, M)
    z = rng.normal(size=(n, M))
    eps = (np.exp(0.9 * z) - np.exp(0.9 * 0.5 * 0.9)) * s[:, None]
    y = f[:, None] + eps
    return y.mean(axis=1), y.std(axis=1, ddof=1) if M > 1 else np.zeros(n)


def run_ce(rng, d, N, K, gens, M_tr, mode, M_val=None, lam=1.0,
           pool_extra=0):
    """One CE run. mode in {'train','robust'}.
    Returns the true objective of the finally selected mean point."""
    mu = rng.uniform(-3, 3, size=d)
    sigma = np.full(d, 2.0)
    best_theta = mu.copy()
    for g in range(gens):
        cand = rng.normal(mu, sigma, size=(N, d))
        tr_mean, _ = noisy_eval(rng, cand, M_tr)
        if mode == 'train':
            elite_idx = np.argsort(-tr_mean)[:K]
        else:  # robust: re-rank a shortlisted pool on independent stream
            # shortlist top (K+pool_extra) by training, then validate them
            short = np.argsort(-tr_mean)[:K + pool_extra]
            val_mean, val_std = noisy_eval(rng, cand[short], M_val)
            R = val_mean - lam * val_std
            order = short[np.argsort(-R)]
            elite_idx = order[:K]
        elite = cand[elite_idx]
        mu = elite.mean(axis=0)
        sigma = elite.std(axis=0) + 1e-3
        best_theta = mu.copy()
    return float(true_objective(best_theta[None, :])[0])


def budget_of(mode, N, K, gens, M_tr, M_val, pool_extra):
    if mode == 'train':
        return gens * N * M_tr
    else:
        return gens * (N * M_tr + (K + pool_extra) * M_val)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', type=int, default=200)
    ap.add_argument('--d', type=int, default=8)
    ap.add_argument('--N', type=int, default=40)
    ap.add_argument('--K', type=int, default=4)
    ap.add_argument('--gens', type=int, default=30)
    ap.add_argument('--lam', type=float, default=1.0)
    ap.add_argument('--pool_extra', type=int, default=6)
    ap.add_argument('--out', type=str, default='synth_results.json')
    args = ap.parse_args()

    # Sweep training budget M_tr; robust uses same M_tr plus validation.
    M_list = [1, 2, 4, 8, 16, 32]
    M_val = 16  # fixed validation budget for robust

    results = {'M_list': M_list, 'd': args.d, 'N': args.N, 'K': args.K,
               'gens': args.gens, 'lam': args.lam, 'M_val': M_val,
               'pool_extra': args.pool_extra, 'runs': args.runs}
    train_curve, robust_curve = [], []
    train_se, robust_se = [], []
    train_budget, robust_budget = [], []

    for M_tr in M_list:
        tr_vals, rb_vals = [], []
        for r in range(args.runs):
            rng_t = np.random.default_rng(1000 + r)
            rng_r = np.random.default_rng(1000 + r)  # same init seed -> paired
            tr_vals.append(run_ce(rng_t, args.d, args.N, args.K, args.gens,
                                  M_tr, 'train'))
            rb_vals.append(run_ce(rng_r, args.d, args.N, args.K, args.gens,
                                  M_tr, 'robust', M_val=M_val, lam=args.lam,
                                  pool_extra=args.pool_extra))
        tr_vals = np.array(tr_vals)
        rb_vals = np.array(rb_vals)
        # report regret = -f (distance below true optimum 0); lower is better
        train_curve.append(float(-tr_vals.mean()))
        robust_curve.append(float(-rb_vals.mean()))
        train_se.append(float(tr_vals.std(ddof=1) / np.sqrt(args.runs)))
        robust_se.append(float(rb_vals.std(ddof=1) / np.sqrt(args.runs)))
        train_budget.append(budget_of('train', args.N, args.K, args.gens,
                                      M_tr, M_val, args.pool_extra))
        robust_budget.append(budget_of('robust', args.N, args.K, args.gens,
                                       M_tr, M_val, args.pool_extra))
        # paired test at this M: robust better <=> higher f <=> rb_vals > tr_vals
        d = rb_vals - tr_vals  # positive means robust achieves higher true f
        md = d.mean()
        se = d.std(ddof=1) / np.sqrt(args.runs)
        t = md / se if se > 0 else float('nan')
        print(f'M_tr={M_tr:3d} | train_regret={-tr_vals.mean():.4f} '
              f'robust_regret={-rb_vals.mean():.4f} '
              f'| paired t={t:.2f} (robust better if t>0)')

    results['train_regret'] = train_curve
    results['robust_regret'] = robust_curve
    results['train_se'] = train_se
    results['robust_se'] = robust_se
    results['train_budget'] = train_budget
    results['robust_budget'] = robust_budget
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
