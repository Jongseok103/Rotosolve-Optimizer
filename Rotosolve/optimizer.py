import random
import numpy as np

class Rotosolve:
    def __init__(self, maxiter=50, tol=1e-8, repeats=1):
        self.maxiter = maxiter
        self.tol = tol
        self.history = []
        self.repeats = repeats

    @staticmethod
    def _wrap_angle(x):
        return (x + np.pi) % (2 * np.pi) - np.pi

    def optimize(self, init_params, cost_fn):
        params = np.array(init_params, dtype=float)
        D = params.size
        for it in range(self.maxiter):
            max_delta = 0.0
            indices = list(range(D))
            random.shuffle(indices)
            for i in indices:
                theta_i = params[i]

                base  = params.copy()
                plus  = params.copy(); plus[i]  += np.pi/2
                minus = params.copy(); minus[i] -= np.pi/2

                batched = np.vstack([base, plus, minus])  # (3, D)
                # Use cost_fn in batched mode if supported
                try:
                    vals = cost_fn(batched)  # expect length-3 array
                    vals = np.asarray(vals, dtype=float).ravel()
                    if vals.shape[0] != 3:
                        raise ValueError("batch output must be length 3")
                    E0, Ep, Em = vals
                except Exception:
                    E0 = float(cost_fn(base))
                    Ep = float(cost_fn(plus))
                    Em = float(cost_fn(minus))

                a = np.arctan2(2.0 * E0 - Ep - Em, Ep - Em)
                theta_opt = -np.pi/2 - a
                theta_opt = self._wrap_angle(theta_opt)

                delta = np.abs(self._wrap_angle(theta_opt - theta_i))
                max_delta = max(max_delta, delta)
                params[i] = theta_opt

            cost = float(cost_fn(params))
            self.history.append(cost)
            if (it == 0) or ((it + 1) % 10 == 0):
                print(f"[Rotosolve] Iter {it+1}/{self.maxiter}, Cost = {cost:.6f}")

            if max_delta < self.tol:
                break

        return params, cost
