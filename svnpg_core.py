"""
Core implementation of SVNPG and baseline methods for sparse learning.
"""
import numpy as np
from scipy.sparse import issparse
from sklearn.utils.extmath import safe_sparse_dot

# ---------------------------------------------------------------------------
# Penalties
# ---------------------------------------------------------------------------

class CappedL1Penalty:
    """Capped-ell_1 penalty: Phi(x) = sum_i min(1, |x_i|/nu)."""
    def __init__(self, lam, nu):
        self.lam = lam
        self.nu = nu

    def value(self, x):
        return self.lam * np.sum(np.minimum(1.0, np.abs(x) / self.nu))

    def dc_decompose(self, x, threshold=None):
        """
        Return DC indicator vector d where:
          d_i = 1 if |x_i| < threshold (inactive)
          d_i = 2 if x_i >= threshold  (active positive)
          d_i = 3 if x_i <= -threshold (active negative)
        Tie-breaking: x_i = threshold -> d_i=2; x_i = -threshold -> d_i=3.
        """
        if threshold is None:
            threshold = self.nu
        d = np.ones(len(x), dtype=int)
        d[x >= threshold] = 2
        d[x <= -threshold] = 3
        return d

    def prox(self, z, gamma, d, box=None):
        """
        Proximal operator of gamma * lam * Phi^d at point z.
        Phi^d(x) = sum_i |x_i|/nu - sum_i theta_{d_i}(x_i).
        box = (l, u) optional box constraints.
        """
        x = np.zeros_like(z)
        nu = self.nu
        coeff = gamma * self.lam / nu
        for i in range(len(z)):
            di = d[i]
            zi = z[i]
            if di == 1:
                # Soft-thresholding
                xi = np.sign(zi) * max(abs(zi) - coeff, 0.0)
            elif di == 2:
                # Minimize coeff*|x| - coeff*x + coeff*nu + 0.5*(x-zi)^2
                # For x >= 0: objective = coeff*nu + 0.5*(x-zi)^2, min at x=zi if zi>0 else x=0
                # For x < 0: objective = -2*coeff*x + coeff*nu + 0.5*(x-zi)^2
                #   derivative: -2*coeff + x - zi = 0  => x = zi + 2*coeff
                if zi > 0:
                    xi = zi
                else:
                    cand = zi + 2 * coeff
                    if cand < 0:
                        val_cand = -2 * coeff * cand + coeff * nu + 0.5 * (cand - zi)**2
                    else:
                        val_cand = np.inf
                    val_0 = coeff * nu + 0.5 * zi**2
                    if val_cand < val_0:
                        xi = cand
                    else:
                        xi = 0.0
            elif di == 3:
                # Minimize coeff*|x| + coeff*x + coeff*nu + 0.5*(x-zi)^2
                # For x <= 0: objective = coeff*nu + 0.5*(x-zi)^2, min at x=zi if zi<0 else x=0
                # For x > 0: objective = 2*coeff*x + coeff*nu + 0.5*(x-zi)^2
                #   derivative: 2*coeff + x - zi = 0  => x = zi - 2*coeff
                if zi < 0:
                    xi = zi
                else:
                    cand = zi - 2 * coeff
                    if cand > 0:
                        val_cand = 2 * coeff * cand + coeff * nu + 0.5 * (cand - zi)**2
                    else:
                        val_cand = np.inf
                    val_0 = coeff * nu + 0.5 * zi**2
                    if val_cand < val_0:
                        xi = cand
                    else:
                        xi = 0.0
            else:
                raise ValueError("Invalid DC indicator")
            # Box projection
            if box is not None:
                l, u = box
                xi = np.clip(xi, l[i], u[i])
            x[i] = xi
        return x


class MCPPenalty:
    """Minimax Concave Penalty for baseline comparison."""
    def __init__(self, lam, gamma=1.0):
        self.lam = lam
        self.gamma = gamma

    def value(self, x):
        val = 0.0
        for xi in np.abs(x):
            if xi <= self.gamma * self.lam:
                val += self.lam * xi - xi**2 / (2 * self.gamma)
            else:
                val += 0.5 * self.gamma * self.lam**2
        return val

    def prox(self, z, step, box=None):
        """Closed-form MCP proximal operator."""
        x = np.zeros_like(z)
        for i in range(len(z)):
            zi = z[i]
            if abs(zi) <= self.lam * step:
                xi = 0.0
            elif abs(zi) <= self.gamma * self.lam * step:
                xi = np.sign(zi) * (abs(zi) - self.lam * step) / (1.0 - step / self.gamma)
            else:
                xi = zi
            if box is not None:
                l, u = box
                xi = np.clip(xi, l[i], u[i])
            x[i] = xi
        return x


# ---------------------------------------------------------------------------
# Losses and Moreau envelopes
# ---------------------------------------------------------------------------

class LADLoss:
    """LAD loss: f(y) = (1/m) sum |y_i - b_i|."""
    def __init__(self, b):
        self.b = np.asarray(b, dtype=float)
        self.m = len(b)

    def value(self, Ax):
        return np.sum(np.abs(Ax - self.b)) / self.m

    def moreau_gradient(self, z, mu, inexact=False, n_inner=10):
        """
        Gradient of Moreau envelope: (z - prox_{mu f}(z)) / mu.
        For LAD, prox has closed form.
        """
        # prox_{mu |.|}(v) = sign(v) * max(|v|-mu, 0)
        v = z - self.b
        p = np.sign(v) * np.maximum(np.abs(v) - mu, 0.0) + self.b
        g = (z - p) / mu
        return g

    def smooth_value(self, z, mu):
        """Huber loss value."""
        v = z - self.b
        val = np.where(np.abs(v) <= mu,
                       v**2 / (2 * mu),
                       np.abs(v) - mu / 2.0)
        return np.sum(val) / self.m


class LogisticLoss:
    """Logistic loss for binary classification."""
    def __init__(self, y):
        self.y = np.asarray(y, dtype=float)
        self.m = len(y)

    def value(self, Ax):
        z = -self.y * Ax
        # Numerically stable log(1+exp(z))
        loss = np.where(z > 0,
                        z + np.log1p(np.exp(-z)),
                        np.log1p(np.exp(z)))
        return np.sum(loss) / self.m

    def moreau_gradient(self, z, mu, inexact=False, n_inner=10):
        """
        Gradient of Moreau envelope of logistic loss.
        prox_{mu f}(z) requires solving a 1D problem for each coordinate.
        """
        p = np.zeros_like(z)
        for i in range(len(z)):
            yi = self.y[i]
            zi = z[i]
            # Solve: min_w log(1+exp(-yi*w)) + 0.5/mu * (w - zi)^2
            # Note: the manuscript uses f_i(w) = log(1+exp(-yi*w)) and the
            # Moreau envelope parameter mu. The proximal subproblem is:
            #   min_w f_i(w) + 1/(2*mu) * (w - zi)^2
            # We solve with Newton's method.
            w = zi
            for _ in range(n_inner if inexact else 50):
                exp_term = np.exp(-yi * w)
                fprime = -yi * exp_term / (1 + exp_term) + (w - zi) / mu
                fsecond = exp_term / (1 + exp_term)**2 + 1.0 / mu
                w = w - fprime / fsecond
            p[i] = w
        g = (z - p) / mu
        return g

    def smooth_value(self, z, mu):
        # Use Moreau envelope value: f_mu(z) = f(prox(z)) + 0.5/mu ||z-prox(z)||^2
        # Approximate using gradient
        g = self.moreau_gradient(z, mu, inexact=True, n_inner=5)
        p = z - mu * g
        return self.value(p) + 0.5 / mu * np.sum((z - p)**2)


# ---------------------------------------------------------------------------
# Gradient utilities
# ---------------------------------------------------------------------------

def batch_gradient(loss, A, x, mu, batch_idx, inexact=False, n_inner=10):
    """Compute mini-batch gradient of f_mu(Ax)."""
    if issparse(A):
        z = safe_sparse_dot(A[batch_idx], x)
    else:
        z = A[batch_idx] @ x
    g_z = loss.moreau_gradient(z, mu, inexact=inexact, n_inner=n_inner)
    if issparse(A):
        g = safe_sparse_dot(A[batch_idx].T, g_z) / len(batch_idx)
    else:
        g = A[batch_idx].T @ g_z / len(batch_idx)
    return g


def full_gradient(loss, A, x, mu, inexact=False, n_inner=10):
    """Compute full gradient of f_mu(Ax)."""
    if issparse(A):
        z = safe_sparse_dot(A, x)
    else:
        z = A @ x
    g_z = loss.moreau_gradient(z, mu, inexact=inexact, n_inner=n_inner)
    if issparse(A):
        g = safe_sparse_dot(A.T, g_z) / loss.m
    else:
        g = A.T @ g_z / loss.m
    return g


# ---------------------------------------------------------------------------
# Algorithms
# ---------------------------------------------------------------------------

class SVNPG:
    """Stochastic Variance-Reduced Nested Proximal Gradient."""
    def __init__(self, A, loss, penalty, x0, mu1, mu2, nu0, nu, alpha=3.0,
                 p_patience=2, bII=1, adaptive_batch=True, box=None,
                 inexact=False, n_inner=10, verbose=False):
        self.A = A
        self.loss = loss
        self.penalty = penalty
        self.x = np.array(x0, dtype=float)
        self.mu1 = mu1
        self.mu2 = mu2
        self.nu0 = nu0
        self.nu = nu
        self.alpha = alpha
        self.p_patience = p_patience
        self.bII = bII
        self.adaptive_batch = adaptive_batch
        self.box = box
        self.inexact = inexact
        self.n_inner = n_inner
        self.verbose = verbose
        self.m = loss.m
        self.n = len(x0)
        self.A_norm = np.linalg.norm(A, 2) if not issparse(A) else np.linalg.norm(A.toarray(), 2)
        self.gammaI = mu1 / (4.0 * self.A_norm**2)
        self.gammaII = mu2 / (8.0 * self.A_norm**2)
        self.R = np.linalg.norm(x0) if box is None else np.max(np.abs(box[1] - box[0]))
        # Margin Delta = lambda/nu - L_col (approximate)
        self.Delta = penalty.lam / nu - np.max(np.abs(full_gradient(loss, A, x0, mu1))) * 1.2
        if self.Delta <= 0:
            self.Delta = penalty.lam / nu * 0.1
        self.Kmax = int(np.ceil(2 * self.R / (self.gammaI * self.Delta))) if self.Delta > 0 else 100
        self.S0 = int(np.ceil(np.log2(nu0 / nu)))
        # Phase tracking
        self.phase = 1
        self.stable_count = 0
        self.d = np.ones(self.n, dtype=int)
        self.support = np.arange(self.n)
        self.history = {'obj': [], 'fscore': [], 'support_size': [], 'epoch': []}

    def _get_batch_size(self, s):
        if self.adaptive_batch:
            # Simple doubling schedule
            b = max(1, int(np.ceil(2**(s-1))))
            b = min(b, self.m)
        else:
            b = 1
        return b

    def _svrg_step(self, x, x_tilde, g_tilde, mu, gamma, b, d):
        """One SVRG proximal gradient step."""
        batch = np.random.choice(self.m, size=b, replace=True)
        g_batch = batch_gradient(self.loss, self.A, x, mu, batch,
                                 inexact=self.inexact, n_inner=self.n_inner)
        g_tilde_batch = batch_gradient(self.loss, self.A, x_tilde, mu, batch,
                                       inexact=self.inexact, n_inner=self.n_inner)
        v = g_batch - g_tilde_batch + g_tilde
        z = x - gamma * v
        x_new = self.penalty.prox(z, gamma, d, box=self.box)
        return x_new

    def run_epoch(self, s, x_star=None):
        """Run one epoch (outer iteration)."""
        if self.phase == 1:
            mu_s = self.mu1
            gamma_s = self.gammaI
            nu_s = max(self.nu0 * (2**(-s)), self.nu)
            b = self._get_batch_size(s)
            # Compute DC indicator from current x
            self.d = self.penalty.dc_decompose(self.x, threshold=nu_s)
            # Snapshot and full gradient
            x_tilde = self.x.copy()
            g_tilde = full_gradient(self.loss, self.A, x_tilde, mu_s,
                                    inexact=self.inexact, n_inner=self.n_inner)
            # Single inner step (T_s=1)
            self.x = self._svrg_step(self.x, x_tilde, g_tilde, mu_s, gamma_s, b, self.d)
            # Check support stability
            support_new = set(np.where(np.abs(self.x) > 1e-12)[0])
            support_old = set(np.where(np.abs(x_tilde) > 1e-12)[0])
            if support_new == support_old:
                self.stable_count += 1
            else:
                self.stable_count = 0
            # Transition to Phase 2?
            if s > self.S0 + self.Kmax and self.stable_count >= self.p_patience:
                self.phase = 2
                self.d = self.penalty.dc_decompose(self.x, threshold=self.nu)
                self.support = np.where(self.d != 1)[0]
                if self.verbose:
                    print(f"  -> Transition to Phase 2 at epoch {s}, support size={len(self.support)}")
        else:
            mu_s = self.mu2
            gamma_s = self.gammaII
            b = self.bII
            T = 4 * self.m
            x_tilde = self.x.copy()
            g_tilde = full_gradient(self.loss, self.A, x_tilde, mu_s,
                                    inexact=self.inexact, n_inner=self.n_inner)
            # Run on active support only
            x_inner = self.x.copy()
            x_sum = np.zeros(self.n)
            for t in range(T):
                x_inner = self._svrg_step(x_inner, x_tilde, g_tilde, mu_s, gamma_s, b, self.d)
                x_sum += x_inner
            # Uniform averaging
            x_avg = x_sum / T
            # Zero out inactive coordinates
            x_avg[self.d == 1] = 0.0
            self.x = x_avg

        # Record history
        obj = self.loss.value(self.A @ self.x) + self.penalty.value(self.x)
        self.history['obj'].append(obj)
        self.history['support_size'].append(len(np.where(np.abs(self.x) > 1e-12)[0]))
        if x_star is not None:
            supp_est = set(np.where(np.abs(self.x) > 1e-12)[0])
            supp_true = set(np.where(np.abs(x_star) > 1e-12)[0])
            if len(supp_true) > 0:
                prec = len(supp_est & supp_true) / max(len(supp_est), 1)
                rec = len(supp_est & supp_true) / max(len(supp_true), 1)
                fscore = 2 * prec * rec / max(prec + rec, 1e-12)
            else:
                fscore = 1.0 if len(supp_est) == 0 else 0.0
            self.history['fscore'].append(fscore)
        self.history['epoch'].append(s)
        return obj


class ProxSGD:
    """Proximal Stochastic Gradient Descent."""
    def __init__(self, A, loss, penalty, x0, gamma0=0.01, box=None, verbose=False):
        self.A = A
        self.loss = loss
        self.penalty = penalty
        self.x = np.array(x0, dtype=float)
        self.gamma0 = gamma0
        self.box = box
        self.verbose = verbose
        self.m = loss.m
        self.history = {'obj': [], 'fscore': [], 'support_size': []}

    def run_epoch(self, epoch, x_star=None):
        gamma = self.gamma0 / np.sqrt(epoch + 1)
        for _ in range(self.m):
            i = np.random.randint(self.m)
            g = batch_gradient(self.loss, self.A, self.x, 1e-3, [i])  # small mu for approx
            z = self.x - gamma * g
            # Use soft-thresholding (ell_1 proxy)
            self.x = np.sign(z) * np.maximum(np.abs(z) - gamma * self.penalty.lam / self.penalty.nu, 0.0)
            if self.box is not None:
                l, u = self.box
                self.x = np.clip(self.x, l, u)
        obj = self.loss.value(self.A @ self.x) + self.penalty.value(self.x)
        self.history['obj'].append(obj)
        self.history['support_size'].append(np.sum(np.abs(self.x) > 1e-12))
        if x_star is not None:
            supp_est = set(np.where(np.abs(self.x) > 1e-12)[0])
            supp_true = set(np.where(np.abs(x_star) > 1e-12)[0])
            if len(supp_true) > 0:
                prec = len(supp_est & supp_true) / max(len(supp_est), 1)
                rec = len(supp_est & supp_true) / max(len(supp_true), 1)
                fscore = 2 * prec * rec / max(prec + rec, 1e-12)
            else:
                fscore = 1.0 if len(supp_est) == 0 else 0.0
            self.history['fscore'].append(fscore)
        return obj


class ProxSVRG_L1:
    """Prox-SVRG with ell_1 penalty (convex surrogate baseline)."""
    def __init__(self, A, loss, lam, x0, mu=0.02, gamma=None, T=None, b=1, box=None, verbose=False):
        self.A = A
        self.loss = loss
        self.lam = lam
        self.x = np.array(x0, dtype=float)
        self.mu = mu
        self.b = b
        self.box = box
        self.verbose = verbose
        self.m = loss.m
        A_norm = np.linalg.norm(A, 2) if not issparse(A) else np.linalg.norm(A.toarray(), 2)
        self.gamma = gamma if gamma is not None else mu / (8 * A_norm**2)
        self.T = T if T is not None else 4 * self.m
        self.history = {'obj': [], 'fscore': [], 'support_size': []}

    def run_epoch(self, epoch, x_star=None):
        x_tilde = self.x.copy()
        g_tilde = full_gradient(self.loss, self.A, x_tilde, self.mu)
        x_inner = self.x.copy()
        x_sum = np.zeros_like(self.x)
        for t in range(self.T):
            batch = np.random.choice(self.m, size=self.b, replace=True)
            g_batch = batch_gradient(self.loss, self.A, x_inner, self.mu, batch)
            g_tilde_batch = batch_gradient(self.loss, self.A, x_tilde, self.mu, batch)
            v = g_batch - g_tilde_batch + g_tilde
            z = x_inner - self.gamma * v
            # ell_1 soft-thresholding
            x_inner = np.sign(z) * np.maximum(np.abs(z) - self.gamma * self.lam, 0.0)
            if self.box is not None:
                l, u = self.box
                x_inner = np.clip(x_inner, l, u)
            x_sum += x_inner
        self.x = x_sum / self.T
        obj = self.loss.value(self.A @ self.x) + self.lam * np.sum(np.abs(self.x))
        self.history['obj'].append(obj)
        self.history['support_size'].append(np.sum(np.abs(self.x) > 1e-12))
        if x_star is not None:
            supp_est = set(np.where(np.abs(self.x) > 1e-12)[0])
            supp_true = set(np.where(np.abs(x_star) > 1e-12)[0])
            if len(supp_true) > 0:
                prec = len(supp_est & supp_true) / max(len(supp_est), 1)
                rec = len(supp_est & supp_true) / max(len(supp_true), 1)
                fscore = 2 * prec * rec / max(prec + rec, 1e-12)
            else:
                fscore = 1.0 if len(supp_est) == 0 else 0.0
            self.history['fscore'].append(fscore)
        return obj


class StochasticPALM:
    """Stochastic Proximal Alternating Linearized Minimization (simplified)."""
    def __init__(self, A, loss, penalty, x0, gamma=0.01, box=None, verbose=False):
        self.A = A
        self.loss = loss
        self.penalty = penalty
        self.x = np.array(x0, dtype=float)
        self.gamma = gamma
        self.box = box
        self.verbose = verbose
        self.m = loss.m
        self.history = {'obj': [], 'fscore': [], 'support_size': []}

    def run_epoch(self, epoch, x_star=None):
        for _ in range(self.m):
            i = np.random.randint(self.m)
            g = batch_gradient(self.loss, self.A, self.x, 1e-3, [i])
            z = self.x - self.gamma * g
            self.x = self.penalty.prox(z, self.gamma, np.ones(len(self.x), dtype=int), box=self.box)
        obj = self.loss.value(self.A @ self.x) + self.penalty.value(self.x)
        self.history['obj'].append(obj)
        self.history['support_size'].append(np.sum(np.abs(self.x) > 1e-12))
        if x_star is not None:
            supp_est = set(np.where(np.abs(self.x) > 1e-12)[0])
            supp_true = set(np.where(np.abs(x_star) > 1e-12)[0])
            if len(supp_true) > 0:
                prec = len(supp_est & supp_true) / max(len(supp_est), 1)
                rec = len(supp_est & supp_true) / max(len(supp_true), 1)
                fscore = 2 * prec * rec / max(prec + rec, 1e-12)
            else:
                fscore = 1.0 if len(supp_est) == 0 else 0.0
            self.history['fscore'].append(fscore)
        return obj
