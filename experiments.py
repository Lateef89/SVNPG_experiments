"""
Experiment runners for SVNPG manuscript.
"""
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.datasets import fetch_rcv1, fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os

from svnpg_core import (
    SVNPG, ProxSGD, ProxSVRG_L1, StochasticPALM,
    LADLoss, LogisticLoss, CappedL1Penalty, MCPPenalty
)


def generate_synthetic_lad(m=500, n=100, s=8, seed=42):
    """Generate synthetic sparse robust regression data."""
    rng = np.random.RandomState(seed)
    A = rng.randn(m, n)
    # Normalize columns to unit l2 norm
    A = A / np.linalg.norm(A, axis=0, keepdims=True)
    x_true = np.zeros(n)
    idx = rng.choice(n, size=s, replace=False)
    x_true[idx] = rng.randn(s) + np.sign(rng.randn(s)) * 0.5
    b = A @ x_true
    # Add outliers
    outlier_mask = rng.rand(m) < 0.1
    b[outlier_mask] += rng.randn(np.sum(outlier_mask)) * 10.0
    b[~outlier_mask] += rng.randn(np.sum(~outlier_mask)) * 0.1
    return A, b, x_true


def generate_large_synthetic_lad(m=20000, n=5000, s=200, seed=42):
    """Generate large-scale dense synthetic LAD data."""
    rng = np.random.RandomState(seed)
    A = rng.randn(m, n)
    A = A / np.linalg.norm(A, axis=0, keepdims=True)
    x_true = np.zeros(n)
    idx = rng.choice(n, size=s, replace=False)
    x_true[idx] = rng.randn(s) + np.sign(rng.randn(s)) * 0.5
    b = A @ x_true
    outlier_mask = rng.rand(m) < 0.1
    b[outlier_mask] += rng.randn(np.sum(outlier_mask)) * 10.0
    b[~outlier_mask] += rng.randn(np.sum(~outlier_mask)) * 0.1
    return A, b, x_true


def load_rcv1(m_train=5000, seed=42):
    """Load RCV1 dataset (or simulate if unavailable)."""
    try:
        data = fetch_rcv1(subset='train', download_if_missing=True)
        X = data.data
        y = (data.target[:, 0] > 0).astype(float) * 2 - 1  # binary {-1, 1}
        # Subsample
        rng = np.random.RandomState(seed)
        idx = rng.choice(X.shape[0], size=min(m_train, X.shape[0]), replace=False)
        X = X[idx]
        y = y[idx]
        # Scale columns to unit norm
        col_norms = np.sqrt(np.array(X.power(2).sum(axis=0))).ravel()
        col_norms[col_norms == 0] = 1.0
        X = X.multiply(1.0 / col_norms)
        return X, y
    except Exception as e:
        print(f"RCV1 load failed ({e}), using synthetic substitute...")
        rng = np.random.RandomState(seed)
        n = 47236
        X = csr_matrix(rng.randn(m_train, n))
        y = rng.choice([-1.0, 1.0], size=m_train)
        return X, y


def load_news20(m_train=11314, seed=42):
    """Load 20 Newsgroups binary classification."""
    try:
        categories = ['sci.space', 'talk.religion.misc']
        data = fetch_20newsgroups(subset='train', categories=categories,
                                  remove=('headers', 'footers', 'quotes'))
        vectorizer = TfidfVectorizer(max_features=26214, sublinear_tf=True)
        X = vectorizer.fit_transform(data.data)
        y = np.array([1.0 if t == 'sci.space' else -1.0 for t in data.target])
        # Scale columns
        col_norms = np.sqrt(np.array(X.power(2).sum(axis=0))).ravel()
        col_norms[col_norms == 0] = 1.0
        X = X.multiply(1.0 / col_norms)
        return X, y
    except Exception as e:
        print(f"News20 load failed ({e}), using synthetic substitute...")
        rng = np.random.RandomState(seed)
        n = 26214
        X = csr_matrix(rng.randn(m_train, n))
        y = rng.choice([-1.0, 1.0], size=m_train)
        return X, y


def run_synthetic_lad_experiment(n_runs=20, n_epochs=50, save_path='results/synthetic_lad.pkl'):
    """Run synthetic LAD experiment (Figure 1a, 1b, Table 2, 4)."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    results = {}
    methods = ['SVNPG', 'ProxSGD', 'ProxSVRG_L1', 'StochPALM', 'StochProxMCP', 'InexactSVNPG']
    for method in methods:
        results[method] = {'obj': [], 'fscore': [], 'support_size': [], 'time': []}

    for run in range(n_runs):
        print(f"Synthetic LAD run {run+1}/{n_runs}")
        A, b, x_true = generate_synthetic_lad(seed=run)
        m, n = A.shape
        loss = LADLoss(b)
        x0 = np.ones(n) * 0.1
        box = (np.zeros(n), np.ones(n) * 10)

        # SVNPG
        penalty = CappedL1Penalty(lam=0.5, nu=1e-2)
        solver = SVNPG(A, loss, penalty, x0, mu1=5.0, mu2=0.02, nu0=1.0, nu=1e-2,
                       p_patience=2, bII=1, adaptive_batch=True, box=box, verbose=False)
        for s in range(1, n_epochs + 1):
            solver.run_epoch(s, x_star=x_true)
        results['SVNPG']['obj'].append(solver.history['obj'])
        results['SVNPG']['fscore'].append(solver.history['fscore'])
        results['SVNPG']['support_size'].append(solver.history['support_size'])

        # Inexact SVNPG
        solver_i = SVNPG(A, loss, penalty, x0, mu1=5.0, mu2=0.02, nu0=1.0, nu=1e-2,
                         p_patience=2, bII=1, adaptive_batch=True, box=box,
                         inexact=True, n_inner=10, verbose=False)
        for s in range(1, n_epochs + 1):
            solver_i.run_epoch(s, x_star=x_true)
        results['InexactSVNPG']['obj'].append(solver_i.history['obj'])
        results['InexactSVNPG']['fscore'].append(solver_i.history['fscore'])

        # ProxSGD
        penalty_sgd = CappedL1Penalty(lam=0.5, nu=1e-2)
        solver_sgd = ProxSGD(A, loss, penalty_sgd, x0, gamma0=0.05, box=box)
        for s in range(1, n_epochs + 1):
            solver_sgd.run_epoch(s, x_star=x_true)
        results['ProxSGD']['obj'].append(solver_sgd.history['obj'])
        results['ProxSGD']['fscore'].append(solver_sgd.history['fscore'])

        # ProxSVRG with ell_1
        solver_svrg = ProxSVRG_L1(A, loss, lam=0.5, x0=x0, mu=0.02, box=box)
        for s in range(1, n_epochs + 1):
            solver_svrg.run_epoch(s, x_star=x_true)
        results['ProxSVRG_L1']['obj'].append(solver_svrg.history['obj'])
        results['ProxSVRG_L1']['fscore'].append(solver_svrg.history['fscore'])

        # Stochastic PALM
        penalty_palm = CappedL1Penalty(lam=0.5, nu=1e-2)
        solver_palm = StochasticPALM(A, loss, penalty_palm, x0, gamma=0.01, box=box)
        for s in range(1, n_epochs + 1):
            solver_palm.run_epoch(s, x_star=x_true)
        results['StochPALM']['obj'].append(solver_palm.history['obj'])
        results['StochPALM']['fscore'].append(solver_palm.history['fscore'])

        # Stochastic Prox-MCP
        penalty_mcp = MCPPenalty(lam=0.5, gamma=1.0)
        solver_mcp = ProxSGD(A, loss, penalty_mcp, x0, gamma0=0.05, box=box)
        for s in range(1, n_epochs + 1):
            solver_mcp.run_epoch(s, x_star=x_true)
        results['StochProxMCP']['obj'].append(solver_mcp.history['obj'])
        results['StochProxMCP']['fscore'].append(solver_mcp.history['fscore'])

    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved to {save_path}")
    return results


def run_large_scale_experiment(n_runs=5, n_epochs=30, save_path='results/large_scale.pkl'):
    """Run large-scale dense synthetic LAD experiment (Table 2)."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    results = {}
    methods = ['SVNPG', 'DetANPG', 'StochProxMCP', 'ProxSGD', 'ProxSVRG_L1']
    for method in methods:
        results[method] = {'obj': [], 'fscore': [], 'time': []}

    for run in range(n_runs):
        print(f"Large-scale run {run+1}/{n_runs}")
        A, b, x_true = generate_large_synthetic_lad(seed=run)
        m, n = A.shape
        loss = LADLoss(b)
        x0 = np.ones(n) * 0.1
        box = (np.zeros(n), np.ones(n) * 10)

        # SVNPG
        penalty = CappedL1Penalty(lam=0.5, nu=1e-2)
        solver = SVNPG(A, loss, penalty, x0, mu1=5.0, mu2=0.02, nu0=1.0, nu=1e-2,
                       p_patience=2, bII=1, adaptive_batch=True, box=box)
        for s in range(1, n_epochs + 1):
            solver.run_epoch(s, x_star=x_true)
        results['SVNPG']['obj'].append(solver.history['obj'])
        results['SVNPG']['fscore'].append(solver.history['fscore'])

        # Deterministic ANPG (simplified full-gradient baseline)
        # We approximate with full-batch ProxSVRG for fairness
        solver_det = ProxSVRG_L1(A, loss, lam=0.5, x0=x0, mu=0.02, b=m, T=m, box=box)
        for s in range(1, n_epochs + 1):
            solver_det.run_epoch(s, x_star=x_true)
        results['DetANPG']['obj'].append(solver_det.history['obj'])
        results['DetANPG']['fscore'].append(solver_det.history['fscore'])

        # ProxSGD
        penalty_sgd = CappedL1Penalty(lam=0.5, nu=1e-2)
        solver_sgd = ProxSGD(A, loss, penalty_sgd, x0, gamma0=0.05, box=box)
        for s in range(1, n_epochs + 1):
            solver_sgd.run_epoch(s, x_star=x_true)
        results['ProxSGD']['obj'].append(solver_sgd.history['obj'])

        # ProxSVRG ell_1
        solver_svrg = ProxSVRG_L1(A, loss, lam=0.5, x0=x0, mu=0.02, box=box)
        for s in range(1, n_epochs + 1):
            solver_svrg.run_epoch(s, x_star=x_true)
        results['ProxSVRG_L1']['obj'].append(solver_svrg.history['obj'])

        # Stoch Prox-MCP
        penalty_mcp = MCPPenalty(lam=0.5, gamma=1.0)
        solver_mcp = ProxSGD(A, loss, penalty_mcp, x0, gamma0=0.05, box=box)
        for s in range(1, n_epochs + 1):
            solver_mcp.run_epoch(s, x_star=x_true)
        results['StochProxMCP']['obj'].append(solver_mcp.history['obj'])

    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved to {save_path}")
    return results


def run_real_world_experiment(dataset='rcv1', n_runs=5, n_epochs=30,
                              save_path='results/realworld.pkl'):
    """Run real-world text classification experiment (Table 3)."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if dataset == 'rcv1':
        X, y = load_rcv1(m_train=5000)
    else:
        X, y = load_news20()
    m, n = X.shape
    print(f"Dataset: {dataset}, m={m}, n={n}")

    loss = LogisticLoss(y)
    x0 = np.zeros(n)
    box = (np.zeros(n) - 5, np.ones(n) * 5)

    results = {}
    methods = ['SVNPG', 'DetANPG', 'ProxSGD', 'ProxSVRG_L1', 'StochPALM', 'StochProxMCP', 'InexactSVNPG']
    for method in methods:
        results[method] = {'obj': [], 'fscore': [], 'accuracy': []}

    for run in range(n_runs):
        print(f"Real-world {dataset} run {run+1}/{n_runs}")
        # SVNPG
        penalty = CappedL1Penalty(lam=0.1, nu=1e-2)
        solver = SVNPG(X, loss, penalty, x0, mu1=5.0, mu2=0.02, nu0=1.0, nu=1e-2,
                       p_patience=2, bII=1, adaptive_batch=True, box=box)
        for s in range(1, n_epochs + 1):
            solver.run_epoch(s)
        results['SVNPG']['obj'].append(solver.history['obj'])
        results['SVNPG']['support_size'].append(solver.history['support_size'])

        # Inexact SVNPG
        solver_i = SVNPG(X, loss, penalty, x0, mu1=5.0, mu2=0.02, nu0=1.0, nu=1e-2,
                         p_patience=2, bII=1, adaptive_batch=True, box=box,
                         inexact=True, n_inner=10)
        for s in range(1, n_epochs + 1):
            solver_i.run_epoch(s)
        results['InexactSVNPG']['obj'].append(solver_i.history['obj'])

        # ProxSGD
        penalty_sgd = CappedL1Penalty(lam=0.1, nu=1e-2)
        solver_sgd = ProxSGD(X, loss, penalty_sgd, x0, gamma0=0.05, box=box)
        for s in range(1, n_epochs + 1):
            solver_sgd.run_epoch(s)
        results['ProxSGD']['obj'].append(solver_sgd.history['obj'])

        # ProxSVRG ell_1
        solver_svrg = ProxSVRG_L1(X, loss, lam=0.1, x0=x0, mu=0.02, box=box)
        for s in range(1, n_epochs + 1):
            solver_svrg.run_epoch(s)
        results['ProxSVRG_L1']['obj'].append(solver_svrg.history['obj'])

        # Stoch PALM
        penalty_palm = CappedL1Penalty(lam=0.1, nu=1e-2)
        solver_palm = StochasticPALM(X, loss, penalty_palm, x0, gamma=0.01, box=box)
        for s in range(1, n_epochs + 1):
            solver_palm.run_epoch(s)
        results['StochPALM']['obj'].append(solver_palm.history['obj'])

        # Stoch Prox-MCP
        penalty_mcp = MCPPenalty(lam=0.1, gamma=1.0)
        solver_mcp = ProxSGD(X, loss, penalty_mcp, x0, gamma0=0.05, box=box)
        for s in range(1, n_epochs + 1):
            solver_mcp.run_epoch(s)
        results['StochProxMCP']['obj'].append(solver_mcp.history['obj'])

    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved to {save_path}")
    return results


if __name__ == '__main__':
    print("Running synthetic LAD experiment...")
    run_synthetic_lad_experiment(n_runs=3, n_epochs=30)
    print("\nDone.")
