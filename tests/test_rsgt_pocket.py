"""Focused tests for the RSGT local pocket module.

These tests import the component file directly so they only require PyTorch.
"""
from pathlib import Path
import importlib.util

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "src/models/components/rsgt.py"
spec = importlib.util.spec_from_file_location("rsgt_component", MODULE_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
LocalPocketRefinement = mod.LocalPocketRefinement


def nparams(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def test_parameter_counts_for_benchmark_class_counts():
    assert nparams(LocalPocketRefinement(13)) == 1642  # S3DIS
    assert nparams(LocalPocketRefinement(15)) == 1826  # KITTI-360
    assert nparams(LocalPocketRefinement(8)) == 1217   # DALES


def test_candidate_rule_and_non_candidates_unchanged():
    m = LocalPocketRefinement(
        num_classes=3, edge_dim=18, hidden_dim=32,
        tau_u=0.6, tau_d=0.6, lambda_p=1.0)

    # Node 0 is very uncertain; nodes 1--2 are confident. The graph gives
    # node 1 two disagreeing neighbors, hence d_1=1.0. Node 2 has one
    # agreeing neighbor and remains unselected.
    z = torch.tensor([
        [0.0, 0.0, 0.0],
        [8.0, 0.0, 0.0],
        [8.0, 0.0, 0.0],
        [0.0, 8.0, 0.0],
    ])
    edge_index = torch.tensor([
        [0, 0, 1, 2],
        [1, 3, 3, 1],
    ])
    edge_attr = torch.zeros(edge_index.shape[1], 18)
    rho = torch.full((edge_index.shape[1],), 0.5)

    refined, d = m(z, edge_index, edge_attr, rho)
    assert bool(d["candidate_mask"][0])
    assert bool(d["candidate_mask"][1])
    assert not bool(d["candidate_mask"][2])
    assert torch.allclose(refined[2], z[2])


def test_beta_normalizes_per_candidate_source():
    torch.manual_seed(0)
    m = LocalPocketRefinement(
        num_classes=3, edge_dim=18, hidden_dim=32,
        tau_u=0.0, tau_d=0.0, lambda_p=1.0)
    z = torch.randn(4, 3)
    edge_index = torch.tensor([[0, 0, 1, 1], [1, 2, 2, 3]])
    edge_attr = torch.randn(4, 18)
    rho = torch.tensor([0.2, 0.9, 0.4, 0.6])

    _, d = m(z, edge_index, edge_attr, rho)
    pe = d["pocket_edge_index"]
    src = edge_index[0, pe]
    beta = d["beta"]
    for s in src.unique():
        assert torch.allclose(beta[src == s].sum(), torch.tensor(1.0), atol=1e-6)


def test_reliability_term_changes_beta_in_expected_direction():
    m = LocalPocketRefinement(
        num_classes=3, edge_dim=18, hidden_dim=32,
        tau_u=0.0, tau_d=0.0, lambda_p=1.0)
    # Make learned phi identical for every edge so only rho controls beta.
    for p in m.score.parameters():
        torch.nn.init.zeros_(p)

    z = torch.zeros(3, 3)
    edge_index = torch.tensor([[0, 0], [1, 2]])
    edge_attr = torch.zeros(2, 18)
    rho = torch.tensor([0.2, 0.8])

    _, d = m(z, edge_index, edge_attr, rho)
    beta = d["beta"]
    assert beta[1] > beta[0]
    assert torch.allclose(beta, torch.tensor([0.2, 0.8]), atol=1e-6)


def test_gradients_flow_through_refinement():
    torch.manual_seed(1)
    m = LocalPocketRefinement(
        num_classes=3, edge_dim=18, hidden_dim=16,
        tau_u=0.0, tau_d=0.0, lambda_p=1.0)
    z = torch.randn(4, 3, requires_grad=True)
    edge_index = torch.tensor([[0, 0, 1, 2], [1, 2, 2, 3]])
    edge_attr = torch.randn(4, 18)
    rho = torch.tensor([0.3, 0.8, 0.5, 0.9], requires_grad=True)

    refined, _ = m(z, edge_index, edge_attr, rho)
    refined.square().mean().backward()

    assert z.grad is not None
    assert rho.grad is not None
    assert m.residual_projection.weight.grad is not None
    assert all(p.grad is not None for p in m.score.parameters())


def test_empty_graph_is_supported():
    m = LocalPocketRefinement(
        num_classes=3, edge_dim=18, hidden_dim=16,
        tau_u=0.6, tau_d=0.6, lambda_p=1.0)
    z = torch.tensor([[8.0, 0.0, 0.0], [0.0, 8.0, 0.0]])
    edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.empty((0, 18))
    rho = torch.empty((0,))

    refined, diagnostics = m(z, edge_index, edge_attr, rho)
    assert torch.allclose(refined, z)
    assert diagnostics["pocket_edge_index"].numel() == 0


def test_self_loops_are_excluded_from_local_pockets():
    m = LocalPocketRefinement(
        num_classes=3, edge_dim=18, hidden_dim=16,
        tau_u=0.0, tau_d=0.0, lambda_p=1.0)
    z = torch.tensor([[8.0, 0.0, 0.0], [0.0, 8.0, 0.0]])
    # Edge 0 is a self-loop at node 0; edge 1 is its actual one-hop neighbor.
    edge_index = torch.tensor([[0, 0], [0, 1]])
    edge_attr = torch.zeros(2, 18)
    rho = torch.tensor([1.0, 0.5])

    _, diagnostics = m(z, edge_index, edge_attr, rho)
    assert diagnostics["pocket_edge_index"].tolist() == [1]
    assert torch.allclose(diagnostics["disagreement"][0], torch.tensor(1.0))
