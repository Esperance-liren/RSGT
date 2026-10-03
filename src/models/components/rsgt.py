"""RSGT-specific model components.

This module implements the decoder-side local pocket refinement used by RSGT.
It is self-contained and uses native PyTorch tensor operations.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn


__all__ = ["LocalPocketRefinement"]


class LocalPocketRefinement(nn.Module):
    """Selective reliability-weighted local correction on coarse logits.

    Parameters
    ----------
    num_classes:
        Number of semantic classes ``C``.
    edge_dim:
        Dimension of the raw inter-superpoint adjacency descriptor. RSGT uses
        18 handcrafted edge attributes.
    hidden_dim:
        Hidden width of the learned pocket scoring function ``phi``. The default
        RSGT configuration uses 32.
    tau_u, tau_d:
        Uncertainty and neighborhood-disagreement thresholds used by the
        candidate rule ``u_i > tau_u OR d_i > tau_d``.
    lambda_p:
        Weight applied to ``log(rho_ij + eps)`` inside the normalized pocket
        score.
    eps:
        Numerical-stability constant used before ``log`` and division.

    Notes
    -----
    The trainable layers are:

    * ``W_p``: bias-free ``C -> C`` residual projection;
    * ``phi``: ``(2C + edge_dim) -> hidden_dim -> 1`` MLP with ReLU.

    For S3DIS (C=13, edge_dim=18, hidden_dim=32), this module has 1,642
    trainable parameters.
    """

    def __init__(
        self,
        num_classes: int,
        edge_dim: int = 18,
        hidden_dim: int = 32,
        tau_u: float = 0.6,
        tau_d: float = 0.6,
        lambda_p: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.edge_dim = int(edge_dim)
        self.hidden_dim = int(hidden_dim)
        self.tau_u = float(tau_u)
        self.tau_d = float(tau_d)
        self.lambda_p = float(lambda_p)
        self.eps = float(eps)

        if self.num_classes <= 1:
            raise ValueError("num_classes must be > 1")
        if self.edge_dim <= 0:
            raise ValueError("edge_dim must be > 0")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")
        if not (0.0 <= self.tau_u <= 1.0):
            raise ValueError("tau_u must lie in [0, 1]")
        if not (0.0 <= self.tau_d <= 1.0):
            raise ValueError("tau_d must lie in [0, 1]")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")

        self.residual_projection = nn.Linear(
            self.num_classes, self.num_classes, bias=False)
        self.score = nn.Sequential(
            nn.Linear(2 * self.num_classes + self.edge_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    @staticmethod
    def _segment_softmax(
        scores: torch.Tensor,
        index: torch.Tensor,
        num_segments: int,
        eps: float,
    ) -> torch.Tensor:
        """Softmax over edges sharing the same source-node index.

        Implemented with native PyTorch so this component does not depend on
        torch-scatter or torch-geometric.
        """
        if scores.numel() == 0:
            return scores
        if scores.dim() != 1 or index.dim() != 1:
            raise ValueError("scores and index must be one-dimensional")
        if scores.shape[0] != index.shape[0]:
            raise ValueError("scores and index must have the same length")

        max_per_segment = torch.full(
            (num_segments,),
            -torch.inf,
            device=scores.device,
            dtype=scores.dtype,
        )
        max_per_segment.scatter_reduce_(
            0, index, scores, reduce="amax", include_self=True)
        stabilized = scores - max_per_segment[index]
        weights = stabilized.exp()

        denom = torch.zeros(
            (num_segments,), device=scores.device, dtype=scores.dtype)
        denom.scatter_add_(0, index, weights)
        return weights / denom[index].clamp_min(eps)

    def forward(
        self,
        coarse_logits: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_reliability: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Refine selected superpoint logits.

        Parameters
        ----------
        coarse_logits:
            ``[N, C]`` decoder logits ``z_i`` before local refinement.
        edge_index:
            Directed level-1 superpoint graph, ``[2, E]``.  Row 0 is the source
            (candidate center) and row 1 is the neighboring target, consistent
            with the SPT attention convention.
        edge_attr:
            Raw 18-D adjacency descriptors ``e_ij``, shape ``[E, edge_dim]``.
        edge_reliability:
            Normalized reliability coefficient ``rho_ij`` for each directed
            edge, shape ``[E]`` or ``[E, H]``.  Head-wise values are averaged
            when the latter is provided.

        Returns
        -------
        refined_logits, diagnostics
            ``refined_logits`` has shape ``[N, C]``. ``diagnostics`` contains
            uncertainty, disagreement, selected edges, weights, and residuals.
        """
        if coarse_logits.dim() != 2:
            raise ValueError("coarse_logits must have shape [N, C]")
        n, c = coarse_logits.shape
        if c != self.num_classes:
            raise ValueError(
                f"expected {self.num_classes} classes, received {c}")
        if edge_index.dim() != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")
        e = edge_index.shape[1]
        if edge_attr.dim() != 2 or edge_attr.shape[0] != e:
            raise ValueError("edge_attr must have shape [E, edge_dim]")
        if edge_attr.shape[1] != self.edge_dim:
            raise ValueError(
                f"expected raw edge dimension {self.edge_dim}, "
                f"received {edge_attr.shape[1]}")

        if edge_reliability.dim() == 2:
            if edge_reliability.shape[0] != e:
                raise ValueError("edge_reliability first dimension must equal E")
            rho = edge_reliability.mean(dim=1)
        elif edge_reliability.dim() == 1:
            if edge_reliability.shape[0] != e:
                raise ValueError("edge_reliability length must equal E")
            rho = edge_reliability
        else:
            raise ValueError("edge_reliability must have shape [E] or [E, H]")
        rho = rho.to(device=coarse_logits.device, dtype=coarse_logits.dtype)
        rho = rho.clamp(0.0, 1.0)

        src = edge_index[0].long()
        dst = edge_index[1].long()
        # SPT adds self-loops for attention. Local pockets, however, are
        # defined over neighboring superpoints, so self-loops are excluded
        # from disagreement statistics and residual refinement.
        non_self = src != dst

        # Eq. (7)--(8): class probabilities and uncertainty.
        prob = coarse_logits.softmax(dim=1)
        uncertainty = 1.0 - prob.max(dim=1).values

        # Eq. (9): source-wise fraction of neighbors whose coarse prediction
        # disagrees with the center prediction.
        pred = coarse_logits.argmax(dim=1)
        disagreement = coarse_logits.new_zeros(n)
        degree = coarse_logits.new_zeros(n)
        if e > 0:
            ns_src = src[non_self]
            ns_dst = dst[non_self]
            edge_disagree = (pred[ns_src] != pred[ns_dst]).to(coarse_logits.dtype)
            disagreement.scatter_add_(0, ns_src, edge_disagree)
            degree.scatter_add_(0, ns_src, torch.ones_like(edge_disagree))
            disagreement = disagreement / degree.clamp_min(1.0)

        # Eq. (10): selective candidate rule.
        candidate_mask = (uncertainty > self.tau_u) | (disagreement > self.tau_d)
        candidate_index = torch.nonzero(candidate_mask, as_tuple=False).flatten()

        # Eq. (11)--(12): one-hop pocket and reliability-weighted normalized
        # score.  We select directed outgoing edges whose source is a candidate.
        if e > 0:
            pocket_edge_mask = candidate_mask[src] & non_self
            pocket_edge_index = torch.nonzero(
                pocket_edge_mask, as_tuple=False).flatten()
        else:
            pocket_edge_index = edge_index.new_empty((0,))

        beta = coarse_logits.new_empty((pocket_edge_index.numel(),))
        delta = torch.zeros_like(coarse_logits)

        if pocket_edge_index.numel() > 0:
            p_src = src[pocket_edge_index]
            p_dst = dst[pocket_edge_index]
            p_edge_attr = edge_attr[pocket_edge_index].to(
                device=coarse_logits.device, dtype=coarse_logits.dtype)
            p_rho = rho[pocket_edge_index]

            score_input = torch.cat(
                (coarse_logits[p_src], coarse_logits[p_dst], p_edge_attr), dim=1)
            learned_score = self.score(score_input).squeeze(1)
            reliability_term = self.lambda_p * torch.log(
                p_rho.clamp_min(self.eps))
            score = learned_score + reliability_term
            beta = self._segment_softmax(score, p_src, n, self.eps)

            residual = self.residual_projection(
                coarse_logits[p_dst] - coarse_logits[p_src])
            delta.index_add_(0, p_src, beta.unsqueeze(1) * residual)

        # Eq. (13): update selected candidates only.
        refined_logits = coarse_logits.clone()
        refined_logits[candidate_mask] = (
            coarse_logits[candidate_mask] + delta[candidate_mask])

        diagnostics = {
            "uncertainty": uncertainty,
            "disagreement": disagreement,
            "candidate_mask": candidate_mask,
            "pocket_index": candidate_index,
            "pocket_edge_index": pocket_edge_index,
            "pocket_weight": beta,
            "beta": beta,
            "reliability": rho,
            "edge_reliability": rho,
            "delta_logits": delta,
        }
        return refined_logits, diagnostics
