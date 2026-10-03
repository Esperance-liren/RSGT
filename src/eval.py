import pyrootutils

root = str(pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "README.md"],
    pythonpath=True,
    dotenv=True))

# ------------------------------------------------------------------------------------ #
# `pyrootutils.setup_root(...)` is an optional line at the top of each entry file
# that helps to make the environment more robust and convenient
#
# you can remove it if you prefer to manage PYTHONPATH & env manually
# ------------------------------------------------------------------------------------ #

# Hack importing pandas here to bypass some conflicts with hydra
import pandas as pd  # noqa: F401

from typing import List, Tuple
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import Logger

from src import utils

# Registering the "eval" resolver allows for advanced config interpolation
if not OmegaConf.has_resolver("eval"):
    OmegaConf.register_new_resolver("eval", eval)

log = utils.get_pylogger(__name__)


def compute_superpoint_uncertainty_and_save(
    predictions: List,
    save_dir: Path,
    alpha: float = 0.5,
    superpoint_level: int = 1,
) -> None:
    """
    Given a list of (NAG, SemanticSegmentationOutput) predictions,
    compute superpoint-level uncertainty scores and save them to disk.

    Each saved file corresponds to one graph (e.g. one room / scene)
    at the chosen superpoint level and contains:
        - u: combined uncertainty (alpha * u1 + (1-alpha) * u2)
        - u1: 1 - max softmax probability
        - u2: 1 - neighbor agreement ratio
        - pred_labels: predicted semantic label per superpoint
    """

    save_dir.mkdir(parents=True, exist_ok=True)

    sample_counter = 0

    for batch_idx, item in enumerate(predictions):
        # SemanticSegmentationModule.predict_step usually returns (nag, output)
        if isinstance(item, (list, tuple)) and len(item) == 2:
            nag, output = item
        else:
            raise RuntimeError(
                "Expected each prediction item to be (NAG, SemanticSegmentationOutput), "
                f"but got type: {type(item)}"
            )

        # --- 1. 取 logits 并计算 softmax 置信度 ---
        if not hasattr(output, "logits"):
            raise RuntimeError("Output object has no attribute 'logits'.")

        logits = output.logits

        # 多阶段输出：通常第 0 个是 P1（细层 superpoint）
        if isinstance(logits, (list, tuple)):
            if len(logits) == 0:
                raise RuntimeError("Output.logits is an empty list/tuple.")
            logits = logits[0]

        # logits: (num_superpoints, num_classes)
        device = logits.device
        probs = torch.softmax(logits, dim=1)
        max_probs, pred_labels = torch.max(probs, dim=1)
        u1 = 1.0 - max_probs  # 置信度型不确定性

        # --- 2. 取 superpoint 图结构（edge_index, batch） ---
        try:
            nag_level = nag[superpoint_level]
        except Exception as e:
            raise RuntimeError(
                f"Failed to access nag level {superpoint_level} from NAG list/tuple. "
                f"Got type: {type(nag)}"
            ) from e

        edge_index = getattr(nag_level, "edge_index", None)
        batch_vec = getattr(nag_level, "batch", None)

        num_nodes = logits.shape[0]

        # 搬到同一 device
        if edge_index is not None:
            edge_index = edge_index.to(device)
        if batch_vec is not None:
            batch_vec = batch_vec.to(device)

        # --- 3. 计算邻居一致性 u2（带安全检查） ---
        if edge_index is None or edge_index.numel() == 0:
            neighbor_ratio = torch.ones(num_nodes, device=device)
        else:
            # 安全检查：防止 edge_index 里的 index 超过节点数
            max_idx = int(edge_index.max().item())
            min_idx = int(edge_index.min().item())
            if max_idx >= num_nodes or min_idx < 0:
                # 图和 logits 不匹配，直接放弃邻居信息，避免越界
                log.warning(
                    f"[Uncertainty] edge_index has out-of-range index "
                    f"(min={min_idx}, max={max_idx}, num_nodes={num_nodes}). "
                    f"Skip neighbor-based uncertainty for this batch."
                )
                neighbor_ratio = torch.ones(num_nodes, device=device)
            else:
                # 将图视作无向：镜像边
                src = edge_index[0]  # (E,)
                dst = edge_index[1]  # (E,)

                same = (pred_labels[src] == pred_labels[dst]).float()  # (E,)

                src_full = torch.cat([src, dst], dim=0)       # (2E,)
                dst_full = torch.cat([dst, src], dim=0)       # (2E,)
                same_full = torch.cat([same, same], dim=0)    # (2E,)

                neighbor_same = torch.zeros(num_nodes, device=device)
                neighbor_total = torch.zeros(num_nodes, device=device)

                neighbor_same.scatter_add_(0, dst_full, same_full)
                ones = torch.ones_like(same_full)
                neighbor_total.scatter_add_(0, dst_full, ones)

                # 没有邻居的点，ratio 设为 1.0（即完全确定）
                neighbor_ratio = torch.where(
                    neighbor_total > 0,
                    neighbor_same / (neighbor_total + 1e-8),
                    torch.ones_like(neighbor_same),
                )

        u2 = 1.0 - neighbor_ratio

        # --- 4. 综合不确定性 ---
        u = alpha * u1 + (1.0 - alpha) * u2

        # --- 5. 如果一个 batch 里有多个图，用 batch 向量拆分 ---
        if batch_vec is None:
            sample_path = save_dir / f"sample_{sample_counter:06d}.pt"
            torch.save(
                {
                    "u": u.detach().cpu(),
                    "u1": u1.detach().cpu(),
                    "u2": u2.detach().cpu(),
                    "pred_labels": pred_labels.detach().cpu(),
                },
                sample_path,
            )
            sample_counter += 1
        else:
            num_graphs = int(batch_vec.max().item()) + 1
            for g in range(num_graphs):
                mask = batch_vec == g
                if mask.sum() == 0:
                    continue

                sample_path = save_dir / f"sample_{sample_counter:06d}.pt"
                torch.save(
                    {
                        "u": u[mask].detach().cpu(),
                        "u1": u1[mask].detach().cpu(),
                        "u2": u2[mask].detach().cpu(),
                        "pred_labels": pred_labels[mask].detach().cpu(),
                    },
                    sample_path,
                )
                sample_counter += 1



@utils.task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[dict, dict]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    assert cfg.ckpt_path

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)
    if float(".".join(torch.__version__.split(".")[:2])) >= 2.0:
        torch.set_float32_matmul_precision(cfg.float32_matmul_precision)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    if cfg.get("compile"):
        log.info("Compiling model!")
        model = torch.compile(model, dynamic=True)

    log.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    # === 新增：计算并保存 superpoint 不确定性（可选） ===
    if cfg.get("dump_uncertainty", False):
        log.info("Computing and saving superpoint uncertainties on the test set...")
        alpha = float(cfg.get("uncertainty_alpha", 0.5))
        out_dir = cfg.get("uncertainty_dir", "uncertainty")
        save_dir = Path(out_dir)
        log.info(f"Uncertainty save directory: {save_dir}")
        predictions = trainer.predict(
            model=model,
            datamodule=datamodule,
            ckpt_path=cfg.ckpt_path,
        )
        compute_superpoint_uncertainty_and_save(
            predictions=predictions,
            save_dir=save_dir,
            alpha=alpha,
        )

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.2", config_path=root + "/configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    evaluate(cfg)


if __name__ == "__main__":
    main()

