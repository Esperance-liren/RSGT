# RSGT: Reliability-aware Superpoint Graph Learning

PyTorch implementation of **Reliability-aware Superpoint Graph Learning for Large-scale Point Cloud Semantic Segmentation: Evaluation on Indoor, Mobile, and Airborne Benchmarks**.

RSGT extends [Superpoint Transformer (SPT)](https://github.com/drprojects/superpoint_transformer) with two lightweight components:

- **Reliability-aware graph propagation:** bounded, head-specific edge modulation before attention softmax.
- **Decoder-side local pocket refinement:** selective one-hop refinement for uncertain or neighborhood-inconsistent superpoints using reliability-weighted context.

## Environment

The implementation uses Python 3.8, PyTorch 2.2.0, and CUDA 11.8.

```bash
bash install.sh
```

Datasets are expected under `data/` by default. The SPT preprocessing and dataset organization are retained.

## Training

```bash
python src/train.py experiment=semantic/rsgt_s3dis
python src/train.py experiment=semantic/rsgt_kitti360
python src/train.py experiment=semantic/rsgt_dales
```

The S3DIS configuration defaults to Area 5. For 6-fold evaluation, run the six folds with the corresponding `datamodule.fold` override.

## Evaluation

```bash
python src/eval.py experiment=semantic/rsgt_s3dis ckpt_path=/path/to/checkpoint.ckpt
```

Use the corresponding KITTI-360 or DALES experiment configuration for the other datasets.

## Main implementation files

- `src/nn/attention.py` — reliability-aware attention modulation
- `src/models/components/rsgt.py` — local pocket refinement
- `src/models/semantic.py` — RSGT integration and training objective
- `configs/model/semantic/rsgt.yaml` — RSGT model configuration

## Acknowledgement

RSGT is implemented on top of the SPT codebase. The original MIT license and attribution are retained in `LICENSE` and `THIRD_PARTY_NOTICES.md`.

## Citation

Please cite the associated RSGT paper and SPT if you use this code. Bibliographic metadata is provided in `CITATION.cff`.
