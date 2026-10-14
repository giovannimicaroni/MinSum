# Running the Modular Pipeline

Run commands from the repository root:

```bash
cd MinSum
uv sync --extra all   # or pick extras: deepface, torch, insightface, dlib
source .venv/bin/activate
```

Use `python -m ... --help` on any step to inspect all options.

To run every step in one go, configure `minsum/cfg_full_pipeline.yaml` and run
`python -m minsum.run_full_pipeline --config <file>` (see the root README). The
per-step commands below are what it runs.


## Optional Extraction Dependencies

The base project dependencies cover clustering, pairs, thresholds, and metrics.
Embedding extraction needs model-specific packages, declared as extras in
`pyproject.toml`:

| Extra | Steps |
|---|---|
| `deepface` | 02a DeepFace clustering, 04d DFace verification |
| `torch` | 02b FairFace, 04b AdaFace, 04e MagFace, 04f ElasticFace, 04g TransFace |
| `insightface` | 04a ArcFace (builds from source; needs a C++ compiler) |
| `dlib` | 04c dlib face_recognition (builds from source; needs cmake and a C++ compiler) |

On Linux the `deepface` and `insightface` extras install `onnxruntime-gpu`,
which also runs on CPU. If GPU setup fails, run InsightFace with `--ctx-id -1`
for CPU.

04d (DFace) needs `tensorflow<=2.12`, because the DeepFace model uses
`LocallyConnected2D`, which newer TensorFlow removed. TF 2.12 supports only
Python <= 3.11 and `numpy<1.24`, so run 04d from a separate environment.

## 01 Build Metadata

```bash
python -m minsum.Experiments.run_01_build_metadata \
  --project-root /home/emily.tomadon/dataset \
  --dataset-root /home/emily.tomadon/dataset/datasets/rfw_split/RFWsummarized_split \
  --dataset-name rfw \
  --parser rfw
```

Outputs:

```text
outputs/metadata/rfw/train_metadata.csv
outputs/metadata/rfw/val_metadata.csv
outputs/metadata/rfw/all_metadata.csv
```

## 02a DeepFace Clustering Embeddings

```bash
python -m minsum.Experiments.run_02a_extract_deepface_clustering \
  --project-root /home/emily.tomadon/dataset \
  --dataset-name rfw \
  --embedding-name deepface_arcface \
  --model-name ArcFace \
  --detector-backend skip \
  --normalization base
```

## 02b FairFace Clustering Embeddings

```bash
python -m minsum.Experiments.run_02b_extract_fairface_clustering \
  --project-root /home/emily.tomadon/dataset \
  --dataset-name rfw \
  --embedding-name fairface_penultimate \
  --weights-path /home/emily.tomadon/dataset/models/fairface/res34_fair_align_multi_7_20190809.pt
```

## 03 Fit KMeans and Assign Validation

DeepFace clusters:

```bash
python -m minsum.Experiments.run_03_fit_clusters \
  --project-root /home/emily.tomadon/dataset \
  --dataset-name rfw \
  --cluster-embedding deepface_arcface \
  --n-clusters 15
```

FairFace clusters:

```bash
python -m minsum.Experiments.run_03_fit_clusters \
  --project-root /home/emily.tomadon/dataset \
  --dataset-name rfw \
  --cluster-embedding fairface_penultimate \
  --n-clusters 15
```

Default run name is `k15_cosine_centroid_no_pca`.

## 04a ArcFace / InsightFace Verification Embeddings

Fast path for aligned/cropped datasets such as RFW:

```bash
python -m minsum.Experiments.run_04a_extract_arcface_verification \
  --project-root /home/emily.tomadon/dataset \
  --dataset-name rfw \
  --embedding-name arcface_prealigned \
  --model-name buffalo_l \
  --ctx-id -1 \
  --prealigned
```

This skips the InsightFace detector and feeds resized image crops directly to the ArcFace recognition model from the `buffalo_l` model pack. Use this when the dataset images are already face-cropped/aligned.

Detector-based path:

```bash
python -m minsum.Experiments.run_04a_extract_arcface_verification \
  --project-root /home/emily.tomadon/dataset \
  --dataset-name rfw \
  --embedding-name arcface \
  --model-name buffalo_l \
  --ctx-id 0
```

Use `--ctx-id -1` for CPU fallback.


## 04b AdaFace Verification Embeddings

Download an official AdaFace checkpoint from the
[AdaFace pretrained-model table](https://github.com/mk-minchul/AdaFace#pretrained-models).
The architecture passed on the command line must match the checkpoint. AdaFace expects
aligned face crops, uses BGR channel order, and resizes inputs to `112x112`.

```bash
python -m minsum.Experiments.run_04b_extract_adaface_verification \
  --project-root /home/emily.tomadon/dataset \
  --dataset-name rfw \
  --embedding-name adaface_ir50_ms1mv2 \
  --weights-path /home/emily.tomadon/dataset/models/adaface/adaface_ir50_ms1mv2.ckpt \
  --architecture ir_50 \
  --device cuda
```

Omit `--device` to select CUDA, MPS, or CPU automatically. The command supports the
same `--splits`, `--checkpoint-every`, `--shard-index`, and `--num-shards` options as
the other embedding extraction steps.

## 04d DFace / DeepFace Verification Embeddings

Use this step when the DeepFace model is the verification backbone. The DeepFace
Python package is the extraction framework used to call
`DeepFace.represent(model_name="DeepFace")`.
For cropped/aligned datasets such as RFW, keep detector skipping enabled:

```bash
python -m minsum.Experiments.run_04d_extract_dface_verification \
  --project-root /home/emily.tomadon/dataset \
  --dataset-name rfw \
  --embedding-name dface_deepface \
  --splits train,val \
  --model-name DeepFace \
  --detector-backend skip \
  --normalization base \
  --checkpoint-every 1000
```

The command supports `--shard-index` and `--num-shards`, so it can be used with
the same batch-array pattern as the other embedding extractors.

## 04f ElasticFace Verification Embeddings

Use this step with an ElasticFace iResNet checkpoint. The extractor reuses the
iResNet backbone family, expects aligned `112x112` face crops, normalizes input
pixels to `[-1, 1]`, and saves L2-normalized embeddings.

```bash
python -m minsum.Experiments.run_04f_extract_elasticface_verification \
  --project-root /home/emily.tomadon/dataset \
  --dataset-name rfw \
  --embedding-name elasticface_iresnet100 \
  --weights-path /home/emily.tomadon/dataset/models/elasticface/elasticface_arcplus_iresnet100.pth \
  --architecture iresnet100 \
  --splits train,val \
  --checkpoint-every 1000
```

The command supports `--shard-index` and `--num-shards`, so it can be used with
the same batch-array pattern as the other embedding extractors.

## 04g TransFace Verification Embeddings

Use this step with an official TransFace ViT checkpoint. TransFace expects
aligned `112x112` face crops, converts images from BGR to RGB, normalizes input
pixels to `[-1, 1]`, and saves L2-normalized 512-D embeddings.

Start with `vit_s_dp005_mask_0` for the quickest smoke test. If you download a
larger checkpoint, make sure `--network` matches the model used by the authors:
`vit_b_dp005_mask_005` for TransFace-B or `vit_l_dp005_mask_005` for TransFace-L.

```bash
python -m minsum.Experiments.run_04g_extract_transface_verification \
  --project-root /home/emily.tomadon/dataset \
  --dataset-name rfw \
  --embedding-name transface_vit_s \
  --weights-path /home/emily.tomadon/dataset/models/transface/transface_vit_s.pt \
  --network vit_s_dp005_mask_0 \
  --splits train,val \
  --checkpoint-every 1000
```

The command supports `--shard-index` and `--num-shards`, so it can be used with
the same batch-array pattern as the other embedding extractors.

## 05 Postprocessing: Pairs, Distances, Thresholds, Evaluation

DeepFace cluster + ArcFace verification:

```bash
python -m minsum.Experiments.run_05_postprocessing \
  --project-root /home/emily.tomadon/dataset \
  --dataset-name rfw \
  --cluster-embedding deepface_arcface \
  --verification-embedding arcface_prealigned \
  --cluster-run-name k15_cosine_centroid_no_pca \
  --pairs-per-class 1000 \
  --fixed-threshold 0.45
```

FairFace cluster + ArcFace verification:

```bash
python -m minsum.Experiments.run_05_postprocessing \
  --project-root /home/emily.tomadon/dataset \
  --dataset-name rfw \
  --cluster-embedding fairface_penultimate \
  --verification-embedding arcface_prealigned \
  --cluster-run-name k15_cosine_centroid_no_pca \
  --pairs-per-class 1000 \
  --fixed-threshold 0.45
```

To evaluate DFace instead of ArcFace or AdaFace, keep the same clustering
arguments and replace only the verification embedding:

```bash
python -m minsum.Experiments.run_05_postprocessing \
  --project-root /home/emily.tomadon/dataset \
  --dataset-name rfw \
  --cluster-embedding deepface_arcface \
  --verification-embedding dface_deepface \
  --cluster-run-name k30_cosine_centroid_no_pca \
  --pairs-per-class 1000 \
  --fixed-threshold 0.75 \
  --num-thresholds 200
```

Outputs:

```text
outputs/pairs/rfw/{cluster_embedding}_clusters/k15_cosine_centroid_no_pca/
outputs/experiments/rfw/{cluster_embedding}_cluster__arcface_prealigned_verify/k15_cosine_centroid_no_pca/
  fixed/
  minimum_sum/
```

The `minimum_sum` method tests `--num-thresholds` candidates, keeps all thresholds with minimum weighted objective, and chooses the median of those tied values. By default the objective is `FPR + FNR`. Use `--minimum-sum-fpr-weights 1,3,5` to also write `minimum_sum_fpr3/` and `minimum_sum_fpr5/`, optimizing `3*FPR + FNR` and `5*FPR + FNR`. `run_05_postprocessing` always runs `fixed` plus the requested minimum-sum variants.

`--pairs-per-class` sets the maximum number of pairs of each class for every train cluster and for the complete validation split. The generator always returns equal counts of same-person and different-person pairs within each generation scope. If fewer valid pairs are available, both classes are reduced to the same available count. The actual train and validation counts are saved in `pair_generation_summary.csv`.

Training pairs are generated within each cluster so that cluster-specific thresholds are learned only from same-cluster comparisons. Validation pairs are generated across the complete validation split and may cross clusters; the threshold associated with `cluster_a` is applied to each pair. Validation remains balanced between same-person and different-person pairs.
