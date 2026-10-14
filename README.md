# MinSum

Pipeline code of MinSum: cluster-adaptive verification thresholds for reducing
demographic bias in face recognition.

A single global threshold for a face verifier gives different error rates for
different demographic groups. MinSum instead:

1. clusters face images using a **clustering embedding** (DeepFace or FairFace),
   without using demographic labels;
2. builds same-person and different-person pairs and scores them with a
   **verification embedding** (ArcFace, AdaFace, MagFace, ElasticFace, TransFace,
   DFace or dlib);
3. calibrates one threshold **per cluster** on the train pairs by minimising
   `alpha * FPR + (1 - alpha) * FNR` (the weighted minimum-sum objective), alongside a fixed global
   threshold as a baseline;
4. assigns validation images to the nearest cluster centroid and evaluates both
   threshold sets on validation pairs, overall, per cluster and per ethnicity.

## Repository layout

```text
minsum/
  Core/                     shared building blocks: output path layout, column schemas,
                            artifact IO, embedding store, extractor base class, validators
  Datasets/                 dataset parsers that turn image folders into metadata tables
                            (RFW filename parser, generic folder-per-identity parser) and
                            identity-disjoint train/val/test splitting
  Preprocessing/
    clustering_embeddings/  DeepFace and FairFace extractors used for clustering
    clustering/             KMeans fitting, cosine centroid assignment, cluster reports
  VerificationEmbeddings/   one extractor per verification model: arcface/, adaface/,
                            magface/, elasticface/, transface/, dface_deepface/,
                            face_recognition/ (dlib); backbones are vendored where needed
  Postprocessing/
    pairs/                  same-person / different-person pair generation
    distances/              cosine and euclidean pair distances
    thresholds/             fixed and minimum-sum threshold selection per cluster
    evaluation/             applying thresholds and computing FPR, FNR, accuracy, etc.
  Experiments/              numbered command-line scripts, one per pipeline step, plus
                            RUNNING.md / BUPT_RUNNING.md with full command examples
  run_full_pipeline.py      runs the whole pipeline from one config file
  utils_full_pipeline.py    config parsing, validation and step planning for the runner
  cfg_full_pipeline.yaml    config for run_full_pipeline.py, initialised to the step defaults
```

## Setup

Requirements: [uv](https://docs.astral.sh/uv/) and Python 3.13.

```bash
git clone git@github.com:giovannimicaroni/MinSum.git
cd MinSum
uv sync                       # base: metadata, clustering, pairs, thresholds, evaluation
uv sync --extra deepface      # add the extras for the extractors you will use
source .venv/bin/activate
```

The embedding extractors are optional extras, so you only install what you run
(`uv sync --extra all` installs everything):

| Extra | Steps | Notes |
|---|---|---|
| `deepface` | 02a DeepFace clustering, 04d DFace verification | TensorFlow |
| `torch` | 02b FairFace, 04b AdaFace, 04e MagFace, 04f ElasticFace, 04g TransFace | PyTorch |
| `insightface` | 04a ArcFace | builds from source; needs a C++ compiler |
| `dlib` | 04c dlib face_recognition | builds from source; needs cmake and a C++ compiler |

**DFace (04d) cannot run in this environment.** The DeepFace model uses a
TensorFlow layer that was removed after TF 2.12, and TF 2.12 only supports
Python <= 3.11 and `numpy<1.24`. Run it from a separate environment and point the
runner at it with `verification_embedding.python` (see below).

## What you need to provide

- **A dataset of face crops** under one root folder, in one of these layouts:
  - `parser: rfw`: `{dataset_root}/train/` and `{dataset_root}/val/`, with files
    named `{ethnicity}_{identity}_{imageNumber}.jpg` (e.g. `0_00012_03.jpg`);
  - `parser: generic-folder`: `{dataset_root}/{split}/{person_id}/image.jpg`;
  - an unsplit tree with `split_by_identity: true`: the metadata step creates
    identity-disjoint train/val/test splits without moving any images.
- **Model weights** for the extractors that need them (`weights_path`): FairFace,
  AdaFace, MagFace, ElasticFace and TransFace checkpoints.
- **Internet access on first use**: DeepFace downloads its weights to
  `~/.deepface`, and InsightFace in `detect` mode downloads the `buffalo_l` pack
  to `~/.insightface/models`. On machines without internet, download these
  beforehand (for InsightFace, pass the pack folder as `model_root`).

## Running the full pipeline

1. Copy the config and edit the copy:

   ```bash
   cp minsum/cfg_full_pipeline.yaml my_run.yaml
   ```

   Fill in at least:
   - `run.project_root`: where outputs are written;
   - `run.dataset_name` and `metadata.dataset_root` / `metadata.parser`;
   - `clustering_embedding.extractor` and `verification_embedding.extractor`,
     plus that extractor's `weights_path` if it needs one;
   - the experiment settings, e.g. `clustering.n_clusters` and the
     `postprocessing` thresholds.

   Every other value is already the default of the step it feeds.

2. Check the config without running anything:

   ```bash
   python -m minsum.run_full_pipeline --config my_run.yaml --dry-run
   ```

   This validates the config (required values, input paths, and every option
   against the step scripts) and prints the exact command for each step.

3. Run it:

   ```bash
   python -m minsum.run_full_pipeline --config my_run.yaml
   ```

   Steps run in order: metadata (01), clustering embeddings (02), clustering
   (03), verification embeddings (04), postprocessing (05). The runner passes
   names between steps (the clustering embedding, cluster run and verification
   embedding names), so they never have to match by hand. It stops at the first
   failing step.

To re-run only some steps on existing outputs, for example after changing the
number of clusters or the threshold settings:

```bash
python -m minsum.run_full_pipeline --config my_run.yaml --steps clustering,postprocessing
```

Each step runs in its own process. To run a step with another interpreter, set
`python` in its section (for example a `tensorflow<=2.12` env for DFace). Every
run saves the resolved config and the exact commands to
`{project_root}/outputs/full_runs/{dataset}/{timestamp}/`.

## Running steps individually

Each step is also a standalone script; use `--help` on any of them:

```text
01  run_01_build_metadata                         build metadata
02a run_02a_extract_deepface_clustering           DeepFace clustering embeddings
02b run_02b_extract_fairface_clustering           FairFace clustering embeddings
03  run_03_fit_clusters                           fit KMeans on train, assign val
04a run_04a_extract_arcface_verification          ArcFace / InsightFace
04b run_04b_extract_adaface_verification          AdaFace
04c run_04c_extract_face_recognition_verification dlib / face_recognition_models
04d run_04d_extract_dface_verification            DFace / DeepFace
04e run_04e_extract_magface_verification          MagFace
04f run_04f_extract_elasticface_verification      ElasticFace
04g run_04g_extract_transface_verification        TransFace
05  run_05_postprocessing                         pairs, distances, thresholds, evaluation
06  run_06_eval_existing_thresholds               evaluate saved thresholds on a new or shared val set
07  run_07_analyze_impostor_similarity            intra-cluster impostor similarity
08  run_08_compare_fixed_sweeps                   fixed-threshold sweeps vs minimum-sum results
```

```bash
python -m minsum.Experiments.run_03_fit_clusters --help
```

For large datasets, extraction steps accept `--shard-index` / `--num-shards`
(for SLURM job arrays); combine the shards with
`python -m minsum.Experiments.run_merge_embedding_shards`. Full command examples
are in `minsum/Experiments/RUNNING.md`, and the BUPT SLURM workflow is in
`minsum/Experiments/BUPT_RUNNING.md`.

## Outputs

Everything is written under `project_root`:

```text
outputs/
  metadata/{dataset}/                                   per-split image metadata
  embeddings/{dataset}/clustering/{embedding}/          clustering embeddings
  embeddings/{dataset}/verification/{embedding}/        verification embeddings
  clusters/{dataset}/{cluster_embedding}/{run_name}/    KMeans model, train clusters, val assignment
  pairs/{dataset}/{cluster_embedding}_clusters/{run_name}/
  experiments/{dataset}/{cluster_embedding}_cluster__{verification_embedding}_verify/{run_name}/
  full_runs/{dataset}/{timestamp}/                      config + commands of each full run
```

Each experiment folder holds the pairs with distances and one subfolder per
threshold method (`fixed/`, `minimum_sum/`, `minimum_sum_fpr3/`, ...), each
containing `thresholds_by_cluster.csv` and train/val metrics overall
(`*_metrics_global.csv`), per cluster and per ethnicity.

## Notes

- **dlib baseline (04c):** for canonical dlib behaviour, extract with
  `normalize: false` and evaluate with `distance_metric: euclidean` and a fixed
  threshold around `0.60`.
- **DFace (04d):** calls `DeepFace.represent(model_name="DeepFace")`. In the
  paper, describe the verifier as DFace/DeepFace, extracted with the DeepFace
  Python package.
