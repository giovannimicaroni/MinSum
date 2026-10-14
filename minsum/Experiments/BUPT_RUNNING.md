# Running BUPT-CBFace-50

This guide runs BUPT-CBFace-50 without copying images. Identities are split
70/15/15 across train, validation, and test. Test is reserved and is not used
by the current train/validation pipeline.

This BUPT copy has no ethnicity annotation. The pipeline therefore produces
global and per-cluster metrics, but no metrics by ethnicity.

## Current status

Run 01 is complete:

~~~text
total: 500000 images, 10000 identities
train: 350000 images, 7000 identities
val:    75000 images, 1500 identities
test:   75000 images, 1500 identities (reserved)
~~~

Primary path:

~~~text
01 metadata (complete)
02a DeepFace shards -> merge
04a ArcFace shards -> merge
04d DFace shards -> merge
03 KMeans k30
05 pairs, distances, fixed 0.70, minimum_sum, validation metrics
~~~

The optional 02b FairFace path is included below.

## Rules

- Submit batch jobs from the login node, without opening salloc.
- Keep dataset-name as bupt_cbface50 so BUPT cannot overwrite RFW.
- Extract only train,val. Test stays untouched.
- Keep NUM_SHARDS unchanged between extraction and merge.
- A merge runs only after every corresponding shard succeeds.
- Prealigned ArcFace assumes cropped/aligned face images.

## 0. Sync and initialize

On the Mac:

~~~bash
cd MinSum
git add minsum README.md
git commit -m "Document and support BUPT batch pipeline"
git pull --rebase origin main
git push origin main
~~~

On the lab login node:

~~~bash
cd /home/emily.tomadon/repos/MinSum
git pull origin main
source /home/emily.tomadon/miniconda3/etc/profile.d/conda.sh
conda activate horus_griaule
mkdir -p logs

export REPO=/home/emily.tomadon/repos/MinSum
export PROJECT=/home/emily.tomadon/dataset
export DATASET=bupt_cbface50
export NUM_SHARDS=20
~~~

## 1. Metadata

~~~bash
export IMGROOT=/home/emily.tomadon/dataset/BUPT-CBFace/BUPT-CBFace-dropbox/BUPT-CBFace-50/BUPT-CBFace-50/images

python -m minsum.Experiments.run_01_build_metadata \
  --project-root "$PROJECT" \
  --dataset-root "$IMGROOT" \
  --dataset-name "$DATASET" \
  --parser generic-folder \
  --split-by-identity \
  --train-ratio 0.70 \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --random-seed 42
~~~

Verify:

~~~bash
META="$PROJECT/outputs/metadata/$DATASET"
column -s, -t < "$META/metadata_build_report.csv"
wc -l "$META/train_metadata.csv" "$META/val_metadata.csv" "$META/test_metadata.csv"
~~~

## 2a. DeepFace clustering embeddings

Submit 20 GPU shards, with at most two running simultaneously:

~~~bash
cd "$REPO"

JOB_DF=$(sbatch --parsable \
  --array=0-19%2 \
  --job-name=bupt_df \
  --output=logs/bupt_df_%A_%a.out \
  --error=logs/bupt_df_%A_%a.err \
  --time=12:00:00 --mem=20G --cpus-per-task=4 --gres=gpu:1 \
  --wrap="bash -lc 'source /home/emily.tomadon/miniconda3/etc/profile.d/conda.sh && conda activate horus_griaule && cd $REPO && export PYTHONUNBUFFERED=1 && python -m minsum.Experiments.run_02a_extract_deepface_clustering --project-root $PROJECT --dataset-name $DATASET --embedding-name deepface_arcface --splits train,val --model-name ArcFace --detector-backend skip --normalization base --checkpoint-every 0 --shard-index \$SLURM_ARRAY_TASK_ID --num-shards $NUM_SHARDS'")

echo "DeepFace array: $JOB_DF"
~~~

Merge after all shards:

~~~bash
JOB_DF_MERGE=$(sbatch --parsable \
  --dependency=afterok:$JOB_DF \
  --job-name=bupt_df_merge \
  --output=logs/bupt_df_merge_%j.out \
  --error=logs/bupt_df_merge_%j.err \
  --time=02:00:00 --mem=32G --cpus-per-task=4 \
  --wrap="bash -lc 'source /home/emily.tomadon/miniconda3/etc/profile.d/conda.sh && conda activate horus_griaule && cd $REPO && python -m minsum.Experiments.run_merge_embedding_shards --project-root $PROJECT --dataset-name $DATASET --purpose clustering --embedding-name deepface_arcface --num-shards $NUM_SHARDS --splits train,val'")

echo "DeepFace merge: $JOB_DF_MERGE"
~~~

## 2b. Optional FairFace clustering embeddings

~~~bash
JOB_FAIR=$(sbatch --parsable \
  --array=0-19%2 \
  --job-name=bupt_fair \
  --output=logs/bupt_fair_%A_%a.out \
  --error=logs/bupt_fair_%A_%a.err \
  --time=12:00:00 --mem=20G --cpus-per-task=4 --gres=gpu:1 \
  --wrap="bash -lc 'source /home/emily.tomadon/miniconda3/etc/profile.d/conda.sh && conda activate horus_griaule && cd $REPO && export PYTHONUNBUFFERED=1 && python -m minsum.Experiments.run_02b_extract_fairface_clustering --project-root $PROJECT --dataset-name $DATASET --embedding-name fairface_penultimate --weights-path $PROJECT/models/fairface/res34_fair_align_multi_7_20190809.pt --device cuda --splits train,val --checkpoint-every 0 --shard-index \$SLURM_ARRAY_TASK_ID --num-shards $NUM_SHARDS'")

JOB_FAIR_MERGE=$(sbatch --parsable \
  --dependency=afterok:$JOB_FAIR \
  --job-name=bupt_fair_merge \
  --output=logs/bupt_fair_merge_%j.out \
  --error=logs/bupt_fair_merge_%j.err \
  --time=02:00:00 --mem=32G --cpus-per-task=4 \
  --wrap="bash -lc 'source /home/emily.tomadon/miniconda3/etc/profile.d/conda.sh && conda activate horus_griaule && cd $REPO && python -m minsum.Experiments.run_merge_embedding_shards --project-root $PROJECT --dataset-name $DATASET --purpose clustering --embedding-name fairface_penultimate --num-shards $NUM_SHARDS --splits train,val'")

echo "FairFace: $JOB_FAIR | merge: $JOB_FAIR_MERGE"
~~~

## 4a. ArcFace verification embeddings

~~~bash
JOB_ARC=$(sbatch --parsable \
  --array=0-19%2 \
  --job-name=bupt_arc \
  --output=logs/bupt_arc_%A_%a.out \
  --error=logs/bupt_arc_%A_%a.err \
  --time=12:00:00 --mem=20G --cpus-per-task=4 --gres=gpu:1 \
  --wrap="bash -lc 'source /home/emily.tomadon/miniconda3/etc/profile.d/conda.sh && conda activate horus_griaule && cd $REPO && export PYTHONUNBUFFERED=1 && python -m minsum.Experiments.run_04a_extract_arcface_verification --project-root $PROJECT --dataset-name $DATASET --embedding-name arcface_prealigned --splits train,val --model-name buffalo_l --model-root /home/emily.tomadon/.insightface/models/buffalo_l --ctx-id 0 --prealigned --checkpoint-every 0 --shard-index \$SLURM_ARRAY_TASK_ID --num-shards $NUM_SHARDS'")

echo "ArcFace array: $JOB_ARC"
~~~

Merge ArcFace:

~~~bash
JOB_ARC_MERGE=$(sbatch --parsable \
  --dependency=afterok:$JOB_ARC \
  --job-name=bupt_arc_merge \
  --output=logs/bupt_arc_merge_%j.out \
  --error=logs/bupt_arc_merge_%j.err \
  --time=02:00:00 --mem=32G --cpus-per-task=4 \
  --wrap="bash -lc 'source /home/emily.tomadon/miniconda3/etc/profile.d/conda.sh && conda activate horus_griaule && cd $REPO && python -m minsum.Experiments.run_merge_embedding_shards --project-root $PROJECT --dataset-name $DATASET --purpose verification --embedding-name arcface_prealigned --num-shards $NUM_SHARDS --splits train,val'")

echo "ArcFace merge: $JOB_ARC_MERGE"
~~~

## 4d. DFace verification embeddings

DFace is extracted through the DeepFace Python package. In reports and papers,
call the verification backbone DFace/DeepFace and mention the DeepFace package
only as the implementation framework.

~~~bash
JOB_DFACE=$(sbatch --parsable \
  --array=0-19%2 \
  --job-name=bupt_dface \
  --output=logs/bupt_dface_%A_%a.out \
  --error=logs/bupt_dface_%A_%a.err \
  --time=12:00:00 --mem=20G --cpus-per-task=4 --gres=gpu:1 \
  --wrap="bash -lc 'source /home/emily.tomadon/miniconda3/etc/profile.d/conda.sh && conda activate horus_griaule && cd $REPO && export PYTHONUNBUFFERED=1 && python -m minsum.Experiments.run_04d_extract_dface_verification --project-root $PROJECT --dataset-name $DATASET --embedding-name dface_deepface --splits train,val --model-name DeepFace --detector-backend skip --normalization base --checkpoint-every 0 --shard-index \$SLURM_ARRAY_TASK_ID --num-shards $NUM_SHARDS'")

echo "DFace array: $JOB_DFACE"
~~~

Merge DFace:

~~~bash
JOB_DFACE_MERGE=$(sbatch --parsable \
  --dependency=afterok:$JOB_DFACE \
  --job-name=bupt_dface_merge \
  --output=logs/bupt_dface_merge_%j.out \
  --error=logs/bupt_dface_merge_%j.err \
  --time=02:00:00 --mem=32G --cpus-per-task=4 \
  --wrap="bash -lc 'source /home/emily.tomadon/miniconda3/etc/profile.d/conda.sh && conda activate horus_griaule && cd $REPO && python -m minsum.Experiments.run_merge_embedding_shards --project-root $PROJECT --dataset-name $DATASET --purpose verification --embedding-name dface_deepface --num-shards $NUM_SHARDS --splits train,val'")

echo "DFace merge: $JOB_DFACE_MERGE"
~~~

## 3. KMeans k30

The primary job waits for the DeepFace merge:

~~~bash
JOB_KMEANS=$(sbatch --parsable \
  --dependency=afterok:$JOB_DF_MERGE \
  --job-name=bupt_kmeans \
  --output=logs/bupt_kmeans_%j.out \
  --error=logs/bupt_kmeans_%j.err \
  --time=24:00:00 --mem=80G --cpus-per-task=16 \
  --wrap="bash -lc 'source /home/emily.tomadon/miniconda3/etc/profile.d/conda.sh && conda activate horus_griaule && cd $REPO && export PYTHONUNBUFFERED=1 && python -m minsum.Experiments.run_03_fit_clusters --project-root $PROJECT --dataset-name $DATASET --cluster-embedding deepface_arcface --n-clusters 30'")

echo "KMeans: $JOB_KMEANS"
~~~

For FairFace, depend on JOB_FAIR_MERGE and replace deepface_arcface with
fairface_penultimate.

## 5. Postprocessing

Wait for KMeans and ArcFace merge:

~~~bash
JOB_POST=$(sbatch --parsable \
  --dependency=afterok:$JOB_KMEANS:$JOB_ARC_MERGE \
  --job-name=bupt_post \
  --output=logs/bupt_post_%j.out \
  --error=logs/bupt_post_%j.err \
  --time=08:00:00 --mem=40G --cpus-per-task=8 \
  --wrap="bash -lc 'source /home/emily.tomadon/miniconda3/etc/profile.d/conda.sh && conda activate horus_griaule && cd $REPO && export PYTHONUNBUFFERED=1 && python -m minsum.Experiments.run_05_postprocessing --project-root $PROJECT --dataset-name $DATASET --cluster-embedding deepface_arcface --verification-embedding arcface_prealigned --cluster-run-name k30_cosine_centroid_no_pca --pairs-per-class 1000 --fixed-threshold 0.70 --num-thresholds 200'")

echo "Postprocessing: $JOB_POST"
~~~

For DFace, keep the same KMeans output and replace only the verification
embedding. Use the same validation-pair protocol as the ArcFace and AdaFace
comparisons.

~~~bash
JOB_POST_DFACE=$(sbatch --parsable \
  --dependency=afterok:$JOB_KMEANS:$JOB_DFACE_MERGE \
  --job-name=bupt_post_dface \
  --output=logs/bupt_post_dface_%j.out \
  --error=logs/bupt_post_dface_%j.err \
  --time=08:00:00 --mem=40G --cpus-per-task=8 \
  --wrap="bash -lc 'source /home/emily.tomadon/miniconda3/etc/profile.d/conda.sh && conda activate horus_griaule && cd $REPO && export PYTHONUNBUFFERED=1 && python -m minsum.Experiments.run_05_postprocessing --project-root $PROJECT --dataset-name $DATASET --cluster-embedding deepface_arcface --verification-embedding dface_deepface --cluster-run-name k30_cosine_centroid_no_pca --pairs-per-class 1000 --fixed-threshold 0.75 --num-thresholds 200'")

echo "Postprocessing DFace: $JOB_POST_DFACE"
~~~

## Monitor

~~~bash
squeue -j "$JOB_DF,$JOB_DF_MERGE,$JOB_ARC,$JOB_ARC_MERGE,$JOB_DFACE,$JOB_DFACE_MERGE,$JOB_KMEANS,$JOB_POST"

sacct -j "$JOB_DF,$JOB_DF_MERGE,$JOB_ARC,$JOB_ARC_MERGE,$JOB_DFACE,$JOB_DFACE_MERGE,$JOB_KMEANS,$JOB_POST" \
  --format=JobID,JobName,State,Elapsed,ExitCode,MaxRSS

tail -f logs/bupt_df_*_0.out
~~~

Use Ctrl+C to leave tail without cancelling jobs.

Expected merged shapes:

~~~text
train: 350000 rows
val:    75000 rows
~~~

Inspect merge reports:

~~~bash
cat "$PROJECT/outputs/embeddings/$DATASET/clustering/deepface_arcface/merged_config.json"
cat "$PROJECT/outputs/embeddings/$DATASET/verification/arcface_prealigned/merged_config.json"
cat "$PROJECT/outputs/embeddings/$DATASET/verification/dface_deepface/merged_config.json"
~~~

## Results

~~~bash
EXP="$PROJECT/outputs/experiments/$DATASET/deepface_arcface_cluster__arcface_prealigned_verify/k30_cosine_centroid_no_pca"

column -s, -t < "$EXP/pair_generation_summary.csv"
column -s, -t < "$EXP/minimum_sum/val_metrics_global.csv"
column -s, -t < "$EXP/fixed/val_metrics_global.csv"
~~~

No val_metrics_by_ethnicity.csv is expected.

## Failures

~~~bash
find logs -name 'bupt_*.err' -size +0 -print
sacct -j "$JOB_DF" --format=JobID,State,Elapsed,ExitCode,MaxRSS
sacct -j "$JOB_ARC" --format=JobID,State,Elapsed,ExitCode,MaxRSS
~~~

If an array task fails, resubmit only that index with the same NUM_SHARDS.
Dependent merges do not run unless all required shards finish successfully.

## RFW compatibility

RFW keeps parser rfw and the normal non-sharded commands. Sharding is enabled
only when shard-index is passed. Dataset-specific output directories prevent
BUPT artifacts from overwriting RFW artifacts.
