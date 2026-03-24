#!/bin/bash
#SBATCH --partition=A40short
#SBATCH --time=4:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --array=0-5%6
#SBATCH --job-name=gallop6
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err

echo "NODE=$(hostname) JOB=$SLURM_JOB_ID CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

module load Python/3.10
module load CUDA/11.7

source venvPatri3.10/bin/activate

cd /home/s33mrodr/DISCO-AE || exit 1

CONFIGS=( \
  gallop_model_seed1 \
  gallop_model_seed2 \
  gallop_model_seed3 \
  gallop_pca_seed1 \
  gallop_pca_seed2 \
  gallop_pca_seed3 \
)

SCRIPTS=( \
  02_train_network.py \
  02_train_network.py \
  02_train_network.py \
  02_train_network_PCA.py \
  02_train_network_PCA.py \
  02_train_network_PCA.py \
)

i=$SLURM_ARRAY_TASK_ID
#echo "Task $i -> ${SCRIPTS[$i]} --config ${CONFIGS[$i]}"

BASE_LOCAL=""
if [ -d "/local/nvme" ] && [ -w "/local/nvme" ]; then
  BASE_LOCAL="/local/nvme/${USER}_${SLURM_JOB_ID}"
elif [ -n "${SLURM_TMPDIR:-}" ] && [ -d "${SLURM_TMPDIR}" ]; then
  BASE_LOCAL="${SLURM_TMPDIR}"
else
  BASE_LOCAL="/tmp/${USER}_${SLURM_JOB_ID}"
fi

#export DISCO_CACHE_DIR="${BASE_LOCAL}/disco_cache/task_${i}"
#mkdir -p "${DISCO_CACHE_DIR}/cache"

#export DISCO_DATALOADER_CACHE_DIR="${DISCO_CACHE_DIR}/dataloader_cache"
#mkdir -p "${DISCO_DATALOADER_CACHE_DIR}"
#echo "DISCO_DATALOADER_CACHE_DIR=${DISCO_DATALOADER_CACHE_DIR}"

DISCO_CACHE_DIR="${BASE_LOCAL}/disco_cache/task_${i}"
rm -rf "${DISCO_CACHE_DIR}"
mkdir -p "${DISCO_CACHE_DIR}/cache"
export DISCO_CACHE_DIR

DISCO_DATALOADER_CACHE_DIR="${DISCO_CACHE_DIR}/dataloader_cache"
rm -rf "${DISCO_DATALOADER_CACHE_DIR}"
mkdir -p "${DISCO_DATALOADER_CACHE_DIR}"
export DISCO_DATALOADER_CACHE_DIR

echo "Start training | job=${SLURM_JOB_ID} | task=${i} | node=${SLURM_NODELIST} | script=${SCRIPTS[$i]} | config=${CONFIGS[$i]}"
echo "Python: $(which python)"
echo "DISCO_CACHE_DIR=${DISCO_CACHE_DIR}"
echo "DISCO_DATALOADER_CACHE_DIR=${DISCO_DATALOADER_CACHE_DIR}"

python -u "${SCRIPTS[$i]}" --config "${CONFIGS[$i]}"
