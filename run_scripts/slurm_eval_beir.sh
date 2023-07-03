#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --account=carper
#SBATCH --job-name="contriever"
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=6
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --exclude=gpu-st-p4d-24xlarge-[73,74,122,123,146,147,280,281,282,283,284,285,286,289,290,291]
#SBATCH --output=/fsx/carper/contriever/BEIR/contriever_deduped_%x_%j.out  # !!SPECIFY THIS 
#SBATCH --comment carper

module load openmpi
source /opt/intel/mpi/latest/env/vars.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/nccl/build/lib:/opt/aws-ofi-nccl-install/lib
export NCCL_PROTO=simple
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/aws-ofi-nccl/lib
export PATH=$PATH:/opt/amazon/efa/bin:/opt/amazon/openmpi/bin
export PATH=/opt/amazon/efa/bin:$PATH
export LD_PRELOAD="/opt/nccl/build/lib/libnccl.so"

export TORCH_CPP_LOG_LEVEL=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_DEBUG=info
export NCCL_TREE_THRESHOLD=0
export PYTHONFAULTHANDLER=1

export CUDA_LAUNCH_BLOCKING=1

export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=1
export FI_EFA_ENABLE_SHM_TRANSFER=0
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64

export OMPI_MCA_mtl_base_verbose=1
export OMPI_MCA_pml="^cm"
export OMPI_MCA_btl="tcp,self"
export OMPI_MCA_btl_tcp_if_exclude="lo,docker1"
export OMPI_MCA_btl_base_verbose=30
export OMPI_MCA_plm_rsh_no_tree_spawn=1

export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export COUNT_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)

echo "Node Count: $COUNT_NODE"
echo "Host Names: $HOSTNAMES"

###############################################################################
# Contriever Setup
###############################################################################
export HF_HOME=/fsx/guac/.cache/huggingface
export TRANSFORMERS_CACHE=/fsx/guac/.cache/huggingface/transformers

TRAIN_PATH=/fsx/carper/contriever
cd $TRAIN_PATH
source $TRAIN_PATH/.env/bin/activate

# NOTE: For msmarco OOM issues see: https://github.com/beir-cellar/beir/issues/109
dataset='msmarco' # 'nq' 'quora' 'fever' 'hotpotqa' 'climate-fever' 'arguana' fiqa' 'trec-covid' 'webis-touche2020' 'dbpedia-entity' 'nfcorpus' 'scidocs' 'scifact' 'cqadupstack'
beir_dir=$TRAIN_PATH/BEIR

echo $dataset

# Step 186_000
# model=/fsx/carper/contriever/checkpoint/pile_deduped/baseline-deduped-16886-average-adamw-bs64-smooth0.0-rmin0.05-rmax0.5-T0.05-8192-0.999-bert-large-uncased-delete-0.1/checkpoint/step-186000

# Step 150_000
# model=/fsx/carper/contriever/checkpoint/pile_deduped/baseline-deduped-16886-average-adamw-bs64-smooth0.0-rmin0.05-rmax0.5-T0.05-8192-0.999-bert-large-uncased-delete-0.1/checkpoint/step-150000

source $TRAIN_PATH/.env/bin/activate  && srun --comment carper --cpu_bind=v --accel-bind=gn python3.8 eval_beir.py \
    --model_name_or_path $model \
    --dataset $dataset \
    --beir_dir $beir_dir \
    --main_port $MASTER_PORT \