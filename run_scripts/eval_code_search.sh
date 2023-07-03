# #!/bin/bash
# #SBATCH --time=24:00:00
# #SBATCH --account=carper
# #SBATCH --job-name="eval-contriever"
# #SBATCH --partition=gpu
# #SBATCH --cpus-per-task=6
# #SBATCH --nodes=1
# #SBATCH --ntasks-per-node=8
# #SBATCH --gres=gpu:8
# #SBATCH --exclusive
# #SBATCH --exclude=gpu-st-p4d-24xlarge-[73,74,122,123,146,147,280,281,282,283,284,285,286,289,290,291]
# #SBATCH --output=/fsx/carper/contriever/CodeSearchNet/contriever_deduped_%x_%j.out  # !!SPECIFY THIS 
# #SBATCH --comment carper

# module load openmpi
# source /opt/intel/mpi/latest/env/vars.sh
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/nccl/build/lib:/opt/aws-ofi-nccl-install/lib
# export NCCL_PROTO=simple
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/aws-ofi-nccl/lib
# export PATH=$PATH:/opt/amazon/efa/bin:/opt/amazon/openmpi/bin
# export PATH=/opt/amazon/efa/bin:$PATH
# export LD_PRELOAD="/opt/nccl/build/lib/libnccl.so"

# export TORCH_CPP_LOG_LEVEL=INFO
# export TORCH_DISTRIBUTED_DEBUG=INFO
# export NCCL_DEBUG=info
# export NCCL_TREE_THRESHOLD=0
# export PYTHONFAULTHANDLER=1

# export CUDA_LAUNCH_BLOCKING=1

# export FI_EFA_FORK_SAFE=1
# export FI_LOG_LEVEL=1
# export FI_EFA_ENABLE_SHM_TRANSFER=0
# export FI_PROVIDER=efa
# export FI_EFA_TX_MIN_CREDITS=64

# export OMPI_MCA_mtl_base_verbose=1
# export OMPI_MCA_pml="^cm"
# export OMPI_MCA_btl="tcp,self"
# export OMPI_MCA_btl_tcp_if_exclude="lo,docker1"
# export OMPI_MCA_btl_base_verbose=30
# export OMPI_MCA_plm_rsh_no_tree_spawn=1

# export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
# export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# export MASTER_PORT=12802
# export COUNT_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)

# echo "Node Count: $COUNT_NODE"
# echo "Host Names: $HOSTNAMES"

###############################################################################
# Contriever Setup
###############################################################################

export HF_HOME=/fsx/guac/.cache/huggingface
export TRANSFORMERS_CACHE=/fsx/guac/.cache/huggingface/transformers

TRAIN_PATH=/fsx/carper/contriever
cd $TRAIN_PATH
source $TRAIN_PATH/.env/bin/activate

CODE_SEARCH_DIR=$TRAIN_PATH/CodeSearchNet

# MODEL_NAME="CarperAI/carptriever-1"
# MODEL_NAME="facebook/contriever"
# MODEL_NAME="microsoft/graphcodebert-base"
# MODEL_NAME="microsoft/deberta-v3-base"
# MODEL_NAME="microsoft/codebert-base"
# MODEL_NAME="huggingface/CodeBERTa-small-v1"
# MODEL_NAME="microsoft/codereviewer"
MODEL_NAME="/fsx/carper/contriever/carptriever-1"
# MODEL_NAME="Salesforce/codet5-large-ntp-py"
# MODEL_NAME="jon-tow/test"

#CANDIDATES=("100 1000 10000")
CANDIDATES=("1000")
SEED="2601934"


for CANDIDATE in $CANDIDATES ; do
    python3.8 csn_ret.py \
        --device gpu \
        --model $MODEL_NAME \
        --subsets "python,go,ruby,php,java,javascript" \
        --candidate_size $CANDIDATE \
        --output_dir $CODE_SEARCH_DIR \
	    --seed $SEED 
done

# source $TRAIN_PATH/.env/bin/activate  && srun --comment carper --cpu_bind=v --accel-bind=gn python3.8 csn_ret.py \
#     --device gpu
#     --model $MODEL_NAME
#     --candidate_size 
#     --subset 
