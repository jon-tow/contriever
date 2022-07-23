#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --job-name="contriever"
#SBATCH --partition=compute-od-gpu
#SBATCH --cpus-per-task=6
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --output=/fsx/carper/contriever/checkpoint/pile/%x_%j.out

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

export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

echo "Node Count: $COUNT_NODE"
echo "Host Names: $HOSTNAMES"

###############################################################################
# Contriever Setup
###############################################################################

TRAIN_PATH=/fsx/carper/contriever

wandb_project="contriever"
wandb_entity="carperai"

per_gpu_batch_size=16
rmin=0.05
rmax=0.5
T=0.05
QSIZE=131072
MOM=0.9995
POOL=average
AUG=delete
PAUG=0.1
LC=0.
_mo=bert-large-uncased
mo=${TRAIN_PATH}/configs/$_mo/
to=bert-base-uncased
mp=none
proj_size=1024 # SET THIS TO HIDDEN SIZE FROM THE MODEL CONFIGS!
name=$SLURM_JOB_ID-$POOL-rmin$rmin-rmax$rmax-T$T-$QSIZE-$MOM-$_mo-$AUG-$PAUG

echo $mo

OUTPUT_DIR=$TRAIN_PATH/checkpoint/pile/$name
# NOTE: DATA_DIR must point to the directory specified in `tokenization_pile_script.sh`
DATA_DIR=$TRAIN_PATH/encoded-data/bert-base-uncased
# NOTE: Uncomment the line below to test on 1 pile slice dataset
#TRAIN_DATASETS=$DATA_DIR/pile/"00"
TRAIN_DATASETS=""
for i in 0{0..9} {10..29}
do 
    TRAIN_DATASETS+="${DATA_DIR}/pile/${i} "
done

source $TRAIN_PATH/.env/bin/activate
cd $TRAIN_PATH

srun --cpu_bind=v --accel-bind=gn python3.8 train.py \
        --model_path $mp \
        --sampling_coefficient $LC \
        --retriever_model_id $mo --pooling $POOL \
        --retriever_tokenizer_id $to \
        --augmentation $AUG --prob_augmentation $PAUG \
        --train_data $TRAIN_DATASETS --loading_mode split \
        --ratio_min $rmin --ratio_max $rmax --chunk_length 256 \
        --momentum $MOM --moco_queue $QSIZE --temperature $T \
        --warmup_steps 20000 --total_steps 500000 --lr 0.00005 \
        --projection_size $proj_size \
        --num_workers 6 \
        --name $name \
        --scheduler linear \
        --optim adamw \
        --per_gpu_batch_size $per_gpu_batch_size \
        --output_dir  $OUTPUT_DIR \
        --main_port $MASTER_PORT \
        --main_addr $MASTER_ADDR \
        --wandb_project $wandb_project \
        --wandb_entity $wandb_entity \
        --random_init
