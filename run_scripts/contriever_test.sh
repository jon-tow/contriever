#!/bin/bash

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
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export PYTHONFAULTHANDLER=1

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

TRAIN_PATH=/fsx/carper/contriever

WANDB_PROJECT="contriever"
WANDB_ENTITY="jon-tow"

PER_GPU_BATCH_SIZE=512
MICRO_BATCH_SIZE=32
QSIZE=32768
MOM=0.9995
T=0.05
RMIN=0.05
RMAX=0.5
POOL=average
AUG=delete
PAUG=0.1
LC=0.
MP=none
TO=bert-base-uncased
_MO=bert-base-uncased
MO=$_MO              # For custom arch configs use: ${TRAIN_PATH}/configs/$_MO/
PROJECTION_SIZE=768  # NOTE: Set this to hidden size from the model configs!
EVAL_FREQ=1000 # (in steps)
OPTIM=adamw
NAME=test-$SLURM_JOB_ID-$POOL-rmin$RMIN-rmax$RMAX-T$T-$QSIZE-$MOM-$_MO-$AUG-$PAUG

OUTPUT_DIR=$TRAIN_PATH/checkpoint/pile/$NAME
EMBED_DIR=$OUTPUT_DIR/embeddings
# NOTE: DATA_DIR must point to the directory specified in `tokenization_pile_script.sh`
DATA_DIR=$TRAIN_PATH/encoded-data/bert-base-uncased
# NOTE: Uncomment the line below to test on 1 pile slice dataset
TRAIN_DATASETS=$DATA_DIR/pile/"00"

source $TRAIN_PATH/.env/bin/activate
cd $TRAIN_PATH

torchrun --standalone --nnodes=1 --nproc_per_node=8 test.py \
    --name $NAME \
    --model_path $MP \
    --sampling_coefficient $LC \
    --retriever_model_id $MO --pooling $POOL \
    --retriever_tokenizer_id $TO \
    --augmentation $AUG --prob_augmentation $PAUG \
    --train_data $TRAIN_DATASETS --loading_mode split \
    --ratio_min $RMIN --ratio_max $RMAX --chunk_length 256 \
    --momentum $MOM --queue_size $QSIZE --temperature $T \
    --warmup_steps 20000 --total_steps 200 --lr 0.00005 \
    --scheduler linear \
    --optim $OPTIM \
    --projection_size $PROJECTION_SIZE \
    --per_gpu_batch_size $PER_GPU_BATCH_SIZE \
    --micro_batch_size $MICRO_BATCH_SIZE \
    --num_workers 6 \
    --output_dir $OUTPUT_DIR \
    --log_embed_dir $EMBED_DIR \
    --log_embed_freq 2 \