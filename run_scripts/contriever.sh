#!/bin/bash

###############################################################################
# Contriever Setup
###############################################################################

TRAIN_PATH=/fsx/carper/contriever

WANDB_PROJECT="contriever"
WANDB_ENTITY="carperai"

PER_GPU_BATCH_SIZE=64
RMIN=0.05
RMAX=0.5
T=0.05
QSIZE=131072
MOM=0.9995
POOL=average
AUG=delete
PAUG=0.1
LC=0.
MP=none
TO=bert-base-uncased
_MO=bert-large-uncased
MO=$_MO              # For custom arch configs use: ${TRAIN_PATH}/configs/$_MO/
PROJECTION_SIZE=1024 # NOTE: Set this to hidden size from the model configs!
EVAL_DATASETS=("nq msmarco")
EVAL_DATASETS_DIR=${TRAIN_PATH}/BEIR/datasets/
EVAL_FREQ=1000 # (in steps)
NAME=$SLURM_JOB_ID-$POOL-rmin$RMIN-rmax$RMAX-T$T-$QSIZE-$MOM-$_MO-$AUG-$PAUG

OUTPUT_DIR=$TRAIN_PATH/checkpoint/pile/$NAME
# NOTE: DATA_DIR must point to the directory specified in `tokenization_pile_script.sh`
DATA_DIR=$TRAIN_PATH/encoded-data/bert-base-uncased
# NOTE: Uncomment the line below to test on 1 pile slice dataset
#TRAIN_DATASETS=$DATA_DIR/pile/"00"
TRAIN_DATASETS=""
for i in 0{0..9} {10..29}; do
    TRAIN_DATASETS+="${DATA_DIR}/pile/${i} "
done

source $TRAIN_PATH/.env/bin/activate
cd $TRAIN_PATH

python3.8 train.py \
    --name $NAME \
    --model_path $MP \
    --sampling_coefficient $LC \
    --retriever_model_id $MO --pooling $POOL \
    --retriever_tokenizer_id $TO \
    --augmentation $AUG --prob_augmentation $PAUG \
    --train_data $TRAIN_DATASETS --loading_mode split \
    --eval_datasets $EVAL_DATASETS --eval_datasets_dir $EVAL_DATASETS_DIR --eval_freq $EVAL_FREQ \
    --ratio_min $RMIN --ratio_max $RMAX --chunk_length 256 \
    --momentum $MOM --queue_size $QSIZE --temperature $T \
    --warmup_steps 20000 --total_steps 500000 --lr 0.00005 \
    --scheduler linear \
    --optim adamw8bit \
    --projection_size $PROJECTION_SIZE \
    --per_gpu_batch_size $PER_GPU_BATCH_SIZE \
    --num_workers 6 \
    --output_dir $OUTPUT_DIR \
    --wandb_project $WANDB_PROJECT \
    --wandb_entity $WANDB_ENTITY
