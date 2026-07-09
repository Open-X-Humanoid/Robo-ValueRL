#!/bin/bash

conda activate robo_valuerl



GPUS_PER_NODE=8
MASTER_ADDR=${MASTER_ADDR:-"10.0.0.1"}
MASTER_PORT=${MASTER_PORT:-"6000"}
NNODES=${WORLD_SIZE:-"2"}
NODE_RANK=${RANK:-"0"}

echo "--------------------------------------------------------"
echo "Node Rank: $NODE_RANK / $NNODES"
echo "Master Addr: $MASTER_ADDR:$MASTER_PORT"
echo "Data Config: $DATA_CONFIG"
echo "Exp Name: $EXP_NAME"
echo "--------------------------------------------------------"


torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train_robo_value_rl_offline_rl.py  \
    stack_all_hours_data_pretraining \
    --exp_name test \
    --batch-size 768 \
    --decay_steps 30000 \
    --lr 1e-4 \
    --sample_from_ratio 1 \
    --only-load-paligemma