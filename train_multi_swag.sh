#!/bin/bash

# Configuration
NUM_MODELS=10
BASE_SEED=42
OUTPUT_DIR="./checkpoint/multi_swag_collection"
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
FINAL_OUTPUT_DIR="${OUTPUT_DIR}/${TIMESTAMP}"

# Create output directory
mkdir -p "${FINAL_OUTPUT_DIR}"

echo "Training ${NUM_MODELS} SWAG models..."
echo "Output directory: ${FINAL_OUTPUT_DIR}"

# Training parameters (modify as needed)
MODEL_DEPTH=32
BATCH_SIZE=256
NORM_TYPE="frn"
OPTIM_LR=0.1
OPTIM_SWA_LR=0.01
OPTIM_MOMENTUM=0.9
OPTIM_WEIGHT_DECAY=1e-4
OPTIM_NUM_EPOCHS=1000
WARMUP_EPOCHS=10
START_SWA_EPOCH=800
SWAG_FREQ=1
SWAG_RANK=20
EVAL_SWAG_FREQ=50
NUM_SWAG_SAMPLES=5

successful_models=0
failed_models=()

for i in $(seq 1 $NUM_MODELS); do
    echo "----------------------------------------"
    echo "Training model ${i}/${NUM_MODELS}..."
    
    # Calculate seed
    seed=$((BASE_SEED + (i-1) * 100))
    
    # Create model-specific directories
    model_id=$(printf "%02d" $i)
    temp_checkpoint_dir="${FINAL_OUTPUT_DIR}/temp_checkpoint_${model_id}"
    temp_swag_dir="${FINAL_OUTPUT_DIR}/temp_swag_${model_id}"
    
    # Train the model
    python train_swag.py \
        --seed $seed \
        --model_depth $MODEL_DEPTH \
        --batch_size $BATCH_SIZE \
        --norm_type $NORM_TYPE \
        --optim_lr $OPTIM_LR \
        --optim_swa_lr $OPTIM_SWA_LR \
        --optim_momentum $OPTIM_MOMENTUM \
        --optim_weight_decay $OPTIM_WEIGHT_DECAY \
        --optim_num_epochs $OPTIM_NUM_EPOCHS \
        --warmup_epochs $WARMUP_EPOCHS \
        --start_swa_epoch $START_SWA_EPOCH \
        --swag_freq $SWAG_FREQ \
        --swag_rank $SWAG_RANK \
        --eval_swag_freq $EVAL_SWAG_FREQ \
        --num_swag_samples $NUM_SWAG_SAMPLES \
        --save_dir "$temp_checkpoint_dir" \
        --swag_save_dir "$temp_swag_dir" \
        --model_id "model_${model_id}_seed_${seed}" \
        --checkpoint_every_n_epochs 10 \
        --max_checkpoints_to_keep 30
    
    if [ $? -eq 0 ]; then
        echo "Model ${i} training completed successfully"
        ((successful_models++))
        
        # Move SWAG state to final location
        if [ -d "$temp_swag_dir" ]; then
            final_swag_dir="${FINAL_OUTPUT_DIR}/swag_state_${model_id}_seed_${seed}"
            mv "$temp_swag_dir" "$final_swag_dir"
        fi
        
        # Clean up checkpoint directory to save space
        # rm -rf "$temp_checkpoint_dir"
        
    else
        echo "Model ${i} training failed"
        failed_models+=($i)
        
        # Clean up on failure
        rm -rf "$temp_checkpoint_dir"
        rm -rf "$temp_swag_dir"
    fi
done

echo "========================================"
echo "Training Summary:"
echo "Successfully trained: ${successful_models}/${NUM_MODELS} models"
if [ ${#failed_models[@]} -gt 0 ]; then
    echo "Failed models: ${failed_models[*]}"
fi
echo "SWAG states saved in: ${FINAL_OUTPUT_DIR}"

# Create summary file
cat > "${FINAL_OUTPUT_DIR}/training_summary.txt" << EOF
Multi-SWAG Training Summary
Training completed: $(date)
Successfully trained: ${successful_models}/${NUM_MODELS} models
Base seed: ${BASE_SEED}
Failed models: ${failed_models[*]}

Training parameters:
MODEL_DEPTH=${MODEL_DEPTH}
BATCH_SIZE=${BATCH_SIZE}
NORM_TYPE=${NORM_TYPE}
OPTIM_LR=${OPTIM_LR}
OPTIM_SWA_LR=${OPTIM_SWA_LR}
OPTIM_MOMENTUM=${OPTIM_MOMENTUM}
OPTIM_WEIGHT_DECAY=${OPTIM_WEIGHT_DECAY}
OPTIM_NUM_EPOCHS=${OPTIM_NUM_EPOCHS}
WARMUP_EPOCHS=${WARMUP_EPOCHS}
START_SWA_EPOCH=${START_SWA_EPOCH}
SWAG_FREQ=${SWAG_FREQ}
SWAG_RANK=${SWAG_RANK}
EVAL_SWAG_FREQ=${EVAL_SWAG_FREQ}
NUM_SWAG_SAMPLES=${NUM_SWAG_SAMPLES}
EOF

echo "Summary saved to: ${FINAL_OUTPUT_DIR}/training_summary.txt"
