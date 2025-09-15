export DISP_NAME="mamba2_2_7b"
export MODEL_DIR="/work/hdd/bcjw/hshen14/huggingface/model_ckpt/mamba2_2_7b/hf"

export LOGIT_DIR="/work/hdd/bcjw/hshen14/logit/${DISP_NAME}"
export OUTPUT_DIR="/work/hdd/bcjw/hshen14/ppl/${DISP_NAME}"

mkdir -p ${LOGIT_DIR}
mkdir -p ${OUTPUT_DIR}

python RULER/scripts/update_experiments.py ${MODEL_DIR}/config.json \
    --set logits_reg false \
    --set logits_dir ${LOGIT_DIR} \
    --set adaptive_upi false \
    --set proper_upi false

torchrun --standalone --nproc_per_node=4 LongBamba/LongBamba/ppl_fsdp.py \
    --model ${MODEL_DIR} \
    --dataset /work/hdd/bcjw/hshen14/huggingface/datasets/pg19 \
    --split validation \
    --streaming \
    --max-length 65537 \
    --batch-size 1 \
    --bf16 \
    --cpu-offload
