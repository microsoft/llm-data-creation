DATA_DIR=$1
DATA_NAME=$2
SETTING=$3
LR=$4
OUT_DIR=$5
MODEL=$6

python train.py \
    --model_name_or_path ${MODEL} \
    --do_train \
    --do_eval \
    --do_predict \
    --data_dir ${DATA_DIR} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUT_DIR} \
    --setting ${SETTING} \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 1 \
    --load_best_model_at_end \
    --metric_for_best_model eval_acc \
    --greater_is_better True \
    --learning_rate ${LR} \
    --num_train_epochs 10 \
    --max_seq_length 256 \
    --per_device_eval_batch_size 4 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --weight_decay 0.01 \
    --warmup_steps 100 \
    --overwrite_output