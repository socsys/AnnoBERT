#!/bin/bash

for i in {1..10}
do
    echo $i
    python3 bert_baseline_trainer.py \
    --text_data_dir ./text_gold_dynabench_6124.csv \
    --crowd_data_dir ./annotation_dynabench_6124.csv \
    --seed $i \
    --output_dir ../outputs/dynabench/bert_baseline_new/seed_$i/ \
    \
    --anno_pool mean \
    --anno_emb_freeze True \
    --max_anno_num 20 \
    \
    --learning_rate 1e-5 \
    --lr_scheduler_type constant \
    --num_train_epochs 4 \
    --per_device_train_batch_size 32 \
    \
    --do_train \
    --do_eval \
    --do_predict \
    --metric_for_best_model macro_f1 \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --label_names label_id \
    --load_best_model_at_end \
    --logging_strategy epoch
done
