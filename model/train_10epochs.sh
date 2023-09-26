#!/bin/bash

for i in {1..10}
do
    echo $i
    python3 training_pipeline_trainer.py \
    --text_data_dir ../data/text_gold_dynabench.csv \
    --crowd_data_dir ../data/annotation_dynabench_idx_coded.csv \
    --seed $i \
    --output_dir ../outputs/dynabench/ctm/seed_$i/ \
    --label_dict {0:_'not'_,_1:'hate'} \
    \
    --anno_pool mean \
    --anno_emb_freeze True \
    --max_anno_num 20 \
    --anno_emb_dir ../embeddings/dynabench_annotator_tensor_ctm_10_train.pt \
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
