CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --model_name_or_path fnlp/bart-base-chinese \
    --do_train \
    --source_lang en \
    --target_lang ro \
    --train_file ./input/all/train.json \
    --num_train_epochs 12 \
    --learning_rate 5e-5 \
    --per_device_train_batch_size 8 \
    --max_source_length 64 \
    --predict_with_generate \
    --generation_max_length 15 \
    --save_steps 2000 \
    --output_dir ./output/all \
    --overwrite_output_dir

CUDA_VISIBLE_DEVICES=0 python generate.py \
    --model_name_or_path ./output/all/checkpoint-12000 \
    --save_path ./output/all/generated_riddles.csv \
    --input_file ./input/all/test.src


CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --model_name_or_path fnlp/bart-base-chinese \
    --do_train \
    --source_lang en \
    --target_lang ro \
    --train_file ./input/wo-expl/train.json \
    --num_train_epochs 12 \
    --learning_rate 5e-5 \
    --per_device_train_batch_size 8 \
    --max_source_length 64 \
    --predict_with_generate \
    --generation_max_length 15 \
    --save_steps 2000 \
    --output_dir ./output/wo-expl \
    --overwrite_output_dir

CUDA_VISIBLE_DEVICES=0 python generate.py \
    --model_name_or_path ./output/wo-expl/checkpoint-10000 \
    --save_path ./output/wo-expl/generated_riddles.csv \
    --input_file ./input/wo-expl/test.src


CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --model_name_or_path fnlp/bart-base-chinese \
    --do_train \
    --source_lang en \
    --target_lang ro \
    --train_file ./input/wo-ids/train.json \
    --num_train_epochs 12 \
    --learning_rate 5e-5 \
    --per_device_train_batch_size 8 \
    --max_source_length 64 \
    --predict_with_generate \
    --generation_max_length 15 \
    --save_steps 2000 \
    --output_dir ./output/wo-ids \
    --overwrite_output_dir

CUDA_VISIBLE_DEVICES=0 python generate.py \
    --model_name_or_path ./output/wo-ids \
    --save_path ./output/wo-ids/generated_riddles.csv \
    --input_file ./input/wo-ids/test.src
