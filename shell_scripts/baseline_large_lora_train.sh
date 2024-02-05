#!/bin/bash

python main.py --is_train 1 \
               --is_test 0 \
               --is_logging 1 \
               --model_name "google/flan-t5-base" \
               --batch_size 4 \
               --epoch 50 \
               --accumulation_step 4\
               --learning_rate 1e-4 \
               --validation_ratio 0.1 \
               --patience 3 \
               --save_name baseline_model_large > current_process.txt


#nohup shell_scripts/baseline_large_lora_train.sh 2>&1 &
#chmod +x sheel_scrips/*.sh