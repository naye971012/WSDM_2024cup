#!/bin/bash

python main.py --is_train 1 \
               --is_test 0 \
               --is_logging 1 \
               --is_init 1 \
               --model_name "google/long-t5-local-large" \
               --batch_size 1 \
               --epoch 50 \
               --accumulation_step 4\
               --learning_rate 1e-4 \
               --validation_ratio 0.1 \
               --patience 5 \
               --save_name longt5_model_large > current_process.txt


#nohup shell_scripts/longt5_large_lora_train.sh 2>&1 &
#chmod +x sheel_scrips/*.sh