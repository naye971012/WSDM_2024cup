#!/bin/bash

python main.py --is_train 1 \
               --is_test 0 \
               --is_logging 0 \
               --model_name "google/long-t5-tglobal-base" \
               --batch_size 2 \
               --epoch 50 \
               --accumulation_step 8\
               --learning_rate 1e-4 \
               --validation_ratio 0.1 \
               --patience 5 \
               --save_name longt5_model_base_2 > current_process.txt


#nohup shell_scripts/longt5_base_lora_train.sh 2>&1 &
#chmod +x sheel_scrips/*.sh