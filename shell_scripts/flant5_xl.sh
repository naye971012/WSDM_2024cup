#!/bin/bash

python main.py --is_train 1 \
               --is_test 0 \
               --is_logging 0 \
               --model_name "google/flan-t5-xl" \
               --batch_size 2 \
               --epoch 50 \
               --accumulation_step 4\
               --learning_rate 1e-4 \
               --validation_ratio 0.1 \
               --patience 5 \
               --save_name flant5_xl_with_lora > current_process.txt


#nohup shell_scripts/longt5_base_lora_train.sh 2>&1 &
#chmod +x sheel_scrips/*.sh