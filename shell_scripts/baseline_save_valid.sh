#!/bin/bash

python main.py --is_train 1 \
               --is_test 0 \
               --is_logging 0 \
               --batch_size 32 \
               --save_name baseline_model > current_process.txt

#nohup shell_scripts/baseline_save_valid.sh 2>&1 &
#chmod +x sheel_scrips/*.sh
#trainer 내부 주석 지우기 필요 (validation에 print부분)