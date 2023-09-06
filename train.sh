#!/bin/bash

date_time=$(date "+%Y_%m%d_%H%M")

cd pytorch/

# git status > ../logs/WDFV@${date_time}.log
# git diff >> ../logs/WDFV@${date_time}.log

nohup python -u train.py >> ../logs/WDFV@${date_time}.log 2>&1 &
tail -f ../logs/WDFV@${date_time}.log
