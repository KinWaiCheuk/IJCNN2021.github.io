#!/bin/bash
# Generating the results as in Table 1 using trained weights
# Change the device according to your computer configuration
echo -e "Which GPU do you want to run on? [0/1/2/3]: "
read GPU_num

python evaluate.py with weight_file=Table1/Attention-MAPS-Mel-feat LSTM=True onset=True device=cuda:$GPU_num
python evaluate.py with weight_file=Table1/Attention-MAPS-Mel-onset LSTM=True onset=True device=cuda:$GPU_num
python evaluate.py with weight_file=Table1/Attention-MAPS-Mel-spec LSTM=True onset=True device=cuda:$GPU_num
python evaluate.py with weight_file=Table1/Original-MAPS-Mel-Null LSTM=True onset=True device=cuda:$GPU_num
