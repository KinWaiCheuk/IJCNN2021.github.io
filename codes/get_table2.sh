#!/bin/bash
# Generating the results as in Table 1 using trained weights
# Change the device according to your computer configuration
echo -e "Which GPU do you want to run on? [0/1/2/3]: "
read GPU_num

python evaluate.py with weight_file=Table2/Attention-MAPS-Mel-spec-no_onset LSTM=True onset=False device=cuda:$GPU_num
python evaluate.py with weight_file=Table2/Attention-MAPS-Mel-spec-no_biLSTM LSTM=False onset=True device=cuda:$GPU_num
python evaluate.py with weight_file=Table2/Original-MAPS-Mel-Null-no_onset LSTM=True onset=False device=cuda:$GPU_num
python evaluate.py with weight_file=Table2/Original-MAPS-Mel-Null-no_biLSTM LSTM=False onset=True device=cuda:$GPU_num