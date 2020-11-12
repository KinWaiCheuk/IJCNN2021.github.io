#!/bin/bash
# Generating the results as in Table 1 using trained weights
# Change the device according to your computer configuration
echo -e "Which GPU do you want to run on? [0/1/2/3]: "
read GPU_num

python evaluate.py with weight_file=Figure1/Simple-MAPS-Mel-Null-D_1  inference=False device=cuda:$GPU_num
python evaluate.py with weight_file=Figure1/Simple-MAPS-Mel-Null-D_5  inference=False device=cuda:$GPU_num
python evaluate.py with weight_file=Figure1/Simple-MAPS-Mel-Null-D_10  inference=False device=cuda:$GPU_num
python evaluate.py with weight_file=Figure1/Simple-MAPS-Mel-Null-D_15 inference=False device=cuda:$GPU_num
python evaluate.py with weight_file=Figure1/Simple-MAPS-Mel-Null-D_20  inference=False device=cuda:$GPU_num
python evaluate.py with weight_file=Figure1/Simple-MAPS-Mel-Null-D_25  inference=False device=cuda:$GPU_num
python evaluate.py with weight_file=Figure1/Simple-MAPS-Mel-Null-D_30  inference=False device=cuda:$GPU_num
python evaluate.py with weight_file=Figure1/Simple-MAPS-Mel-Null-D_0 Simple_attention=False cat_feat=False  inference=False device=cuda:$GPU_num