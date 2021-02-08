# IJCNN2021
[Source code](https://github.com/KinWaiCheuk/IJCNN2021.github.io/tree/main/codes) and [supplementary information](https://kinwaicheuk.github.io/IJCNN2021.github.io/) for IJCNN2021.

The supplementary information is available at: https://kinwaicheuk.github.io/IJCNN2021.github.io/

The source code is available under the `codes` folder.

## Training the mode
### Step 1: Preparing Dataset
The MAPS dataset can be downloaded via https://amubox.univ-amu.fr/index.php/s/iNG0xc5Td1Nv4rR.
Download and unzip the dataset into the `codes` folder.

After unzipping all the files inside, the `ENSTDkAm1` and `ENSTDkAm2` folder should be combined as one single folder `ENSTDkAm`.

### Step 2: Training the models
Run the following scripts to train different models:

* `python train_original.py with <args>`
* `python train_fast_local_attent.py with <args>`
* `python train_simple.py with <args>`

The following arguments are available:

* `device`: choose what device to use. Can be `cpu`, `cuda:0` or any device that is available in your PC. Default `cuda:0`.
* `LSTM`: Train the model with or without the LSTM layer. Either `True` or `False`. Default `True`.
* `onset_stack`: Train the model with or without the onset stack. Either `True` or `False`. Default `True`.

* `batch_size`: Setting the batch size. Default `16`.

The following arguments are for `train_fast_local_attent.py` only

* `w_size`: The attention window size. Default: `30`.
* `attention_mode`: Choosing which feature to attend to. Either `onset` or `activation` or `spec`. Default `onset`.

The PyTorch dataset class `MAPS()` inside each script will process and prepare the dataset if you are running it for the first time.


## Using pre-trained models
### Step 1: Downloading weights
The weights can be downloaded [here](https://sutdapac-my.sharepoint.com/:f:/g/personal/kinwai_cheuk_mymail_sutd_edu_sg/Em-RhkuS7S9Oq9iGU25KixcBQ-Ylh-z1miYa9xmrQZ4KYg?e=Y9vl7X)

### Step 2: Generating the results
The bash files contain all the commanands to obtain the results reported in the paper. Run the following bash scripts to get all the results

* `bash get_table1.sh`
* `bash get_figure1.sh`
* `bash get_table2.sh`

The accuracy reports and the midi files will be saved in the `results` folder.
