# WDFV

## prerequisties
The code is developed and tested with:
* Python **3.7.15**
* Pytorch **1.13.0**
* More details are available in **requirements.txt**

## Data Preparation

### Datasets
We use FoD500 datast and DDFF-12 dataset in our experiments:
* [FoD500](https://github.com/dvl-tum/defocus-net) can be downloaded [here](https://drive.google.com/file/d/1bR-WZQf44s0nsScC27HiEwaXPyEQ3-Dw/view?usp=sharing).
* [DDFF-12](https://github.com/soyers/ddff-pytorch) can be downloaded [here](https://pennstateoffice365-my.sharepoint.com/:u:/g/personal/fuy34_psu_edu/ERBeMZVm8UhNnQNIg1zXe6IBfLVpTxJtYuPymgU1TqjAbQ?e=g9u9kX).

### Pre-processing
For we directly use pre-processed data, so no pre-processing is needed.

## Training
1. Choose the dataset you want to train with:
   - To train with **DDFF-12** dataset, you may change the following lines in `WDFV/pytorch/train.py`
   ```C++
   // ./pytorch/train.py
   train_with_ddff = True
   train_with_fod = False
   ```
   - To train with **FoD500** dataset, you may change the following lines in `WDFV/pytorch/train.py`
   ```C++
   // ./pytorch/train.py
   train_with_ddff = False
   train_with_fod = True
   ```
2. Run `python train.py` under `WDFV/pytorch/` folder ***OR*** If you are using a **LINUX** system, you can directy execute `train.sh` under `WDFV/` folder.

## Evaluation
* If you want to evaluate with our models:
    1. Please download them from the links below first, and place them under `WDFV/saved_models/` folder.
        * [FoD500 model](https://drive.google.com/file/d/1YWEfN7I9judYGOxXdA0LCD87zMoTfIKd/view?usp=drive_link)
        * [DDFF-12 model](https://drive.google.com/file/d/1bCieUrhisVGLRBsPPTUD_i85xWl-ZcSP/view?usp=drive_link)
    2. If you want to evaluate on DDFF-12 dataset, run `python eval_DDFF12.py` under `WDFV/pytorch/eval/` folder ***OR*** directly execute `eval_DDFF12.sh` under `WDFV/` folder.
    3. If you want to evaluate on FoD500 dataset, run `python FoD_test.py` under `WDFV/pytorch/eval/` folder ***OR*** directly execute `eval_FoD500.sh` under `WDFV/` folder.
* If you want to train your own models, you can find your models also under `WDFV/saved_models/` folder.
    1. You shold first change the `mdl_path` in `WDFV/pytorch/eval/eval_DDFF12.py` and `WDFV/pytorch/eval/FoD_test.py` to the following:
        ```
        epoch = 233*3 // change this to the number of epoch you want to evaluate on
        // ...
        mdl_path = "../../saved_models/WDFV_e{:0>3d}.depth_net.mdl".format(epoch)
        # mdl_path = "../../saved_models/WDFV-DDFF_431.depth_net.mdl"
        ```
    3. If you want to evaluate on DDFF-12 dataset, run `python eval_DDFF12.py` under `WDFV/pytorch/eval/` folder ***OR*** directly execute `eval_DDFF12.sh` under `WDFV/` folder.
    4. If you want to evaluate on FoD500 dataset, run `python FoD_test.py` under `WDFV/pytorch/eval/` folder ***OR*** directly execute `eval_FoD500.sh` under `WDFV/` folder.

# Acknowledgement
Parts of the code are developed from [DFV](https://github.com/fuy34/DFV), [DDFF-pytorch](https://github.com/soyers/ddff-pytorch), [DDFF-toolbox](https://github.com/hazirbas/ddff-toolbox) and [DefocusNet](https://github.com/dvl-tum/defocus-net).
