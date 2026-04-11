# HSPGNN

PyTorch implementation of **HSPGNN / HSPGCN** for spatio-temporal missing value imputation.

This repository has been cleaned up for GitHub sharing so that other researchers can more easily:
- install dependencies
- understand the project structure
- reproduce training runs
- cite and extend the code

## Project structure

```text
.
├── HSPGCN.py                # training entrypoint
├── model.py                 # model definitions: HSPGCN / HSPGCN_L
├── utils.py                 # core neural-network blocks
├── densities.py             # flow-related density functions
├── fit_flow.py              # normalizing-flow training utilities
├── normalizing_flows.py     # flow modules
├── src/
│   ├── normalizing_flows.py # normalizing flow
│   └── utils.py             # normalizing flow helpers
├── save_params/
│   ├── block-HSPGCN_L_pems_bay_epoch_48_0.08.params # saved block missing pems-bay
│   ├── point-HSPGCN_L_pems_bay_epoch_39_0.08.params # saved point missing pems-bay
│   ├── block---HSPGCN_Electricity_epoch_35_0.01.params # saved block missing Electricity
│   ├── point--HSPGCN_Electricity_epoch_50_0.01.params # saved point missing Electricity
│   ├── HSPGCN_AQI_epoch_50_2.02.paramss             # saved missing AQI
│   └── HSPGCN_AQI36_epoch_38_1.18.params            # saved missing AQI36
├── lib/
│   ├── data_preparation.py  # dataset loading and mask generation
│   ├── metrics.py           # evaluation metrics
│   └── utils.py             # training / evaluation helpers
├── data/                    # datasets and adjacency files
└── requirements.txt
```

## Environment

Recommended Python version:
- Python 3.9–3.11

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data preparation

The code currently supports these dataset names:
- `pems_bay`
- `Electricity`
- `AQI`
- `AQI36`

Place the corresponding raw data and adjacency files under the `data/` directory expected by the scripts.

## For datasets download
As for the pems-bay dataset and AQI,you can refer to 
https://github.com/Graph-Machine-Learning-Group/grin/tree/main
As for Electricity dataset, you can refer to
https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014

or you can download them from cloud storage （pan.Baidu.com） as well as the adjacent matrix through:
https://pan.baidu.com/s/1V6gp5VhpUxJ8wnzGBbA0wQ?pwd=kpxm access code: kpxm


### Notes on missing-value simulation

Missing masks are generated in `lib/data_preparation.py`:
- Line 152 ---`pems_bay_mask_generator()` for **pems_bay**
- Line 79 --- `Electricity_mask_generator()` for **Electricity**

block missing is:
pems_bay_mask_generator(data_seq.shape, p_fault=0.0015, p_noise=0.05, min_seq=12, max_seq=12 * 4)
Electricity_mask_generator(data_seq1.shape, p_fault=0.0015, p_noise=0.05, min_seq=12, max_seq=12 * 4)

point missing is:
pems_bay_mask_generator(data_seq.shape, p_fault=0.0, p_noise=0.25, min_seq=12, max_seq=12 * 4)
Electricity_mask_generator(data_seq1.shape, p_fault=0.0, p_noise=0.25, min_seq=12, max_seq=12 * 4)

You can change the corruption settings by editing:
- `p_fault`
- `p_noise`
- `min_seq`
- `max_seq`

## Training

Example:

```bash
python HSPGCN.py \
  --model HSPGCN_L \
  --mode train \
  --data_name pems_bay \
  --num_point 325 \
  --device cuda:0 \
  --max_epoch 50
```

```bash
python HSPGCN.py \
  --model HSPGCN \
  --mode train \
  --data_name AQI36 \
  --num_point 36 \
  --device cuda:0 \
  --max_epoch 50
```

```bash
python HSPGCN.py \
  --model HSPGCN \
  --mode train \
  --data_name AQI \
  --num_point 437 \
  --device cuda:0 \
  --max_epoch 50
```

```bash
python HSPGCN.py \
  --model HSPGCN \
  --mode train \
  --data_name Electricity \
  --num_point 370 \
  --device cuda:0 \
  --max_epoch 50
```


Main options:
- `--model`: `HSPGCN` or `HSPGCN_L`
- `--mode`: `train` or `test`
- `--data_name`: dataset name
- `--num_point`: number of nodes
- `--device`: `cuda:0`, `cuda:1`, or `cpu`
- `--max_epoch`: training epochs
- `--learning_rate`: optimizer learning rate
- `--decay`: exponential LR decay
- `--force`: overwrite existing experiment directory
- `--seed`: random seed



## Outputs

Training produces:
- checkpoint files under `experiment_<MODEL>/<MODEL>_<DATASET>/`
- saved normalization statistics
- prediction `.npz` results in the project root



## Citation

If you use this repository in your work, please cite the corresponding paper and consider citing the code repository as well.

Suggested BibTeX template for the repository:

```bibtex
@inproceedings{liang2024higher,
  title={Higher-order spatio-temporal physics-incorporated graph neural network for multivariate time series imputation},
  author={Liang, Guojun and Tiwari, Prayag and Nowaczyk, S{\l}awomir and Byttner, Stefan},
  booktitle={Proceedings of the 33rd ACM international conference on information and knowledge management},
  pages={1356--1366},
  year={2024}
}
```

## Limitations

This repository is a cleaned research codebase, not yet a packaged library. Some dataset paths are still hard-coded in the original preprocessing logic and may need adjustment for your local environment.


## For the Inference and the imputation Outputs
If you want to skip the training process, you can apply our trained models as:
```bash
python HSPGCN.py \
  --model HSPGCN_L \
  --mode test \
  --checkpoint ./save_params/block-HSPGCN_L_pems_bay_epoch_48_0.08.params\
  --data_name pems_bay \
  --num_point 325 \
  --device cuda:0 \
  --max_epoch 50
```

```bash
python HSPGCN.py \
  --model HSPGCN \
  --mode test \
  --checkpoint ./save_params/HSPGCN_AQI36_epoch_38_1.18.params\  
  --data_name AQI36 \
  --num_point 36 \
  --device cuda:0 \
  --max_epoch 50
```

```bash
python HSPGCN.py \
  --model HSPGCN \
  --mode test \
  --checkpoint ./save_params/HSPGCN_AQI_epoch_50_2.02.params\    
  --data_name AQI \
  --num_point 437 \
  --device cuda:0 \
  --max_epoch 50
```

```bash
python HSPGCN.py \
  --model HSPGCN \
  --mode test \
  --checkpoint  ./save_params/block---HSPGCN_Electricity_epoch_35_0.01.params\
  --data_name Electricity \
  --num_point 370 \
  --device cuda:0 \
  --max_epoch 50
```
 
If you are interested in training and the reproduced results, you can download the saved results and training parameters at each epoch through:
https://pan.baidu.com/s/1oApZ1IYns4-GoQCrSmFmSg?pwd=y65x access code: y65x

