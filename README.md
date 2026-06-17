# Radar-Based Human Activity Recognition With 1-D Dense Attention Network

This repository contains the PyTorch implementation for the paper **"Radar-Based Human Activity Recognition With 1-D Dense Attention Network"**, published in *IEEE Geoscience and Remote Sensing Letters*.

The code implements a 1-D convolutional dense attention network for radar-based human activity recognition. The input radar signal is converted to a time-frequency representation with STFT and then classified into seven activity classes.

## Keywords

radar, human activity recognition, PyTorch, 1-D CNN, dense attention network, attention mechanism, STFT, time-frequency analysis

## Overview

The repository includes:

- The proposed 1-D Dense Attention Network with BAM-style attention modules.
- A baseline 1-D CNN model.
- An ablation model without the first attention module.
- Training and evaluation scripts for 5-fold cross-validation.
- Utilities for generating train/test lists and loading radar `.mat` data.

## Repository Structure

```text
.
|-- dense_bam_max_avg_wo_bottle.py          # Attention module implementation
|-- kim_300x153.py                          # Baseline 1-D CNN
|-- kim_dense_bam_300x153.py                # Proposed dense attention model
|-- kim_dense_bam_300x153_wo_first_att.py   # Ablation model
|-- logger.py                               # CSV training logger
|-- my_function_1.py                        # Data loading and list generation utilities
|-- run.py                                  # Training, testing, and checkpoint helpers
|-- train_kim_300x153.py                    # Train baseline model
|-- train_dense_bam_300x153.py              # Train proposed model
`-- train_dense_bam_300x153_wo_first_att.py # Train ablation model
```

## Requirements

The code was developed with Python and PyTorch. A typical environment includes:

```text
python
torch
numpy
scipy
scikit-learn
matplotlib
opencv-python
```

You can install the main dependencies with:

```bash
pip install torch numpy scipy scikit-learn matplotlib opencv-python
```

Please install a PyTorch version that matches your CUDA environment. See the official PyTorch installation page if GPU support is needed.

## Dataset

The dataset is not included in this repository.

The current data loader expects radar samples stored as MATLAB `.mat` files. Each `.mat` file should contain a variable named `d`, which is loaded by `scipy.io.loadmat`.

The expected dataset directory structure is:

```text
dataset_root/
|-- walk_hold/
|   |-- subject_1/
|   |   |-- sample_001.mat
|   |   `-- sample_002.mat
|-- crawl/
|-- boxing/
|-- walk/
|-- walk_boxing/
|-- run/
`-- sit/
```

The default label mapping used in `my_function_1.py` is:

```text
0: walk_hold
1: crawl
2: boxing
3: walk
4: walk_boxing
5: run
6: sit
```

## Data Lists

Training uses text files under `list/7act_5fold/`. Each line follows this format:

```text
path people_name label
```

Example:

```text
/path/to/dataset/walk_hold/subject_1/sample_001.mat 0 0
```

The repository contains code for generating the data lists automatically. To regenerate them, edit the following variables in the training script:

```python
Generate_List = True
Data_Path = '/path/to/your/radar_7act_data_I_Q/'
```

After list generation, you can set `Generate_List = False` for later training runs.

## Training

Before training, update the GPU setting and dataset path in the training script:

```python
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
Data_Path = '/path/to/your/radar_7act_data_I_Q/'
```

Train the proposed 1-D Dense Attention Network:

```bash
python train_dense_bam_300x153.py
```

Train the baseline 1-D CNN:

```bash
python train_kim_300x153.py
```

Train the ablation model without the first attention module:

```bash
python train_dense_bam_300x153_wo_first_att.py
```

The default training configuration is:

```text
Input shape: 300 x 153
Number of classes: 7
Cross-validation: 5 folds
Batch size: 64
Epochs: 300
Optimizer: Adam
Initial learning rate: 1e-4
Weight decay: 4e-5
NFFT: 51
Overlap: 12
Padding length: 300
```

## Outputs

Training logs and checkpoints are saved under:

```text
results/<timestamp>/
```

Each fold writes a CSV log file:

```text
fold0results.csv
fold1results.csv
...
```

Model checkpoints are saved as:

```text
fold<k>checkpoint.pth.tar
model_best.pth.tar
```

Intermediate list files and copied split files may also be written under `keras_result/` and `list/`.

## Notes

- The file paths in `list/7act_5fold/*.txt` may contain absolute paths from the original environment. Regenerate the lists or edit the paths before running on a new machine.
- The current training scripts call `.cuda()` directly, so a CUDA-capable GPU is expected. For CPU training, the scripts need minor modifications.
- Large datasets, trained weights, virtual environments, and generated results should usually be excluded from Git.

## Citation

If this code is useful for your research, please cite the paper:

```bibtex
@ARTICLE{9312969,
  author={Lai, Guoji and Lou, Xin and Ye, Wenbin},
  journal={IEEE Geoscience and Remote Sensing Letters},
  title={Radar-Based Human Activity Recognition With 1-D Dense Attention Network},
  year={2022},
  volume={19},
  number={},
  pages={1-5},
  keywords={Convolution;Radar;Spectrogram;Time-frequency analysis;Radar imaging;Activity recognition;Feature extraction;1-D convolutional network;attention mechanism;human activity recognition;radar},
  doi={10.1109/LGRS.2020.3045176}
}
```

## License

Please add a license file if you plan to publicly release or redistribute this code.
