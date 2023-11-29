# M2TR: Multi-modal Multi-scale Transformers for DeepfakeDetection

## Disclosure

This is a copy of the original repository [M2TR](https://github.com/wangjk666/M2TR-Multi-modal-Multi-scale-Transformers-for-Deepfake-Detection) with some minor changes to get and save predictions on a customized set of images.

## Quick Start

Create a new environment and install the requirements:

```bash
conda create -n m2tr python=3.10
conda activate m2tr
pip install torch tochvision
pip install -r requirements.txt
```

Download the weights provided by the authors in the [original repository](https://github.com/wangjk666/M2TR-Multi-modal-Multi-scale-Transformers-for-Deepfake-Detection). Weights' path should be specified in the config file in the `cfg[TEST][CHECKPOINT_TEST_PATH]` key

Specify the folder containing the images (or the folders containing the images) for which you want to get prediction in the `cfg[DATASET][ROOT_DIR]` key

Run the python script:

```bash
python run.py --cfg <config_file_name>.yaml
```

