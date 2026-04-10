# Learning Robustness at Test Time

This repository contains a compact PyTorch implementation of Teacher-guided Robust Adaptation (TgRA) for CIFAR-10 and ImageNet.
The main experiment script can:

- load a pretrained CIFAR-10 or ImageNet classifier,
- finetune it with loss: **TGRA**
- apply compound corruptions during training/evaluation,
- report natural and PGD robust top-1/top-5 accuracy


---

## Files

- `finetuning.py`  
  Main experiment runner:
  - load a pretrained CIFAR-10 or ImageNet model,
  - finetuning it at test-time based on the loss 
  - runs evaluation

- `loss.py`  
  Robust finetuning losses:
  - TGRA and TGRA-FGSM,
  - TRADES and TRADES-U,
  - MART,
  - PGD adversarial training,
  - DKL finetuning utilities.

- `eval.py`  
  Evaluation helpers:
  - natural accuracy,
  - adversarial accuracy with PGD-20, PGN, or Square attack.

- `data_augmentation.py`  
  Compound corruption pipeline with Gaussian noise, blur, and color jitter controlled by `--severity`.

- `util.py`  
  Reproducibility, model loading, and dataset splitting helpers.

- `models/`  
  Normalization wrappers and the bundled WideResNet implementation.

---

## Setup

Create an environment with the core dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision numpy scikit-learn filelock tqdm torchattacks
```

`eval.py` also imports `TransferAttack` for PGN evaluation, so keep that package available in your environment if you use the script as-is.

Model notes:

- The bundled CIFAR-10 model implementation is `WideResnet`.
- A WideResNet demo run expects a pretrained checkpoint at `models/cifar10_models/state_dicts/WideResnet.pt`.
- The other CIFAR-10 model names in the CLI (`VGG11`, `VGG16`, `Resnet18`, and so on) expect the corresponding modules to exist under `models/cifar10_models/`.
- ImageNet experiments use torchvision models and expect ImageNet data at the path configured in `finetuning.py`.

---

## Demo

After placing a pretrained WideResNet checkpoint in `models/cifar10_models/state_dicts/WideResnet.pt`, run a small CIFAR-10 finetuning experiment:

```bash
python finetuning.py \
  --dataset cifar10 \
  --model WideResnet \
  --loss tgra \
  --beta 6 \
  --eps 8 \
  --step_size 2 \
  --num_step 5 \
  --epoch 30 \
  --frq_test 3 \
  --split 50 \
  --corruption 1 \
  --severity 1 \
  --lr 1e-3 \
  --scheduler vanilla \
  --seed 0
```

This demo:

- downloads CIFAR-10 into `./data`,
- trains on half test-set of CIFAR-10,
- applies compound corruption with severity `1`,
- evaluates natural accuracy and PGD robust accuracy,
- saves a checkpoint under `models/cifar10_models/state_dicts/`


## unsupervised variant

**TRADESU**:

```bash
python finetuning.py --dataset cifar10 --model WideResnet --loss tradesU --beta 6 --eps 8 --step_size 2 --num_step 5 --epoch 30 --frq_test 3 --split 50 --corruption 1 --severity 1
```

## Supervised approach:

**PGD**:

```bash
python finetuning.py --dataset cifar10 --model WideResnet --loss pgd --eps 8 --step_size 2 --num_step 5 --epoch 30 --frq_test 3 --split 50
```

**TRADES**:

```bash
python finetuning.py --dataset cifar10 --model WideResnet --loss trades --eps 8 --beta 6 --step_size 2 --num_step 5 --epoch 30 --frq_test 3 --split 50
```

**MART**:

```bash
python finetuning.py --dataset cifar10 --model WideResnet --loss mart --eps 8 --beta 6 --step_size 2 --num_step 5 --epoch 30 --frq_test 3 --split 50
```

**DKL** finetuning: 

```bash
python finetuning.py --dataset cifar10 --model WideResnet --loss dkl --eps 8 --step_size 2 --num_step 5 --epoch 30 --beta 20 --frq_test 3 --split 50
```

