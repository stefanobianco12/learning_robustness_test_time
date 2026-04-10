import torch
import numpy as np
import random
import numpy as np
import importlib
from torch.utils.data import Dataset,Subset
from sklearn.model_selection import StratifiedShuffleSplit

def split_dataset(dataset, split_ratio):
    if isinstance(dataset, Subset):
        targets = np.array(dataset.dataset.targets)
    elif isinstance(dataset, Dataset):
        targets = np.array(dataset.targets)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=split_ratio[1], train_size=split_ratio[0], random_state=42)
    train_indices, test_indices = next(sss.split(np.zeros(len(targets)), targets))
    subset_1 = Subset(dataset, train_indices)
    subset_2 = Subset(dataset, test_indices)
    return subset_1, subset_2

def split_cifar10_fixed(dataset, train_frac_of_full, test_frac=0.10,
                                seed_test=42, seed_train=123):
    """
    dataset: CIFAR10(train=False) dataset (transform can be None)
    train_frac_of_full: fraction of FULL dataset used for training (e.g. 0.5, 0.8, 0.9)
    test_frac: fixed evaluation fraction (always 0.10)

    Returns: train_idx, fixed_test_idx
    """
    y = np.array(dataset.targets)

    if train_frac_of_full > 1.0 - test_frac + 1e-12:
        raise ValueError(f"train_frac must be <= {1.0 - test_frac:.2f} because test is fixed at {test_frac:.2f}")

    # 1) Fixed 10% test set (stratified, reproducible)
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed_test)
    pool_idx, fixed_test_idx = next(sss_test.split(np.zeros(len(y)), y))

    # 2) Train from the remaining pool
    pool_y = y[pool_idx]
    train_size_within_pool = train_frac_of_full / (1.0 - test_frac)

    if np.isclose(train_size_within_pool, 1.0):
        train_idx = pool_idx
    else:
        sss_train = StratifiedShuffleSplit(n_splits=1, train_size=train_size_within_pool, random_state=seed_train)
        train_rel_idx, _ = next(sss_train.split(np.zeros(len(pool_idx)), pool_y))
        train_idx = pool_idx[train_rel_idx]

    return train_idx, fixed_test_idx

def _get_targets(dataset):
    if isinstance(dataset, Subset):
        base = dataset.dataset
        idx = np.array(dataset.indices)
    else:
        base = dataset
        idx = None

    if hasattr(base, "targets"):
        y = np.array(base.targets)
    elif hasattr(base, "samples"):
        y = np.array([t for _, t in base.samples])
    else:
        raise ValueError("Dataset must have .targets or .samples")

    return y[idx] if idx is not None else y


def split_train_test_fixed_test(dataset, train_frac, test_frac=0.10,
                                seed_test=42, seed_train=123):
    """
    train_frac is fraction of FULL dataset (e.g. 0.5, 0.8, 0.9)
    test_frac is fixed (default 0.10)
    """
    y = _get_targets(dataset)

    # ---- fixed test split ----
    sss_test = StratifiedShuffleSplit(
        n_splits=1, test_size=test_frac, random_state=seed_test
    )
    pool_idx, test_idx = next(sss_test.split(np.zeros(len(y)), y))

    if train_frac > 1.0 - test_frac:
        raise ValueError(f"train_frac must be ≤ {1.0 - test_frac:.2f}")

    # ---- variable train split from remaining pool ----
    pool_y = y[pool_idx]
    train_size_in_pool = train_frac / (1.0 - test_frac)

    if np.isclose(train_size_in_pool, 1.0):
        train_idx = pool_idx
    else:
        sss_train = StratifiedShuffleSplit(
            n_splits=1, train_size=train_size_in_pool, random_state=seed_train
        )
        train_rel_idx, _ = next(
            sss_train.split(np.zeros(len(pool_idx)), pool_y)
        )
        train_idx = pool_idx[train_rel_idx]

    return Subset(dataset, train_idx), Subset(dataset, test_idx)



model_list = {
    "WideResnet": ('models.cifar10_models.wideresnet', 'WideResNet'),
    "VGG11": ('models.cifar10_models.vgg', 'vgg11_bn'),
    "VGG16": ('models.cifar10_models.vgg', 'vgg16_bn'),
    "VGG19": ('models.cifar10_models.vgg', 'vgg19_bn'),
    "DenseNet161": ('models.cifar10_models.densenet', 'densenet161'),
    "InceptionV3": ('models.cifar10_models.inception', 'inception_v3'),
    "Resnet18": ('models.cifar10_models.resnet', 'resnet18'),
    "Resnet50": ('models.cifar10_models.resnet', 'resnet50'),
}


def load_model(model_name,pre=True):
    if model_name not in model_list:
        raise Exception('Unspported surrogate model {}'.format(model_name))
    module_path, class_name = model_list[model_name]
    module = importlib.import_module(module_path, __package__)
    model_class = getattr(module, class_name)
    if "WideResnet" in model_name:
        model = model_class()
    else:
        model = model_class(pretrained=pre)
    return model



def set_seed(seed):
    # Set Python random seed
    random.seed(seed)
    
    # Set numpy random seed
    np.random.seed(seed)
    
    # Set torch random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Set deterministic behavior (optional)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




