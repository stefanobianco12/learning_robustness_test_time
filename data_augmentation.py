import torch
import torchvision.transforms as T

def gaussian_noise_transform(severity):
    # sigma values for severity 0..5
    sigma = [0, 0.03, 0.06, 0.12, 0.2, 0.5][severity]

    if sigma == 0:
        return T.Lambda(lambda x: x)

    return T.Lambda(lambda x: x + sigma * torch.randn_like(x))

def blur_transform(severity):
    kernel_sizes = [1, 3, 5, 7, 9, 11]
    return T.GaussianBlur(kernel_size=kernel_sizes[severity])


def color_jitter_transform(severity):
    amounts = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    a = amounts[severity]

    if a == 0:
        return T.Lambda(lambda x: x)

    return T.ColorJitter(
        brightness=a,
        contrast=a,
        saturation=a,
        hue=0
    )

def get_compound_aug(severity):
    return T.Compose([
        gaussian_noise_transform(severity),
        blur_transform(severity),
        color_jitter_transform(severity),
    ])