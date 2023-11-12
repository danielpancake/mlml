from numpy.lib import stride_tricks

import numpy as np


def pad(x: np.ndarray, padding: int) -> np.ndarray:
    """
    Pads the input tensor with zeros on the last two dimensions

    Args:
        x (np.ndarray): Input tensor of shape (B, C, H, W)
        padding (int): Padding size

    Returns:
        np.ndarray: Padded tensor of shape (B, C, H + 2 * padding, W + 2 * padding)
    """
    return np.pad(
        x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant"
    )


def dilate(x: np.ndarray, dilation: int) -> np.ndarray:
    """
    Dilates the input tensor on the last two dimensions

    Args:
        x (np.ndarray): Input tensor of shape (B, C, H, W)
        dilation (int): Dilation factor

    Returns:
        np.ndarray: Dilated tensor of shape (B, C, H + (H - 1) * (D - 1), W + (W - 1) * (D - 1))
    """
    B, C, H, W = x.shape
    D = dilation

    x_dilated = np.zeros((B, C, H + (H - 1) * (D - 1), W + (W - 1) * (D - 1)))
    x_dilated[:, :, ::D, ::D] = x

    return x_dilated


def im2col(x, output_dims, kernel_size, padding=0, stride=1, dilation=0):
    """
    Converts the input tensor to a matrix of sliding local blocks
    """
    # Make some aliases for brevity
    K, P, S, D = kernel_size, padding, stride, dilation
    B, C, _, _ = x.shape
    _, _, Ho, Wo = output_dims

    # Dilate if necessary
    if D > 0:
        x = dilate(x, D + 1)

    # Pad if necessary
    if P > 0:
        x = pad(x, P)

    Bs, Cs, Ksh, Ksw = x.strides

    # Creates a 6D tensor containing all of KxK parts of the input image
    intermid_shape = (B, C, Ho, Wo, K, K)
    strides = (Bs, Cs, S * Ksh, S * Ksw, Ksh, Ksw)

    return stride_tricks.as_strided(x, intermid_shape, strides)
