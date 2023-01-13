from PIL import Image

import numpy as np
from skimage.transform import resize

import dicomsdl as dicom

def image_normalization_min_max(matrix: np.array, EPSILON : np.float32 = 1e-6) -> np.array:
    """Image normalization by using the min and max."""
    m_max, m_min = matrix.max(), matrix.min()
    matrix = (matrix - m_min) / (m_max - m_min + EPSILON)
    matrix = matrix.astype(np.float16)
    return matrix

def open_dicom_image(path: str, width: int = 224, height: int = 224):
    """Open the dicom image from the path. Normalize & resize."""
    dset = dicom.open(path)
    data = dset.pixelData()
    img  = image_normalization_min_max(data)
    img  = resize(img, (width, height), anti_aliasing=True)
    img  = np.stack((img,) * 3, axis=-1)
    img  = img.reshape(width, height, 3)
    return img

def open_png_image(path: str, width: int = 224, height: int = 224):
    img = Image.open(path)
    img = np.asarray(img)
    img = image_normalization_min_max(img)
    img = resize(img, (width, height), anti_aliasing=True)
    img = np.stack((img,) * 3, axis=-1)
    img = img.reshape(width, height, 3)
    img = img.astype(np.uint8)
    return img


