
import glob as glob

import cv2
import dicomsdl as dicom
import matplotlib.pyplot as plt
import numpy as np
import pydicom
from PIL import Image
from pydicom.pixel_data_handlers.util import apply_voi_lut
from skimage.transform import resize

# Additional information on DICOM images 
## https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.11.2.html#sect_C.11.2.1.2.1
## https://dicom.innolitics.com/ciods/ct-image/voi-lut/00281056


def image_normalization_min_max(matrix: np.array, EPSILON : np.float32 = 1e-6) -> np.array:
    """Image normalization by using the min and max.

    Args:
        matrix (np.array): Input images.
        EPSILON (np.float32, optional): Minimal value possible. Defaults to 1e-6.

    Returns:
        np.array: Normalized image.
    """
    m_max, m_min = matrix.max(), matrix.min()
    matrix = (matrix - m_min) / (m_max - m_min + EPSILON)
    # matrix = matrix.astype(np.float16)
    matrix = matrix * 255
    matrix = matrix.astype(np.uint8)
    return matrix

def dicom2array(path: str, voi_lut = True, fix_monochrome = True) -> np.array:
    """Transform a dicom file to an array by using the pydicom library. 

    Args:
        path (str): Path to the dicom file.
        
        voi_lut (bool, optional): Apply VOI LUT transformation. Defaults to True.
            VOI LUT (Value of Interest - Look Up Table) : The idea is to have a larger representation 
            of the data. Since, dicom files have larger pixel display range than usuall pictures. The 
            idea is to keep a larger representation in order to better see the subtle differences.
    
        fix_monochrome (bool, optional): Indicate if we fix the pixel value for specific files. 
            Defaults to True. Some images have MONOCHROME1 interpretation, which means that higher 
            pixel values corresponding to the dark instead of the white.

    Returns:
        np.array: Extracted numpy array.
    """
    
    dicom = pydicom.read_file(path)
    
    # Apply the VOI LUT
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
        
    # Fix the representation
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
    
    return data

def dicom2array_dicomsdl(path: str, voi_lut = True, fix_monochrome = True) -> np.array:
    """Transform a dicom file to an array by using the dicomsdl library.
    It should be faster than using the pydicom library.

    References: https://www.kaggle.com/code/bobdegraaf/dicomsdl-voi-lut

    Args:
        path (str): Path of the dicom file.
        voi_lut (bool, optional): Apply VOI LUT transformation. Defaults to True.
        fix_monochrome (bool, optional): Indicate if we fix the pixel value for specific files. 
            Defaults to True.

    Returns:
        np.array: Extracted numpy array.
    """
    dataset = dicom.open(path)
    img = dataset.pixelData()
    
    if voi_lut:
        # Load only the variables we need
        center = dataset["WindowCenter"]
        width = dataset["WindowWidth"]
        bits_stored = dataset["BitsStored"]
        voi_lut_function = dataset["VOILUTFunction"]

        # For sigmoid it's a list, otherwise a single value
        if isinstance(center, list):
            center = center[0]
        if isinstance(width, list):
            width = width[0]

        # Set y_min, max & range
        y_min = 0
        y_max = float(2**bits_stored - 1)
        y_range = y_max

        # Function with default LINEAR (so for Nan, it will use linear)
        if voi_lut_function == "SIGMOID":
            img = y_range / (1 + np.exp(-4 * (img - center) / width)) + y_min
        else:
            # Checks width for < 1 (in our case not necessary, always >= 750)
            center -= 0.5
            width -= 1

            below = img <= (center - width / 2)
            above = img > (center + width / 2)
            between = np.logical_and(~below, ~above)

            img[below] = y_min
            img[above] = y_max
            if between.any():
                img[between] = (
                    ((img[between] - center) / width + 0.5) * y_range + y_min
                )
    
    if fix_monochrome and dataset["PhotometricInterpretation"] == "MONOCHROME1":
        img = np.amax(img) - img

    return img

def crop_image_roi(img: np.array) -> np.array:
    """Given a mammography, we can crop the images based on the region of interest. 
    Additionnaly of some of the images, we can have a letter on it. This process allow us 
    to remove them.
    
    References:
        - https://www.kaggle.com/code/fabiendaniel/dicom-cropped-resized-png-jpg
        - https://www.kaggle.com/code/davidbroberts/mammography-remove-letter-markers
        - https://www.kaggle.com/code/theoviel/dicom-resized-png-jpg
    
    Args:
        img (np.array): Normalized mammography with pixel values of [0;255]

    Returns:
        np.array: Cropped mammography.
    """

    # Binarize the image
    bin_pixels = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)[1]
   
    # Make contours around the binarized image, keep only the largest contour
    contours, _ = cv2.findContours(bin_pixels, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)

    # Create a mask from the largest contour
    mask = np.zeros(img.shape, np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
   
    # Use bitwise_and to get masked part of the original image
    out = cv2.bitwise_and(img, mask)
    
    # get bounding box of contour
    y1, y2 = np.min(contour[:, :, 1]), np.max(contour[:, :, 1])
    x1, x2 = np.min(contour[:, :, 0]), np.max(contour[:, :, 0])
    
    x1 = int(0.99 * x1)
    x2 = int(1.01 * x2)
    y1 = int(0.99 * y1)
    y2 = int(1.01 * y2)
    
    return out[y1:y2, x1:x2]

def image_enhancement_clahe(img: np.array, clipLimit=30.0, tileGridSize=(8,8)):
    """Apply CLAHE on images.
    
    References: 
        - https://www.kaggle.com/code/rerere/covid-19-image-enhancement

    Args:
        img (np.array): Input image normalized.
        clipLimit (float, optional): _description_. Defaults to 30.0.
        tileGridSize (tuple, optional): _description_. Defaults to (8,8).

    Returns:
        np.array: Enhanced image.
    """
    clahe = cv2.createCLAHE(
        clipLimit    = clipLimit, 
        tileGridSize = tileGridSize
    )
    return clahe.apply(img)

def preprocess_mammography(path: str, width = 224, height = 224):
    img = dicom2array_dicomsdl(path)
    img = image_normalization_min_max(img)
    img = crop_image_roi(img)
    
    dim1 = img
    dim3 = image_enhancement_clahe(img, clipLimit=1.0)
    dim2 = image_enhancement_clahe(img)
    
    img = cv2.merge((dim1,dim2,dim3))
    
    
    # TODO :: De we want to keep image distortion by croping ?
    img = cv2.resize(img, (width, height))
    
    # TODO :: In resize think about maybe padding then resize to avoid image distortion
    # TODO :: Or to have a model that took directly images with larger shaper for one side
    # TODO :: Possiblity to flip the breast on the same size
    
    return img
    
    

def test_open_png_image(path: str, width: int = 224, height: int = 224):
    """Used only for testing purpose. Prefer to use the original dicom file."""
    img = Image.open(path)
    img = np.asarray(img)
    img = image_normalization_min_max(img)
    img = crop_image_roi(img)
    
    dim1 = img
    dim3 = image_enhancement_clahe(img, clipLimit=1.0)
    dim2 = image_enhancement_clahe(img)
    
    img = cv2.merge((dim1,dim2,dim3))
    img = cv2.resize(img, (width, height))
    return img


    
    
if __name__ == "__main__":
    
    FOLDER = "./data/dicom/"
    TMP_SHOW = "./tmp/"
    
    files = glob.glob(FOLDER + "*.dcm")
    
    for i, file in enumerate(files):
        final_img = preprocess_mammography(file, 512, 512)
        plt.imsave(f"./{TMP_SHOW}/{i}_final.png", final_img)
