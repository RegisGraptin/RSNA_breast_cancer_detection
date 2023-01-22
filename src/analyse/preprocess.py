
import glob as glob
import os

import cv2
import dicomsdl as dicom
import numpy as np
import pydicom
from PIL import Image
from pydicom.pixel_data_handlers.util import apply_voi_lut

class MammographyPreprocess:
    """Mammography pre-process class.
  
    Additionnal information on the DICOM images:
    - https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.11.2.html#sect_C.11.2.1.2.1  
    - https://dicom.innolitics.com/ciods/ct-image/voi-lut/00281056
        
    """
    
    def _image_normalization_min_max(self, matrix: np.array, EPSILON : np.float32 = 1e-6) -> np.array:
        """Image normalization by using the min and max.

        Args:
            matrix (np.array): Input images.
            EPSILON (np.float32, optional): Minimal value possible. Defaults to 1e-6.

        Returns:
            np.array: Normalized image.
        """
        m_max, m_min = matrix.max(), matrix.min()
        matrix = (matrix - m_min) / (m_max - m_min + EPSILON)
        matrix = matrix * 255
        matrix = matrix.astype(np.uint8)
        return matrix

    def _dicom2array(self, path: str, voi_lut = True, fix_monochrome = True) -> np.array:
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

    def _dicom2array_dicomsdl(self, path: str, voi_lut = True, fix_monochrome = True) -> np.array:
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

    def _crop_image_roi(self, img: np.array) -> np.array:
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

    def _image_enhancement_clahe(self, img: np.array, clipLimit=30.0, tileGridSize=(8,8)) -> np.array:
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

    def _determine_breast_side(self, img: np.array) -> str:
        """Determine the breast side from an image.

        Args:
            img (np.array): Input image

        Returns:
            str: Return the side of the breast on the image.
        """
        col_sums_split = np.array_split(np.sum(img, axis=0), 2)
        left_col_sum   = np.sum(col_sums_split[0])
        right_col_sum  = np.sum(col_sums_split[1])
        if left_col_sum > right_col_sum:
            return 'L'
        else:
            return 'R'

    def _flip_breast_side(self, img: np.array, breast_side: str = 'L') -> np.array:
        """Flip the breast horizontally on the chosen side.

        https://www.kaggle.com/code/paulbacher/custom-preprocessor-rsna-breast-cancer/notebook

        Args:
            img (np.array): Input image.
            breast_side (str, optional): Select the side for the breast image. Defaults to 'L'.

        Returns:
            np.array: Image flipped on the selected breast side.
        """
        img_breast_side = self._determine_breast_side(img)
        if img_breast_side == breast_side:
            return img
        else:
            return np.fliplr(img)

    def _process_png_image(path: str, width: int, height: int) -> np.array:
        """Process png image.

        Args:
            path (str): Path of the image.
            width (int): Width of the output image
            height (int): Height of the output image.

        Returns:
            np.array: Processed the input image.
        """
        img = Image.open(path)
        img = np.asarray(img)
        img = cv2.resize(img, (width, height))
        return img

    def _process_dicom_image(self, path: str, width: int, height: int) -> np.array:
        """Process a dicom image.

        Args:
            path (str): Path of the dicom file.
            width (int): Width of the output image.
            height (int): Height of the output image.

        Returns:
            np.array: Image pre processed.
        """
        img = self._dicom2array_dicomsdl(path)
        img = self._image_normalization_min_max(img)
        img = self._flip_breast_side(img)
        img = self._crop_image_roi(img)

        # Create the image with the channel
        img = cv2.merge((
            img,
            self._image_enhancement_clahe(img, clipLimit=1.0),
            self._image_enhancement_clahe(img)
        ))
        
        img = cv2.resize(img, (width, height))
        return img

    def preprocess_mammography(self, path: str, width: int = 224, height: int = 224) -> np.array:
        """Pre process the mammography image.

        Args:
            path (str): Path of the image.
            width (int, optional): Width of the output image. Defaults to 224.
            height (int, optional): Height of the output image. Defaults to 224.

        Raises:
            Exception: If the image extention is not `.dcm` or `.png`.

        Returns:
            np.array: Processed image.
        """
        _, file_extension = os.path.splitext(path)
        
        # Read the dicome raw file
        if file_extension == ".dcm":
            return self._process_dicom_image(path, width, height)
        
        # Read the image already preprocess
        elif file_extension == ".png":
            return self._process_png_image(path, width, height)

        else:
            raise Exception("The file type is not supported.")

    def __init__(self) -> None:
        pass
