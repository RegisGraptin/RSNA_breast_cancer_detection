# RSNA - Breast Cancer Detection 

This repository aim to provide a solution for the kaggle competition name [`RSNA Screening Mammography Breast Cancer Detection`](https://www.kaggle.com/competitions/rsna-breast-cancer-detection). The goal of this competition is to identify cases of breast cancer in mammograms from screening exams. 
Cancer is a disease caused by an **uncontrolled** division of **abnormal** cells in a part of the body. In this competition, it is important to identify cases of cancer, but false positives also have downsides for patients. As millions of women get mammograms each year, a useful machine learning tool could help a great many people.

In this repository, we decided to decompose the different components of our network in sub modules. Thus, it will be more modular to change our approach by simply changing the component by implementing another one or by using another configuration. Also, by modularity, it will be more easy to transfer this approach to other use cases.

## Project initialization

In this project, we use the python package manager [`Poetry`](https://python-poetry.org/). If you are not familiar with it, you can simply take a look on the documentation in the website and [install it](https://python-poetry.org/docs/#installing-with-the-official-installer). Then, to install the dependencies of this project run:

```bash
poetry install
```

## Data

With the Kaggle competition, some data are provided. The RAW data are from DICOM files. However, due to large amount of data (more than 300 GB), I could not use them directly with my computer. An alternative solution is to use extracted data already resized. In order to achieve this, I download one of the [dataset from Radek Osmulski](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/discussion/369282) that provide JPEG images converted from DICOM files. You can find it [here](https://www.kaggle.com/datasets/radek1/rsna-mammography-images-as-pngs?select=images_as_pngs_cv2_dicomsdl_256). This approach will be use in the testing part for the model, from my side, but you can download the real data and use them.

In our dataset, we can find multiples data in the `data` folder:
- train.csv : extracted from kaggle
- train_images_processed_cv2_dicomsdl_256 : extracted from https://www.kaggle.com/datasets/radek1/rsna-mammography-images-as-pngs?select=images_as_pngs_cv2_dicomsdl_256
- dicom: Some raw dicom extracted from the kaggle datasets.

Additionally, for the data, we used a Data Version Control with the [`dvc` librairy](https://dvc.org/doc). It allows us to do data versioning. You can use it to control the data that you download in the `data` folder.

*__Note__*: Be aware that the DICOM format is not similar as PNG format. In the DICOM format we store the pixel intensity. This pixel is not define on 8 bits as we used to see for png images, but it can be defined by 12~16 bits of information per pixel. If you want more information, you can have a look at the article of [Amrit Virdee](https://towardsdatascience.com/a-matter-of-grayscale-understanding-dicom-windows-1b44344d92bd).

### Additionally Source of Data

As we are using Deep Learning, the data is one of the key factor for the model improvement. Through some research, we can found additional database online providing new data:

- [Cancer Imaging Archive](https://www.cancerimagingarchive.net/collections/) have a large number of images available. I extracted some of them that are interesting, but didn't have the time to check them.
    - [ACRIN-Contralateral-Breast-MR (ACRIN 6667)](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70225026)
    - [Categorized Digital Database for Low energy and Subtracted Contrast Enhanced Spectral Mammography images (CDD-CESM)](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=109379611)
    - [Breast Cancer Screening – Digital Breast Tomosynthesis (BCS-DBT)](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=64685580)
    - [The Chinese Mammography Database (CMMD)](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70230508)
    - [The Cancer Genome Atlas Breast Invasive Carcinoma Collection (TCGA-BRCA)](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=3539225)
    - *[Curated Breast Imaging Subset of Digital Database for Screening Mammography (CBIS-DDSM)](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=22516629) - [Paper](https://www.nature.com/articles/sdata2017177)
    - [Multi-center breast DCE-MRI data and segmentations from patients in the I-SPY 1/ACRIN 6657 trials (ISPY1)](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=20643859)
    - [BREAST-DIAGNOSIS](https://wiki.cancerimagingarchive.net/display/Public/BREAST-DIAGNOSIS)
    - [Dynamic contrast-enhanced magnetic resonance images of breast cancer patients with tumor locations (Duke-Breast-Cancer-MRI)](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70226903)
    - [ACRIN 6698/I-SPY2 Breast DWI (ACRIN 6698)](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=50135447)
    - [I-SPY 2 Breast Dynamic Contrast Enhanced MRI (I-SPY2 TRIAL)](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70230072)
    - [The VICTRE Trial: Open-Source, In-Silico Clinical Trial For Evaluating Digital Breast Tomosynthesis](https://wiki.cancerimagingarchive.net/display/Public/The+VICTRE+Trial%3A+Open-Source%2C+In-Silico+Clinical+Trial+For+Evaluating+Digital+Breast+Tomosynthesis)
    - [QIN-BREAST](https://wiki.cancerimagingarchive.net/display/Public/QIN-Breast)
    - [QIN-BREAST-02](https://wiki.cancerimagingarchive.net/display/Public/QIN-BREAST-02)
    - [ACRIN-FLT-Breast (ACRIN 6688)](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=30671268)
    - [Single site breast DCE-MRI data and segmentations from patients undergoing neoadjuvant chemotherapy (Breast-MRI-NACT-Pilot)](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=22513764)


- [The mini-MIAS database of mammograms](http://peipa.essex.ac.uk/info/mias.html) containing `pgm` images.

- [VinDr-Mammo: A large-scale benchmark dataset for computer-aided detection and diagnosis in full-field digital mammography](https://physionet.org/content/vindr-mammo/1.0.0/)

- [DDSM: Digital Database for Screening Mammography](http://www.eng.usf.edu/cvprg/Mammography/Database.html)

*__Note__*: I listed some of the database that I found. I am not so sure that all of them can be used for our use case. Aditionally, it will requiere some pre-processing for each database to match these data to the competition. I currently do not have the time for that. But, feel free to use them according their license.

### Image preprocessing 

Given a dicom file, we decided to use multiple pre-processing on our images. First, we are using the `dicomsdl` librairy as it is more fast to extract the pixel data. Then, we apply [VOI LUT transformation](https://dicom.innolitics.com/ciods/ct-image/voi-lut/00281056) and fix monochrome on some of the images. 
Then, regarding the pixels values, we apply different treatments:
- Normalize the pixel value by using min-max approach.
- Crop on the area of interest.
- Image enhancement by using CLAHE.
- Resizing the image.

*__Note__*: 
- The resizing process is a part that will requiere some changes, in my opinion. Indeed, by resizing, we lose information on the image because it is distorted. I think it could be interesting to found a solution about it. 

- Additionally, it could be interesting to flip only on one side the breast images, as it could maybe help the model to learn only on one side. This is a hypothesis that could be interesting to test.

## Model

We will use a deep learning model. TODO...

### Approach


> Possible model optimization
BCE Loss
Adam optimizer lr 1->10 0.0001 10-15 => 0.00005
image 256x256
batchsize = 8



### Training

```bash
export PYTHONPATH="${PYTHONPATH}:./src/"
poetry run python ./src/train.py
```

#### Tracking models

For tracking the training of the model, we used, in this repository, MLFlow. If you want to see the evolution of your model during the training, you simply need to start mlflow as follow:

```bash
poetry run mlflow server
```

Then, you should have a webservice available at http://127.0.0.1:5000/.

*__NOTE__*: It will create a `mlruns` folder where the data logs will be.
 

## Improvement 

> Increase training dataset by using additional datasets.
> Data augmentation: 
    - Affine transformation + Horizontal Flip
    - (X-ray) ShiftScaleRotate, IAAAffine, Blur/GaussianBlur/MedianBlur, RandomBrightnessContrast, IAAAdditiveGaussianNoise/GaussNoise, HorizontalFlip
    - Mild rotations (up to 6 deg), shift, scale, shear and h_flip, for some images random level of blur and noise and gamma changes.
    - rotation, translation, scaling, and horizontal flipping + random constants
> Model Ensemble: Using in cross-validation models (Validation + training folds) => Ensemble prediction
> Increas image size



Read about: DBT (Digital Breast Tomosynthesis)
- https://radiopaedia.org/articles/digital-breast-tomosynthesis
- https://pubs.rsna.org/doi/full/10.1148/radiol.2019180760



# TODO List

- [] Clean the README
- [x] Preprocess the dicom data 
- [x] PyTorch Model, create Timm model
- [x] Add validation dataset during the traning
- [] Have Stratified split dataset
- [] Multiple model k-fold
- [x] See for image transformation 
- [] Add data augmentation 
- [] Clean code
- [] See loss + Read paper + Link
- [] Try hyperparameters
- [] [WeightedRandomSampler](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/discussion/376472)
- [] See for resizing issue
- [] First model -> Run kaggle (train + inference)
- [] Get more information on kaggle (post + previous competition) 
- [] Analyse Dicomm format / Get and use additionnal data from it


## Papers to read ?

- [] https://arxiv.org/abs/2301.01931
- [] https://arxiv.org/abs/2301.02554
- [] https://arxiv.org/abs/2301.02166
- [] Shen et al. "An interpretable classifier for high-resolution breast cancer screening images utilizing weakly supervised localisation."
- [] Wu et al: "The NYU breast cancer screening dataset" - 2019


## Ressources

- [Coursera - Yale - Introduction to Breast Cancer](https://www.coursera.org/learn/breast-cancer-causes-prevention)
