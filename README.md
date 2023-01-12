# RSNA - Breast Cancer Detection 

This repository aim to provide a solution for the kaggle competition name [`RSNA Screening Mammography Breast Cancer Detection`](https://www.kaggle.com/competitions/rsna-breast-cancer-detection). The goal of this competition is to identify cases of breast cancer in mammograms from screening exams. 

It is important to identify cases of cancer for obvious reasons, but false positives also have downsides for patients. As millions of women get mammograms each year, a useful machine learning tool could help a great many people.

In this repository, we decided to decompose the different components of our network in sub module. Thus, it will be more modular to change this approach by simply changing the component by implementing another method. Also, by modularity, it will be more easy to transfer this approach to other use cases.

## Project initialization

In this project, we use the python package manager [`Poetry`](https://python-poetry.org/). If you are not familiar with it, you can simply take a look on the documentation in the website and [install it](https://python-poetry.org/docs/#installing-with-the-official-installer). Then, to install the dependencies of this project run:

```bash
poetry install
```

## Data

With the Kaggle competition, some data are provided. The RAW data are from DICOM files. However, due to large amount of data (more than 300 GB), I could not use them directly with my computer. An alternative solution is to use extracted data already resized. In order to achieve this, I download one of the [dataset from Radek Osmulski](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/discussion/369282) that provide JPEG images converted from DICOM files. You can find it [here](https://www.kaggle.com/datasets/radek1/rsna-mammography-images-as-pngs?select=images_as_pngs_cv2_dicomsdl_256).

*__Note__*: Be aware that the DICOM format is not similar as JPEG format. In the DICOM format we store the pixel intensity. This pixel is not define on 8 bits as we used to see for png images, but it can be defined by 12~16 bits of information per pixel. If you want more information, you can have a look at the article of [Amrit Virdee](https://towardsdatascience.com/a-matter-of-grayscale-understanding-dicom-windows-1b44344d92bd).


In our dataset, we can find two data:
- train.csv : extracted from kaggle
- train_images_processed_cv2_dicomsdl_256 : extracted from https://www.kaggle.com/datasets/radek1/rsna-mammography-images-as-pngs?select=images_as_pngs_cv2_dicomsdl_256


TODO :: We use dvc for data version controlling.

## Tracking models

For tracking the training of the model, we used, in this repository, MLFlow. If you want to see the evolution of your model during the training, you simply need to start mlflow as follow:

```bash
poetry run mlflow server
```

Then, you should have a webservice available at http://127.0.0.1:5000/.

*__NOTE__* It will create a `mlruns` folder where the data logs will be there.
 

## Training 



```
export PYTHONPATH="${PYTHONPATH}:./src/"
poetry run python ./src/main.py
```


# TODO List

- Clean the README
- PyTorch Model, create Timm model
- See for image transformation 
- Improve data pre processing 
- Clean code
- See loss + Read paper + Link
- Try hyperparameters

- When first pipeline process -> Go to kaggle experiment (train + inference)
- Read Kaggle post / previous competitions 

- Analyse Dicomm format / Get additionnal data from it



