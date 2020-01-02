<br/>
<p align="center">
<img src="https://raw.githubusercontent.com/stevensmiley1989/Kaggle-WisconsinBreastCancer/master/GitHub_Images/BreastCancer_TitleGIF.gif">
</p>

# Kaggle-Wisconsin-Breast-Cancer
## Repository by Steven Smiley

This respository hosts the files I used to create my Kaggle submission for using Machine Learning to Diagnose Breast Cancer in Python using the Kaggle dataset from the Wisconsin Breast Cancer study.


# Table of Contents to Repository 

* [1. Background](#1_Background)
* [2. Jupyter Notebook](#2_Jupyter_Notebook)
* [3. Inputs](#3_Inputs)
   * [data.csv](#3_Inputs)
* [4. Outputs](#4_Outputs)
   * [Models](#4.1_Models)
   * [Figures](#4.2_Figures)
* [5. credits](#5_credits)
* [6. contact-info](#6_contact-info)
* [7. license](#7_license)


## 1_Background
### 1.1_Description of Kaggle Project 

Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image. n the 3-dimensional space is that described in: [K. P. Bennett and O. L. Mangasarian: "Robust Linear Programming Discrimination of Two Linearly Inseparable Sets", Optimization Methods and Software 1, 1992, 23-34].

This database is also available through the UW CS ftp server: ftp ftp.cs.wisc.edu cd math-prog/cpo-dataset/machine-learn/WDBC/

Also can be found on UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29

Attribute Information:

1) ID number 2) Diagnosis (M = malignant, B = benign) 3-32)

Ten real-valued features are computed for each cell nucleus:

* a) radius (mean of distances from center to points on the perimeter) 
* b) texture (standard deviation of gray-scale values) 
* c) perimeter 
* d) area 
* e) smoothness (local variation in radius lengths) 
* f) compactness (perimeter^2 / area - 1.0) 
* g) concavity (severity of concave portions of the contour) 
* h) concave points (number of concave portions of the contour) 
* i) symmetry 
* j) fractal dimension ("coastline approximation" - 1)

The mean, standard error and "worst" or largest (mean of the three largest values) of these features were computed for each image, resulting in 30 features. For instance, field 3 is Mean Radius, field 13 is Radius SE, field 23 is Worst Radius.

All feature values are recoded with four significant digits.

Missing attribute values: none

Class distribution: 357 benign, 212 malignant

### 1.2_Problem Statement:

Find a Machine Learning (ML) model that accurately predicts breast cancer based on the 30 features extracted.

## 2_Jupyter_Notebook
Jupyter Notebook(s) written in Python.

| Notebook | Description |
|--------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [ML_for_Diagnosing_Breast_Cancer-Steven_Smiley.ipynb](http://nbviewer.ipython.org/github/stevensmiley1989/Kaggle-WisconsinBreastCancer/blob/master/ML_for_Diagnosing_Breast_Cancer-Steven_Smiley.ipynb) | My Jupyter notebook written in Python. |

## [3_Inputs](https://github.com/stevensmiley1989/Kaggle-WisconsinBreastCancer/tree/master/Inputs)
Single input file (data.csv) contains all of the information for the Wisconsin Breast Cancer dataset.

[data.csv](https://github.com/stevensmiley1989/Kaggle-WisconsinBreastCancer/blob/master/Inputs/data.csv)

## [4_Outputs](https://github.com/stevensmiley1989/Kaggle-WisconsinBreastCancer/tree/master/Outputs)
The Outputs from the Jupyter notebook are placed in the following two folders: Models & Figures

### [4.1_Models](https://github.com/stevensmiley1989/Kaggle-WisconsinBreastCancer/tree/master/Outputs/Models)




### [4.2_Figures](https://github.com/stevensmiley1989/Kaggle-WisconsinBreastCancer/tree/master/Outputs/Figures)

<br/>
<p align="center">
<img src="https://raw.githubusercontent.com/stevensmiley1989/Kaggle-WisconsinBreastCancer/master/Outputs/Figures/Figure1.Heatmap.png">
</p>

<p align="center">
<img src="https://raw.githubusercontent.com/stevensmiley1989/Kaggle-WisconsinBreastCancer/master/Outputs/Figures/Figure2.A_LR_Confusion_Matrix.png">
<br/>
</p>

<p align="center">
<img src="https://raw.githubusercontent.com/stevensmiley1989/Kaggle-WisconsinBreastCancer/master/Outputs/Figures/Figure3.A_SVM_Confusion_Matrix.png">
<br/>
</p>

<p align="center">
<img src="https://raw.githubusercontent.com/stevensmiley1989/Kaggle-WisconsinBreastCancer/master/Outputs/Figures/Figure4.A_MLP_Confusion_Matrix.png">
<br/>
</p>

<p align="center">
<img src="https://raw.githubusercontent.com/stevensmiley1989/Kaggle-WisconsinBreastCancer/master/Outputs/Figures/Figure5.A_RF_Confusion_Matrix.png">
<br/>
</p>

<p align="center">
<img src="https://raw.githubusercontent.com/stevensmiley1989/Kaggle-WisconsinBreastCancer/master/Outputs/Figures/Figure5.B_RF_Variable_Importance_Plot.png">
<br/>
</p>

<p align="center">
<img src="https://raw.githubusercontent.com/stevensmiley1989/Kaggle-WisconsinBreastCancer/master/Outputs/Figures/Figure6.A_GB_Confusion_Matrix.png">
<br/>
</p>

<p align="center">
<img src="https://raw.githubusercontent.com/stevensmiley1989/Kaggle-WisconsinBreastCancer/master/Outputs/Figures/Figure6.B_GB_Variable_Importance_Plot.png">
<br/>
</p>


<p align="center">
<img src="https://raw.githubusercontent.com/stevensmiley1989/Kaggle-WisconsinBreastCancer/master/Outputs/Figures/Figure7.A_XGB_Confusion_Matrix.png">
<br/>
</p>

<p align="center">
<img src="https://raw.githubusercontent.com/stevensmiley1989/Kaggle-WisconsinBreastCancer/master/Outputs/Figures/Figure7.B_XGB_Variable_Importance_Plot.png">
<br/>
</p>

## 5_credits

* [Kaggle](https://www.kaggle.com/)

## 6_contact-info

Feel free to contact me to discuss any issues, questions, or comments.

* Email: [stevensmiley1989@gmail.com](mailto:stevensmiley1989@gmail.com)
* GitHub: [stevensmiley1989](https://github.com/stevensmiley1989)
* LinkedIn: [stevensmiley1989](https://www.linkedin.com/in/stevensmiley1989)
* Kaggle: [stevensmiley](https://www.kaggle.com/stevensmiley)

### 7_License

This repository contains a variety of content; some developed by Steven Smiley, and some from third-parties.  The third-party content is distributed under the license provided by those parties.

The content developed by Steven Smiley is distributed under the following license:

*I am providing code and resources in this repository to you under an open source license.  Because this is my personal repository, the license you receive to my code and resources is from me and not my employer. 

   Copyright 2020 Steven Smiley

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
   
