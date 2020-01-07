# Kaggle-WisconsinBreastCancer
## Repository by Steven Smiley

This respository hosts the files I used to create my Kaggle submission for using Machine Learning to Diagnose Breast Cancer in Python using the Kaggle dataset from the Wisconsin Breast Cancer study.


# Table of Contents to Repository 

* [1. Background](#1)
* [2. Jupyter Notebook](#2)
* [3. Inputs](#3)
   * [data.csv](#3)
* [4. Outputs](#4)
   * [Models](#4.1)
   * [Figures](#4.2)
* [5. credits](#5)
* [6. contact-info](#6)
* [7. license](#7)


## 1 Background<a class="anchor" id="1"></a>
### 1.1 Description of Kaggle Project <a class="anchor" id="1.1"></a>

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

### 1.2 Problem Statement:<a class="anchor" id="1.2"></a>

Find a Machine Learning (ML) model that accurately predicts breast cancer based on the 30 features extracted.

## 2 Jupyter_Notebook<a class="anchor" id="2"></a>
Jupyter Notebook(s) written in Python.

| Notebook | Description |
|--------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [ML_for_Diagnosing_Breast_Cancer-Steven_Smiley.ipynb](https://github.com/stevensmiley1989/Kaggle-WisconsinBreastCancer/blob/master/ML_for_Diagnosing_Breast_Cancer-Steven_Smiley.ipynb) | My Jupyter notebook written in Python. |



## [3 Inputs](https://github.com/stevensmiley1989/Kaggle-WisconsinBreastCancer/tree/master/Inputs)<a class="anchor" id="3"></a>
Single input file (data.csv) contains all of the information for the Wisconsin Breast Cancer dataset.

[data.csv](https://github.com/stevensmiley1989/Kaggle-WisconsinBreastCancer/blob/master/Inputs/data.csv)

## [4 Outputs](https://github.com/stevensmiley1989/Kaggle-WisconsinBreastCancer/tree/master/Outputs)<a class="anchor" id="4"></a>
The Outputs from the Jupyter notebook are placed in the following two folders: Models & Figures

## 5 credits<a class="anchor" id="5"></a>

* [Kaggle](https://www.kaggle.com/)

## 6 contact-info<a class="anchor" id="6"></a>

Feel free to contact me to discuss any issues, questions, or comments.

* Email: [stevensmiley1989@gmail.com](mailto:stevensmiley1989@gmail.com)
* GitHub: [stevensmiley1989](https://github.com/stevensmiley1989)
* LinkedIn: [stevensmiley1989](https://www.linkedin.com/in/stevensmiley1989)
* Kaggle: [stevensmiley](https://www.kaggle.com/stevensmiley)

### 7 License <a class="anchor" id="7"></a>

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
   
