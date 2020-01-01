#!/usr/bin/env python
# coding: utf-8

# #     Using Machine Learning to Diagnose Breast Cancer in Python
# ## by:  Steven Smiley

# # Problem Statement:
# 
# Find a Machine Learning (ML) model that accurately predicts breast cancer based on the 30 features described below.

# # 1.  Background:
# 
# Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image. n the 3-dimensional space is that described in: [K. P. Bennett and O. L. Mangasarian: "Robust Linear Programming Discrimination of Two Linearly Inseparable Sets", Optimization Methods and Software 1, 1992, 23-34].
# 
# This database is also available through the UW CS ftp server: ftp ftp.cs.wisc.edu cd math-prog/cpo-dataset/machine-learn/WDBC/
# 
# Also can be found on UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29
# 
# Attribute Information:
# 
# 1) ID number 2) Diagnosis (M = malignant, B = benign) 3-32)
# 
# Ten real-valued features are computed for each cell nucleus:
# 
# a) radius (mean of distances from center to points on the perimeter) b) texture (standard deviation of gray-scale values) c) perimeter d) area e) smoothness (local variation in radius lengths) f) compactness (perimeter^2 / area - 1.0) g) concavity (severity of concave portions of the contour) h) concave points (number of concave portions of the contour) i) symmetry j) fractal dimension ("coastline approximation" - 1)
# 
# The mean, standard error and "worst" or largest (mean of the three largest values) of these features were computed for each image, resulting in 30 features. For instance, field 3 is Mean Radius, field 13 is Radius SE, field 23 is Worst Radius.
# 
# All feature values are recoded with four significant digits.
# 
# Missing attribute values: none
# 
# Class distribution: 357 benign, 212 malignant

# # 2. Abstract:
# 
#   When it comes to diagnosing breast cancer, we want to make sure we don't have too many false positives (you have cancer, but told you dont) or false negatives (you don't have cancer, but told you do and go on treatments). Therefore, the highest overall accuracy model was chosen, which was the Gradient Boosted model.  
#   
#   Several different models were evaluated through k-crossfold validation and GridSearchCV, which iterates on different algorithm's hyperparameters:
#   * Logistic Regression 
#   * Support Vector Machine
#   * Neural Network
#   * Random Forest
#   * Gradient Boost
#   * eXtreme Gradient Boost
# 
#    
#    All of the models performed well after fine tunning their hyperparameters, but the best model was the Gradient Boosted model as shown with an accuracy of ~97.4%. Out of the 20% of data witheld in this test (114 random individuals), only 3 were misdiagnosed. Two of which were misdiagnosed via False Positive, which means they had cancer, but told they didn't. One was misdiganosed via False Negative, which means they didn't have cancer, but told they did. No model is perfect, but I am happy about how accurate my model is here. If on average only 3 people out of 114 are misdiagnosed, that is a good start for making a model. Furthermore, the Feature Importance plots show that the "concave points mean" was by far the most significant feature to extract from a biopsy and should be taken each time if possible for predicting breast cancer.
# 
# 

# # 3.  Import Libraries 

# In[1]:


import warnings
import os # Get Current Directory
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd # data processing, CSV file I/O (e.i. pd.read_csv)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from time import time
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from scipy import stats
import subprocess
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.multiclass import unique_labels
import itertools


# # 3.  Hide Warnings

# In[2]:


warnings.filterwarnings("ignore")
pd.set_option('mode.chained_assignment', None)


# # 4. Get Current Directory

# In[3]:


currentDirectory=os.getcwd()
print(currentDirectory)


# # 5. Import and View Data

# In[4]:


#data= pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')
data=os.path.join(currentDirectory,'data.csv')
data= pd.read_csv(data)
data.head(10) # view the first 10 columns


# ## 5.1 Import and View Data:  Check for Missing Values
# 
# As the background stated, no missing values should be present.  The following verifies that.  The last column doesn't hold any information and should be removed.  In addition, the diagnosis should be changed to a binary classification of 0= benign and 1=malignant.

# In[5]:


data.isnull().sum()


# In[6]:


# Drop Unnamed: 32 variable that has NaN values.
data.drop(['Unnamed: 32'],axis=1,inplace=True)


# In[7]:


# Convert Diagnosis for Cancer from Categorical Variable to Binary
diagnosis_num={'B':0,'M':1}
data['diagnosis']=data['diagnosis'].map(diagnosis_num)


# In[8]:


# Verify Data Changes, look at first 5 rows 
data.head(5)


# # 6.  Split Data for Training 
# 
# A good rule of thumb is to hold out 20 percent of the data for testing.  

# In[9]:


X = data.drop(['id','diagnosis'], axis= 1)
y = data.diagnosis

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)
# Use Pandas DataFrame
X_train = pd.DataFrame(X_train)
X_test=pd.DataFrame(X_test)
y_train = pd.DataFrame(y_train)
y_test=pd.DataFrame(y_test)

tr_features=X_train
tr_labels=y_train

val_features = X_test
val_labels=y_test


# Verify the data was split correctly

# In[10]:


print('X_train - length:',len(X_train), 'y_train - length:',len(y_train))
print('X_test - length:',len(X_test),'y_test - length:',len(y_test))
print('Percent heldout for testing:', round(100*(len(X_test)/len(data)),0),'%')


# # 7. Machine Learning:
# 
# In order to find a good model, several algorithms are tested on the training dataset. A senstivity study using different Hyperparameters of the algorithms are iterated on with GridSearchCV in order optimize each model. The best model is the one that has the highest accuracy without overfitting by looking at both the training data and the validation data results. Computer time does not appear to be an issue for these models, so it has little weight on deciding between models.

# ## GridSearch CV
# 
# class sklearn.model_selection.GridSearchCV(estimator, param_grid, scoring=None, n_jobs=None, iid='deprecated', refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs', error_score=nan, return_train_score=False)[source]¶
# 
# Exhaustive search over specified parameter values for an estimator.
# 
# Important members are fit, predict.
# 
# GridSearchCV implements a “fit” and a “score” method. It also implements “predict”, “predict_proba”, “decision_function”, “transform” and “inverse_transform” if they are implemented in the estimator used.
# 
# The parameters of the estimator used to apply these methods are optimized by cross-validated grid-search over a parameter grid.

# #### Function: print_results

# In[11]:


def print_results(results,name,filename_pr):
    with open(filename_pr, mode='w') as file_object:
        print(name,file=file_object)
        print(name)
        print('BEST PARAMS: {}\n'.format(results.best_params_),file=file_object)
        print('BEST PARAMS: {}\n'.format(results.best_params_))
        means = results.cv_results_['mean_test_score']
        stds = results.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, results.cv_results_['params']):
            print('{} {} (+/-{}) for {}'.format(name,round(mean, 3), round(std * 2, 3), params),file=file_object)
            print('{} {} (+/-{}) for {}'.format(name,round(mean, 3), round(std * 2, 3), params))


# In[12]:


print(GridSearchCV)


# ## 7.1 Machine Learning Models:  Logistic Regression

# ## Logistic Regression:  Hyperparameter used in GridSearchCV
# ### HP1, C:  float, optional (default=1.0)
# Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.
# ##### Details
# Regularization is when a penality is applied with increasing value to prevent overfitting.  The inverse of regularization strength means as the value of C goes up, the value of the regularization strength goes down and vice versa.  
# ##### Values chosen
# 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]

# In[13]:


LR_model_dir=os.path.join(currentDirectory,'LR_model.pkl')
if os.path.exists(LR_model_dir) == False:
    lr = LogisticRegression()
    parameters = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
            }
    cv=GridSearchCV(lr, parameters, cv=5)
    cv.fit(tr_features,tr_labels.values.ravel())      
    print_results(cv,'Logistic Regression (LR)','LR_GridSearchCV_results.txt')
    cv.best_estimator_
    LR_model_dir=os.path.join(currentDirectory,'LR_model.pkl')
    joblib.dump(cv.best_estimator_,LR_model_dir)
else:
    print('Already have LR')


# # 7.2 Machine Learning Models:  Support Vector Machine

# ## Support Vector Machine:  
# ### Hyperparameter used in GridSearchCV
# #### HP1,  kernelstring, optional (default=’rbf’)
# Specifies the kernel type to be used in the algorithm. It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable. If none is given, ‘rbf’ will be used. If a callable is given it is used to pre-compute the kernel matrix from data matrices; that matrix should be an array of shape (n_samples, n_samples).
# ###### Details
# A linear kernel type is good when the data is Linearly seperable, which means it can be separated by a single Line.
# A radial basis function (rbf) kernel type is an expontential function of the squared Euclidean distance between two vectors and a constant.  Since the value of RBF kernel decreases with distance and ranges between zero and one, it has a ready interpretation as a similiarity measure.  
# ###### Values chosen
# 'kernel': ['linear','rbf']
# 
# #### HP2,  C:  float, optional (default=1.0)
# Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive. The penalty is a squared l2 penalty.
# ###### Details
# Regularization is when a penality is applied with increasing value to prevent overfitting.  The inverse of regularization strength means as the value of C goes up, the value of the regularization strength goes down and vice versa.  
# ###### Values chosen
# 'C': [0.1, 1, 10]

# In[14]:


print(SVC())


# In[15]:


SVM_model_dir=os.path.join(currentDirectory,'SVM_model.pkl')
if os.path.exists(SVM_model_dir) == False:
    svc = SVC()
    parameters = {
            'kernel': ['linear','rbf'],
            'C': [0.1, 1, 10]
            }
    cv=GridSearchCV(svc,parameters, cv=5)
    cv.fit(tr_features, tr_labels.values.ravel())
    print_results(cv,'Support Vector Machine (SVM)','SVM_GridSearchCV_results.txt')
    cv.best_estimator_
    SVM_model_dir=os.path.join(currentDirectory,'SVM_model.pkl')
    joblib.dump(cv.best_estimator_,SVM_model_dir)
else:
    print('Already have SVM')


# # 7.3 Machine Learning Models:  Neural Network

# ## Neural Network:  (sklearn)
# ### Hyperparameter used in GridSearchCV
# #### HP1, hidden_layer_sizes:  tuple, length = n_layers - 2, default (100,)
# The ith element represents the number of neurons in the ith hidden layer.
# ###### Details
# A rule of thumb is (2/3)*(# of input features) = neurons per hidden layer. 
# ###### Values chosen
# 'hidden_layer_sizes': [(10,),(50,),(100,)]
# 
# #### HP2, activation:  {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}, default ‘relu’
# Activation function for the hidden layer.
# ###### Details
# * ‘identity’, no-op activation, useful to implement linear bottleneck, returns f(x) = x
# * ‘logistic’, the logistic sigmoid function, returns f(x) = 1 / (1 + exp(-x)).
# * ‘tanh’, the hyperbolic tan function, returns f(x) = tanh(x).
# * ‘relu’, the rectified linear unit function, returns f(x) = max(0, x)   
# ###### Values chosen
# 'hidden_layer_sizes': [(10,),(50,),(100,)]
# 
# #### HP3, learning_rate:  {‘constant’, ‘invscaling’, ‘adaptive’}, default ‘constant’
# Learning rate schedule for weight updates.
# ###### Details
# * ‘constant’ is a constant learning rate given by ‘learning_rate_init’.
# * ‘invscaling’ gradually decreases the learning rate at each time step ‘t’ using an inverse scaling exponent of ‘power_t’. effective_learning_rate = learning_rate_init / pow(t, power_t)
# * ‘adaptive’ keeps the learning rate constant to ‘learning_rate_init’ as long as training loss keeps decreasing. Each time two consecutive epochs fail to decrease training loss by at least tol, or fail to increase validation score by at least tol if ‘early_stopping’ is on, the current learning rate is divided by 5.
# 
# Only used when solver='sgd'.
#   
# ###### Values chosen
# 'learning_rate': ['constant','invscaling','adaptive']

# In[16]:


print(MLPClassifier())


# In[17]:


MLP_model_dir=os.path.join(currentDirectory,'MLP_model.pkl')
if os.path.exists(MLP_model_dir) == False:
    mlp = MLPClassifier()
    parameters = {
            'hidden_layer_sizes': [(10,),(50,),(100,)],
            'activation': ['relu','tanh','logistic'],
            'learning_rate': ['constant','invscaling','adaptive']
            }
    cv=GridSearchCV(mlp, parameters, cv=5)
    cv.fit(tr_features, tr_labels.values.ravel())
    print_results(cv,'Neural Network (MLP)','MLP_GridSearchCV_results.txt')
    cv.best_estimator_
    MLP_model_dir=os.path.join(currentDirectory,'MLP_model.pkl')
    joblib.dump(cv.best_estimator_,MLP_model_dir)
else:
    print('Already have MLP')


# # 7.4 Machine Learning Models:  Random Forest

# ## Random Forest:  
# ### Hyperparameter used in GridSearchCV
# #### HP1, n_estimators:  integer, optional (default=100)
# The number of trees in the forest.
# 
# Changed in version 0.22: The default value of n_estimators changed from 10 to 100 in 0.22.
# ###### Details
# Usually 500 does the trick and the accuracy and out of bag error doesn't change much after. 
# ###### Values chosen
# 'n_estimators': [500],
# 
# #### HP2, max_depth:  integer or None, optional (default=None)
# The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
# ###### Details
# None usually does the trick, but a few shallow trees are tested. 
# ###### Values chosen
# 'max_depth': [5,7,9, None]

# In[18]:


print(RandomForestClassifier())


# In[19]:


RF_model_dir=os.path.join(currentDirectory,'RF_model.pkl')
if os.path.exists(RF_model_dir) == False:
    rf = RandomForestClassifier(oob_score=False)
    parameters = {
            'n_estimators': [500],
            'max_depth': [5,7,9, None]
            }
    cv = GridSearchCV(rf, parameters, cv=5)
    cv.fit(tr_features, tr_labels.values.ravel())
    print_results(cv,'Random Forest (RF)','RF_GridSearchCV_results.txt')
    cv.best_estimator_
    RF_model_dir=os.path.join(currentDirectory,'RF_model.pkl')
    joblib.dump(cv.best_estimator_,RF_model_dir)
else:
    print('Already have RF')


# # 7.4 Machine Learning Models:  Gradient Boosting

# ## Gradient Boosting:  
# ### Hyperparameter used in GridSearchCV
# #### HP1, n_estimators:  int (default=100)
# The number of boosting stages to perform. Gradient boosting is fairly robust to over-fitting so a large number usually results in better performance.
# ###### Details
# Usually 500 does the trick and the accuracy and out of bag error doesn't change much after. 
# ###### Values chosen
# 'n_estimators': [5, 50, 250, 500],
# 
# #### HP2, max_depth:  integer, optional (default=3)
# maximum depth of the individual regression estimators. The maximum depth limits the number of nodes in the tree. Tune this parameter for best performance; the best value depends on the interaction of the input variables.
# ###### Details
# A variety of shallow trees are tested. 
# ###### Values chosen
# 'max_depth': [1, 3, 5, 7, 9],
# 
# #### HP3, learning_rate:  float, optional (default=0.1)
# learning rate shrinks the contribution of each tree by learning_rate. There is a trade-off between learning_rate and n_estimators.
# ###### Details
# A variety was chosen because of the trade-off.
# ###### Values chosen
# 'learning_rate': [0.01, 0.1, 1]

# In[20]:


print(GradientBoostingClassifier())


# In[21]:


GB_model_dir=os.path.join(currentDirectory,'GB_model.pkl')
if os.path.exists(GB_model_dir) == False:
    gb = GradientBoostingClassifier()
    parameters = {
            'n_estimators': [5, 50, 250, 500],
            'max_depth': [1, 3, 5, 7, 9],
            'learning_rate': [0.01, 0.1, 1]
            }
    cv=GridSearchCV(gb, parameters, cv=5)
    cv.fit(tr_features, tr_labels.values.ravel())
    print_results(cv,'Gradient Boost (GB)','GR_GridSearchCV_results.txt')
    cv.best_estimator_
    GB_model_dir=os.path.join(currentDirectory,'GB_model.pkl')
    joblib.dump(cv.best_estimator_,GB_model_dir)
else:
    print('Already have GB') 


# # 7.5 Machine Learning Models:  eXtreme Gradient Boosting

# ## eXtreme Gradient Boosting:  
# ### Hyperparameter used in GridSearchCV
# #### HP1, n_estimators:  (int) – Number of trees to fit.
# ###### Details
# Usually 500 does the trick and the accuracy and out of bag error doesn't change much after. 
# ###### Values chosen
# 'n_estimators': [5, 50, 250, 500],
# 
# #### HP2, max_depth:  (int) – 
# Maximum tree depth for base learners.
# ###### Details
# A variety of shallow trees are tested. 
# ###### Values chosen
# 'max_depth': [1, 3, 5, 7, 9],
# 
# #### HP3, learning_rate: (float) – 
# Boosting learning rate (xgb’s “eta”)
# ###### Details
# A variety was chosen because of the trade-off.
# ###### Values chosen
# 'learning_rate': [0.01, 0.1, 1]

# In[22]:


XGB_model_dir=os.path.join(currentDirectory,'XGB_model.pkl')
if os.path.exists(XGB_model_dir) == False:
    xgb = XGBClassifier()
    parameters = {
            'n_estimators': [5, 50, 250, 500],
            'max_depth': [1, 3, 5, 7, 9],
            'learning_rate': [0.01, 0.1, 1]
            }
    cv=GridSearchCV(xgb, parameters, cv=5)
    cv.fit(tr_features, tr_labels.values.ravel())
    print_results(cv,'eXtreme Gradient Boost (XGB)','XGB_GridSearchCV_results.txt')
    cv.best_estimator_
    XGB_model_dir=os.path.join(currentDirectory,'XGB_model.pkl')
    joblib.dump(cv.best_estimator_,XGB_model_dir)
else:
    print('Already have XGB')  


# # 8. Evaluate Models

# In[23]:


## all models
models = {}

#for mdl in ['LR', 'SVM', 'MLP', 'RF', 'GB','XGB']:
for mdl in ['LR', 'SVM', 'MLP', 'RF', 'GB','XGB']:
    model_path=os.path.join(currentDirectory,'{}_model.pkl')
    models[mdl] = joblib.load(model_path.format(mdl))


# #### Function: evaluate_model

# In[24]:


def evaluate_model(name, model, features, labels, y_test_ev, fc):
        start = time()
        pred = model.predict(features)
        end = time()
        y_truth=y_test_ev
        accuracy = round(accuracy_score(labels, pred), 3)
        precision = round(precision_score(labels, pred), 3)
        recall = round(recall_score(labels, pred), 3)
        print('{} -- Accuracy: {} / Precision: {} / Recall: {} / Latency: {}ms'.format(name,
                                                                                       accuracy,
                                                                                       precision,
                                                                                       recall,
                                                                                       round((end - start)*1000, 1)))
        
        
        pred=pd.DataFrame(pred)
        pred.columns=['diagnosis']
        # Convert Diagnosis for Cancer from Binary to Categorical
        diagnosis_name={0:'Benign',1:'Malginant'}
        y_truth['diagnosis']=y_truth['diagnosis'].map(diagnosis_name)
        pred['diagnosis']=pred['diagnosis'].map(diagnosis_name)
        class_names = ['Benign','Malginant']        
        cm = confusion_matrix(y_test_ev, pred, class_names)
        
        FP_L='False Positive'
        FP = cm[0][1]
        #print(FP_L)
        #print(FP)
        FN_L='False Negative'
        FN = cm[1][0]
        #print(FN_L)
        #print(FN)
        TP_L='True Positive'
        TP = cm[1][1]
        #print(TP_L)
        #print(TP)
        TN_L='True Negative'
        TN = cm[0][0]
        #print(TN_L)
        #print(TN)

        #TPR_L= 'Sensitivity, hit rate, recall, or true positive rate'
        TPR_L= 'Sensitivity'
        TPR = round(TP/(TP+FN),3)
        #print(TPR_L)
        #print(TPR)
        #TNR_L= 'Specificity or true negative rate'
        TNR_L= 'Specificity'
        TNR = round(TN/(TN+FP),3) 
        #print(TNR_L)
        #print(TNR)
        #PPV_L= 'Precision or positive predictive value'
        PPV_L= 'Precision'
        PPV = round(TP/(TP+FP),3)
        #print(PPV_L)
        #print(PPV)
        #NPV_L= 'Negative predictive value'
        NPV_L= 'NPV'
        NPV = round(TN/(TN+FN),3)
        #print(NPV_L)
        #print(NPV)
        #FPR_L= 'Fall out or false positive rate'
        FPR_L= 'FPR'
        FPR = round(FP/(FP+TN),3)
        #print(FPR_L)
        #print(FPR)
        #FNR_L= 'False negative rate'
        FNR_L= 'FNR'
        FNR = round(FN/(TP+FN),3)
        #print(FNR_L)
        #print(FNR)
        #FDR_L= 'False discovery rate'
        FDR_L= 'FDR'
        FDR = round(FP/(TP+FP),3)
        #print(FDR_L)
        #print(FDR)

        ACC_L= 'Accuracy'
        ACC = round((TP+TN)/(TP+FP+FN+TN),3)
        #print(ACC_L)
        #print(ACC)
        
        stats_data = {'Name':name,
                      ACC_L:ACC,
                      FP_L:FP,
                     FN_L:FN,
                     TP_L:TP,
                     TN_L:TN,
                     TPR_L:TPR,
                     TNR_L:TNR,
                     PPV_L:PPV,
                     NPV_L:NPV,
                     FPR_L:FPR,
                     FNR_L:FDR}
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(cm,cmap=plt.cm.gray_r)
        plt.title('Figure {}.A: {} Confusion Matrix on Unseen Test Data'.format(fc,name),y=1.08)
        fig.colorbar(cax)
        ax.set_xticklabels([''] + class_names)
        ax.set_yticklabels([''] + class_names)
        # Loop over data dimensions and create text annotations.
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                text = ax.text(j, i, cm[i, j],
                               ha="center", va="center", color="r")
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig('Figure{}.A_{}_Confusion_Matrix.png'.format(fc,name),dpi=400,bbox_inches='tight')
        #plt.show()
        
        if  name == 'RF' or name == 'GB' or name == 'XGB': 
            # Get numerical feature importances
            importances = list(model.feature_importances_)
            importances=100*(importances/max(importances))               
            feature_list = list(features.columns)
            sorted_ID=np.argsort(importances)   
            plt.figure()
            plt.barh(sort_list(feature_list,importances),importances[sorted_ID],align='center')
            plt.title('Figure {}.B: {} Variable Importance Plot'.format(fc,name))
            plt.xlabel('Relative Importance')
            plt.ylabel('Feature') 
            plt.savefig('Figure{}.B_{}_Variable_Importance_Plot.png'.format(fc,name),dpi=300,bbox_inches='tight')
            #plt.show()
        
        return accuracy,name, model, stats_data
        


# #### Function:  sort_list

# In[25]:


def sort_list(list1, list2): 
    zipped_pairs = zip(list2, list1)   
    z = [x for _, x in sorted(zipped_pairs)]       
    return z 


# ### Search for best model using test features

# In[27]:


ev_accuracy=[None]*len(models)
ev_name=[None]*len(models)
ev_model=[None]*len(models)
ev_stats=[None]*len(models)
count=1
for name, mdl in models.items():
        y_test_ev=y_test
        ev_accuracy[count-1],ev_name[count-1],ev_model[count-1], ev_stats[count-1] = evaluate_model(name,mdl,val_features, val_labels, y_test_ev,count)
        diagnosis_name={'Benign':0,'Malginant':1}
        y_test['diagnosis']=y_test['diagnosis'].map(diagnosis_name)
        count=count+1

    


# In[28]:


best_name=ev_name[ev_accuracy.index(max(ev_accuracy))]    #picks the maximum accuracy
print('Best Model:',best_name,'with Accuracy of ',max(ev_accuracy))   
best_model=ev_model[ev_accuracy.index(max(ev_accuracy))]    #picks the maximum accuracy

if best_name == 'RF' or best_name == 'GB' or best_name == 'XGB': 
    # Get numerical feature importances
    importances = list(best_model.feature_importances_)
    importances=100*(importances/max(importances))               
    feature_list = list(X.columns)
    sorted_ID=np.argsort(importances)   
    plt.figure()
    plt.barh(sort_list(feature_list,importances),importances[sorted_ID],align='center')
    plt.title('Figure 7:  Variable Importance Plot -- {}'.format(best_name))
    plt.xlabel('Relative Importance')
    plt.ylabel('Feature') 
    plt.savefig('Figure7.png',dpi=300,bbox_inches='tight')
    plt.show()


#  # 9. Conclusions 
#   When it comes to diagnosing breast cancer, we want to make sure we don't have too many false positives (you have cancer, but told you dont) or false negatives (you don't have cancer, but told you do and go on treatments).  Therefore, the highest overall accuracy model is chosen.  
# 
#   All of the models performed well after fine tunning their hyperparameters, but the best model was the Gradient Boosted model as shown with an accuracy of ~97.4%.  Out of the 20% of data witheld in this test (114 random individuals), only 3 were misdiagnosed.  Two of which were misdiagnosed via False Positive, which means they had cancer, but told they didn't.  One was misdiganosed via False Negative, which means they didn't have cancer, but told they did.  No model is perfect, but I am happy about how accurate my model is here.  If on average only 3 people out of 114 are misdiagnosed, that is a good start for making a model.  Furthermore, the Feature Importance plots show that the "concave points mean" was by far the most significant feature to extract from a biopsy and should be taken each time if possible for predicting breast cancer.   

# In[29]:


ev_stats=pd.DataFrame(ev_stats)
print(ev_stats.head(10))


# In[ ]:




