# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 19:17:20 2018

@author: isogb
@ Machine Learning Classifers Evaluation


"""
import numpy
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import resample
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import auc, accuracy_score, f1_score, mean_squared_error, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import make_scorer
from sklearn.externals import joblib
from sklearn.utils import shuffle
from sklearn.model_selection import learning_curve
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from keras.callbacks import ModelCheckpoint 
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import numpy.ma as ma
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectFromModel
import plotly as py
import plotly.graph_objs as go
from plotly.offline import plot, iplot

from keras.regularizers import L1L2
#from tensorflow.python.keras._impl.keras.utils.generic_utils import deserialize_keras_object




# In[1]:

csv_path = (r"C:\Users\isogb\Documents\Final Year\Final Year Project\Final Year Investigative Project\Bank Marketing Data Set\bank-additional\bank_additional_full.csv")

#Dataset Preprocessing
bank_data = pd.read_csv(csv_path, sep=';')
categorical = ['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome', 'y']
numerical = ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
bank_data['y'] = bank_data['y'].map({'no':0, 'yes':1}) 
    
# Normalization of numeric features
scaler = MinMaxScaler() # default=(0, 1)
bank_data[numerical] = scaler.fit_transform(bank_data[numerical])

# One Hot Encoding of the categorical features
bank_data_processed = pd.get_dummies(bank_data)
bank_data_processed = bank_data_processed.drop('duration', axis=1)
#bank_data_processed.info()
#scatter_matrix(bank_data_processed, figsize=(30,15))
 
# In[2]:
# Shuffle Split for Training, Validation and Testion dataset  
numpy.random.seed(100)    
split = StratifiedShuffleSplit(n_splits=10, test_size = 0.3, random_state=42)    
for train_index, test_index in split.split(bank_data_processed, bank_data_processed['y']):
    train_val_set_1 = bank_data_processed.loc[train_index]
    testing_1 = bank_data_processed.loc[test_index]
    training_1, validation_1 = train_test_split(train_val_set_1, test_size = 0.3, random_state = 42)
print (len(training_1), "training +", len(validation_1), "validation +", len(testing_1), "test")  #prints number of training, validation and test set entries      

#Statistics for the training, validation and testing data sets
training_1.info()
print(training_1['y'].value_counts())
#print(training_1['y'].value_counts()/len(training_1))
print(validation_1['y'].value_counts())
#validation_1['y'].value_counts()/len(validation_1)
print(testing_1['y'].value_counts())

        
# In[3]:      
#Handling data imbalance for training set
        
# Separate majority and minority classes
train_negative = training_1[training_1['y']==0] #majority class - no
train_positive = training_1[training_1['y']==1] #minority class - yes

# Upsample minority class
train_positive_upsample = resample(train_positive, 
                                  replace=True,     # sample with replacement
                                  n_samples=17897,    # to match majority class
                                  random_state=42) # reproducible results
  
# Combine majority class with upsampled minority class
train_upsample = pd.concat([train_negative, train_positive_upsample])
    
# Display new class counts
#print(train_upsample['y'].value_counts()/len(train_upsample))  
#print(train_upsample['y'].value_counts()) 

# In[4]:    
#Handling data imbalance for validation set
        
# Separate majority and minority classes
validation_negative = validation_1[validation_1['y']==0] #majority class - no
validation_positive = validation_1[validation_1['y']==1] #minority class - yes

# Upsample minority class
validation_positive_upsample = resample(validation_positive, 
                                  replace=True,     # sample with replacement
                                  n_samples=7686,    # to match majority class
                                  random_state=42) # reproducible results
  
# Combine majority class with upsampled minority class
validation_upsample = pd.concat([validation_negative, validation_positive_upsample])
    
# Display new class counts
#print(validation_upsample['y'].value_counts()/len(validation_upsample))
#print(validation_upsample['y'].value_counts()) 
 


# In[5]:  
#Handling data imbalance for training and validation set
        
# Separate majority and minority classes
train_val_negative = train_val_set_1[train_val_set_1['y']==0] #majority class - no
train_val_positive = train_val_set_1[train_val_set_1['y']==1] #minority class - yes

# Upsample minority class
train_val_positive_upsample = resample(train_val_positive, 
                                  replace=True,     # sample with replacement
                                  n_samples=25583,    # to match majority class
                                  random_state=42) # reproducible results
  
# Combine majority class with upsampled minority class
train_val_upsample = pd.concat([train_val_negative, train_val_positive_upsample])

# In[5]:
# create X, y for imbalanced train set used for performance validation
X_training_imb = training_1.drop('y', axis=1)
y_training_imb = training_1['y']
#print("Imbalanced Training Shape without FS:", X_training_imb.shape)

X_validation_imb = validation_1.drop('y', axis=1)
y_validation_imb = validation_1['y']
#print("Imbalanced Training Shape without FS:", X_validation_imb.shape)

# create X, y for upsampled Training
X_training = train_upsample.drop('y', axis=1)
#print("Balanced Training Shape without FS:", X_training.shape)
y_training = train_upsample['y']
#print("Balanced Training target Shape without FS:", y_training.shape)
#return X_training.info() 
   
# create X, y for upsampled Validation
X_validation = validation_upsample.drop('y', axis=1)
#print("Balanced Validation Shape without FS:", X_validation.shape)
y_validation = validation_upsample['y']
#print("Balanced Validation target Shape without FS:", y_validation.shape)
#return X_validation.info()   
   
# create X, y for Testing - No upsampling for test set
X_testing = testing_1.drop('y', axis=1)
#print("Test Shape without FS:", X_testing.shape)
y_testing = testing_1['y'] 
#print(y_testing.value_counts())
#return X_training.info()

# create X, y for Traning + validation set
X_train_val = train_val_upsample.drop('y', axis=1)
#print("Balanced Training and val set:", X_train_val.shape)
y_train_val = train_val_upsample['y'] 
#print("Balanced Training and val set target:", y_train_val.shape)
    
    
# In[6]:
#  Feature selection using RFECV and Randomforest and accuracy scoring
'''
clf_rfecv10 = RandomForestClassifier()    #10-fold cross-validation
rfecv = RFECV(estimator= clf_rfecv10, step=1, cv=10, scoring='roc_auc') 
rfecv = rfecv.fit(X_validation, y_validation)
rfecv = rfecv.fit(X_training, y_training)
print('Optimal number of features :', rfecv.n_features_)
print('Best features :', X_training.columns[rfecv.support_])
print('Best features :', X_validation.columns[rfecv.support_])
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score of number of selected features")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()
'''    
all_features = X_training.columns.tolist() 

selected_features = ['age', 'campaign', 'pdays', 'previous', 'emp.var.rate',
       'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed',
       'job_admin.', 'job_blue-collar', 'job_entrepreneur', 'job_management',
       'job_self-employed', 'job_services', 'job_technician',
       'marital_divorced', 'marital_married', 'marital_single',
       'education_basic.4y', 'education_basic.6y', 'education_basic.9y',
       'education_high.school', 'education_professional.course',
       'education_university.degree', 'education_unknown', 'default_no',
       'default_unknown', 'housing_no', 'housing_yes', 'loan_no', 'loan_yes',
       'contact_cellular', 'contact_telephone', 'month_may', 'day_of_week_fri',
       'day_of_week_mon', 'day_of_week_thu', 'day_of_week_tue',
       'day_of_week_wed', 'poutcome_failure', 'poutcome_success']
    
#dropped_features = []    
#for i in selected_features:
#    for j in all_features:
#        if j not in selected_features:
#            dropped_features.append(j)
#print (dropped_features)

#dropped_features = ['job_housemaid', 'job_retired', 'job_student', 'job_unemployed', 'job_unknown', 'marital_unknown', 'education_illiterate', 'default_yes', 'housing_unknown', 'loan_unknown', 'month_apr', 'month_aug', 'month_dec', 'month_jul', 'month_jun', 'month_mar', 'month_nov', 'month_oct', 'month_sep', 'poutcome_nonexistent'] 

#X_training_FS = X_training.drop(dropped_features, axis=1)
#X_training_RFEFS = rfecv.transform(X_training)
#X_validation_RFEFS = rfecv.transform(X_validation)
#X_testing_RFEFS = rfecv.transform(X_testing)
#print("Training Shape with FS:", X_training_FS.shape)
#X_validation_FS = X_validation.drop(dropped_features, axis=1)
#print("Validation Shape with FS:", X_validation_FS.shape)
#return X_training_FS.info()
#X_testing_FS = X_testing.drop(dropped_features, axis=1)
#print("Validation Shape with FS:", X_validation_FS.shape)


# In[6.1]:
# Feature Selction using Kbest means method

#selector = SelectKBest(chi2, k=63)
#X_training_k = selector.fit(X_training, y_training)
#names = X_training.columns.values[selector.get_support()]
#scores = selector.scores_[selector.get_support()]
#names_scores = list(zip(names, scores))
#ns_df = pd.DataFrame(names_scores, columns = ['Features', 'Score'])
#ns_df_sorted = ns_df.sort_values(['Score', 'Features'], ascending = [False, True])
#print(names)
#print(ns_df_sorted)
#print("K Best Training Shape with FS:", ns_df_sorted.shape)
#print(ns_df_sorted.values.tolist())

all_features = X_training.columns.tolist() 
selected_k_features = ['poutcome_success', 'euribor3m','contact_telephone', 'emp.var.rate',
 'nr.employed','month_may','default_unknown','month_oct','month_sep','month_mar','contact_cellular',
 'previous','poutcome_nonexistent', 'job_blue-collar','job_retired','month_apr','pdays','job_student', 
 'education_basic.9y','month_dec','marital_single','cons.price.idx','default_no', 'education_university.degree',
 'job_services','marital_married','month_jul','job_admin.','poutcome_failure','job_unemployed','campaign',
 'cons.conf.idx', 'day_of_week_mon', 'month_nov','education_basic.6y','month_aug','day_of_week_tue',
 'education_unknown', 'housing_yes','housing_no','day_of_week_fri','education_basic.4y', 'education_high.school',
 'age','housing_unknown','loan_unknown','job_housemaid', 'day_of_week_wed','job_technician', 'job_entrepreneur',
 'day_of_week_thu','marital_divorced','education_professional.course','loan_yes','job_self-employed', 
 'loan_no','education_illiterate','marital_unknown','month_jun','job_unknown', 'job_management','default_yes']    

dropped_features = ['education_professional.course','loan_yes','job_self-employed', 
 'loan_no','education_illiterate','marital_unknown','month_jun','job_unknown', 'job_management','default_yes'] 

#dropped_features = ['job_housemaid', 'day_of_week_wed','job_technician', 'job_entrepreneur',
# 'day_of_week_thu','marital_divorced','education_professional.course','loan_yes','job_self-employed', 
# 'loan_no','education_illiterate','marital_unknown','month_jun','job_unknown', 'job_management','default_yes'] 
#
X_training_FS = X_training.drop(dropped_features, axis=1)
##print("Training Shape with FS:", X_training_FS.shape)

X_validation_FS = X_validation.drop(dropped_features, axis=1)
##print("Validation Shape with FS:", X_validation_FS.shape)

X_testing_FS = X_testing.drop(dropped_features, axis=1)
#print("Validation Shape with FS:", X_validation_FS.shape)

# In[7]:
# create function to fit classifier using GridsearchCV and report metrics score on train dataset
def fit_classifier(model, X, y, parameters=None, scorer_metrics=None):

    # Perform grid search on the classifier using scorer_metrics as the scoring method
    grid_obj = GridSearchCV(estimator = model, param_grid = parameters, scoring=make_scorer(scorer_metrics), cv=5)

    # Fit the grid search object to the training data and find the optimal parameters using fit()
    grid_fit = grid_obj.fit(X, y)

    # Get the estimator
    model_estimator = grid_fit.best_estimator_

    # Report the metrics scores on train data
    model_estimator.fit(X, y)
    y_pred = model_estimator.predict(X)
    grid_scores = grid_obj.grid_scores_
    to_vary = "C"
    grid_search(grid_obj.grid_scores_, to_vary)
    plt.show()
    
#    plt.grid_search(grid_obj.grid_scores_,change='C', kind = 'bar')
#    cvres = grid_fit.cv_results_
#    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
#        print (numpy.sqrt(-mean_score), params)

    print("\n")
    print("\nModel performance on training set\n------------------------")
    print("Final accuracy score on the training data: {:.4f}".format(accuracy_score(y, y_pred)))
    print("Final precision score on training data: {:.4f}".format(precision_score(y, y_pred)))
    print("Final Recall score on training data: {:.4f}".format(recall_score(y, y_pred)))
    print("Final ROC AUC score on training data: {:.4f}".format(roc_auc_score(y, y_pred)))
    print("Final mean squared error score on training data: {:.4f}".format(mean_squared_error(y, y_pred)))
    print("Final f1 score on training data: {:.4f}".format(f1_score(y, y_pred)))
    print("\n")
    print("The best parameters are: {}".format(model_estimator))

    return model_estimator


# In[8]:
    
def classifier_train(model_train, X, y):
    model_train.fit(X, y)
    y_pred_train = model_train.predict(X)
    print("\n")
    print("\n Model performance on training set\n------------------------")
    print("Final accuracy score on the training data: {:.4f}".format(accuracy_score(y, y_pred_train)))
    print("Final precision score on training data: {:.4f}".format(precision_score(y, y_pred_train)))
    print("Final Recall score on training data: {:.4f}".format(recall_score(y, y_pred_train)))
    print("Final ROC AUC score on training data: {:.4f}".format(roc_auc_score(y, y_pred_train)))
    print("Final mean squared error score on training data: {:.4f}".format(mean_squared_error(y, y_pred_train)))
    print("Final f1 score on training data: {:.4f}".format(f1_score(y, y_pred_train)))
    #return y_pred_train
    
def classifier_val(model_val, X, y):
    model_val.fit(X, y)
    y_pred_val = model_val.predict(X)
    print("\n")
    print("\nLogistic Regression model on validation set\n------------------------")
    print("Final accuracy score on the validation data: {:.4f}".format(accuracy_score(y, y_pred_val)))
    print("Final precision score on validation data: {:.4f}".format(precision_score(y, y_pred_val)))
    print("Final Recall score on validation data: {:.4f}".format(recall_score(y, y_pred_val)))
    print("Final ROC AUC score on validation data: {:.4f}".format(roc_auc_score(y, y_pred_val)))
    print("Final mean squared error score on validation data: {:.4f}".format(mean_squared_error(y, y_pred_val)))
    print("Final f1 score on validation data: {:.4f}".format(f1_score(y, y_pred_val)))
    #return y_pred_val

# In[9]:
    
# create function to use fitted model to report metrics score on test dataset
# return predicted classification on test dataset
def classifier_test(model_fit, X, y):
    y_pred = model_fit.predict(X)
    print("\n")
    print("\nModel performance on test set\n------------------------")
    print("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y, y_pred)))
    print("Final precision score on testing data: {:.4f}".format(precision_score(y, y_pred)))
    print("Final Recall score on testing data: {:.4f}".format(recall_score(y, y_pred)))
    print("Final ROC AUC score on testing data: {:.4f}".format(roc_auc_score(y, y_pred)))
    print("Final mean squared error score on testing data: {:.4f}".format(mean_squared_error(y, y_pred)))
    print("Final f1 score on testing data: {:.4f}".format(f1_score(y, y_pred)))
    #return y_pred

# In[10]:

def plot_learning_curves(model, Xtrain, ytrain, Xval, yval):
    Xt, yt = shuffle(Xtrain, ytrain)
    Xv, yv = shuffle(Xval, yval)
    train_errors, val_errors = [], []
    for m in range(1, len(Xt), 100):
        model.fit(Xt[:m], yt[:m])
        y_train_predict = model.predict(Xt[:m])
        y_val_predict = model.predict(Xv)
        train_errors.append(mean_squared_error(yt[:m], y_train_predict))
        val_errors.append(mean_squared_error(yv, y_val_predict))
    plt.plot((train_errors), "r--", linewidth=1, label="training")
    plt.plot((val_errors), "b-", linewidth=1, label="validation")
    plt.legend(loc="upper right", fontsize=14)   
    plt.xlabel("Training set size", fontsize=14)
    plt.ylabel("MSE", fontsize=14)
        
# In[11]:

# create function to plot ROC curve
from sklearn.metrics import roc_curve

def roc_curve_plot(model, X, y,label=None):
    # make sure positive class prediction is in the second column of binary prediction
    if label=='Neural Network':
        y_score = model.predict_proba(X)[:,0]
    else:
        y_score = model.predict_proba(X)[:,1]
    
    # generate ROC curve data
    roc = roc_curve(y, y_score)
    
    plt.plot(roc[0], roc[1], label=label)
    plt.plot([0,1],[0,1], 'k--')
    plt.axis([0,1,0,1])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    
    roc_score = auc(roc[0],roc[1])
    print('AUC score of %s is %.4f.' % (label, roc_score))
    
    
# In[12]:
import itertools   
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

#This function prints and plots the confusion matrix.
#Normalization can be applied by setting `normalize=True`.
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = numpy.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# In[XX]:
    
### DEFINE AND TRAIN MODELS###

# In[16]:
#Naive Bayes Model
    
# Naive Bayes model without GridserachCV on imbalanced training set
model_GNB_imb = GaussianNB()
model_GNB_imb.fit(X_training_imb, y_training_imb)
y_pred_GNB_imb = model_GNB_imb.predict(X_training_imb)
print("\nGaussian Naive Bayes model on imbalanced training set\n------------------------")
print("Final accuracy score on the training data: {:.4f}".format(accuracy_score(y_training_imb, y_pred_GNB_imb)))
print("Final precision score on training data: {:.4f}".format(precision_score(y_training_imb, y_pred_GNB_imb)))
print("Final Recall score on training data: {:.4f}".format(recall_score(y_training_imb, y_pred_GNB_imb)))
print("Final ROC AUC score on training data: {:.4f}".format(roc_auc_score(y_training_imb, y_pred_GNB_imb)))
print("Final mean squared error score on training data: {:.4f}".format(mean_squared_error(y_training_imb, y_pred_GNB_imb)))
print("Final f1 score on training data: {:.4f}".format(f1_score(y_training_imb, y_pred_GNB_imb)))

#Naive Bayes model without GridserachCV on imbalanced validation set
model_GNB_imb.fit(X_validation_imb, y_validation_imb)
y_pred_GNB_imb = model_GNB_imb.predict(X_validation_imb)
print("\nGaussian Naive Bayes model on imbalanced validation set\n------------------------")
print("Final accuracy score on the validation dataa: {:.4f}".format(accuracy_score(y_validation_imb, y_pred_GNB_imb)))
print("Final precision score on validation data: {:.4f}".format(precision_score(y_validation_imb, y_pred_GNB_imb)))
print("Final Recall score on validation data: {:.4f}".format(recall_score(y_validation_imb, y_pred_GNB_imb)))
print("Final ROC AUC score on validation data: {:.4f}".format(roc_auc_score(y_validation_imb, y_pred_GNB_imb)))
print("Final mean squared error score on validation data: {:.4f}".format(mean_squared_error(y_validation_imb, y_pred_GNB_imb)))
print("Final f1 score on validation data: {:.4f}".format(f1_score(y_validation_imb, y_pred_GNB_imb)))

# Naive Bayes model without GridserachCV on balanced training set
model_GNB_bal = GaussianNB()
model_GNB_bal.fit(X_training, y_training)
y_pred_GNB_bal = model_GNB_bal.predict(X_training)
model_GNB_bal.fit(X_training_FS, y_training)
y_pred_GNB_bal = model_GNB_bal.predict(X_training_FS)
print("\nGaussian Naive Bayes model on balanced training set\n------------------------")
print("Final accuracy score on the training data: {:.4f}".format(accuracy_score(y_training, y_pred_GNB_bal)))
print("Final precision score on training data: {:.4f}".format(precision_score(y_training, y_pred_GNB_bal)))
print("Final Recall score on training data: {:.4f}".format(recall_score(y_training, y_pred_GNB_bal)))
print("Final ROC AUC score on training data: {:.4f}".format(roc_auc_score(y_training, y_pred_GNB_bal)))
print("Final mean squared error score on training data: {:.4f}".format(mean_squared_error(y_training, y_pred_GNB_bal)))
print("Final f1 score on training data: {:.4f}".format(f1_score(y_training, y_pred_GNB_bal)))

###Naive Bayes model without GridserachCV on balanced validation set
model_GNB_bal.fit(X_validation, y_validation)
y_pred_GNB_bal = model_GNB_bal.predict(X_validation)
model_GNB_bal.fit(X_validation_FS, y_validation)
y_pred_GNB_bal = model_GNB_bal.predict(X_validation_FS)
print("\nGaussian Naive Bayes model on balanced validation set\n------------------------")
print("Final accuracy score on the validation data: {:.4f}".format(accuracy_score(y_validation, y_pred_GNB_bal)))
print("Final precision score on validation data: {:.4f}".format(precision_score(y_validation, y_pred_GNB_bal)))
print("Final Recall score on validation data: {:.4f}".format(recall_score(y_validation, y_pred_GNB_bal)))
print("Final ROC AUC score on validation data: {:.4f}".format(roc_auc_score(y_validation, y_pred_GNB_bal)))
print("Final mean squared error score on validation data: {:.4f}".format(mean_squared_error(y_validation, y_pred_GNB_bal)))
print("Final f1 score on validation data: {:.4f}".format(f1_score(y_validation, y_pred_GNB_bal)))

# In[17]:
# Plotting learning curve for Naive Bayes
   
train_sizes, train_scores, validation_scores = learning_curve(
    estimator = model_GNB_bal, X = X_train_val,
    y = y_train_val, train_sizes = numpy.linspace(0.01, 1.0, 25), cv = 10, shuffle=True, random_state = 42, 
    scoring = 'neg_mean_squared_error')
train_scores_mean = -train_scores.mean(axis = 1)
validation_scores_mean = -validation_scores.mean(axis = 1)
plt.plot(train_sizes, train_scores_mean, "r-", linewidth=2, label = 'Training Error')
plt.plot(train_sizes, validation_scores_mean, "b--", linewidth=2, label = 'Validation Error')
plt.ylabel('MSE', fontsize = 14)
plt.xlabel('Training set size', fontsize = 14)
plt.title('Naive Bayes Learning Curve', fontsize = 18, y = 1.03)
plt.legend()
plt.savefig('Naive_Bayes_Learning_Curve.jpeg')

# In[18]:
# Testing the Naive Bayes model
    
#Balanced Testing
y_pred_test_GNB = model_GNB_bal.predict(X_testing)
y_pred_test_GNB = model_GNB_bal.predict(X_testing_FS)
print("\n")
print("\nGaussian Naive Bayes model on test set with balanced training\n------------------------")
print("Final accuracy score on the test data: {:.4f}".format(accuracy_score(y_testing, y_pred_test_GNB)))
print("Final Precision score on test data: {:.4f}".format(precision_score(y_testing, y_pred_test_GNB)))
print("Final Recall score on test data: {:.4f}".format(recall_score(y_testing, y_pred_test_GNB)))
print("Final ROC AUC score on test data: {:.4f}".format(roc_auc_score(y_testing, y_pred_test_GNB)))
print("Final mean squared error score on test data: {:.4f}".format(mean_squared_error(y_testing, y_pred_test_GNB)))
print("Final f1 score on testdata: {:.4f}".format(f1_score(y_testing, y_pred_test_GNB)))

# In[19]:
# save the Final Naive Bayes model to current directory
joblib.dump(model_GNB_bal,r'C:\Users\isogb\Documents\FYP Software\Saved Models\Gaussian_NB_model.pkl')

confusion_matrix_GNB = confusion_matrix(y_testing, y_pred_test_GNB)
plt.figure(figsize=(6,6))
plot_confusion_matrix(confusion_matrix_GNB, classes=['not subscribe','subscribe'],
                     title= 'Gaussian GNB Confusion matrix')

# In[20]:
    
##Logistic Regression Models
    
#Logistic Regression model without GridserachCV on imbalanced training set
model_LR_imb = LogisticRegression(random_state=42)
model_LR_imb.fit(X_training_imb, y_training_imb)
y_pred_LR_imb = model_LR_imb.predict(X_training_imb)
print("\nLogistic Regression model on imbalanced training set\n------------------------")
print("Final accuracy score on the training data: {:.4f}".format(accuracy_score(y_training_imb, y_pred_LR_imb)))
print("Final precision score on training data: {:.4f}".format(precision_score(y_training_imb, y_pred_LR_imb)))
print("Final Recall score on training data: {:.4f}".format(recall_score(y_training_imb, y_pred_LR_imb)))
print("Final ROC AUC score on training data: {:.4f}".format(roc_auc_score(y_training_imb, y_pred_LR_imb)))
print("Final mean squared error score on training data: {:.4f}".format(mean_squared_error(y_training_imb, y_pred_LR_imb)))
print("Final f1 score on training data: {:.4f}".format(f1_score(y_training_imb, y_pred_LR_imb)))


### Logistic Regression model without GridserachCV on imbalanced validation set
model_LR_imb.fit(X_validation_imb, y_validation_imb)
y_pred_LR_imb = model_LR_imb.predict(X_validation_imb)
print("\nLogistic Regression model on imbalanced validation set\n------------------------")
print("Final accuracy score on the validation dataa: {:.4f}".format(accuracy_score(y_validation_imb, y_pred_LR_imb)))
print("Final precision score on validation data: {:.4f}".format(precision_score(y_validation_imb, y_pred_LR_imb)))
print("Final Recall score on validation data: {:.4f}".format(recall_score(y_validation_imb, y_pred_LR_imb)))
print("Final ROC AUC score on validation data: {:.4f}".format(roc_auc_score(y_validation_imb, y_pred_LR_imb)))
print("Final mean squared error score on training data: {:.4f}".format(mean_squared_error(y_validation_imb, y_pred_LR_imb)))
print("Final f1 score on training data: {:.4f}".format(f1_score(y_validation_imb, y_pred_LR_imb)))

# In[21]:
#Libnear Solver Logistic Regression
    
#Logistic Regression model without GridserachCV on balanced training set

#model_LR_bal = LogisticRegression(random_state=42) 
model_LR_bal = LogisticRegression(random_state=42, C = 0.5, penalty ='l1')
model_LR_bal.fit(X_training, y_training)
y_pred_LR_bal = model_LR_bal.predict(X_training)
model_LR_bal.fit(X_training_FS, y_training)
y_pred_LR_bal = model_LR_bal.predict(X_training_FS)
print("\nLogistic Regression model on balanced training set\n------------------------")
print("Final accuracy score on the training data: {:.4f}".format(accuracy_score(y_training, y_pred_LR_bal)))
print("Final precision score on training data: {:.4f}".format(precision_score(y_training, y_pred_LR_bal)))
print("Final Recall score on training data: {:.4f}".format(recall_score(y_training, y_pred_LR_bal)))
print("Final ROC AUC score on training data: {:.4f}".format(roc_auc_score(y_training, y_pred_LR_bal)))
print("Final mean squared error score on training data: {:.4f}".format(mean_squared_error(y_training, y_pred_LR_bal)))
print("Final f1 score on training data: {:.4f}".format(f1_score(y_training, y_pred_LR_bal)))

#Logistic Regression model without GridserachCV on balanced validation set
model_LR_bal.fit(X_validation, y_validation)
y_pred_LR_bal = model_LR_bal.predict(X_validation)
model_LR_bal.fit(X_validation_FS, y_validation)
y_pred_LR_bal = model_LR_bal.predict(X_validation_FS)
print("\nLogistic Regression model on balanced validation set\n------------------------")
print("Final accuracy score on the validation data: {:.4f}".format(accuracy_score(y_validation, y_pred_LR_bal)))
print("Final precision score on validation data: {:.4f}".format(precision_score(y_validation, y_pred_LR_bal)))
print("Final Recall score on validation data: {:.4f}".format(recall_score(y_validation, y_pred_LR_bal)))
print("Final ROC AUC score on validation data: {:.4f}".format(roc_auc_score(y_validation, y_pred_LR_bal)))
print("Final mean squared error score on training data: {:.4f}".format(mean_squared_error(y_validation, y_pred_LR_bal)))
print("Final f1 score on training data: {:.4f}".format(f1_score(y_validation, y_pred_LR_bal)))

# In[23]:

#Learning curve for Logistic Regression Model pre optimization
train_sizes, train_scores, validation_scores = learning_curve(
    estimator = model_LR_bal, X = X_train_val,
    y = y_train_val, train_sizes = numpy.linspace(0.01, 1.0, 25), cv = 7, shuffle=True, random_state = 42, 
    scoring = 'neg_mean_squared_error')
train_scores_mean = -train_scores.mean(axis = 1)
validation_scores_mean = -validation_scores.mean(axis = 1)
plt.plot(train_sizes, train_scores_mean, "r-", linewidth=2, label = 'Training Error')
plt.plot(train_sizes, validation_scores_mean, "b--", linewidth=2, label = 'Validation Error')
plt.ylabel('MSE', fontsize = 14)
plt.xlabel('Training set size', fontsize = 14)
plt.title('Optimized Logistic Regression Learning Curve', fontsize = 18, y = 1.03)
plt.legend()
plt.savefig('Logistic_Regression_Learning_Curve.jpeg')    

# In[24]:
#Logistic Regression Optimization on Balanced training set
    
#Create the parameters list
#
Cs = [0.1, 0.3, 0.5, 0.7, 1]
penalty = ['l1', 'l2']
Cs = [0.5]
penalty = ['l1']
grid_obj = GridSearchCV(LogisticRegression(random_state=42),
                        dict(C=Cs, penalty = penalty), scoring= 'roc_auc', cv=3)

##### Fit the grid search object to the training data and find the optimal parameters using fit()
grid_fit = grid_obj.fit(X_training, y_training)
grid_fit = grid_obj.fit(X_training_FS, y_training)
##
#### Get the estimator
model_LR_opt = grid_fit.best_estimator_
##
##
print("The best parameters are: {}".format(model_LR_opt))

#Report the metrics scores on train data
model_LR_opt.fit(X_training, y_training)
y_pred = model_LR_opt.predict(X_training) 
model_LR_opt.fit(X_training_FS, y_training)
y_pred = model_LR_opt.predict(X_training_FS) 
print("\n")
print("\nOptimal LR Model performance on training set\n------------------------")
print("Final accuracy score on the training data: {:.4f}".format(accuracy_score(y_training, y_pred)))  
print("Final precision score on training data: {:.4f}".format(precision_score(y_training, y_pred)))
print("Final Recall score on training data: {:.4f}".format(recall_score(y_training, y_pred)))
print("Final ROC AUC score on training data: {:.4f}".format(roc_auc_score(y_training, y_pred)))
print("Final mean squared error score on training data: {:.4f}".format(mean_squared_error(y_training, y_pred)))
print("Final f1 score on training data: {:.4f}".format(f1_score(y_training, y_pred)))
print("\n")

#Evaluate optimal model against Validation Set 
model_LR_opt.fit(X_validation, y_validation)
y_pred_LR = model_LR_opt.predict(X_validation)
model_LR_opt.fit(X_validation_FS, y_validation)
y_pred_LR = model_LR_opt.predict(X_validation_FS)
print("\n")
print("\nOptimal Logistic Regression model on balanced validation set\n------------------------")
print("Final accuracy score on the validation data: {:.4f}".format(accuracy_score(y_validation, y_pred_LR)))
print("Final precision score on validation data: {:.4f}".format(precision_score(y_validation, y_pred_LR)))
print("Final Recall score on validation data: {:.4f}".format(recall_score(y_validation, y_pred_LR)))
print("Final ROC AUC score on validation data: {:.4f}".format(roc_auc_score(y_validation, y_pred_LR)))
print("Final mean squared error score on validation data: {:.4f}".format(mean_squared_error(y_validation, y_pred_LR)))
print("Final f1 score on validation data: {:.4f}".format(f1_score(y_validation, y_pred_LR)))
print(y_pred_LR)
#
results = grid_obj.cv_results_
for mean_score, params in zip(results["mean_test_score"], results["params"]):
    print((mean_score), params)
    
LR_data = [go.Contour(z = results["mean_test_score"],
        x = results['param_C'], y = results['param_penalty'],
        colorscale ='Jet',contours = dict(showlabels = True,labelfont = dict(
        family = 'Raleway',size = 15,color = 'black',)), colorbar = dict( title = 'Score',
        titleside = 'right', titlefont = dict( size =16, family = 'Arial, sans-serif',)))]
py.offline.plot(LR_data)

# In[25]:

#Learning curve for Logistic Regression Model post optimization

train_sizes, train_scores, validation_scores = learning_curve(
    estimator = model_LR_opt, X = X_train_val,
    y = y_train_val, train_sizes = numpy.linspace(0.01, 1.0, 25), cv = 3, shuffle=True, random_state = 42, 
    scoring = 'neg_mean_squared_error')
train_scores_mean = -train_scores.mean(axis = 1)
validation_scores_mean = -validation_scores.mean(axis = 1)
plt.plot(train_sizes, train_scores_mean, "r-", linewidth=2, label = 'Training Error')
plt.plot(train_sizes, validation_scores_mean, "b--", linewidth=2, label = 'Validation Error')
plt.ylabel('MSE', fontsize = 14)
plt.xlabel('Training set size', fontsize = 14)
plt.title('Optimized Logistic Regression Learning Curve', fontsize = 18, y = 1.03)
plt.legend()
plt.savefig('Opt_Logistic_Regression_Learning_Curve.jpeg')  

#
# In[26]:
#Testing the Logistic Regression Model
  
### Logistic Regression Testing with Balanced training set
y_pred_test_LR = model_LR_bal.predict(X_testing)
y_pred_test_LR = model_LR_bal.predict(X_testing_FS)
print("\n")
print("\nLogistic Regression optimal model on test set with balanced training\n------------------------")
print("Final accuracy score on the test data: {:.4f}".format(accuracy_score(y_testing, y_pred_test_LR)))
print("Final Precision score on test data: {:.4f}".format(precision_score(y_testing, y_pred_test_LR)))
print("Final Recall score on test data: {:.4f}".format(recall_score(y_testing, y_pred_test_LR)))
print("Final ROC AUC score on test data: {:.4f}".format(roc_auc_score(y_testing, y_pred_test_LR)))
print("Final mean squared error score on test data: {:.4f}".format(mean_squared_error(y_testing, y_pred_test_LR)))
print("Final f1 score on test data: {:.4f}".format(f1_score(y_testing, y_pred_test_LR)))

# In[27]:
# save the model to current directory

joblib.dump(model_LR_opt, r'C:\Users\isogb\Documents\FYP Software\Saved Models\Logistic_Regression_model.pkl')

Plot Confusion Matrix for Logistic Regression
confusion_matrix_LR = confusion_matrix(y_testing, y_pred_test_LR)
plt.figure(figsize=(6,6))
plot_confusion_matrix(confusion_matrix_LR, classes=['not subscribe','subscribe'],
                      title='Logistic Regression Confusion matrix')

# In[28]:
# Decision Tree Models

## Decision tree model without GridserachCV on imbalanced training set
model_DT_imb = DecisionTreeClassifier(random_state=42)
model_DT_imb.fit(X_training_imb, y_training_imb)
y_pred_DT_imb = model_DT_imb.predict(X_training_imb)
print("\nDecision Tree model on imbalanced training set\n------------------------")
print("Final accuracy score on the training data: {:.4f}".format(accuracy_score(y_training_imb, y_pred_DT_imb)))
print("Final precision score on training data: {:.4f}".format(precision_score(y_training_imb, y_pred_DT_imb)))
print("Final Recall score on training data: {:.4f}".format(recall_score(y_training_imb, y_pred_DT_imb)))
print("Final ROC AUC score on training data: {:.4f}".format(roc_auc_score(y_training_imb, y_pred_DT_imb)))
print("Final mean squared error score on training data: {:.4f}".format(mean_squared_error(y_training_imb, y_pred_DT_imb)))
print("Final f1 score on training data: {:.4f}".format(f1_score(y_training_imb, y_pred_DT_imb)))

## Decision tree model without GridserachCV on imbalanced validation set
model_DT_imb.fit(X_validation_imb, y_validation_imb)
y_pred_DT_imb = model_DT_imb.predict(X_validation_imb)
print("\nDecision Tree model on imbalanced validation set\n------------------------")
print("Final accuracy score on the validation dataa: {:.4f}".format(accuracy_score(y_validation_imb, y_pred_DT_imb)))
print("Final precision score on validation data: {:.4f}".format(precision_score(y_validation_imb, y_pred_DT_imb)))
print("Final Recall score on validation data: {:.4f}".format(recall_score(y_validation_imb, y_pred_DT_imb)))
print("Final ROC AUC score on validation data: {:.4f}".format(roc_auc_score(y_validation_imb, y_pred_DT_imb)))
print("Final mean squared error score on training data: {:.4f}".format(mean_squared_error(y_validation_imb, y_pred_DT_imb)))
print("Final f1 score on training data: {:.4f}".format(f1_score(y_validation_imb, y_pred_DT_imb)))

# Decision tree model without GridserachCV on balanced training set
model_DT_bal = DecisionTreeClassifier(random_state=42, max_depth = 100)
model_DT_bal = DecisionTreeClassifier(random_state=42, max_depth = 5, min_samples_leaf =2, min_samples_split = 2)
model_DT_bal.fit(X_training, y_training)
y_pred_DT_bal = model_DT_bal.predict(X_training)
model_DT_bal.fit(X_training_FS, y_training)
y_pred_DT_bal = model_DT_bal.predict(X_training_FS)
print("\nDecision Tree model on balanced training set\n------------------------")
print("Final accuracy score on the training data: {:.4f}".format(accuracy_score(y_training, y_pred_DT_bal)))
print("Final precision score on training data: {:.4f}".format(precision_score(y_training, y_pred_DT_bal)))
print("Final Recall score on training data: {:.4f}".format(recall_score(y_training, y_pred_DT_bal)))
print("Final ROC AUC score on training data: {:.4f}".format(roc_auc_score(y_training, y_pred_DT_bal)))
print("Final mean squared error score on training data: {:.4f}".format(mean_squared_error(y_training, y_pred_DT_bal)))
print("Final f1 score on training data: {:.4f}".format(f1_score(y_training, y_pred_DT_bal)))

##### Decision tree model without GridserachCV on balanced validation set
model_DT_bal.fit(X_validation, y_validation)
y_pred_DT_bal = model_DT_bal.predict(X_validation)
model_DT_bal.fit(X_validation_FS, y_validation)
y_pred_DT_bal = model_DT_bal.predict(X_validation_FS)
print("\nDecision Tree model on balanced validation set\n------------------------")
print("Final accuracy score on the validation data: {:.4f}".format(accuracy_score(y_validation, y_pred_DT_bal)))
print("Final precision score on validation data: {:.4f}".format(precision_score(y_validation, y_pred_DT_bal)))
print("Final Recall score on validation data: {:.4f}".format(recall_score(y_validation, y_pred_DT_bal)))
print("Final ROC AUC score on validation data: {:.4f}".format(roc_auc_score(y_validation, y_pred_DT_bal)))
print("Final mean squared error score on training data: {:.4f}".format(mean_squared_error(y_validation, y_pred_DT_bal)))
print("Final f1 score on training data: {:.4f}".format(f1_score(y_validation, y_pred_DT_bal)))

# In[29]: 

#Plotting Decision Tree Learning Curves pre optimization

train_sizes, train_scores, validation_scores = learning_curve(
    estimator = model_DT_bal, X = X_train_val,
    y = y_train_val, train_sizes = numpy.linspace(0.01, 1.0, 25), cv = 5, shuffle=True, random_state = 42, 
    scoring = 'neg_mean_squared_error')
train_scores_mean = -train_scores.mean(axis = 1)
validation_scores_mean = -validation_scores.mean(axis = 1)
plt.plot(train_sizes, train_scores_mean, "r-", linewidth=2, label = 'Training Error')
plt.plot(train_sizes, validation_scores_mean, "b--", linewidth=2, label = 'Validation Error')
plt.ylabel('MSE', fontsize = 14)
plt.xlabel('Training set size', fontsize = 14)
plt.title('Decision Tree Learning Curve', fontsize = 18, y = 1.03)
plt.legend()
plt.savefig('DT_Learning_Curve.jpeg')  


# In[30]:

# Decision Tree Optimization on Balanced Training set

# Create the parameters list
parameters_DT = {'max_depth': [6,7,8],
                 'min_samples_leaf': [2,3],
                 'min_samples_split': [2,3]}


max_depth = [20,30,40,50,60,70,80]
max_depth = [30,32,35,36,38,40,42,44,46,48,50]
min_samples_leaf = [2,3,4]
min_samples_split = [2]
grid_obj = GridSearchCV(DecisionTreeClassifier(random_state=42),
                        dict(max_depth =max_depth,min_samples_leaf = min_samples_leaf
                             ,min_samples_split = min_samples_split), 
                             scoring= 'roc_auc', cv=5)

### Fit the grid search object to the training data and find the optimal parameters using fit()
grid_fit = grid_obj.fit(X_training, y_training)
grid_fit = grid_obj.fit(X_training_FS, y_training)

## Get the estimator
model_DT_opt = grid_fit.best_estimator_

print("The best parameters are: {}".format(model_DT_opt))
##
results = grid_obj.cv_results_
for mean_score, params in zip(results["mean_test_score"], results["params"]):
    print((mean_score), params)
#    
DT_data = [go.Contour(z = results["mean_test_score"],
        x = results['param_max_depth'], y = results['param_min_samples_leaf'],
        colorscale ='Jet',contours = dict(showlabels = True,labelfont = dict(
        family = 'Raleway',size = 15,color = 'black',)), colorbar = dict( title = 'Score',
        titleside = 'right', titlefont = dict( size =16, family = 'Arial, sans-serif',)))]
py.offline.plot(DT_data)


##Report the metrics scores on train data
model_DT_opt.fit(X_training, y_training)
model_DT_opt.fit(X_training_FS, y_training)
y_pred = model_DT_opt.predict(X_training)    
y_pred = model_DT_opt.predict(X_training_FS) 
print("\n")
print("\nOptimal LR Model performance on training set\n------------------------")
print("Final accuracy score on the training data: {:.4f}".format(accuracy_score(y_training, y_pred)))  
print("Final precision score on training data: {:.4f}".format(precision_score(y_training, y_pred)))
print("Final Recall score on training data: {:.4f}".format(recall_score(y_training, y_pred)))
print("Final ROC AUC score on training data: {:.4f}".format(roc_auc_score(y_training, y_pred)))
print("Final mean squared error score on training data: {:.4f}".format(mean_squared_error(y_training, y_pred)))
print("Final f1 score on training data: {:.4f}".format(f1_score(y_training, y_pred)))
print("\n")



## Optimal Decision tree model with GridserachCV on balanced validation set
model_DT_opt.fit(X_validation, y_validation)
y_pred_DT = model_DT_opt.predict(X_validation)
model_DT_opt.fit(X_validation_FS, y_validation)
y_pred_DT = model_DT_opt.predict(X_validation_FS)
print("\nDecision Tree model on balanced validation set\n------------------------")
print("Final accuracy score on the validation data: {:.4f}".format(accuracy_score(y_validation, y_pred_DT)))
print("Final precision score on validation data: {:.4f}".format(precision_score(y_validation, y_pred_DT)))
print("Final Recall score on validation data: {:.4f}".format(recall_score(y_validation, y_pred_DT)))
print("Final ROC AUC score on validation data: {:.4f}".format(roc_auc_score(y_validation, y_pred_DT)))
print("Final mean squared error score on training data: {:.4f}".format(mean_squared_error(y_validation, y_pred_DT)))
print("Final f1 score on training data: {:.4f}".format(f1_score(y_validation, y_pred_DT)))

# In[31]: 

#Plotting Decision Tree Learning Curves post optimization
train_sizes, train_scores, validation_scores = learning_curve(
    estimator = model_DT_opt, X = X_train_val,
    y = y_train_val, train_sizes = numpy.linspace(0.01, 1.0, 35), cv = 5, shuffle=True, random_state = 42, 
    scoring = 'neg_mean_squared_error')
train_scores_mean = -train_scores.mean(axis = 1)
validation_scores_mean = -validation_scores.mean(axis = 1)
plt.plot(train_sizes, train_scores_mean, "r-", linewidth=2, label = 'Training Error')
plt.plot(train_sizes, validation_scores_mean, "b--", linewidth=2, label = 'Validation Error')
plt.ylabel('MSE', fontsize = 14)
plt.xlabel('Training set size', fontsize = 14)
plt.title('Optimized DT Learning Curve', fontsize = 18, y = 1.03)
plt.legend()
plt.savefig('Opt_DT_Learning_Curve.jpeg')  

    
# In[32]:
### Decision Tree Testing 
##
y_pred_test_DT = model_DT_bal.predict(X_testing_FS)
y_pred_test_DT = model_DT_opt.predict(X_testing)
print("\n")
print("\nDecision Tree optimal model on test set with balanced training\n------------------------")
print("Final accuracy score on the test data: {:.4f}".format(accuracy_score(y_testing, y_pred_test_DT)))
print("Final Precision score on test data: {:.4f}".format(precision_score(y_testing, y_pred_test_DT)))
print("Final Recall score on test data: {:.4f}".format(recall_score(y_testing, y_pred_test_DT)))
print("Final ROC AUC score on test data: {:.4f}".format(roc_auc_score(y_testing, y_pred_test_DT)))
print("Final mean squared error score on test data: {:.4f}".format(mean_squared_error(y_testing, y_pred_test_DT)))
print("Final f1 score on test data: {:.4f}".format(f1_score(y_testing, y_pred_test_DT)))

## In[33]:
#
## save the Optimal Decision Tree model to current directory
joblib.dump(model_DT_opt, r'C:\Users\isogb\Documents\FYP Software\Saved Models\Decision_Tree_model.pkl')

confusion_matrix_DT = confusion_matrix(y_testing, y_pred_test_DT)
plt.figure(figsize=(6,6))
plot_confusion_matrix(confusion_matrix_DT, classes=['not subscribe','subscribe'],
                      title='Decision Tree Confusion matrix')
# In[34]:

# SVM Models

 SVM model without GridserachCV on imbalanced training set
model_SVM_bal = SGDClassifier(loss= 'hinge', random_state=42, max_iter = 100)
model_SVM_imb = LinearSVC(loss= 'hinge', random_state=42)
model_SVM_imb.fit(X_training_imb, y_training_imb)
y_pred_SVM_imb = model_SVM_imb.predict(X_training_imb)
print("\nSVM on imbalanced training set\n------------------------")
print("Final accuracy score on the training data: {:.4f}".format(accuracy_score(y_training_imb, y_pred_SVM_imb)))
print("Final precision score on training data: {:.4f}".format(precision_score(y_training_imb, y_pred_SVM_imb)))
print("Final Recall score on training data: {:.4f}".format(recall_score(y_training_imb, y_pred_SVM_imb)))
print("Final ROC AUC score on training data: {:.4f}".format(roc_auc_score(y_training_imb, y_pred_SVM_imb)))
print("Final mean squared error score on training data: {:.4f}".format(mean_squared_error(y_training_imb, y_pred_SVM_imb)))
print("Final f1 score on training data: {:.4f}".format(f1_score(y_training_imb, y_pred_SVM_imb)))
#
## SVM model without GridserachCV on imbalanced validation set
model_SVM_imb.fit(X_validation_imb, y_validation_imb)
y_pred_SVM_imb = model_SVM_imb.predict(X_validation_imb)
print("\nSVM model on imbalanced validation set\n------------------------")
print("Final accuracy score on the validation dataa: {:.4f}".format(accuracy_score(y_validation_imb, y_pred_SVM_imb)))
print("Final precision score on validation data: {:.4f}".format(precision_score(y_validation_imb, y_pred_SVM_imb)))
print("Final Recall score on validation data: {:.4f}".format(recall_score(y_validation_imb, y_pred_SVM_imb)))
print("Final ROC AUC score on validation data: {:.4f}".format(roc_auc_score(y_validation_imb, y_pred_SVM_imb)))
print("Final mean squared error score on validation data: {:.4f}".format(mean_squared_error(y_validation_imb, y_pred_SVM_imb)))
print("Final f1 score on validation data: {:.4f}".format(f1_score(y_validation_imb, y_pred_SVM_imb)))

# In[35]:

#rbf/linear SVM solver

## SVM model without GridserachCV on balanced training set
model_SVM_bal = SVC(kernel="linear", random_state=42)
model_SVM_bal.fit(X_training, y_training)
y_pred_SVM_bal = model_SVM_bal.predict(X_training)
print("\nSVM model on balanced training set\n------------------------")
print("Final accuracy score on the training data: {:.4f}".format(accuracy_score(y_training, y_pred_SVM_bal)))
print("Final precision score on training data: {:.4f}".format(precision_score(y_training, y_pred_SVM_bal)))
print("Final Recall score on training data: {:.4f}".format(recall_score(y_training, y_pred_SVM_bal)))
print("Final ROC AUC score on training data: {:.4f}".format(roc_auc_score(y_training, y_pred_SVM_bal)))
print("Final mean squared error score on training data: {:.4f}".format(mean_squared_error(y_training, y_pred_SVM_bal)))
print("Final f1 score on training data: {:.4f}".format(f1_score(y_training, y_pred_SVM_bal)))
#
## SVM model without GridserachCV on balanced validation set
model_SVM_bal.fit(X_validation, y_validation)
y_pred_SVM_bal = model_SVM_bal.predict(X_validation)
print("\nSVM model on balanced validation set\n------------------------")
print("Final accuracy score on the validation data: {:.4f}".format(accuracy_score(y_validation, y_pred_SVM_bal)))
print("Final precision score on validation data: {:.4f}".format(precision_score(y_validation, y_pred_SVM_bal)))
print("Final Recall score on validation data: {:.4f}".format(recall_score(y_validation, y_pred_SVM_bal)))
print("Final ROC AUC score on validation data: {:.4f}".format(roc_auc_score(y_validation, y_pred_SVM_bal)))
print("Final mean squared error score on validation data: {:.4f}".format(mean_squared_error(y_validation, y_pred_SVM_bal)))
print("Final f1 score on validation data: {:.4f}".format(f1_score(y_validation, y_pred_SVM_bal)))


# In[36]:

#SGD SVM solver

#model_SVM_bal = SGDClassifier(loss= 'hinge', random_state=42, max_iter = 100)
model_SVM_bal = LinearSVC(loss= 'hinge', random_state = 42, C=80, penalty = 'l2')
model_SVM_bal.fit(X_training, y_training)
y_pred_LR_bal = model_SVM_bal.predict(X_training)
model_SVM_bal.fit(X_training_FS, y_training)
y_pred_LR_bal = model_SVM_bal.predict(X_training_FS)
print("\nSVM model on balanced training set\n------------------------")
print("Final accuracy score on the training data: {:.4f}".format(accuracy_score(y_training, y_pred_LR_bal)))
print("Final precision score on training data: {:.4f}".format(precision_score(y_training, y_pred_LR_bal)))
print("Final Recall score on training data: {:.4f}".format(recall_score(y_training, y_pred_LR_bal)))
print("Final ROC AUC score on training data: {:.4f}".format(roc_auc_score(y_training, y_pred_LR_bal)))
print("Final mean squared error score on training data: {:.4f}".format(mean_squared_error(y_training, y_pred_LR_bal)))
print("Final f1 score on training data: {:.4f}".format(f1_score(y_training, y_pred_LR_bal)))


model_SVM_bal.fit(X_validation, y_validation)
y_pred_LR_bal = model_SVM_bal.predict(X_validation)
model_SVM_bal.fit(X_validation_FS, y_validation)
y_pred_LR_bal = model_SVM_bal.predict(X_validation_FS)
print("\nSVM model on balanced validation set\n------------------------")
print("Final accuracy score on the validation data: {:.4f}".format(accuracy_score(y_validation, y_pred_LR_bal)))
print("Final precision score on validation data: {:.4f}".format(precision_score(y_validation, y_pred_LR_bal)))
print("Final Recall score on validation data: {:.4f}".format(recall_score(y_validation, y_pred_LR_bal)))
print("Final ROC AUC score on validation data: {:.4f}".format(roc_auc_score(y_validation, y_pred_LR_bal)))
print("Final mean squared error score on validation data: {:.4f}".format(mean_squared_error(y_validation, y_pred_LR_bal)))
print("Final f1 score on validation data: {:.4f}".format(f1_score(y_validation, y_pred_LR_bal)))

# In[37]: 

#Plotting SVM Learning Curves pre optimization

train_sizes, train_scores, validation_scores = learning_curve(
    estimator = model_SVM_bal, X = X_train_val,
    y = y_train_val, train_sizes = numpy.linspace(0.01, 1.0, 25), cv = 7, shuffle=True, random_state = 42, 
    scoring = 'neg_mean_squared_error')
train_scores_mean = -train_scores.mean(axis = 1)
validation_scores_mean = -validation_scores.mean(axis = 1)
plt.plot(train_sizes, train_scores_mean, "r-", linewidth=2, label = 'Training Error')
plt.plot(train_sizes, validation_scores_mean, "b--", linewidth=2, label = 'Validation Error')
plt.ylabel('MSE', fontsize = 14)
plt.xlabel('Training set size', fontsize = 14)
plt.title('Linear SVM Learning Curve', fontsize = 18, y = 1.03)
plt.legend()
plt.savefig('Linear_SVC_Learning_Curve.jpeg')  

# In[38]:

# SVM Optimization on Balanced Training set

# Create the parameters list

Cs = [10,20,30,40,50,60,70,80,90,100,110,120]
Cs = [80]
penalty = ['l2']
#
grid_obj = GridSearchCV(LinearSVC(loss= 'hinge', random_state=42),
                        dict(C=Cs,penalty=penalty), 
                           scoring= 'roc_auc', cv=3)
###
###### Fit the grid search object to the training data and find the optimal parameters using fit()
grid_fit = grid_obj.fit(X_training, y_training)
##grid_fit = grid_obj.fit(X_training_FS, y_training)
####
##
#### Get the estimator
model_SVM_opt = grid_fit.best_estimator_
print("The best parameters are: {}".format(model_SVM_opt)) 
#
results = grid_obj.cv_results_
for mean_score, params in zip(results["mean_test_score"], results["params"]):
    print((mean_score), params)
#    
SVC_data = [go.Contour(z = results["mean_test_score"],
        x = results['param_C'], y = results['param_penalty'],
        colorscale ='Jet',contours = dict(showlabels = True,labelfont = dict(
        family = 'Raleway',size = 15,color = 'black',)), colorbar = dict( title = 'Score',
        titleside = 'right', titlefont = dict( size =16, family = 'Arial, sans-serif',)))]
py.offline.plot(SVC_data)
#
####Report the metrics scores on train data
model_SVM_opt.fit(X_training, y_training)
model_SVM_opt.fit(X_training_FS, y_training)
y_pred = model_SVM_opt.predict(X_training_FS)  
print("The best parameters are: {}".format(model_SVM_opt))  
print("\n")
print("\nOptimal SVM Model performance on training set\n------------------------")
print("Final accuracy score on the training data: {:.4f}".format(accuracy_score(y_training, y_pred)))  
print("Final precision score on training data: {:.4f}".format(precision_score(y_training, y_pred)))
print("Final Recall score on training data: {:.4f}".format(recall_score(y_training, y_pred)))
print("Final ROC AUC score on training data: {:.4f}".format(roc_auc_score(y_training, y_pred)))
print("Final mean squared error score on training data: {:.4f}".format(mean_squared_error(y_training, y_pred)))
print("Final f1 score on training data: {:.4f}".format(f1_score(y_training, y_pred)))
print("\n")

###Evaluate optimal model against Validation Set 
y_pred_LR = model_SVM_opt.predict(X_validation)
y_pred_LR = model_SVM_opt.predict(X_validation_FS)
print("\n")
print("\nOptimal SVM model on balanced validation set\n------------------------")
print("Final accuracy score on the validation data: {:.4f}".format(accuracy_score(y_validation, y_pred_LR)))
print("Final precision score on validation data: {:.4f}".format(precision_score(y_validation, y_pred_LR)))
print("Final Recall score on validation data: {:.4f}".format(recall_score(y_validation, y_pred_LR)))
print("Final ROC AUC score on validation data: {:.4f}".format(roc_auc_score(y_validation, y_pred_LR)))
print("Final mean squared error score on validation data: {:.4f}".format(mean_squared_error(y_validation, y_pred_LR)))
print("Final f1 score on validation data: {:.4f}".format(f1_score(y_validation, y_pred_LR)))

# In[39]: 

#Plotting SVM Learning Curves post optimization

train_sizes, train_scores, validation_scores = learning_curve(
    estimator = model_SVM_opt, X = X_train_val,
    y = y_train_val, train_sizes = numpy.linspace(0.01, 1.0, 15), cv = 7, shuffle=True, random_state = 42, 
    scoring = 'neg_mean_squared_error')
train_scores_mean = -train_scores.mean(axis = 1)
validation_scores_mean = -validation_scores.mean(axis = 1)
plt.plot(train_sizes, train_scores_mean, "r-", linewidth=2, label = 'Training Set')
plt.plot(train_sizes, validation_scores_mean, "b--", linewidth=2, label = 'Validation Set')
plt.ylabel('MSE', fontsize = 14)
plt.xlabel('Training set size', fontsize = 14)
plt.title('Optimized Linear SVC Model Learning Curve', fontsize = 18, y = 1.03)
plt.legend()
plt.savefig('Opt_LinearSVC_Learning_Curve.jpeg')  

# In[40]:
## SVM Testing

y_pred_test_SVM = model_SVM_bal.predict(X_testing_FS)
y_pred_test_SVM = model_SVM_bal.predict(X_testing)
print("\n")
print("\nSVM optimal model on test set with balanced training\n------------------------")
print("Final accuracy score on the test data: {:.4f}".format(accuracy_score(y_testing, y_pred_test_SVM)))
print("Final Precision score on test data: {:.4f}".format(precision_score(y_testing, y_pred_test_SVM)))
print("Final Recall score on test data: {:.4f}".format(recall_score(y_testing, y_pred_test_SVM)))
print("Final ROC AUC score on test data: {:.4f}".format(roc_auc_score(y_testing, y_pred_test_SVM)))
print("Final mean squared error score on validation data: {:.4f}".format(mean_squared_error(y_testing, y_pred_test_SVM)))
print("Final f1 score on validation data: {:.4f}".format(f1_score(y_testing, y_pred_test_SVM)))

## In[41]:
#
##save the optimal SVM model to current directory
joblib.dump(model_SVM_opt, r'C:\Users\isogb\Documents\FYP Software\Saved Models\SVM_model.pkl')
##
confusion_matrix_SVC = confusion_matrix(y_testing, y_pred_test_SVM)
plt.figure(figsize=(6,6))
plot_confusion_matrix(confusion_matrix_SVC, classes=['not subscribe','subscribe'],
                      title='Linear SVC Confusion matrix')

					  
# In[42]:
#Neural Nets

##Converting panda dataframe to numpy array for Keras
X_train_NN = X_training.values
X_train_NN = X_training_FS.values
y_train_NN = y_training.values
X_val_NN = X_validation.values
X_val_NN = X_validation_FS.values
y_val_NN = y_validation.values
X_test_NN = X_testing.values
X_test_NN = X_testing_FS.values
y_test_NN = y_testing.values

#
X_train_NN = X_training_imb.values
y_train_NN = y_training_imb.values
X_val_NN = X_validation_imb.values
y_val_NN = y_validation_imb.values


#Neural Net model without GridserachCV on balanced training set    

#Building and training the Neural Net Model 
model_NN_bal = Sequential()
model_NN_bal.add(Dense(5, activation='sigmoid', 
                       init ='glorot_uniform',input_shape=(52,)))
model_NN_bal.add(Dense(10, activation='sigmoid', init ='glorot_uniform', 
                       kernel_regularizer = l2(0.0003), input_shape=(52,)))
model_NN_bal.add(Dropout(.20))
model_NN_bal.add(Dense(1, activation='sigmoid'))


#### Compiling the model
model_NN_bal.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['mse'])
model_NN_bal.summary()


#### Training the model
epochs = 50
checkpointer = ModelCheckpoint(filepath= r'C:\Users\isogb\Documents\FYP Software\Saved Models\weights.best.from_scratch.hdf5', monitor= 'mse',
                               verbose=1, save_best_only=False, mode='max')

earlystopping = EarlyStopping(monitor='mean_squared_error', patience=10, verbose=1, mode='auto')
NN_classifier = model_NN_bal.fit(X_train_NN, y_train_NN, 
          validation_data=(X_val_NN, y_val_NN),
          epochs=epochs, batch_size=10, callbacks=[earlystopping, checkpointer], verbose=1)

		  
print(NN_classifier.history.keys())

## Make predictions with the model
y_pred_NN_train = model_NN_bal.predict(X_train_NN)
y_pred_NN_train = (y_pred_NN_train > 0.5)
y_pred_NN_val = model_NN_bal.predict(X_val_NN)
y_pred_NN_val = (y_pred_NN_val > 0.5)
y_pred_NN_test = model_NN_bal.predict(X_test_NN)
y_pred_NN_test = (y_pred_NN_test > 0.5)


### evaluation on training set
print("\nNeural Network model on train set\n------------------------")
print("Final accuracy score on train data: {:.4f}".format(accuracy_score(y_train_NN, y_pred_NN_train)))
print("Final Precision score on train data: {:.4f}".format(precision_score(y_train_NN, y_pred_NN_train)))
print("Final Recall score on train data: {:.4f}".format(recall_score(y_train_NN, y_pred_NN_train)))
print("Final ROC AUC score on train data: {:.4f}".format(roc_auc_score(y_train_NN, y_pred_NN_train)))
print("Final mean squared error on train data: {:.4f}".format(mean_squared_error(y_train_NN, y_pred_NN_train)))
print("Final f1 score on train data: {:.4f}".format(f1_score(y_train_NN, y_pred_NN_train)))

### evaluation on validation set
print("\nNeural Network model on validation set\n------------------------")
print("Final accuracy score on validation data: {:.4f}".format(accuracy_score(y_val_NN, y_pred_NN_val)))
print("Final Precision score on validation data: {:.4f}".format(precision_score(y_val_NN, y_pred_NN_val)))
print("Final Recall score on validation data: {:.4f}".format(recall_score(y_val_NN, y_pred_NN_val)))
print("Final ROC AUC score on validation data: {:.4f}".format(roc_auc_score(y_val_NN, y_pred_NN_val)))
print("Final mean squared error on validation data: {:.4f}".format(mean_squared_error(y_val_NN, y_pred_NN_val)))
print("Final f1 score on  validation data: {:.4f}".format(f1_score(y_val_NN, y_pred_NN_val)))

#
### In[43]:
#Plotting learning curve for Neural Net pre optimization
##
plt.plot(NN_classifier.history['mean_squared_error'], "r-", linewidth=2, label = 'Training error')
plt.plot(NN_classifier.history['val_mean_squared_error'], "b-", linewidth=2, label = 'Validation error')
plt.ylabel('MSE', fontsize = 14)
plt.xlabel('Epochs', fontsize = 14)
plt.title('Optimized Neural Network MSE vs Number of Epochs with Dropout and Earlystopping', fontsize = 18, y = 1.03)
plt.legend()
plt.savefig('NeuralNet_MSE_Epoch_Curve.jpeg')  


# In[44]:
#Grid Search CV optimization

#Converting panda dataframe to numpy array for Keras
X_train_NN = X_training.values
y_train_NN = y_training.values
X_val_NN = X_validation.values
y_val_NN = y_validation.values
X_test_NN = X_testing.values
y_test_NN = y_testing.values


def create_model(kernel_regularizer = l2(0.0005), neurons = 5):
    model_NN_bal = Sequential()
    model_NN_bal.add(Dense(neurons, activation='sigmoid',init ='glorot_uniform', 
                           kernel_regularizer = kernel_regularizer,
                          input_shape=(52,)))
    model_NN_bal.add(Dropout(.2))
    model_NN_bal.add(Dense(1, activation='sigmoid'))

	# Compiling the model
    model_NN_bal.compile(loss = 'binary_crossentropy', optimizer= 'adam',
                          metrics=['mse'])
    return model_NN_bal

## Training the model

epochs = 40
earlystopping = EarlyStopping(monitor='mean_squared_error', patience=5, verbose=1, mode='auto')
NN_classifier = model_NN_bal.fit(X_train_NN, y_train_NN, 
          validation_data=(X_val_NN, y_val_NN),
          epochs=epochs, batch_size=10, callbacks=[earlystopping], verbose=1)
  
model_NN_bal = KerasClassifier(build_fn=create_model, epochs= 20, batch_size = 40,
                              verbose = 1)


neurons = [1,10,40]
kernel_regularizer = [l2(0.0001), l2(0.0005), l2(0.005)]
#
grid_obj = GridSearchCV(estimator = model_NN_bal,
                       param_grid = dict(kernel_regularizer = kernel_regularizer,
                                         neurons = neurons), scoring = 'neg_mean_squared_error')

grid_classifier = grid_obj.fit(X_train_NN, y_train_NN)
NN_opt = grid_classifier.best_estimator_

results = grid_obj.cv_results_

for mean_score, params in zip(results["mean_test_score"], results["params"]):
    print((mean_score), params)

print(grid_obj.best_params_)   

## Visualising the Model
x = numpy.ma.array(results['param_kernel_regularizer'])
new_array = []
for i in x :
   p = i.get_config()['l2']
   new_array.append(p)  
       
NN_data = [go.Contour(z = results["mean_test_score"],
       x = new_array, y = results['param_neurons'],
        colorscale ='Jet',contours = dict(showlabels = True,labelfont = dict(
        family = 'Raleway',size = 15,color = 'black',)), colorbar = dict( title = 'Score',
        titleside = 'right', titlefont = dict( size =16, family = 'Arial, sans-serif',)))]
py.offline.plot(NN_data)

###
epochs = 30
checkpointer = ModelCheckpoint(filepath= r'C:\Users\isogb\Documents\FYP Software\Saved Models\weights.best.from_scratch.hdf5', monitor= 'mse',
                               verbose=1, save_best_only=False, mode='max')
earlystopping = EarlyStopping(monitor='mean_squared_error', patience=9, verbose=1, mode='auto')
NN_opt_1 = NN_opt.fit(X_train_NN, y_train_NN, 
          validation_data=(X_val_NN, y_val_NN),
          epochs=epochs, batch_size=10, callbacks=[checkpointer], verbose=1)


## Make predictions with the optimal model
y_pred_NN_train = NN_opt.predict(X_train_NN)
y_pred_NN_train = (y_pred_NN_train > 0.5)
y_pred_NN_val = NN_opt.predict(X_val_NN)
y_pred_NN_val = (y_pred_NN_val > 0.5)
y_pred_NN_test = NN_opt.predict(X_test_NN)
y_pred_NN_test = (y_pred_NN_test > 0.5)


## evaluation on training set
print("\nNeural Network model on train set\n------------------------")
print("Final accuracy score on train data: {:.4f}".format(accuracy_score(y_train_NN, y_pred_NN_train)))
print("Final Precision score on train data: {:.4f}".format(precision_score(y_train_NN, y_pred_NN_train)))
print("Final Recall score on train data: {:.4f}".format(recall_score(y_train_NN, y_pred_NN_train)))
print("Final ROC AUC score on train data: {:.4f}".format(roc_auc_score(y_train_NN, y_pred_NN_train)))
print("Final mean squared error on train data: {:.4f}".format(mean_squared_error(y_train_NN, y_pred_NN_train)))
print("Final f1 score on train data: {:.4f}".format(f1_score(y_train_NN, y_pred_NN_train)))
print("\n")

# evaluation on validation set
print("\nNeural Network model on validation set\n------------------------")
print("Final accuracy score on validation data: {:.4f}".format(accuracy_score(y_val_NN, y_pred_NN_val)))
print("Final Precision score on validation data: {:.4f}".format(precision_score(y_val_NN, y_pred_NN_val)))
print("Final Recall score on validation data: {:.4f}".format(recall_score(y_val_NN, y_pred_NN_val)))
print("Final ROC AUC score on validation data: {:.4f}".format(roc_auc_score(y_val_NN, y_pred_NN_val)))
print("Final mean squared error on  validation data: {:.4f}".format(mean_squared_error(y_val_NN, y_pred_NN_val)))
print("Final f1 score on  validation data: {:.4f}".format(f1_score(y_val_NN, y_pred_NN_val)))

# In[44]:
#
print("\nNeural Network model on Test set\n------------------------")
print("Final accuracy score on test data: {:.4f}".format(accuracy_score(y_test_NN, y_pred_NN_test)))
print("Final Precision score on test data: {:.4f}".format(precision_score(y_test_NN, y_pred_NN_test)))
print("Final Recall score on test data: {:.4f}".format(recall_score(y_test_NN, y_pred_NN_test)))
print("Final ROC AUC score on test data: {:.4f}".format(roc_auc_score(y_test_NN, y_pred_NN_test)))
print("Final mean squared error on test data: {:.4f}".format(mean_squared_error(y_test_NN, y_pred_NN_test)))
print("Final f1 score on test data: {:.4f}".format(f1_score(y_test_NN, y_pred_NN_test)))

# In[45]:
# Plotting learning curve for Neural Net post optimization
plt.plot(NN_opt_1.history['mean_squared_error'], "r-", linewidth=2, label = 'Training error')
plt.plot(NN_opt_1.history['val_mean_squared_error'], "b-", linewidth=2, label = 'Validation error')
plt.ylabel('MSE', fontsize = 14)
plt.xlabel('Epochs', fontsize = 14)
plt.title('Opimized Neural Network MSE vs Number of Epochs', fontsize = 18, y = 1.03)
plt.legend()
plt.savefig('NeuralNet_MSE_Epoch_Curve.jpeg')  

# In[46]:
#
confusion_matrix_NN = confusion_matrix(y_testing, y_pred_NN_test)
plt.figure(figsize=(6,6))
plot_confusion_matrix(confusion_matrix_NN, classes=['not subscribe','subscribe'],
                      title='Neural Network Confusion matrix')

# In[49]:
##Bagging Classifier Performance

bag_clf =BaggingClassifier(LinearSVC(loss= 'hinge', random_state=42, C=0.5,penalty='l1')
, n_estimators = 10, max_samples = 10, bootstrap = True, n_jobs = -1)
bag_clf =BaggingClassifier(DecisionTreeClassifier(random_state=42, max_depth = 7, min_samples_leaf =1, min_samples_split = 8)
,n_estimators = 200, max_samples = 30, bootstrap = True)

bag_clf.fit(X_training, y_training) 
y_pred_bag_bal = bag_clf.predict(X_training)
print("\nBagging model on balanced training set\n------------------------")
print("Final accuracy score on the training data: {:.4f}".format(accuracy_score(y_training, y_pred_bag_bal)))
print("Final precision score on training data: {:.4f}".format(precision_score(y_training, y_pred_bag_bal)))
print("Final Recall score on training data: {:.4f}".format(recall_score(y_training, y_pred_bag_bal)))
print("Final ROC AUC score on training data: {:.4f}".format(roc_auc_score(y_training, y_pred_bag_bal)))
print("Final mean squared error score on training data: {:.4f}".format(mean_squared_error(y_training, y_pred_bag_bal)))
print("Final f1 score on training data: {:.4f}".format(f1_score(y_training, y_pred_bag_bal)))


bag_clf.fit(X_validation, y_validation)
y_pred_bag_bal = bag_clf.predict(X_validation)
print("\nBagging model on balanced validation set\n------------------------")
print("Final accuracy score on the validation data: {:.4f}".format(accuracy_score(y_validation, y_pred_bag_bal)))
print("Final precision score on validation data: {:.4f}".format(precision_score(y_validation, y_pred_bag_bal)))
print("Final Recall score on validation data: {:.4f}".format(recall_score(y_validation, y_pred_bag_bal)))
print("Final ROC AUC score on validation data: {:.4f}".format(roc_auc_score(y_validation, y_pred_bag_bal)))
print("Final mean squared error score on validation data: {:.4f}".format(mean_squared_error(y_validation, y_pred_bag_bal)))
print("Final f1 score on validation data: {:.4f}".format(f1_score(y_validation, y_pred_bag_bal)))
#
y_pred_bagging = bag_clf.predict(X_testing)
print("\n")
print("\nBagging optimal model on test set with balanced training\n------------------------")
print("Final accuracy score on the test data: {:.4f}".format(accuracy_score(y_testing, y_pred_bagging)))
print("Final Precision score on test data: {:.4f}".format(precision_score(y_testing, y_pred_bagging)))
print("Final Recall score on test data: {:.4f}".format(recall_score(y_testing, y_pred_bagging)))
print("Final ROC AUC score on test data: {:.4f}".format(roc_auc_score(y_testing, y_pred_bagging)))
print("Final mean squared error score on test data: {:.4f}".format(mean_squared_error(y_testing, y_pred_bagging)))
print("Final f1 score on test data: {:.4f}".format(f1_score(y_testing, y_pred_bagging)))

# In[50]:

##Boosting Classifier Performance
ada_clf =AdaBoostClassifier(DecisionTreeClassifier(random_state=42, max_depth = 5),
learning_rate = 1, n_estimators = 400, algorithm = "SAMME.R")
#ada_clf =AdaBoostClassifier(DecisionTreeClassifier(random_state=42, max_depth = 1),algorithm = "SAMME.R")
#
ada_clf.fit(X_training, y_training) 
y_pred_ada_bal = ada_clf.predict(X_training)
ada_clf.fit(X_training_FS, y_training) 
y_pred_ada_bal = ada_clf.predict(X_training_FS)
print("\nAda Boosting model on balanced training set\n------------------------")
print("Final accuracy score on the training data: {:.4f}".format(accuracy_score(y_training, y_pred_ada_bal)))
print("Final precision score on training data: {:.4f}".format(precision_score(y_training, y_pred_ada_bal)))
print("Final Recall score on training data: {:.4f}".format(recall_score(y_training, y_pred_ada_bal)))
print("Final ROC AUC score on training data: {:.4f}".format(roc_auc_score(y_training, y_pred_ada_bal)))
print("Final mean squared error score on training data: {:.4f}".format(mean_squared_error(y_training, y_pred_ada_bal)))
print("Final f1 score on training data: {:.4f}".format(f1_score(y_training, y_pred_ada_bal)))

y_pred_ada_bal = ada_clf.predict(X_validation)
y_pred_ada_bal = ada_clf.predict(X_validation_FS)
print("\nAda Boosting model on balanced validation set\n------------------------")
print("Final accuracy score on the validation data: {:.4f}".format(accuracy_score(y_validation, y_pred_ada_bal)))
print("Final precision score on validation data: {:.4f}".format(precision_score(y_validation, y_pred_ada_bal)))
print("Final Recall score on validation data: {:.4f}".format(recall_score(y_validation, y_pred_ada_bal)))
print("Final ROC AUC score on validation data: {:.4f}".format(roc_auc_score(y_validation, y_pred_ada_bal)))
print("Final mean squared error score on validation data: {:.4f}".format(mean_squared_error(y_validation, y_pred_ada_bal)))
print("Final f1 score on validation data: {:.4f}".format(f1_score(y_validation, y_pred_ada_bal)))

###
train_sizes, train_scores, validation_scores = learning_curve(
    estimator = ada_clf, X = X_train_val,
    y = y_train_val, train_sizes = numpy.linspace(0.01, 1.0, 25), cv = 5, shuffle=True, random_state = 42, 
    scoring = 'neg_mean_squared_error')
train_scores_mean = -train_scores.mean(axis = 1)
validation_scores_mean = -validation_scores.mean(axis = 1)
plt.plot(train_sizes, train_scores_mean, "r-", linewidth=2, label = 'Training Set')
plt.plot(train_sizes, validation_scores_mean, "b--", linewidth=2, label = 'Validation Set')
plt.ylabel('MSE', fontsize = 14)
plt.xlabel('Training set size', fontsize = 14)
plt.title('Ada Boosting Classifier Learning Curve', fontsize = 18, y = 1.03)
plt.legend()

# In[38]:

# Ada boosting Optimization on Balanced Training set

# Create the parameters list

n_estimators = [50,100,150,200,250,300,350,400]
learning_rate = [0.1,0.3,0.5,0.7,1]
max_depth =[5,10,30,50]


grid_obj = GridSearchCV(AdaBoostClassifier(DecisionTreeClassifier(random_state=42, max_depth = 1),algorithm = "SAMME.R"),
                      dict(n_estimators=n_estimators,learning_rate = learning_rate), 
                           scoring= 'roc_auc', cv=3)

### Fit the grid search object to the training data and find the optimal parameters using fit()
grid_fit = grid_obj.fit(X_training, y_training)
grid_fit = grid_obj.fit(X_training_FS, y_training)

### Get the estimator
model_ADA_opt = grid_fit.best_estimator_

print("The best parameters are: {}".format(model_ADA_opt))

results = grid_obj.cv_results_
for mean_score, params in zip(results["mean_test_score"], results["params"]):
    print((mean_score), params)
    
ADA_data = [go.Contour(z = results["mean_test_score"],
        x = results['param_n_estimators'], y = results['param_learning_rate'],
        colorscale ='Jet',contours = dict(showlabels = True,labelfont = dict(
        family = 'Raleway',size = 15,color = 'black',)), colorbar = dict( title = 'Score',
        titleside = 'right', titlefont = dict( size =16, family = 'Arial, sans-serif',)))]
py.offline.plot(ADA_data)

##Report the metrics scores on train data
model_ADA_opt.fit(X_training, y_training)
y_pred_ada = model_ADA_opt.predict(X_training)    
print("\n")
print("\nOptimal ADA Model performance on training set\n------------------------")
print("Final accuracy score on the training data: {:.4f}".format(accuracy_score(y_training, y_pred_ada)))  
print("Final precision score on training data: {:.4f}".format(precision_score(y_training, y_pred_ada)))
print("Final Recall score on training data: {:.4f}".format(recall_score(y_training, y_pred_ada)))
print("Final ROC AUC score on training data: {:.4f}".format(roc_auc_score(y_training, y_pred_ada)))
print("Final mean squared error score on training data: {:.4f}".format(mean_squared_error(y_training, y_pred_ada)))
print("Final f1 score on training data: {:.4f}".format(f1_score(y_training, y_pred_ada)))
print("\n")

#
##Evaluate optimal model against Validation Set 
y_pred_ada = model_ADA_opt.predict(X_validation)
y_pred_ada = model_ADA_opt.predict(X_validation_FS)
print("\n")
print("\nOptimal ADA model on balanced validation set\n------------------------")
print("Final accuracy score on the validation data: {:.4f}".format(accuracy_score(y_validation, y_pred_ada)))
print("Final precision score on validation data: {:.4f}".format(precision_score(y_validation, y_pred_ada)))
print("Final Recall score on validation data: {:.4f}".format(recall_score(y_validation, y_pred_ada)))
print("Final ROC AUC score on validation data: {:.4f}".format(roc_auc_score(y_validation, y_pred_ada)))
print("Final mean squared error score on validation data: {:.4f}".format(mean_squared_error(y_validation, y_pred_ada)))
print("Final f1 score on validation data: {:.4f}".format(f1_score(y_validation, y_pred_ada)))

# In[38]:

y_pred_ada = model_ADA_opt.predict(X_testing)
y_pred_ada = model_ADA_opt.predict(X_testing_FS)
y_pred_ada = ada_clf.predict(X_testing_FS)
print("\nAda Boosting model on test set with balanced training set\n------------------------")
print("Final accuracy score on the validation data: {:.4f}".format(accuracy_score(y_testing, y_pred_ada)))
print("Final precision score on validation data: {:.4f}".format(precision_score(y_testing, y_pred_ada)))
print("Final Recall score on validation data: {:.4f}".format(recall_score(y_testing, y_pred_ada)))
print("Final ROC AUC score on validation data: {:.4f}".format(roc_auc_score(y_testing, y_pred_ada)))
print("Final mean squared error score on validation data: {:.4f}".format(mean_squared_error(y_testing, y_pred_ada)))
print("Final f1 score on validation data: {:.4f}".format(f1_score(y_testing, y_pred_ada)))

train_sizes, train_scores, validation_scores = learning_curve(
    estimator = model_ADA_opt, X = X_train_val,
    y = y_train_val, train_sizes = numpy.linspace(0.01, 1.0, 25), cv = 5, shuffle=True, random_state = 42, 
    scoring = 'neg_mean_squared_error')
train_scores_mean = -train_scores.mean(axis = 1)
validation_scores_mean = -validation_scores.mean(axis = 1)
plt.plot(train_sizes, train_scores_mean, "r-", linewidth=2, label = 'Training Set')
plt.plot(train_sizes, validation_scores_mean, "b--", linewidth=2, label = 'Validation Set')
plt.ylabel('MSE', fontsize = 14)
plt.xlabel('Training set size', fontsize = 14)
plt.title(' Optimized Ada Boosting Classifier Learning Curve', fontsize = 18, y = 1.03)
plt.legend()

confusion_matrix_ADA = confusion_matrix(y_testing, y_pred_ada)
plt.figure(figsize=(6,6))
plot_confusion_matrix(confusion_matrix_ADA, classes=['not subscribe','subscribe'],
                      title='AdaBoost Confusion matrix')

# In[48]:
##Voting CLassifier Performance 
#
voting_clf = VotingClassifier(estimators=[('lr', model_LR_bal),
('gnb', model_GNB_bal),('dt', model_DT_bal),('svc', model_SVM_bal), ('dtb', ada_clf)], voting= 'hard')
#('gnb', model_GNB_bal),('dt', model_DT_bal),('dtb', ada_clf)], voting= 'soft')

##voting_clf.fit(X_training_FS, y_training)
##y_pred_vote_bal = voting_clf.predict(X_training_FS)
voting_clf.fit(X_training, y_training)
y_pred_vote_bal = voting_clf.predict(X_training)
print("\nVoting model on balanced training set\n------------------------")
print("Final accuracy score on the training data: {:.4f}".format(accuracy_score(y_training, y_pred_vote_bal)))
print("Final precision score on training data: {:.4f}".format(precision_score(y_training, y_pred_vote_bal)))
print("Final Recall score on training data: {:.4f}".format(recall_score(y_training, y_pred_vote_bal)))
print("Final ROC AUC score on training data: {:.4f}".format(roc_auc_score(y_training, y_pred_vote_bal)))
print("Final mean squared error score on training data: {:.4f}".format(mean_squared_error(y_training, y_pred_vote_bal)))
print("Final f1 score on training data: {:.4f}".format(f1_score(y_training, y_pred_vote_bal)))

#y_pred_vote_bal = voting_clf.predict(X_validation_FS)
y_pred_vote_bal = voting_clf.predict(X_validation)
print("\nVoting model on balanced validation set\n------------------------")
print("Final accuracy score on the validation data: {:.4f}".format(accuracy_score(y_validation, y_pred_vote_bal)))
print("Final precision score on validation data: {:.4f}".format(precision_score(y_validation, y_pred_vote_bal)))
print("Final Recall score on validation data: {:.4f}".format(recall_score(y_validation, y_pred_vote_bal)))
print("Final ROC AUC score on validation data: {:.4f}".format(roc_auc_score(y_validation, y_pred_vote_bal)))
print("Final mean squared error score on validation data: {:.4f}".format(mean_squared_error(y_validation, y_pred_vote_bal)))
print("Final f1 score on validation data: {:.4f}".format(f1_score(y_validation, y_pred_vote_bal)))

train_sizes, train_scores, validation_scores = learning_curve(
    estimator = voting_clf, X = X_train_val,
    y = y_train_val, train_sizes = numpy.linspace(0.01, 1.0, 25), cv = 5, shuffle=True, random_state = 42, 
    scoring = 'neg_mean_squared_error')
train_scores_mean = -train_scores.mean(axis = 1)
validation_scores_mean = -validation_scores.mean(axis = 1)
plt.plot(train_sizes, train_scores_mean, "r-", linewidth=2, label = 'Training Set')
plt.plot(train_sizes, validation_scores_mean, "b--", linewidth=2, label = 'Validation Set')
plt.ylabel('MSE', fontsize = 14)
plt.xlabel('Training set size', fontsize = 14)
plt.title('Voting Classifier Learning Curve', fontsize = 18, y = 1.03)
plt.legend()
#plt.savefig('Opt_LinearSVC_Learning_Curve.jpeg')  


#y_pred_voting = voting_clf.predict(X_testing_FS )
y_pred_voting = voting_clf.predict(X_testing)
print("\nVoting model on test set with balanced training\n------------------------")
print("Final accuracy score on the test data: {:.4f}".format(accuracy_score(y_testing, y_pred_voting)))
print("Final precision score on test data: {:.4f}".format(precision_score(y_testing, y_pred_voting)))
print("Final Recall score on test data: {:.4f}".format(recall_score(y_testing, y_pred_voting)))
print("Final ROC AUC score on test data: {:.4f}".format(roc_auc_score(y_testing, y_pred_voting)))
print("Final mean squared error score on test data: {:.4f}".format(mean_squared_error(y_testing, y_pred_voting)))
print("Final f1 score on testdata: {:.4f}".format(f1_score(y_testing, y_pred_voting)))

confusion_matrix_Voting = confusion_matrix(y_testing, y_pred_voting)
plt.figure(figsize=(6,6))
plot_confusion_matrix(confusion_matrix_Voting, classes=['not subscribe','subscribe'],
                      title='Voting Confusion matrix')

## In[51]:
'''
from sklearn.linear_model import Perceptron
per_clf = Perceptron(random_state = 0, max_iter = 100)
per_clf.fit(X_training, y_training)
y_pred = per_clf.predict(X_testing)

print("\nPerceptron on test set with balanced training\n------------------------")
print("Final accuracy score on the test data: {:.4f}".format(accuracy_score(y_testing, y_pred)))
print("Final precision score on test data: {:.4f}".format(precision_score(y_testing, y_pred)))
print("Final Recall score on test data: {:.4f}".format(recall_score(y_testing, y_pred)))
print("Final ROC AUC score on test data: {:.4f}".format(roc_auc_score(y_testing, y_pred)))
print("Final mean squared error score on test data: {:.4f}".format(mean_squared_error(y_testing, y_pred)))
print("Final f1 score on testdata: {:.4f}".format(f1_score(y_testing, y_pred)))

confusion_matrix_Perceptron = confusion_matrix(y_testing, y_pred)
plt.figure(figsize=(6,6))
plot_confusion_matrix(confusion_matrix_Perceptron, classes=['not subscribe','subscribe'],
                      title='Perceptron Confusion matrix')
'''

'''
model_DT = joblib.load(r'C:\Users\isogb\Documents\FYP Software\Saved Models\Decision_Tree_model.pkl')
model_GNB = joblib.load(r'C:\Users\isogb\Documents\FYP Software\Saved Models\Gaussian_NB_model.pkl')
model_LR = joblib.load(r'C:\Users\isogb\Documents\FYP Software\Saved Models\Logistic_Regression_model.pkl')
model_SVM = joblib.load(r'C:\Users\isogb\Documents\FYP Software\Saved Models\SVM_model.pkl')
model_NN_bal.load_weights(r'C:\Users\isogb\Documents\FYP Software\Saved Models\weights.best.from_scratch.hdf5')


roc_curve_plot(model_DT, X_testing, y_testing, label='Decision Tree')
roc_curve_plot(model_GNB, X_testing, y_testing, label='Gaussian Naive Bayes')
roc_curve_plot(model_LR, X_testing, y_testing, label='Logistic Regression')
roc_curve_plot(model_SVM, X_testing, y_testing, label='Linear SVC')
roc_curve_plot(model_NN_bal, X_testing.values, y_testing.values, label='Neural Network')
'''
