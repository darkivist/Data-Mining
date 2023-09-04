# %%%%%%%%%%%%% Machine Learning %%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%% Authors  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Dr. Amir Jafari------>Email: amir.h.jafari@okstate.edu
# Deepak Agarwal------>Email:deepakagarwal@gwmail.gwu.edu
# %%%%%%%%%%%%% Date:
# V1 June - 13 - 2018
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%% K-Nearest Neighbor  %%%%%%%%%%%%%%%%%%%%%
#%%-----------------------------------------------------------------------
# Exercise
#%%-----------------------------------------------------------------------
# Specify what are your features and targets. Why this is a classification
# 1- Use the chronic_kidney disease dataset.
# 2- Specify what are your features and targets.
# 3- Why this is a classification problem.
# 4- Run the K-Nearest Neighbor algorithm.
# 5- Explain your findings and write down a paragraph to explain all the results.
# 6- Explain the differences between Logistic Regression  and K-Nearest Neighbor.
#%%-----------------------------------------------------------------------

# 1-
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

#read in data
col_names = ['age', 'bp', 'sg', 'al', 'su','rbc','pc','pcc','ba','bgr','bu','sc','sod','pot','hemo','pcv','wbcc','rbcc','htn','dm','cad','appet','pe','ane','class']
kidney = pd.read_csv("chronic_kidney.csv", header=None, names=col_names)

#inspect data
kidney.head()

#check unique values of columns
for col in kidney:
    print(kidney[col].unique())

#clean data by replacing '?' and other bad values, removing NA
kidney_clean = kidney.replace(['?','\t8400', '\t6200'], float('nan'))
kidney_clean = kidney_clean.dropna()

#clean data by converting boolean category values to 0,1
kidney_clean = kidney_clean.replace(['normal','present','yes','good'], int(0))
kidney_clean = kidney_clean.replace(['abnormal','notpresent','no','poor'], int(1))

#check unique values of columns
for col in kidney_clean:
    print(kidney_clean[col].unique())

#printing the dataset shape
print("Dataset No. of Rows (Original): ", kidney.shape[0])
print("Dataset No. of Columns (Original): ", kidney.shape[1])

#printing the structure of the dataset
print("Dataset info:\n ")
print(kidney_clean.info())
print("Dataset No. of Rows (Clean): ", kidney_clean.shape[0])
print("Dataset No. of Columns (Clean): ", kidney_clean.shape[1])

#correct datatypes
kidney_clean['age'] = kidney_clean.age.astype(int)
kidney_clean['bp'] = kidney_clean.bp.astype(int)
kidney_clean['sg'] = kidney_clean.sg.astype(float)
kidney_clean['al'] = kidney_clean.al.astype(int)
kidney_clean['su'] = kidney_clean.su.astype(int)
kidney_clean['rbc'] = kidney_clean.rbc.astype(object)
kidney_clean['pc'] = kidney_clean.pc.astype(object)
kidney_clean['pcc'] = kidney_clean.pcc.astype(object)
kidney_clean['ba'] = kidney_clean.ba.astype(object)
kidney_clean['bgr'] = kidney_clean.bgr.astype(float)
kidney_clean['bu'] = kidney_clean.bu.astype(float)
kidney_clean['sc'] = kidney_clean.sc.astype(float)
kidney_clean['sod'] = kidney_clean.sod.astype(float)
kidney_clean['pot'] = kidney_clean.pot.astype(float)
kidney_clean['hemo'] = kidney_clean.hemo.astype(float)
kidney_clean['pcv'] = kidney_clean.pcv.astype(float)
kidney_clean['wbcc'] = kidney_clean.wbcc.astype(float)
kidney_clean['rbcc'] = kidney_clean.rbcc.astype(float)
kidney_clean['htn'] = kidney_clean.htn.astype(object)
kidney_clean['dm'] = kidney_clean.dm.astype(object)
kidney_clean['cad'] = kidney_clean.cad.astype(object)
kidney_clean['appet'] = kidney_clean.appet.astype(object)
kidney_clean['pe'] = kidney_clean.pe.astype(object)
kidney_clean['ane'] = kidney_clean.ane.astype(object)


#%%-----------------------------------------------------------------------
# 2-

#The target of this model is the variable CLASS, which identifies if an observation is
#from a chronic kidney disease patient (class = 'ckd') or not (class = 'notckd').
#The remaining variables are features. They are, in order:

#'age': Age
#'bp': Blood pressure
#'sg': Specific gravity
#'al': Albumin
#'su': Sugar
#'rbc': Red blood cells
#'pc': Pus cell
#'pcc': Pus cell clumps
#'ba': Bacteria
#'bgr': Blood glucose random
#'bu': Blood urea
#'sc': Serum creatinine
#'sod': Sodium
#'pot': Potassium
#'hemo': Haemoglobin
#'pcv': Packed cell volume
#'wbcc': White blood cell count
#'rbcc': Red blood cell count
#'htn': Hypertension
#'dm': Diabetes mellitus
#'cad': Coronary artery disease
#'appet': Appetite
#'pe': Pedal edema
#'ane': Amnemia

#%%-----------------------------------------------------------------------
# 3-

#Classifcation models aim to categorize observations into one or more outcome categories based on variable values.
#In the CKD dataset, we aim to predict whether a new observation will be a patient with chronic kidney disease or not,
#based on the levels of their feature variables.

#%%-----------------------------------------------------------------------
# 4-

#printing the summary statistics of the dataset
print(kidney_clean.describe(include='all'))

#split the dataset
#separate the target variable

X = kidney_clean.values[:, :-1]
Y = kidney_clean.values[:, -1]

#data preprocessing
#encode the target variable
class_le = LabelEncoder()

y = class_le.fit_transform(Y)
y

#split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100, stratify=y)

#data preprocessing
#standardize the data
stdsc = StandardScaler()

stdsc.fit(X_train)

X_train_std = stdsc.transform(X_train)
X_test_std = stdsc.transform(X_test)

#perform training
#creating the classifier object

import math
#cite https://saravananthirumuruganathan.wordpress.com/2010/05/17/a-detailed-introduction-to-k-nearest-neighbor-knn-algorithm/
k = math.sqrt(155)
k


clf = KNeighborsClassifier(n_neighbors=13)

#Cite:

#performing training
clf.fit(X_train_std, y_train)

#make predictions

#predicton on test
y_pred = clf.predict(X_test_std)


#calculate metrics

print("\n")
print("Classification Report: ")
print(classification_report(y_test,y_pred))
print("\n")


print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)
print("\n")

#confusion matrix

conf_matrix = confusion_matrix(y_test, y_pred)
class_names = kidney_clean['class'].unique()


df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )

plt.figure(figsize=(5,5))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=45, ha='right', fontsize=10)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=10)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
plt.tight_layout()
plt.show()

#%%-----------------------------------------------------------------------
# 5-

#I used 13 for number of neighbors, based on the suggestion of the paper cited. The number of observations in the set after cleaning is 155; sqrt(155) for k to control for overfitting (k too small) and underfitting (k too large). Additionally, I selected 13 instead of 12 for number of neighbors based on the suggestion of using an odd number if the number of classes is 2. With this, we have an accuracy score of 93.75.

#%%-----------------------------------------------------------------------
# 6-

#KNN is a distance-based algorithm. Linear regression is a parametric model that is used to predict linear outcomes.
#KNN outputs categorical labels for a prediction, while LR outputs numerical predictions for linear-based values.
