# %%%%%%%%%%%%% Machine Learning %%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%% Authors  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Dr. Amir Jafari------>Email: amir.h.jafari@okstate.edu
# Deepak Agarwal------>Email:deepakagarwal@gwmail.gwu.edu
# %%%%%%%%%%%%% Date:
# V1 June - 05 - 2018
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%% Naive Bayes  %%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%-----------------------------------------------------------------------
# Exercise
#%%-----------------------------------------------------------------------
# Specify what are your features and targets. Why this is a classification
# 1- Use the bank adult dataset.
# 2- Specify what are your features and targets.
# 3- Why this is a classification problem.
# 4- Run the Naive Bayes algorithm.
# 5- Explain your findings and write down a paragraph to explain all the results.
# 6- Explain the differences between Naive Bayes and Decision tree.
#%%-----------------------------------------------------------------------
# 1-

import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder

col_names = ['age',
'workclass',
'fnlwgt',
'education',
'educationnum',
'maritalstatus',
'occupation',
'relationship',
'race',
'sex',
'capitalgain',
'capitalloss',
'hoursperweek',
'nativecountry',
'income']
adult = pd.read_csv("adult.data.csv", header=None, names=col_names)

adult.head()
adult.tail()

#check unique values of columns
for col in adult:
    print(adult[col].unique())

#eliminate white space
adult['workclass'] = adult['workclass'].str.strip()
adult['education'] = adult['education'].str.strip()
adult['maritalstatus'] = adult['maritalstatus'].str.strip()
adult['occupation'] = adult['occupation'].str.strip()
adult['relationship'] = adult['relationship'].str.strip()
adult['race'] = adult['race'].str.strip()
adult['sex'] = adult['sex'].str.strip()
adult['nativecountry'] = adult['nativecountry'].str.strip()
adult['income'] = adult['income'].str.strip()

#correct '?' values, 'South' country value, drop NA
#'South' is not a country; dropping this value as it is only about 80 records.

adult_clean = adult.replace(['?', 'South'], float('nan'))
adult_clean = adult_clean.dropna()

#recode 'Hong' country value to 'HongKong'
adult_clean = adult_clean.replace(['Hong'], str('HongKong'))

#check unique values of columns
for col in adult_clean:
    print(adult_clean[col].unique())

#printing the dataset shape
print("Dataset No. of Rows (Original): ", adult.shape[0])
print("Dataset No. of Columns: ", adult.shape[1])

print("Dataset No. of Rows: (Clean)", adult_clean.shape[0])
print("Dataset No. of Columns: ", adult_clean.shape[1])

#How much data did we lose?
#We are left with 92% of the total records.
#Note that removing observations with missing data can result in a biased model;
#however, initial checks of the data have 24% of original data is observations with >50K;
#25% of 'cleaned' data is observations with >50K, suggesting the dataset has not been biased against target variable.
30091/32561

#printing the structure of the dataset
print("Dataset info:\n ")
print(adult_clean.info())

#correct datatypes.
adult_clean['workclass'] = adult_clean.workclass.astype('category')
adult_clean['education'] = adult_clean.education.astype('category')
adult_clean['maritalstatus'] = adult_clean.maritalstatus.astype('category')
adult_clean['occupation'] = adult_clean.occupation.astype('category')
adult_clean['relationship'] = adult_clean.relationship.astype('category')
adult_clean['race'] = adult_clean.race.astype('category')
adult_clean['sex'] = adult_clean.sex.astype('category')
adult_clean['nativecountry'] = adult_clean.nativecountry.astype('category')
adult_clean['income'] = adult_clean.income.astype('object')

#printing the summary statistics of the dataset
print(adult_clean.describe(include='all'))

#change all category columns to numeric values
#cite https://stackoverflow.com/questions/32011359/convert-categorical-data-in-pandas-dataframe
category_cols = adult_clean.select_dtypes(['category']).columns
category_cols

adult_clean[category_cols] = adult_clean[category_cols].apply(lambda x: x.cat.codes)

#%%-----------------------------------------------------------------------
# 2-

#As specified in the ReadMe file, the features and target variables are as follows:
#Target:
#- Income (>50K, <=50K) - Categorical

#Features:
#- age - Age (years) - Numeric
#- workclass - Working class - Categorical
#- fnlwgt - final weight - Numeric
#- education - education - Categorical
#- education-num - education number - Numeric
#- marital-status - marital status - Categorical
#- occupation - occupation - Categorical
# relationship - relationship - Categorical
#- race - race - Categorical
#- sex - sex - Categorical
#- capital-gain - capital gain - Numeric
#- capital-loss - capital loss - Numeric
#- hours-per-week - hours work per week - Numeric
#- native-country - native country - Categorical

#%%-----------------------------------------------------------------------
# 3-

#This is a classification problem because classification models aim to categorize observations into one or more outcome categories based on variable values.
#In the adult dataset, we aim to predict whether a new observation will be an individual with an income of >50K based on the levels of their feature variables.

#%%-----------------------------------------------------------------------

# 4- Run the Naive Bayes algorithm.

#separate the target variable
X = adult_clean.values[:, :-1]
Y = adult_clean.values[:, -1]

#split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

#perform training

#creating the classifier object
clf = GaussianNB()

#performing training
clf.fit(X_train, y_train)

#make predictions

#predicton on test
y_pred = clf.predict(X_test)

y_pred_score = clf.predict_proba(X_test)

#calculate metrics

print("\n")

print("Classification Report: ")
print(classification_report(y_test,y_pred))
print("\n")

print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)
print("\n")

print("ROC_AUC : ", roc_auc_score(y_test,y_pred_score[:,1]) * 100)
print("\n")

#confusion matrix

conf_matrix = confusion_matrix(y_test, y_pred)
class_names = adult_clean['income'].unique()

df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )

plt.figure(figsize=(5,5))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)

#show heat map
plt.tight_layout()
plt.show()

#run some calculations
Total = 6422+337+1611+658
TruePos = 658/Total
FalsePos = 337/Total
FalseNeg = 1611/Total
TrueNeg = 6422/Total

TrueFP = (6422+658)/Total

print(Total)
print(TrueFP)

#%%-----------------------------------------------------------------------
# 5-

#After removal of unknowns and correcting variable values in the training and test sets, the resulting Naive Bayes model has an Accuracy of 78.42%. The ROC_AUC is 82.7. These are relatively good scores
#that could be improved with further tuning of features selected in the model. The confusion matrix shows that, of 9028 total observations in the testing set,
#658 observations were correctly categorized as having income >50K, 6422 were correctly categorized as having an income <=50K; the combined total of these gives us the Accuracy score of 78%.

#%%-----------------------------------------------------------------------
# 6-

# Cite https://www.datasciencecentral.com/comparing-classifiers-decision-trees-knn-naive-bayes/

#Niave Bayes and Decision tree are both supervised learning models. Niave Bayes is a linear classification model. It can be tuned by two hyperparameters, alpha and beta.
#It is based on using the stastistical theorem, Bayes Theorem, assuming all variables are independent. Decision Tree is a flowchart-like structural classification model represented by decision nodes with rule sets
#dictating where an observation is categorized.