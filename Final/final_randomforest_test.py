import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# read in data

#define column names to get rid of hyphens
col_names = ['id',
             'age',
             'workclass',
             'people_responded',
             'education',
             'educational_level',
             'marital_status',
             'occupation',
             'relationship',
             'race',
             'gender',
             'money_gain',
             'money_loss',
             'hours_per_week',
             'country',
             'comfortability']

comfort = pd.read_csv("Test_submission_netid.csv", header=None, names=col_names)

#print dataframe head

comfort.head()

#check unique values of columns

for col in comfort:
    print(comfort[col].unique())

# print dataframe shape

print("Dataset No. of Rows: ", comfort.shape[0])
print("Dataset No. of Columns: ", comfort.shape[1])

comfort_clean = comfort.replace([''], float('nan'))
comfort_clean = comfort_clean.dropna()

print("Dataset No. of Rows: ", comfort_clean.shape[0])
print("Dataset No. of Columns: ", comfort_clean.shape[1])

#print dataframe structure

print("Dataset info:\n ")
print(comfort_clean.info())


#changing datatypes in preparation for modeling

comfort_clean['workclass'] = comfort_clean.workclass.astype('category')
comfort_clean['education'] = comfort_clean.education.astype('category')
comfort_clean['educational_level'] = comfort_clean.educational_level.astype('category')
comfort_clean['marital_status'] = comfort_clean.marital_status.astype('category')
comfort_clean['occupation'] = comfort_clean.occupation.astype('category')
comfort_clean['relationship'] = comfort_clean.relationship.astype('category')
comfort_clean['race'] = comfort_clean.race.astype('category')
comfort_clean['gender'] = comfort_clean.gender.astype('category')
comfort_clean['country'] = comfort_clean.country.astype('category')
comfort_clean['comfortability'] = comfort_clean.comfortability.astype('str')

comfort_clean = comfort_clean.replace(['0'], str('0'))
comfort_clean = comfort_clean.replace(['1'], str('1'))

comfort_clean['comfortability'] = comfort_clean.comfortability.astype('category')

# change category columns to numeric values

category_cols = comfort_clean.select_dtypes(['category']).columns

comfort_clean[category_cols] = comfort_clean[category_cols].apply(lambda x: x.cat.codes)
print(comfort_clean.info())

# separate the target variable of comfortability
X = comfort_clean.values[:, :-1]
Y = comfort_clean.values[:, -1]

# data preprocessing
# encode the target variable

class_le = LabelEncoder()

y = class_le.fit_transform(Y)

# split the dataset into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100, stratify=y)

#create the classifier object using random forest

clf = RandomForestClassifier()
clf_2 = RandomForestClassifier()


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
class_names = comfort_clean['comfortability'].unique()

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

pd.DataFrame(y_pred).to_csv('y_pred.csv')

















