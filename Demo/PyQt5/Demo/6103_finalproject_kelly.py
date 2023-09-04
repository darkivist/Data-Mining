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

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# read in data


# 1- Use the bank adult dataset.
col_names = ['class',
'cap_shape',
'cap_surface',
'cap_color',
'bruises',
'odor',
'gill_attachment',
'gill_spacing',
'gill_size',
'gill_color',
'stalk_shape',
'stalk_root', #?
'stalk_surface_above_ring',
'stalk_surface_below_ring',
'stalk_color_above_ring',
'stalk_color_below_ring',
'veil_type',
'veil_color',
'ring_number',
'ring_type',
'spore_print_color',
'population',
'habitat']
mushrooms = pd.read_csv("mushrooms.csv", header=None, names=col_names)

mushrooms.head()

#Check unique values of columns
for col in mushrooms:
    print(mushrooms[col].unique())

# printing the dataset shape
print("Dataset No. of Rows (Original): ", mushrooms.shape[0])
print("Dataset No. of Columns: ", mushrooms.shape[1])

#print("Dataset No. of Rows: (Clean)", mushrooms_clean.shape[0])
#print("Dataset No. of Columns: ", mushrooms_clean.shape[1])

# Correct '?' values, drop NA
mushrooms_clean = mushrooms.replace(['?'], float('nan'))
mushrooms_clean = mushrooms_clean.dropna()

5644/8124

4208/(4208+3916)

mushrooms_clean['class'].value_counts()

3488/(3488+2156)

mushrooms['stalk_root'].value_counts()

mushrooms_clean['stalk_root'].value_counts()

# How much data did we lose?
# We are left with 69% of the total records.
# Note that removing observations with missing data can result in a biased model;
5644/8124

# printing the structure of the dataset
print("Dataset info:\n ")
print(mushrooms_clean.info())



mushrooms_clean['class'] = mushrooms_clean['class'].astype('object')
mushrooms_clean['cap_shape'] = mushrooms_clean['cap_shape'].astype('category')
mushrooms_clean['cap_surface'] = mushrooms_clean['cap_surface'].astype('category')
mushrooms_clean['cap_color'] = mushrooms_clean['cap_color'].astype('category')
mushrooms_clean['bruises'] = mushrooms_clean['bruises'].astype('category')
mushrooms_clean['odor'] = mushrooms_clean['odor'].astype('category')
mushrooms_clean['gill_attachment'] = mushrooms_clean['gill_attachment'].astype('category')
mushrooms_clean['gill_spacing'] = mushrooms_clean['gill_spacing'].astype('category')
mushrooms_clean['gill_size'] = mushrooms_clean['gill_size'].astype('category')
mushrooms_clean['gill_color'] = mushrooms_clean['gill_color'].astype('category')
mushrooms_clean['stalk_shape'] = mushrooms_clean['stalk_shape'].astype('category')
mushrooms_clean['stalk_root'] = mushrooms_clean['stalk_root'].astype('category')
mushrooms_clean['stalk_surface_above_ring'] =mushrooms_clean['stalk_surface_above_ring'] .astype('category')
mushrooms_clean['stalk_surface_below_ring'] =mushrooms_clean['stalk_surface_below_ring'] .astype('category')
mushrooms_clean['stalk_color_above_ring'] =mushrooms_clean['stalk_color_above_ring'] .astype('category')
mushrooms_clean['stalk_color_below_ring'] =mushrooms_clean['stalk_color_below_ring'] .astype('category')
mushrooms_clean['veil_type'] =mushrooms_clean['veil_type'] .astype('category')
mushrooms_clean['veil_color'] =mushrooms_clean['veil_color'] .astype('category')
mushrooms_clean['ring_number'] =mushrooms_clean['ring_number'] .astype('category')
mushrooms_clean['ring_type'] =mushrooms_clean['ring_type'] .astype('category')
mushrooms_clean['spore_print_color'] =mushrooms_clean['spore_print_color'].astype('category')
mushrooms_clean['population'] =mushrooms_clean['population'].astype('category')
mushrooms_clean['habitat'] =mushrooms_clean['habitat'].astype('category')

# printing the summary statistics of the dataset
print(mushrooms_clean.describe(include='all'))


# Change all category columns to numeric values
category_cols = mushrooms_clean.select_dtypes(['category']).columns

mushrooms_clean[category_cols] = mushrooms_clean[category_cols].apply(lambda x: x.cat.codes)

category_cols

mushrooms_clean.to_csv('mushrooms_clean')

## separate the target variable
X = mushrooms_clean.values[:, 1:]
Y = mushrooms_clean.values[:, 0]

# data preprocessing
# encode the target variable
class_le = LabelEncoder()

y = class_le.fit_transform(Y)

# split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100, stratify=y)

# Run each of the supervised learning algorithms against.
#Make a case for which one(s) to use.

# Run the Naive Bayes algorithm.
# perform training

# creating the classifier object
clf = GaussianNB()

# performing training
clf.fit(X_train, y_train)

# printing the dataset shape
print("Dataset No. of Rows: ", X.shape[0])
print("Dataset No. of Columns: ", X.shape[1])

# make predictions

# prediction on test
y_pred = clf.predict(X_test)

y_pred_score = clf.predict_proba(X_test)

# calculate metrics

print("\n")

print("Classification Report: ")
print(classification_report(y_test,y_pred))
print("\n")


print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)
print("\n")

print("ROC_AUC : ", roc_auc_score(y_test,y_pred_score[:,1]) * 100)
print("\n")

# confusion matrix

conf_matrix = confusion_matrix(y_test, y_pred)
class_names = mushrooms_clean['class'].unique()


df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )

plt.figure(figsize=(5,5))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
# Show heat map
plt.tight_layout()
plt.show()



