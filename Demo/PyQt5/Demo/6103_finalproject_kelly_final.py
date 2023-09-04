#Paul Kelly
#Final Project: All Mushrooms are edible, but some only once in a lifetime
#06/19/2022
#DATS 6103

#Cite: code is based upon class notes at https://github.com/amir-jafari/Data-Mining

import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.graphics.mosaicplot import mosaic
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


# Use the mushroom dataset from https://www.kaggle.com/datasets/uciml/mushroom-classification
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
mushroom = pd.read_csv("mushrooms.csv", header=None, names=col_names)

mushroom.head()

# Write in values for codes in dataset
mushroom.loc[mushroom["class"] == "p", "class"] = "poisonous"
mushroom.loc[mushroom["class"] == "e", "class"] = "edible"

mushroom.loc[mushroom["cap_shape"]== "b","cap_shape"] = "bell"
mushroom.loc[mushroom["cap_shape"]== "c","cap_shape"] = "conical"
mushroom.loc[mushroom["cap_shape"]== "x","cap_shape"] = "convex"
mushroom.loc[mushroom["cap_shape"]== "f","cap_shape"] = "flat"
mushroom.loc[mushroom["cap_shape"]== "k","cap_shape"] = "knobbed"
mushroom.loc[mushroom["cap_shape"]== "s","cap_shape"] = "sunken"

mushroom.loc[mushroom["cap_surface"]== "f","cap_surface"] = "fibrous"
mushroom.loc[mushroom["cap_surface"]== "g","cap_surface"] = "grooves"
mushroom.loc[mushroom["cap_surface"]== "y","cap_surface"] = "scaly"
mushroom.loc[mushroom["cap_surface"]== "s","cap_surface"] = "smooth"

mushroom.loc[mushroom["cap_color"]== "n","cap_color"] = "brown"
mushroom.loc[mushroom["cap_color"]== "b","cap_color"] = "buff"
mushroom.loc[mushroom["cap_color"]== "c","cap_color"] = "cinnamon"
mushroom.loc[mushroom["cap_color"]== "g","cap_color"] = "gray"
mushroom.loc[mushroom["cap_color"]== "r","cap_color"] = "green"
mushroom.loc[mushroom["cap_color"]== "p","cap_color"] = "pink"
mushroom.loc[mushroom["cap_color"]== "u","cap_color"] = "purple"
mushroom.loc[mushroom["cap_color"]== "e","cap_color"] = "red"
mushroom.loc[mushroom["cap_color"]== "w","cap_color"] = "white"
mushroom.loc[mushroom["cap_color"]== "y","cap_color"] = "yellow"

mushroom.loc[mushroom["bruises"]== "t","bruises"] = "yes"
mushroom.loc[mushroom["bruises"]== "f","bruises"] = "no"

mushroom.loc[mushroom["odor"]== "a","odor"] = "almond"
mushroom.loc[mushroom["odor"]== "l","odor"] = "anise"
mushroom.loc[mushroom["odor"]== "c","odor"] = "creosote"
mushroom.loc[mushroom["odor"]== "y","odor"] = "fishy"
mushroom.loc[mushroom["odor"]== "f","odor"] = "foul"
mushroom.loc[mushroom["odor"]== "m","odor"] = "musty"
mushroom.loc[mushroom["odor"]== "n","odor"] = "none"
mushroom.loc[mushroom["odor"]== "p","odor"] = "pungent"
mushroom.loc[mushroom["odor"]== "s","odor"] = "spicy"

mushroom.loc[mushroom["gill_attachment"]== "a","gill_attachment"] = "attached"
mushroom.loc[mushroom["gill_attachment"]== "d","gill_attachment"] = "descending"
mushroom.loc[mushroom["gill_attachment"]== "f","gill_attachment"] = "free"
mushroom.loc[mushroom["gill_attachment"]== "n","gill_attachment"] = "notched"

mushroom.loc[mushroom["gill_spacing"]== "c","gill_spacing"] = "close"
mushroom.loc[mushroom["gill_spacing"]== "w","gill_spacing"] = "crowded"
mushroom.loc[mushroom["gill_spacing"]== "d","gill_spacing"] = "distant"

mushroom.loc[mushroom["gill_size"]== "b","gill_size"] = "broad"
mushroom.loc[mushroom["gill_size"]== "n","gill_size"] = "narrow"

mushroom.loc[mushroom["gill_color"]== "k","gill_color"] = "black"
mushroom.loc[mushroom["gill_color"]== "n","gill_color"] = "brown"
mushroom.loc[mushroom["gill_color"]== "b","gill_color"] = "buff"
mushroom.loc[mushroom["gill_color"]== "h","gill_color"] = "chocolate"
mushroom.loc[mushroom["gill_color"]== "g","gill_color"] = "gray"
mushroom.loc[mushroom["gill_color"]== "r","gill_color"] = "green"
mushroom.loc[mushroom["gill_color"]== "o","gill_color"] = "orange"
mushroom.loc[mushroom["gill_color"]== "p","gill_color"] = "pink"
mushroom.loc[mushroom["gill_color"]== "u","gill_color"] = "purple"
mushroom.loc[mushroom["gill_color"]== "e","gill_color"] = "red"
mushroom.loc[mushroom["gill_color"]== "w","gill_color"] = "white"
mushroom.loc[mushroom["gill_color"]== "y","gill_color"] = "yellow"

mushroom.loc[mushroom["stalk_shape"]== "e","stalk_shape"] = "enlarging"
mushroom.loc[mushroom["stalk_shape"]== "t","stalk_shape"] = "tapering"

mushroom.loc[mushroom["stalk_color_above_ring"]== "c","stalk_color_above_ring"] = "cinnamon"
mushroom.loc[mushroom["stalk_color_above_ring"]== "n","stalk_color_above_ring"] = "brown"
mushroom.loc[mushroom["stalk_color_above_ring"]== "b","stalk_color_above_ring"] = "buff"
mushroom.loc[mushroom["stalk_color_above_ring"]== "h","stalk_color_above_ring"] = "chocolate"
mushroom.loc[mushroom["stalk_color_above_ring"]== "g","stalk_color_above_ring"] = "gray"
mushroom.loc[mushroom["stalk_color_above_ring"]== "r","stalk_color_above_ring"] = "green"
mushroom.loc[mushroom["stalk_color_above_ring"]== "o","stalk_color_above_ring"] = "orange"
mushroom.loc[mushroom["stalk_color_above_ring"]== "p","stalk_color_above_ring"] = "pink"
mushroom.loc[mushroom["stalk_color_above_ring"]== "u","stalk_color_above_ring"] = "purple"
mushroom.loc[mushroom["stalk_color_above_ring"]== "e","stalk_color_above_ring"] = "red"
mushroom.loc[mushroom["stalk_color_above_ring"]== "w","stalk_color_above_ring"] = "white"
mushroom.loc[mushroom["stalk_color_above_ring"]== "y","stalk_color_above_ring"] = "yellow"

mushroom.loc[mushroom["stalk_color_below_ring"]== "c","stalk_color_below_ring"] = "cinnamon"
mushroom.loc[mushroom["stalk_color_below_ring"]== "n","stalk_color_below_ring"] = "brown"
mushroom.loc[mushroom["stalk_color_below_ring"]== "b","stalk_color_below_ring"] = "buff"
mushroom.loc[mushroom["stalk_color_below_ring"]== "h","stalk_color_below_ring"] = "chocolate"
mushroom.loc[mushroom["stalk_color_below_ring"]== "g","stalk_color_below_ring"] = "gray"
mushroom.loc[mushroom["stalk_color_below_ring"]== "r","stalk_color_below_ring"] = "green"
mushroom.loc[mushroom["stalk_color_below_ring"]== "o","stalk_color_below_ring"] = "orange"
mushroom.loc[mushroom["stalk_color_below_ring"]== "p","stalk_color_below_ring"] = "pink"
mushroom.loc[mushroom["stalk_color_below_ring"]== "u","stalk_color_below_ring"] = "purple"
mushroom.loc[mushroom["stalk_color_below_ring"]== "e","stalk_color_below_ring"] = "red"
mushroom.loc[mushroom["stalk_color_below_ring"]== "w","stalk_color_below_ring"] = "white"
mushroom.loc[mushroom["stalk_color_below_ring"]== "y","stalk_color_below_ring"] = "yellow"

mushroom.loc[mushroom["spore_print_color"]== "k","spore_print_color"] = "black"
mushroom.loc[mushroom["spore_print_color"]== "c","spore_print_color"] = "cinnamon"
mushroom.loc[mushroom["spore_print_color"]== "n","spore_print_color"] = "brown"
mushroom.loc[mushroom["spore_print_color"]== "b","spore_print_color"] = "buff"
mushroom.loc[mushroom["spore_print_color"]== "h","spore_print_color"] = "chocolate"
mushroom.loc[mushroom["spore_print_color"]== "g","spore_print_color"] = "gray"
mushroom.loc[mushroom["spore_print_color"]== "r","spore_print_color"] = "green"
mushroom.loc[mushroom["spore_print_color"]== "o","spore_print_color"] = "orange"
mushroom.loc[mushroom["spore_print_color"]== "p","spore_print_color"] = "pink"
mushroom.loc[mushroom["spore_print_color"]== "u","spore_print_color"] = "purple"
mushroom.loc[mushroom["spore_print_color"]== "e","spore_print_color"] = "red"
mushroom.loc[mushroom["spore_print_color"]== "w","spore_print_color"] = "white"
mushroom.loc[mushroom["spore_print_color"]== "y","spore_print_color"] = "yellow"

mushroom.loc[mushroom["veil_color"]== "k","veil_color"] = "black"
mushroom.loc[mushroom["veil_color"]== "c","veil_color"] = "cinnamon"
mushroom.loc[mushroom["veil_color"]== "n","veil_color"] = "brown"
mushroom.loc[mushroom["veil_color"]== "b","veil_color"] = "buff"
mushroom.loc[mushroom["veil_color"]== "h","veil_color"] = "chocolate"
mushroom.loc[mushroom["veil_color"]== "g","veil_color"] = "gray"
mushroom.loc[mushroom["veil_color"]== "r","veil_color"] = "green"
mushroom.loc[mushroom["veil_color"]== "o","veil_color"] = "orange"
mushroom.loc[mushroom["veil_color"]== "p","veil_color"] = "pink"
mushroom.loc[mushroom["veil_color"]== "u","veil_color"] = "purple"
mushroom.loc[mushroom["veil_color"]== "e","veil_color"] = "red"
mushroom.loc[mushroom["veil_color"]== "w","veil_color"] = "white"
mushroom.loc[mushroom["veil_color"]== "y","veil_color"] = "yellow"

mushroom.loc[mushroom["stalk_surface_above_ring"]== "s","stalk_surface_above_ring"] = "smooth"
mushroom.loc[mushroom["stalk_surface_above_ring"]== "f","stalk_surface_above_ring"] = "fibrous"
mushroom.loc[mushroom["stalk_surface_above_ring"]== "k","stalk_surface_above_ring"] = "silky"
mushroom.loc[mushroom["stalk_surface_above_ring"]== "y","stalk_surface_above_ring"] = "scaly"

mushroom.loc[mushroom["stalk_surface_below_ring"]== "s","stalk_surface_below_ring"] = "smooth"
mushroom.loc[mushroom["stalk_surface_below_ring"]== "f","stalk_surface_below_ring"] = "fibrous"
mushroom.loc[mushroom["stalk_surface_below_ring"]== "k","stalk_surface_below_ring"] = "silky"
mushroom.loc[mushroom["stalk_surface_below_ring"]== "y","stalk_surface_below_ring"] = "scaly"

mushroom.loc[mushroom["stalk_root"]== "b","stalk_root"] = "bulbous"
mushroom.loc[mushroom["stalk_root"]== "c","stalk_root"] = "club"
mushroom.loc[mushroom["stalk_root"]== "u","stalk_root"] = "cup"
mushroom.loc[mushroom["stalk_root"]== "e","stalk_root"] = "equal"
mushroom.loc[mushroom["stalk_root"]== "z","stalk_root"] = "rhizomorphs"
mushroom.loc[mushroom["stalk_root"]== "r","stalk_root"] = "rooted"

mushroom.loc[mushroom["veil_type"]== "p","veil_type"] = "partial"
mushroom.loc[mushroom["veil_type"]== "u","veil_type"] = "universal"

mushroom.loc[mushroom["ring_number"]== "n","ring_number"] = 0
mushroom.loc[mushroom["ring_number"]== "o","ring_number"] = 1
mushroom.loc[mushroom["ring_number"]== "t","ring_number"] = 2

mushroom.loc[mushroom["ring_type"]== "c","ring_type"] = "cobwebby"
mushroom.loc[mushroom["ring_type"]== "e","ring_type"] = "evanescent"
mushroom.loc[mushroom["ring_type"]== "f","ring_type"] = "flaring"
mushroom.loc[mushroom["ring_type"]== "l","ring_type"] = "large"
mushroom.loc[mushroom["ring_type"]== "n","ring_type"] = "none"
mushroom.loc[mushroom["ring_type"]== "p","ring_type"] = "pendant"
mushroom.loc[mushroom["ring_type"]== "s","ring_type"] = "sheathing"
mushroom.loc[mushroom["ring_type"]== "z","ring_type"] = "zone"

mushroom.loc[mushroom["population"]== "a","population"] = "abundant"
mushroom.loc[mushroom["population"]== "c","population"] = "clustered"
mushroom.loc[mushroom["population"]== "n","population"] = "numerous"
mushroom.loc[mushroom["population"]== "s","population"] = "scattered"
mushroom.loc[mushroom["population"]== "v","population"] = "several"
mushroom.loc[mushroom["population"]== "y","population"] = "solitary"

mushroom.loc[mushroom["habitat"]== "g","habitat"] = "grasses"
mushroom.loc[mushroom["habitat"]== "l","habitat"] = "leaves"
mushroom.loc[mushroom["habitat"]== "m","habitat"] = "meadows"
mushroom.loc[mushroom["habitat"]== "p","habitat"] = "paths"
mushroom.loc[mushroom["habitat"]== "u","habitat"] = "urban"
mushroom.loc[mushroom["habitat"]== "w","habitat"] = "waste"
mushroom.loc[mushroom["habitat"]== "d","habitat"] = "woods"

#EDA Graphs

mushroom['class'].value_counts().plot(kind='bar')
mushroom['odor'].value_counts().plot(kind='bar')
mushroom['habitat'].value_counts().plot(kind='bar')
mushroom['cap_shape'].value_counts().plot(kind='bar')
mushroom['cap_color'].value_counts().plot(kind='bar')
mushroom['population'].value_counts().plot(kind='bar')

# cite https://stackoverflow.com/questions/31029560/plotting-categorical-data-with-pandas-and-matplotlib

#plt.rcParams['font.size'] = 10.0
mosaic(mushroom, ['population', 'class'])

#cite https://datascience.stackexchange.com/questions/89692/plot-two-categorical-variables
sns.histplot(binwidth=0.5, x="class", hue="habitat", data=mushroom, stat="count", multiple="stack")

sns.histplot(binwidth=0.5, x="class", hue="odor", data=mushroom, stat="count", multiple="stack")
#sns.histplot(binwidth=0.5, x="class", hue="gill_size", data=mushroom, stat="count", multiple="stack")
#sns.histplot(binwidth=0.5, x="class", hue="stalk_surface_below_ring", data=mushroom, stat="count", multiple="stack")
#sns.histplot(binwidth=0.5, x="class", hue="stalk_shape", data=mushroom, stat="count", multiple="stack")
#sns.histplot(binwidth=0.5, x="class", hue="spore_print_color", data=mushroom, stat="count", multiple="stack")

#Check unique values of columns
for col in mushroom:
    print(col,":", mushroom[col].unique())


# Correct '?' values, drop NA
mushroom_clean = mushroom.replace(['?'], float('nan'))
mushroom_clean = mushroom_clean.dropna()

# printing the dataset shape
print("Dataset No. of Rows (Original): ", mushroom.shape[0])
print("Dataset No. of Columns: ", mushroom.shape[1])

print("Dataset No. of Rows: (Clean)", mushroom_clean.shape[0])
print("Dataset No. of Columns: ", mushroom_clean.shape[1])

mushroom['class'].value_counts()
mushroom_clean['class'].value_counts()

# How much data did we lose?
# We are left with 69% of the total records.
# Note that removing observations with missing data can result in a biased model;
5644/8124


# printing the structure of the dataset
print("Dataset info:\n ")
print(mushroom_clean.info())

mushroom_clean['class'] = mushroom_clean['class'].astype('object')
mushroom_clean['cap_shape'] = mushroom_clean['cap_shape'].astype('category')
mushroom_clean['cap_surface'] = mushroom_clean['cap_surface'].astype('category')
mushroom_clean['cap_color'] = mushroom_clean['cap_color'].astype('category')
mushroom_clean['bruises'] = mushroom_clean['bruises'].astype('category')
mushroom_clean['odor'] = mushroom_clean['odor'].astype('category')
mushroom_clean['gill_attachment'] = mushroom_clean['gill_attachment'].astype('category')
mushroom_clean['gill_spacing'] = mushroom_clean['gill_spacing'].astype('category')
mushroom_clean['gill_size'] = mushroom_clean['gill_size'].astype('category')
mushroom_clean['gill_color'] = mushroom_clean['gill_color'].astype('category')
mushroom_clean['stalk_shape'] = mushroom_clean['stalk_shape'].astype('category')
mushroom_clean['stalk_root'] = mushroom_clean['stalk_root'].astype('category')
mushroom_clean['stalk_surface_above_ring'] = mushroom_clean['stalk_surface_above_ring'] .astype('category')
mushroom_clean['stalk_surface_below_ring'] = mushroom_clean['stalk_surface_below_ring'] .astype('category')
mushroom_clean['stalk_color_above_ring'] = mushroom_clean['stalk_color_above_ring'] .astype('category')
mushroom_clean['stalk_color_below_ring'] = mushroom_clean['stalk_color_below_ring'] .astype('category')
mushroom_clean['veil_type'] = mushroom_clean['veil_type'] .astype('category')
mushroom_clean['veil_color'] = mushroom_clean['veil_color'] .astype('category')
mushroom_clean['ring_number'] = mushroom_clean['ring_number'] .astype('category')
mushroom_clean['ring_type'] = mushroom_clean['ring_type'] .astype('category')
mushroom_clean['spore_print_color'] = mushroom_clean['spore_print_color'].astype('category')
mushroom_clean['population'] = mushroom_clean['population'].astype('category')
mushroom_clean['habitat'] = mushroom_clean['habitat'].astype('category')

# printing the summary statistics of the dataset
print(mushroom_clean.describe(include='all'))


# Change all category columns to numeric values
category_cols = mushroom_clean.select_dtypes(['category']).columns

mushroom_clean[category_cols] = mushroom_clean[category_cols].apply(lambda x: x.cat.codes)

category_cols

## separate the target variable
X = mushroom_clean.values[:, 1:]
Y = mushroom_clean.values[:, 0]


# data preprocessing
# encode the target variable
class_le = LabelEncoder()

y = class_le.fit_transform(Y)

# split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100, stratify=y)

y

# Run each of the supervised learning algorithms against cleaned data.
#Make a case for which one(s) to use.


#NAIVE BAYES
# Run the Naive Bayes algorithm.
# perform training

# creating the classifier object
clf = GaussianNB()

# performing training
clf.fit(X_train, y_train)

# prediction on test
y_pred = clf.predict(X_test)

y_pred_score = clf.predict_proba(X_test)

# calculate metrics

print("\n")
print("Results Using Naive Bayes: \n")
print("Classification Report: ")
print(classification_report(y_test,y_pred))
print("\n")


print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)
print("\n")

print("ROC_AUC : ", roc_auc_score(y_test,y_pred_score[:,1]) * 100)
print("\n")

# confusion matrix

conf_matrix = confusion_matrix(y_test, y_pred)
class_names = mushroom_clean['class'].unique()


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

######### TREE
# perform training with giniIndex.
# creating the classifier object
clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5)

# performing training
clf_gini.fit(X_train, y_train)

# perform training with entropy.
# Decision tree with entropy
clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=3, min_samples_leaf=5)

# Performing training
clf_entropy.fit(X_train, y_train)

# make predictions
# predicton on test using gini
y_pred_gini = clf_gini.predict(X_test)

# predicton on test using entropy
y_pred_entropy = clf_entropy.predict(X_test)

# calculate metrics gini model
print("\n")
print("Decision Tree: \n")
print("Results Using Gini Index: \n")
print("Classification Report: ")
print(classification_report(y_test,y_pred_gini))
print("\n")
print("Accuracy : ", accuracy_score(y_test, y_pred_gini) * 100)
print("\n")
print ('-'*80 + '\n')
# calculate metrics entropy model
print("\n")
print("Decision Tree: \n")
print("Results Using Entropy: \n")
print("Classification Report: ")
print(classification_report(y_test,y_pred_entropy))
print("\n")
print("Accuracy : ", accuracy_score(y_test, y_pred_entropy) * 100)
print ('-'*80 + '\n')

# confusion matrix for gini model
conf_matrix = confusion_matrix(y_test, y_pred_gini)
class_names = mushroom_clean['class'].unique()
df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )

plt.figure(figsize=(5,5))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
plt.tight_layout()
plt.show()


# confusion matrix for entropy model
conf_matrix = confusion_matrix(y_test, y_pred_entropy)
class_names = mushroom_clean['class'].unique()
df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )

plt.figure(figsize=(5,5))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
plt.tight_layout()
plt.show()

#RANDOM FOREST

clf = RandomForestClassifier(n_estimators=100, random_state=100)

#performing training

clf.fit(X_train, y_train)

#make predictions

#predicton on test
y_pred = clf.predict(X_test)

y_pred_score = clf.predict_proba(X_test)

#calculate metrics

print("\n")
print("Results Using Random Forest: \n")
print("Classification Report: ")
print(classification_report(y_test,y_pred))
print("\n")

print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)
print("\n")

print("ROC_AUC : ", roc_auc_score(y_test,y_pred_score[:,1]) * 100)
print("\n")

#confusion matrix

conf_matrix = confusion_matrix(y_test, y_pred)
class_names = mushroom['class'].unique()

df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)

plt.figure(figsize=(5,5))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)

#show heat map
plt.tight_layout()
plt.show()
