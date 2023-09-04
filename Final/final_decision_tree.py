import pandas as pd
from sklearn.tree import DecisionTreeClassifier
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
comfort = pd.read_csv("train.csv", header=None, names=col_names)

comfort.head()

#Check unique values of columns
for col in comfort:
    print(comfort[col].unique())

# printing the dataset shape
print("Dataset No. of Rows: ", comfort.shape[0])
print("Dataset No. of Columns: ", comfort.shape[1])

comfort_clean = comfort.replace([''], float('nan'))
comfort_clean = comfort_clean.dropna()

print("Dataset No. of Rows: ", comfort_clean.shape[0])
print("Dataset No. of Columns: ", comfort_clean.shape[1])

#printing the structure of the dataset
print("Dataset info:\n ")
print(comfort_clean.info())


#changing datatypes
#comfort_clean['age'] = comfort_clean.age.astype('int')
comfort_clean['workclass'] = comfort_clean.workclass.astype('category')
comfort_clean['education'] = comfort_clean.education.astype('category')
comfort_clean['educational_level'] = comfort_clean.educational_level.astype('category')
comfort_clean['marital_status'] = comfort_clean.marital_status.astype('category')
comfort_clean['occupation'] = comfort_clean.occupation.astype('category')
comfort_clean['relationship'] = comfort_clean.relationship.astype('category')
comfort_clean['race'] = comfort_clean.race.astype('category')
comfort_clean['gender'] = comfort_clean.gender.astype('category')
#comfort_clean['money_game'] = comfort_clean.money_gain.astype('int')
#comfort_clean['money_loss'] = comfort_clean.money_loss.astype('int')
#comfort_clean['hours_per_week'] = comfort_clean.hours_per_week.astype('int')
comfort_clean['country'] = comfort_clean.country.astype('category')
comfort_clean['comfortability'] = comfort_clean.comfortability.astype('str')

comfort_clean = comfort_clean.replace(['0'], str('0'))
comfort_clean = comfort_clean.replace(['1'], str('1'))

comfort_clean['comfortability'] = comfort_clean.comfortability.astype('category')

# Change all category columns to numeric values
category_cols = comfort_clean.select_dtypes(['category']).columns

comfort_clean[category_cols] = comfort_clean[category_cols].apply(lambda x: x.cat.codes)
print(comfort_clean.info())

## separate the target variable
X = comfort_clean.values[:, :-1]
Y = comfort_clean.values[:, -1]

# data preprocessing
# encode the target variable
class_le = LabelEncoder()

y = class_le.fit_transform(Y)


# split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100, stratify=y)

# perform training with giniIndex.
# creating the classifier object
# removed parameters random_state=100, max_depth=3, min_samples_leaf=5 to run in default state (this was the only way I could see all variables in resulting graph
clf_gini = DecisionTreeClassifier(criterion="gini")

# performing training
clf_gini.fit(X_train, y_train)

# perform training with entropy.
# decision tree with entropy
# removed parameters random_state=100, max_depth=3, min_samples_leaf=5 to run in default state (this was the only way I could see all variables in resulting graph
clf_entropy = DecisionTreeClassifier(criterion="entropy")

# Performing training
clf_entropy.fit(X_train, y_train)

# make predictions
# predicton on test using gini
y_pred_gini = clf_gini.predict(X_test)

# predicton on test using entropy
y_pred_entropy = clf_entropy.predict(X_test)

y_pred_entropy

# calculate metrics gini model
print("\n")
print("Results Using Gini Index: \n")
print("Classification Report: ")
print(classification_report(y_test,y_pred_gini))
print("\n")
print("Accuracy : ", accuracy_score(y_test, y_pred_gini) * 100)
print("\n")
print ('-'*80 + '\n')
# calculate metrics entropy model
print("\n")
print("Results Using Entropy: \n")
print("Classification Report: ")
print(classification_report(y_test,y_pred_entropy))
print("\n")
print("Accuracy : ", accuracy_score(y_test, y_pred_entropy) * 100)
print ('-'*80 + '\n')

# confusion matrix for gini model
conf_matrix = confusion_matrix(y_test, y_pred_gini)
comfortability = comfort.comfortability.unique()
df_cm = pd.DataFrame(conf_matrix, index=comfortability, columns=comfortability)

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
#purchase_decision = social.comfortability.unique()
#purchase_decision = purchase_decision.astype(str)
df_cm = pd.DataFrame(conf_matrix, index=comfortability, columns=comfortability)

plt.figure(figsize=(5,5))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
plt.tight_layout()
plt.show()

# display decision tree
dot_data = export_graphviz(clf_gini, filled=True, rounded=True, class_names=comfortability, feature_names=comfort.iloc[:, 0:3].columns, out_file=None)

graph = graph_from_dot_data(dot_data)
graph.write_pdf("6103_quiz4_decision_tree_gini_kelly.pdf")
webbrowser.open_new(r'6103_quiz4_decision_tree_gini_kelly.pdf')

print(clf_entropy)

# display decision tree

dot_data = export_graphviz(clf_entropy, filled=True, rounded=True, class_names=purchase_decision, feature_names=social.iloc[:, 0:3].columns, out_file=None)

graph = graph_from_dot_data(dot_data)
graph.write_pdf("6103_quiz4_decision_tree_entropy_kelly.pdf")
webbrowser.open_new(r'6103_quiz4_decision_tree_entropy_kelly.pdf')

print("Decision trees made with default parameters to ensure all variables displayed in Decision Trees.")
print ('-'*40 + 'End Console' + '-'*40 + '\n')


















