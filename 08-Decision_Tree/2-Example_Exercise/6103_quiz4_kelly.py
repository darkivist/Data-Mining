#cite: https://github.com/amir-jafari/Data-Mining/blob/master/08-Decision_Tree/2-Example_Exercise/DT_1.py

#import libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# import os
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

# libraries to display decision tree
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
import webbrowser

# read in data
social = pd.read_csv("social_media.csv")

# examine shape
shape = social.shape
print(shape)

#print dataframe head
print(social.head())

#print dataframe tail
print(social.tail())

# print rows and column values
print("Dataset No. of Rows: ", social.shape[0])
print("Dataset No. of Columns: ", social.shape[1])

# printing the dataset observations
print("Dataset first few rows:\n ")
print(social.head())

# printing the structure of the dataset
print("Dataset info:\n ")
print(social.info())
print ('-'*80 + '\n')

# no missing values.
print("No missing values observed. Further preprocess data.")

# make gender binary integer values.
social.loc[social.Gender == 'Female', 'Gender'] = 1
social.loc[social.Gender == 'Male', 'Gender'] = 0

# drop User ID.
social = social.drop('User ID',1)

# correct datatypes.
social.Gender = social.Gender.astype('category')
social.Purchased = social.Purchased.astype('object')

# printing the summary statistics of the dataset
print(social.describe(include='all'))

# LabelEncode for DecisionTree.
social[['Purchased']] = social[['Purchased']].apply(LabelEncoder().fit_transform)

# split the dataset
# separate the target variable

X = social.values[:, 0:3]
y = social.values[:, 3]

# check to make sure variables are correct.
print(X)

# encoding the class with sklearn's LabelEncoder
class_le = LabelEncoder()

# fit and transform the class
y = class_le.fit_transform(y)

# split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

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
purchase_decision = social.Purchased.unique()
print(social.Purchased.unique())
df_cm = pd.DataFrame(conf_matrix, index=purchase_decision, columns=purchase_decision)

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
purchase_decision = social.Purchased.unique()
purchase_decision = purchase_decision.astype(str)
df_cm = pd.DataFrame(conf_matrix, index=purchase_decision, columns=purchase_decision )

plt.figure(figsize=(5,5))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
plt.tight_layout()
plt.show()

# display decision tree
dot_data = export_graphviz(clf_gini, filled=True, rounded=True, class_names=purchase_decision, feature_names=social.iloc[:, 0:3].columns, out_file=None)

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
