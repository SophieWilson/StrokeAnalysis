# Random Forest Classifier

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, GridSearchCV, KFold, StratifiedKFold
import re
import seaborn as sns # nicer (easier) visualisation

# Importing the datasets
datasets = pd.read_csv('C:/Users/Mischa/Documents/Uni Masters/Module 6 - Group proj/finalSMOTEIMPUTED.csv')
datasets['classnum'] = datasets['origin'].replace(['synthetic 1', 'synthetic 0', 'original 1', 
'original 0'], [1, 0, 1, 0])
X = datasets.drop(['ID', 'classnum', 'origin'],axis=1)
y = datasets['classnum']
print(y)
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Grid search
parameters = {
    'criterion': ['gini','entropy'], 
    'n_estimators': [2,3,5,10, 200], 
    'max_depth':[1,2,3,4,5, 100],
    'min_samples_leaf':[2,5,7,10, 20],
}

random_f_model = RandomForestClassifier() 
rf_grid_search = GridSearchCV(random_f_model, parameters, cv=5,scoring='balanced_accuracy') # weighted == F1 Measure for multi-class
grid_search = rf_grid_search.fit(X_train, y_train)
best_random_f_model = rf_grid_search.best_estimator_ # best model according to grid search 
#print(best_random_f_model.get_params())

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
# Cross fold validation, five
skf = StratifiedKFold(n_splits = 5)
all_accuracies = cross_val_score(estimator=best_random_f_model, X=X_train, y=y_train, cv=skf)

# Feature Importance
df_importance = pd.DataFrame(list(zip(X.columns.values,best_random_f_model.feature_importances_)),columns=['column_name','feature_importance'])
df_importance = df_importance.set_index(['column_name'])
df_importance.sort_values(['feature_importance'],ascending=False,inplace=True)
#df_importance[df_importance['feature_importance']]
print(df_importance)
plt.figure(figsize=(20,10))
sns.barplot(x='column_name',y='feature_importance',data=df_importance.reset_index(),palette='muted')
ticks_information = plt.xticks(rotation=65)
plt.show()

# Predicting the test set results
y_pred = best_random_f_model.predict(X_test)
# Probabilities for each class
rf_probs = best_random_f_model.predict_proba(X_test)[:, 1]
import sklearn.metrics as metrics
from sklearn.metrics import roc_curve, roc_auc_score
# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_test, rf_probs)
roc_auc = metrics.auc(fpr,tpr)
# method I: plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
# Making the Confusion Matrix 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# plot it
# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Oranges):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
#     """
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')

#     print(cm)

#     # Plot the confusion matrix
#     plt.figure(figsize = (10, 10))
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title, size = 24)
#     plt.colorbar(aspect=4)
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45, size = 14)
#     plt.yticks(tick_marks, classes, size = 14)

#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
    
#     # Labeling the plot
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt), fontsize = 20,
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
        
#     plt.grid(None)
#     plt.tight_layout()
#     plt.ylabel('True label', size = 18)
#     plt.xlabel('Predicted label', size = 18)

# plot_confusion_matrix(cm, classes = ['Poor Health', 'Good Health'],
#                       title = 'Health Confusion Matrix')


from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Mean Accuracy', np.mean(all_accuracies))


# from treeinterpreter import treeinterpreter as ti
# prediction, bias, contributions = ti.predict(estimator, X_test[6:7])
# N = 22 # no of entries in plot , 4 ---> features & 1 ---- class label
# fitter = []
# non_fitter = []

# for i in range(2):
#     list_ = [fitter, non_fitter]
#     for i in range(3):
#         val = contributions[0,i,j]
#         list_[j].append(val)
           
# fitter.append(prediction[0,0]/5)
# non_fitter.append(prediction[0,1]/5)
# fig, ax = plt.subplots()
# ind = np.arange(N)   
# width = 0.15        
# p1 = ax.bar(ind, setosa, fitter, color='red', bottom=0)
# p2 = ax.bar(ind+width, non_fitter, width, color='green', bottom=0)
# ax.set_title('Contribution of all feature for a particular \n sample of flower ')
# ax.set_xticks(ind + width / 2)
# ax.set_xticklabels(col, rotation = 90)
# ax.legend((p1[0], p2[0] ,p3[0]), ('fitter', 'non_fitter' ) , bbox_to_anchor=(1.04,1), loc="upper left")
# ax.autoscale_view()
# plt.show()
