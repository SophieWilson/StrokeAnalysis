# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, GridSearchCV, KFold, StratifiedKFold
import re
import seaborn as sns 

### Preprocessing
# Importing the dataset
datasets = pd.read_csv('C:/Users/Mischa/Documents/Uni Masters/Module 6 - Group proj/finalSMOTEIMPUTED.csv')
# subsetting real data
outliers = datasets[datasets.origin != 'synthetic 1']
# 1 is non-fitter, 0 is fitter
outliers['classnum'] = outliers['origin'].replace(['original 1', 'original 0'], [1, 0])
outliers_X = outliers.drop(['ID', 'classnum', 'origin'],axis=1)
outliers_y = outliers['classnum']

# making class labels numeric 
datasets['classnum'] = datasets['origin'].replace(['synthetic 1', 'synthetic 0', 'original 1', 
'original 0'], [1, 0, 1, 0])
# setting x and y
X = datasets.drop(['ID', 'classnum', 'origin'],axis=1)
y = datasets['classnum']

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = 0.25, random_state = 0)

# Grid search
def grid_search(X_train, y_train):
    ''' Grid search to find best parameters'''
    random_f_model = RandomForestClassifier()
    parameters = {
        'criterion': ['gini','entropy'], 
        'n_estimators': [2,3,5,10, 200], 
        'max_depth':[1,2,3,4,5, 100],
        'min_samples_leaf':[2,5,7,10, 20],
    }
    # weighted == F1 Measure for multi-class
    rf_grid_search = GridSearchCV(random_f_model, parameters, cv=5,scoring='balanced_accuracy') 
    grid_search = rf_grid_search.fit(X_train, y_train)
    # best model according to grid search 
    best_random_f_model = rf_grid_search.best_estimator_ 
    # can print params if wanted.
    #print(best_random_f_model.get_params())
    return best_random_f_model

best_random_f_model = grid_search(X_train, y_train)

# Cross fold validation, five iterations
from sklearn.model_selection import cross_val_score, StratifiedKFold
skf = StratifiedKFold(n_splits = 5)
all_accuracies = cross_val_score(estimator=best_random_f_model, X=X_train, y=y_train, cv=skf)

# Feature Importance
def feature_importance(X, best_random_f_model):
    ''' Plot feature importance graph '''
    df_importance = pd.DataFrame(list(zip(X.columns.values,best_random_f_model.feature_importances_)),columns=['column_name','feature_importance'])
    df_importance = df_importance.set_index(['column_name'])
    df_importance.sort_values(['feature_importance'],ascending=False,inplace=True)
    # export list
    df_importance.to_csv(r'C:\Users\Mischa\Documents\Uni Masters\Module 6 - Group proj\importancelist.csv')
    plt.figure(figsize=(20,10))
    sns.barplot(x='column_name',y='feature_importance',data=df_importance.reset_index(),palette='muted')
    #ticks_information = plt.xticks(rotation=65)
    plt.show()

feature_importance(X, best_random_f_model)

# Predicting the test set results
y_pred = best_random_f_model.predict(X_test)
# Probabilities for each class
rf_probs = best_random_f_model.predict_proba(X_test)[:, 1]

# ROC auc plot for predictions
def prediction(best_random_f_model, X_test, y_pred, rf_probs):
    ''' Plot ROC AUC for predictions using best model '''
    import sklearn.metrics as metrics
    from sklearn.metrics import roc_curve, roc_auc_score
    # calculate roc curve
    prediction.fpr, prediction.tpr, thresholds = roc_curve(y_test, rf_probs)
    roc_auc = metrics.auc(prediction.fpr,prediction.tpr)
    # Plot ROC
    fig, ax = plt.subplots(figsize=(10,7))
    ax.plot(prediction.fpr, prediction.tpr, label='Random forest')
    ax.plot(np.linspace(0, 1, 100),
         np.linspace(0, 1, 100),
         label='Baseline',
         linestyle='--')
    plt.title('Receiver Operating Characteristic Curve', fontsize=18)
    plt.ylabel('True positive rate', fontsize=16)
    plt.xlabel('False positive rate', fontsize=16)
    plt.legend(fontsize=12)
    #plt.show()

prediction(best_random_f_model,X_test, y_pred, rf_probs)

def confusion_mat(y_test, y_pred):
    ''' Make confusion matrix '''
    # Making the Confusion Matrix 
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    #plot it
    np.set_printoptions(precision=2)

def plot_confusionmat(best_random_f_model, X_test, y_test):
    ''' Plot confusion matrix '''
    from sklearn.metrics import plot_confusion_matrix
     # Plot non-normalized confusion matrix
    titles_options = [("Confusion matrix, without normalization", None),
                    ("Normalized confusion matrix", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(best_random_f_model, X_test, y_test,
                                    cmap=plt.cm.Blues,
                                    normalize=normalize, labels=[0, 1])
        disp.ax_.set_title(title)
        print(title)
        print(disp.confusion_matrix)
    plt.show()

confusion_mat(y_test, y_pred)
plot_confusionmat(best_random_f_model, X_test, y_test)

def metrics_report(y_test, y_pred, cross_val_metrics, prediction_tpr, prediction_fpr):
    ''' y_test and y_pred are labels, cross_val_metrics is the output of cross_val_score sklearn prediction_tpr/fpr are outputs from roc_curve function sklearn '''   
    from sklearn import metrics
    from sklearn.metrics import classification_report
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Mean Accuracy', np.mean(cross_val_metrics))
    print('Sensitivity', prediction_tpr/(prediction_tpr + prediction_fpr))
    metrics_report.classification_rep = classification_report(y_test, y_pred, labels= [0, 1], output_dict=True)
    print(classification_report)

def outputreport(classification_rep):
    classification_df = pd.DataFrame(classification_rep).transpose()
    classification_df.to_csv(r'C:\Users\Mischa\Documents\Uni Masters\Module 6 - Group proj\classification_report.csv')


metrics_report(y_test, y_pred, all_accuracies, prediction.tpr, prediction.fpr)

outputreport(metrics_report.classification_rep)   

##### To try and get variable importance in group classification, this is pretty 
# complicated and not very reliable as random forest isn't made for it. Probably best to use plsda.

#  from treeinterpreter import treeinterpreter as ti
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
