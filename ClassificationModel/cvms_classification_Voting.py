from csv import reader
import os,glob
import pandas as pd
import numpy as np
import json
from joblib import dump, load

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score

from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV


# Create Confusion Matrix report
def confusion_matrix_graph(y_test, y_pred, labels, name_img):
    print('Confusion Matrix')
    sns.heatmap(metrics.confusion_matrix(y_test, y_pred, labels=labels), 
                annot=True, annot_kws={"size": 16}, 
                square=True, cmap='Reds', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.savefig(name_img,bbox_inches='tight')

# Rename of dataset
def simplify(tlabel):
    if "CVMS_FiveClass" in tlabel:
        tlabel = '5_Class'
    elif "Clinic" in tlabel:
        tlabel = '4_Class'
    else:
        tlabel = '6_Class'
    return tlabel


def simplify_name(tlabel):
    if "Ton" in tlabel:
        name_type = 'Rater1'
    elif "Jam" in tlabel:
        name_type = 'Rater2'
    else:
        name_type = 'Rater3'
    return name_type

 


# Import Feature extraction data
dataframe = pd.read_csv(os.getcwd()+'/CheckData_Class.csv').set_index('No.')

# seperate categorical col and numeric col
cat_col = set(['CVM Pitipat','CVM Prinya', 'CVM Supakit'])
num_col = list(set(dataframe._get_numeric_data().columns.tolist()) - cat_col)
col = num_col

# Type of dataframe: 1 reader, 3 reader ( Voting agreement), 3 reader ( Completed Agreement) 

temp_D = dataframe[dataframe['Level of dentists agreement'] != 'Disagreement']
name_type_D = 'Majority'
temp_T = dataframe[dataframe['Level of dentists agreement'] == 'Totally Agreement']
name_type_T = 'Completed'
temp_O = dataframe
name_type_O = 'One_'


for temp, name_type in zip([temp_D,temp_T],[name_type_D,name_type_T]): 

    # Rescaling on data with Standard scaler
    scaler =  StandardScaler()
    feature = temp.reset_index(drop=True).loc[:,col]
    scale_feature = scaler.fit_transform(feature.replace(np.inf, 0))
    scale_df = pd.DataFrame(scale_feature, columns=feature.columns).fillna(0)

    for tlabel in ['CVMS_Class']:

        label = temp.reset_index(drop=True).loc[:, tlabel]

        noclass = simplify(tlabel)
        name_feat = os.getcwd()+'/July2023/Prepare_Stage2/Data_'+name_type+'_'+noclass+'.csv'
        pd.concat([scale_df,label],axis=1).to_csv(name_feat)
        print('Dataframe shape', temp.shape)

        # Train Test spliting
        X_train, X_test, y_train, y_test = train_test_split(scale_df, label, test_size=0.3, random_state=7)

        # Feature selection by using Random forest model

        # 1. Full features
        feat_labels = feature.columns
        clf = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
        clf.fit(X_train, y_train)

        # Apply The Full Featured Classifier To The Test Data
        y_pred = clf.predict(X_test)

        # View the accuracy Of Full features
        print('Number of features: ', feature.columns.shape)
        print('Accuracy score : ', accuracy_score(y_test, y_pred))

        # 2. Limited features
        # Create a selector object that will use the random forest classifier to identify
        # features that have an importance of more than 0.015
        sfm = SelectFromModel(clf, threshold=0.015)

        # Train the selector
        sfm.fit(X_train, y_train)

        print('Number of features: ', feature.columns[sfm.get_support(indices=True)].shape)
        # Print the names of the most important features
        feat = []
        for feature_list_index in sfm.get_support(indices=True):
            feat = feat + [feat_labels[feature_list_index]]
            print(feat_labels[feature_list_index])

        # tlabel = simplify(tlabel)
        name_feat = os.getcwd()+'/July2023/FeatureImportance/Feature_'+name_type+'_'+noclass+'.csv'
        pd.DataFrame(feat).to_csv(name_feat)
            
        X_important_train = sfm.transform(X_train)
        X_important_test = sfm.transform(X_test)
        # Create a new random forest classifier for the most important features
        clf_important = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
        # Train the new classifier on the new dataset containing the most important features
        clf_important.fit(X_important_train, y_train)
        # Apply The Full Featured Classifier To The Test Data
        y_important_pred = clf_important.predict(X_important_test)

        # View The Accuracy Of Our Limited Feature Model
        print('Accuracy score : ', accuracy_score(y_test, y_important_pred))
        print('-' * 100, '\n')

        # X_important = sfm.fit(scale_df)
        name_feat = os.getcwd()+'/July2023/Prepare_Stage2/DataImportance_'+name_type+'_'+noclass+'.csv'
        pd.concat([scale_df.loc[:,feat],label],axis=1).to_csv(name_feat)

        labels = sorted(list(set(label)))
        model_score=[]

        # Selected single model: KNeighborsClassifier, Support Vector Machine, LogistricRegression
        # Selected ensemble model: RandomForest, GradientBoostingClassifier
        algo = [
                [KNeighborsClassifier(), 'KNeighbors'],
                [SVC(probability=True), 'SVM'],
                [RandomForestClassifier(), 'RForest'],
                [LogisticRegression(max_iter=1000000), 'LogReg'],
                [GradientBoostingClassifier(), 'GraBoost'],
            ]

        parameters = [ 
            [{ 'n_neighbors' : [5,6,7,9,11,13,15], 'weights' : ['uniform','distance'], 
            'metric' : ['minkowski','euclidean','manhattan']}],
            
            [{'C': (0.1, 200.0, 'log-uniform'),'gamma': (0.1, 200.0, 'log-uniform'),
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'degree': (1,10)}],
            
            
            [{'max_depth': (5,30), 'min_samples_split': [2, 5, 10],'n_estimators': (1,100),
            'max_features': ['auto', 'sqrt'],'min_samples_leaf': (1,5),'bootstrap': [True, False]}],
            
            
            [{'penalty' : ['l2'],'C' : np.logspace(-4, 4, 20), 'solver' : ['lbfgs','newton-cg','liblinear','sag','saga'] }],

            [{'n_estimators': [10, 50, 100, 500], 'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1.0], 
            'subsample':[0.5, 0.7, 1.0], 'max_depth': (3, 30)}]
                    
                    ]


        for a,b in zip(algo,parameters):
            model=a[0]
            params = b[0]
            fig = plt.figure()
            # Cross validation
            cv = RepeatedKFold(n_splits=10, n_repeats=1, random_state=1)
            # define the search
            search = BayesSearchCV(estimator=model, search_spaces=params, n_jobs=-1, cv=cv, scoring='accuracy')
            # # perform the search
            search.fit(X_important_train, y_train)


            name_model = os.getcwd()+'/July2023/Model/'+name_type+'_'+noclass+'_'+a[1]+'.joblib'
            dump(search, name_model) 

            # # report the best result
            print('Accuracy for training:', round(search.best_score_,3))
            print('Best_parameter:', search.best_params_)


            name_para = os.getcwd()+'/July2023/BestParameters/'+name_type+'_'+noclass+'_'+a[1]+'_bestPara.csv'
            pd.DataFrame.from_dict(search.best_params_, orient="index").to_csv(name_para)
            
            y_pred=search.predict(X_important_test) # step 3: predict
            score=search.score(X_important_test, y_test)
            model_score.append([a[1],score])
            print(f'{a[1]} score = {score}') # step 4: score



            name_img = os.getcwd()+'/July2023/ConfusionMatrix/'+name_type+'_'+noclass+'_'+a[1]+'.png'
            confusion_matrix_graph(y_test, y_pred, labels, name_img)
            
            print('\n Classification Report')
            print(metrics.classification_report(y_test, y_pred))
            report = metrics.classification_report(y_test, y_pred, output_dict=True)


            name = os.getcwd()+'/July2023/Classification_report/'+name_type+'_'+noclass+'_'+a[1]+'_class_report.csv'
            pd.DataFrame(report).transpose().to_csv(name)
            print('-' * 100)

        #############################################################################################################

        # Multi Layer Perceptron
        fig = plt.figure()
        mlp_gs = MLPClassifier(max_iter=1000000)
        parameter_space = {
            'hidden_layer_sizes': [(16,16),(16,16,16),(10,10,10),(5,10),(10,)],
            'activation': ['tanh', 'relu'],
            'solver': ['sgd', 'adam'],
            'alpha': [0.0001, 0.05],
            'learning_rate': ['constant','adaptive'],
        }


        clf = GridSearchCV(mlp_gs, parameter_space, n_jobs=-1, cv=10)
        clf.fit(X_important_train, y_train) # X is train samples and y is the corresponding labels


        name_model = os.getcwd()+'/July2023/Model/'+name_type+'_'+noclass+'_'+'MLP.joblib'
        dump(clf, name_model) 

        print('Best_parameter:', clf.best_params_)

        name_para = os.getcwd()+'/July2023/BestParameters/'+name_type+'_'+noclass+'_MLP_bestPara.csv'
        pd.DataFrame.from_dict(clf.best_params_, orient="index").to_csv(name_para)
        score = clf.score(X_important_test,y_test)
        print('MLP score =',  score)


        name_img = os.getcwd()+'/July2023/ConfusionMatrix/'+name_type+'_'+noclass +'_MLP.png'
        confusion_matrix_graph(y_test, y_pred, labels, name_img)


        print('\n Classification Report')
        print(metrics.classification_report(y_test, y_pred))
        report = metrics.classification_report(y_test, y_pred, output_dict=True)


        name = os.getcwd()+'/July2023/Classification_report/'+name_type+'_'+noclass+'_MLP_class_report.csv'
        pd.DataFrame(report).transpose().to_csv(name)
        print('-' * 100)

        model_score.append(['MLP',score])

        dscore=pd.DataFrame(model_score, columns=['classifier', 'score'])
        final = dscore.sort_values('score', ascending=False)
        print(final)


        final.to_csv(os.getcwd()+'/July2023/ModelAccuracy/'+ name_type + '_CVMS_'+ noclass + '.csv')
