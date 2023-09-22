#Imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder ,LabelEncoder ,StandardScaler
from sklearn.model_selection import GridSearchCV,cross_val_score
from sklearn.feature_selection import SelectPercentile,f_regression,SelectFromModel,RFECV,SequentialFeatureSelector
from sklearn.metrics import f1_score,accuracy_score,recall_score,roc_auc_score
from imblearn.over_sampling import SMOTE

# models
from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC

# ignore warnings
from warnings import filterwarnings,warn
from sklearn.exceptions import ConvergenceWarning
filterwarnings("ignore", category=ConvergenceWarning)
filterwarnings('ignore')  
warn('ignore')

#misc
sns.set_style('darkgrid')

# Dataset
churn = pd.read_csv("\Churn_Modelling.csv",delimiter=",")

"""
User Defined Functions
"""
#plot continues features in a plot
def continues_plot(continuous,dataset):
    dataset[continuous].hist(figsize=(12, 10),
                            bins=20,
                            layout=(2, 2)
                            )
    plt.show()

# To find best two model with base data
def basemodels(features, target ,algo_list):
    
    mean =[]
  
    for algo in algo_list:
        cross = cross_val_score(estimator=algo[1],
                                X=features,
                                y=target,
                                cv=5,
                                n_jobs=-1
                                ) 
        # will use mean to find the best                                  
        mean.append(cross.mean())
           # zip mean with name (algo_list item zero) then sort them using lambda for mean in reverse order to get max to min
           # useing slicing to get top two values    
    return sorted(list(zip(list(zip(*algo_list))[0],mean)),key=lambda x : x[1] ,reverse=True)[:2]

# GraphSearchCV to optimize the base model that resulted in best results
def graphCV(feature,target,models):
    entries = []

    for model in models:
    # Create model
        clf = model["estimater"]
        # Instantiate the grid search model
        grid_search = GridSearchCV(estimator = clf, param_grid = model["params"], 
                                cv = 5 ,  n_jobs=-1)
        grid_search.fit(feature, target)

        entries.append([clf,grid_search.best_score_,grid_search.best_params_])

    return entries

# using results of gragpsearchCV to create models with best possible features
def BestModel(features, target, algo_list):
    
    result =[]
    
    for algo in algo_list:  
        _score = []
        for score in ["accuracy", "recall" , "roc_auc", "f1" ]:
            cross = cross_val_score(estimator=algo[1],
                                    X=features,
                                    y=target,
                                    cv=5,
                                    n_jobs=-1,
                                    scoring=score
                                    ).mean() 
            _score.append((score,cross))
        result.append((algo[0],_score))

    return result

# Univariate Feature Selection 
def UFS(features,target):
    
    UF_selector = SelectPercentile(f_regression,percentile=25).fit(features,target)

    # display best features using slicing as get support returns a boolean array with True matching the above condition
    print("--------------------------------- \n")
    print(f"Columns with the best features according to Univariate Feature Selection: \n{features.columns[UF_selector.get_support()]}")

    # to remove the rest of the features
    return UF_selector.transform(features)

# Tree–Based Feature Selection
def TBS(features,target):
    
    #SelectFrommodel will select those features which are of importance
    Tree_Selector = SelectFromModel(RandomForestClassifier(n_estimators=250,random_state=0)).fit(features,target)

    # display best features using slicing as get support returns a boolean array with True matching the above condition
    print("--------------------------------- \n")
    print(f"Columns with the best features according to Tree–Based Feature Selection: \n{features.columns[Tree_Selector.get_support()]}")

    # to remove the rest of the features
    return Tree_Selector.transform(features)

#Greedy Feature Selection
def GS(features, target):

    estimator = LogisticRegression(multi_class='auto', solver ='lbfgs')
    Greedy_Selector = RFECV(estimator, cv=10)
    Greedy_Selector.fit(features, target)

    # display best features using slicing as support returns a boolean array with True matching the above condition
    print("--------------------------------- \n")
    print(f"Columns with the best features according to Greedy Feature Selection: \n{features.columns[Greedy_Selector.support_]}")

    # to remove the rest of the features
   
    return Greedy_Selector.transform(features)

#Logistic Regression Coefficients
def LRCS(features,target):

    #The magnitude of the coefficients is directly influenced by the scale of the features. 
    # Therefore, to compare coefficients across features, it is important that all features are on a similar scale. 
    # This is why normalisation is important for variable importance and feature selection in linear models.
    # Hence , we will scale the variables, so we fit a scaler
    scalor = StandardScaler().fit(features)

    LRC_Selector = SelectFromModel(LogisticRegression(C=1000, penalty='l2', max_iter=300, random_state=10))

    LRC_Selector.fit(scalor.transform(features),target)
    print("--------------------------------- \n")
    print('features with coefficients greater than the mean coefficient: {}'.format(
            np.sum(
                np.abs(LRC_Selector.estimator_.coef_) > np.abs(
                    LRC_Selector.estimator_.coef_).mean())))

    # display best features using slicing as get support returns a boolean array with True matching the above condition
    print("--------------------------------- \n")
    print(f"Columns with the best features according to Logistic Regression Feature Selection: \n{features.columns[LRC_Selector.get_support()]}")

    #select features where coefficient is above the mean
    LRC_Selector_features_coef = pd.DataFrame(LRC_Selector.transform(features))
  
    # add the columns name
    LRC_Selector_features_coef.columns = features.columns[(LRC_Selector.get_support())]
  
    return LRC_Selector_features_coef

#Sequential Feature Selection
def SFS(features, target):

    knn = KNeighborsClassifier(n_neighbors=3)
    sfs_selector = SequentialFeatureSelector(knn, n_features_to_select='auto').fit(features, target)
    
    # display best features using slicing as support returns a boolean array with True matching the above condition
    print("--------------------------------- \n")
    print(f"Columns with the best features according to Sequential Feature Selection: \n{features.columns[sfs_selector.get_support()]}")

    return sfs_selector.transform(features)

#Feature Selection by Random Shuffling
def randomshuffel(features, target, model) :
    
    model.fit(features, target)
    performance_shift = []
    initial_Score = f1_score(target, (model.predict(features)))
    
    for feature in features.columns:

        features_c = features.copy()

        # shuffle individual feature
        features_c[feature] = features_c[feature].sample(
            frac=1, random_state=10).reset_index(drop=True)
        #calculating affect of shuffling and measure how much the permutation (or shuffling of its values) decreases the accuracy
        shuff_score = f1_score(target, model.predict(features_c))
        drift = initial_Score - shuff_score
        
        performance_shift.append(drift)

    feature_importance = pd.Series(performance_shift)
    # add variable names in the index
    feature_importance.index = features.columns

    # capture the selected features using filter
    selected_features = feature_importance[feature_importance > 0].index
    print("--------------------------------- \n")
    print(f"Columns with the best features for a {model} with Random Shuffling: \n{selected_features}")

    model.fit(features[selected_features], target)
    f1 = f1_score(target, model.predict(features[selected_features]))
    recall = recall_score(target, model.predict(features[selected_features]))
    accuracy = accuracy_score(target,model.predict(features[selected_features]))
    roc = roc_auc_score(target,model.predict(features[selected_features])) 
    
    return [accuracy,recall,roc,f1]

#one hot encoder
def onehotencoding(categorical_cols,dataset):
    le = LabelEncoder()
    dataset[categorical_cols] = dataset[categorical_cols].apply(lambda col: le.fit_transform(col))    
    encoder = OneHotEncoder()
    encoded_data = pd.DataFrame(encoder.fit_transform(churn[categorical_cols]))
    dataset.drop(columns=categorical_cols)
    dataset.join(encoded_data)
    print("---------------------------------")
    print(f"After OneHotEncoding: \n{dataset.describe()}")


"""
Check DataSet
"""
#Dataset shape
print("---------------------------------")
print('DataFrame contains {} rows and {} columns.'.format(churn.shape[0], churn.shape[1]))
print("\n")

# Check columns for missing values
print("---------------------------------")
print("Check Null Values:") 
print(churn.isnull().sum()) # No null values found
print("\n")

# using describe to find data types for further analysis
print("---------------------------------")
print("Dataset Info:")
print(churn.describe().T)
print("\n")

# using info to find data types for further analysis
print("---------------------------------")
print("Dataset Info:")
print(churn.info())
print("\n")


"""
Exploratory Data Analysis
"""
# Check Class label for imbalance
labels = 'Customer left', 'Customer Retained'
sizes = [churn.Exited[churn['Exited']==1].count(), churn.Exited[churn['Exited']==0].count()]
explode = (0, 0.1)
fig1, ax1 = plt.subplots(figsize=(10, 8))
ax1.pie(sizes, explode=explode, labels=labels,autopct='%1.1f%%')
plt.title("Customer Left Vs Customer Retained")
plt.show()

#Relation of Continues Values
continuous = ['Age', 'CreditScore', 'Balance', 'EstimatedSalary']
continues_plot(continuous,churn)

#Relation of Catagorical Values
categorical = ['Geography', 'Gender', 'Tenure', 'NumOfProducts', 'HasCrCard', 'IsActiveMember']
fig, ax = plt.subplots(3, 2, figsize=(20, 12))
sns.boxplot(y='CreditScore',x = 'Exited', data = churn, ax=ax[0][0])
sns.boxplot(y='Age',x = 'Exited', data = churn , ax=ax[0][1])
sns.boxplot(y='Tenure',x = 'Exited', data = churn, ax=ax[1][0])
sns.boxplot(y='Balance',x = 'Exited', data = churn, ax=ax[1][1])
sns.boxplot(y='NumOfProducts',x = 'Exited', data = churn, ax=ax[2][0])
sns.boxplot(y='EstimatedSalary',x = 'Exited',data = churn, ax=ax[2][1])
plt.show()

""""
Data Processing
"""
# we can determine features 'RowNumber', 'CustomerId', and 'Surname' are specific to each customer and can be dropped 
# as they have no effect on the classification
churn.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)
print("---------------------------------")
print(f"After Droping Columns left: {churn.columns.values}")
print("\n")

# models wont work with non-encoded categorical (which were found using info), so applying one hot encoder to catagorecal values
categorical_cols = ["Gender","Geography"]
onehotencoding(categorical_cols,churn)

#scaling  to normalise the ranges of CreditScore', 'Age', 'Balance' in a dataset 
scaler = StandardScaler()
scl_columns = ['CreditScore', 'Age', 'Balance']
churn[scl_columns] = scaler.fit_transform(churn[scl_columns])

#split test data
X = churn.drop(columns="Exited")
y = churn["Exited"]

#fixing imbalance
over = SMOTE(sampling_strategy='auto', random_state=0)
X,y = over.fit_resample(X,y)

""""
Base Line
"""

# Base Value of the models
base_algo_list =[('GaussianNB', GaussianNB() ) ,
                ('LRegress',LogisticRegression(random_state=0)) , 
                ('KNN',KNeighborsClassifier(n_neighbors=3)),
                ('DTree',DecisionTreeClassifier(random_state=0)) ,
                ('RForest',RandomForestClassifier(max_depth=2, random_state=0)), 
                #dual = False to remove exception 
                ('LinearSVC',LinearSVC(dual=False,random_state=0, tol=0.0001))
                ]
base_value = basemodels(X, y,base_algo_list)
print("---------------------------------")
print(f"Two Best Base Value of Models: \n {base_value}")

"""
# Two Best Base Value of Models
 
Two Best Base Value of Models:
 [('DTree', 0.8094389208674923), ('RForest', 0.7973817161226362)]

"""

"""
Hypermeter Tuning
"""
models = [
        { "estimater": LinearSVC(),
            "params": {
            "loss": ['hinge','squared_hinge'],
            "multi_class": ['ovr'],
            "fit_intercept": [True, False],
            "max_iter": [2000,3000],
            "dual" : [False],
            "tol" : [0.0001],
            "random_state" : [0]
            }
        },
       { "estimater": RandomForestClassifier(),
        "params": {
          "criterion": ['gini','entropy'],
           "max_depth": [None,2,5],
          "max_features": [None,"sqrt","log2"],
          "n_estimators": [100, 120, 90],
          "random_state" : [0]          
      }
    }
]

graph_results = graphCV(X,y,models)
print("---------------------------------")
print(graph_results)

"""
tuning results :
[[LinearSVC(), 0.771826071874498, {'dual': False, 'fit_intercept': True, 'loss': 'squared_hinge', 'max_iter': 2000, 'multi_class': 'ovr', 'random_state': 0, 'tol': 0.0001}], 
[RandomForestClassifier(), 0.862498726275966, {'criterion': 'entropy', 'max_depth': None, 'max_features': None, 'n_estimators': 100, 'random_state': 0}]]
"""


"""
Feature Selection
"""

optimized_algo_list =[
            ('RForest',RandomForestClassifier(max_depth=None,criterion="entropy", max_features=None, n_estimators=100 ,random_state=0 )), 
                #dual = False to remove exception 
            ('LinearSVC',LinearSVC(dual=False, tol=0.0001, fit_intercept= True, loss='squared_hinge',max_iter=2000, multi_class='ovr',random_state=0))
                ]

# Base Results
base_results = BestModel(X,y,optimized_algo_list)
print("--------------------------------- \n")
print(f"Base Results with basic data processing : \n{base_results}")

# Univariate Feature Selection 
UFS_X=UFS(X,y)
UFS_result = BestModel(UFS_X,y,optimized_algo_list)
print("--------------------------------- \n")
print(f"Univariate Feature Selection Results : \n{UFS_result}")

# Tree–Based Feature Selection
TBS_X = TBS(X,y)
TFS_result = BestModel(TBS_X,y,optimized_algo_list)
print("--------------------------------- \n")
print(f"Tree–Based Feature Selection Results : \n{TFS_result}")

# Greedy Feature Selection
GS_X= GS(X, y)
GS_result = BestModel(GS_X,y,optimized_algo_list)
print("--------------------------------- \n")
print(f"Greedy Feature Selection Results : \n{GS_result}")

# Logistic Regression Coefficients
LRC_X = LRCS(X,y)
LRCS_result = BestModel(LRC_X,y,optimized_algo_list)
print("--------------------------------- \n")
print(f"Logistic Regression Coefficients Feature Selection Results : \n{LRCS_result}")

# Sequential Feature Selection.
SFS_X = SFS(X,y)
SFS_result = BestModel(SFS_X,y,optimized_algo_list)
print("--------------------------------- \n")
print(f"Sequential Feature Selection Results : \n{SFS_result}")

#Feature Selection by Random Shuffling
print("--------------------------------- \n")
acc , recall , roc , f1  = randomshuffel(X,y,optimized_algo_list[0][1])   
print("RF Results:",acc , recall , roc , f1)
acc , recall , roc , f1  = randomshuffel(X,y,optimized_algo_list[1][1])   
print("SVC Results:",acc , recall , roc , f1)

