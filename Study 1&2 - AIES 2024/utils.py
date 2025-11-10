import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import datetime
import statsmodels.api as sm
import scipy.stats as sci
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, XGBRFClassifier
# from tensorflow.keras.layers import Flatten, BatchNormalization, Dense, Activation, Dropout
# from tensorflow.keras.models import Sequential
# import tensorflow
# from tensorflow.keras import backend as K
# from fasttrees import FastFrugalTreeClassifier
import copy

from sklearn.model_selection import GridSearchCV
# from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import RandomizedSearchCV
from tqdm.notebook import tqdm
from sklearn.pipeline import Pipeline
import scipy.stats as stats

model_labels = ['XGBoost', 'XGBRF', 'MLP', 'KNeighbors','DecisionTree', 'LogisticR', 'RandForest']

def initialize_estimators():
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='auc', random_state=42)
    xgbrf = XGBRFClassifier(use_label_encoder=False, random_state=42)
    
    xgb = XGBClassifier(eval_metric='auc', random_state=42)
    xgbrf = XGBRFClassifier(random_state=42)
    mlp = MLPClassifier(random_state=42, max_iter=1000)
    knn = KNeighborsClassifier()
    dt = DecisionTreeClassifier(random_state=42)
    lr = LogisticRegression(random_state=42, solver='liblinear')
    # el = ElasticNetCV(l1_ratio = [.1, .5, .7, .9, .95, .99, 1], max_iter = 5000,random_state=42)
    rf = RandomForestClassifier(random_state=42)

    # return [rf, xgb, xgbrf, mlp, knn, dt, lr, el]

    models = [xgb, xgbrf, mlp, knn, dt, lr, rf]
    
    return dict(zip(model_labels, models))


def estimator_params():
    xg_params = {
        'clf__max_depth': stats.randint(3, 10),
        'clf__learning_rate': stats.uniform(0.01, 0.1),
        'clf__subsample': stats.uniform(0.5, 0.5),
        'clf__n_estimators':stats.randint(50, 200)
    }
    
    mlp_params = {
        'clf__hidden_layer_sizes': [(20, 5), (20, 5, 2), (100,)],
        'clf__activation': ['tanh', 'relu'],
        'clf__solver': ['sgd', 'adam'],
        'clf__alpha': [0.0001, 0.01, 0.1, 0.5, 1],
        'clf__learning_rate': ['constant','adaptive'],
    }

    knn_params = {'clf__n_neighbors' : [5,7,9,11,13,15],
               'clf__weights' : ['uniform','distance'],
               'clf__metric' : ['minkowski','euclidean','manhattan']}

    lr_params = {'clf__C': [0.001, 0.01, 0.1, 0.5, 1, 2, 3], 
                 'clf__penalty': ['l1', 'l2']}

    rf_params = {'clf__bootstrap': [True, False],
                 'clf__max_depth': [1, 5, 10, None],
                 'clf__min_samples_leaf': [1, 2, 4],
                 'clf__min_samples_split': [2, 5, 10],
                 'clf__n_estimators': [100, 200, 300, 400, 500]}
    
    dt_params = {
        'clf__max_depth': stats.randint(1, 20),
        'clf__min_samples_split': stats.randint(2, 20),
        'clf__min_samples_leaf': stats.randint(1, 20)
    }

    params = [xg_params, xg_params, mlp_params, knn_params, dt_params, lr_params, rf_params]

    return dict(zip(model_labels, params))


def predictive_modeling(df, study="one", reps=5):
    if study=="one":
        main_feat_cols = ['uid', 'lalco', 'ldep', 'llife', 'lcrim', 'ralco', 'rdep', 'rlife', 'rcrim', 
                     'alcodiff', 'depdiff', 'lifediff', 'crimdiff']
    if study=="two":
        main_feat_cols = ['id', 'l_elderlyDep', 'l_lifeYearsGained', 'l_obesity', 'l_weeklyWorkhours', 'l_yearsWaiting', 
                     'r_elderlyDep', 'r_lifeYearsGained', 'r_obesity', 'r_weeklyWorkhours', 'r_yearsWaiting', 
                     'eldepdiff', 'lifediff', 'obesdiff', 'workdiff', 'waitdiff']
    
    user_metrics = {}
    user_models = {}
    user_scalers = {}
    label_to_params = estimator_params()
    model_labels = list(label_to_params.keys())
    params = list(label_to_params.values())

    idvar = "uid" if study=="one" else "id"
    users = set(df[idvar])
    
    for user in tqdm(users):
    
        df_user = df[df[idvar] == user]
        if len(df_user) == 0:
            continue
        
        accuracies, f1s, best_estimators = {}, {}, {}
        estimators = initialize_estimators()
        accs_by_diff_scores, accs_by_response_times = {}, {}

        for i, mlabel in enumerate((model_labels)):
            estimator = estimators[mlabel]
            accs, ests = [], []
            accs_by_diff, accs_by_time = [[], [], []], [[], []]
            for _ in range(reps):
    
                df_train, df_test = train_test_split(df_user, test_size=0.3)
                X_train = df_train.drop('chosen', axis=1)
                y_train = df_train['chosen']
                X_test = df_test.drop('chosen', axis=1)
                y_test = df_test['chosen']
                
                param = label_to_params[mlabel]
                model = Pipeline([('scalar', StandardScaler()), ('clf', estimator)])
                scoring = ['accuracy','f1_macro']
                model = RandomizedSearchCV(estimator=model, param_distributions=param, cv=5, scoring='accuracy', n_iter=10)
                # model = GridSearchCV(estimator=model, param_grid=param, cv=5, scoring='accuracy')
                model.fit(X_train[main_feat_cols].to_numpy(), y_train)
                best_estimator = model.best_estimator_
    
                accs.append(model.best_estimator_.score(X_test[main_feat_cols].to_numpy(), y_test))
                
                X_test = df_test[df_test['first_quantile'] == True].drop('chosen', axis=1)
                y_test = df_test[df_test['first_quantile'] == True]['chosen']
                accs_by_diff[0].append(model.best_estimator_.score(X_test[main_feat_cols].to_numpy(), y_test))

                X_test = df_test[df_test['second_quantile'] == True].drop('chosen', axis=1)
                y_test = df_test[df_test['second_quantile'] == True]['chosen']
                accs_by_diff[1].append(model.best_estimator_.score(X_test[main_feat_cols].to_numpy(), y_test))

                X_test = df_test[df_test['third_quantile'] == True].drop('chosen', axis=1)
                y_test = df_test[df_test['third_quantile'] == True]['chosen']
                accs_by_diff[2].append(model.best_estimator_.score(X_test[main_feat_cols].to_numpy(), y_test))
                
                X_test = df_test[df_test['first_half'] == True].drop('chosen', axis=1)
                y_test = df_test[df_test['first_half'] == True]['chosen']
                accs_by_time[0].append(model.best_estimator_.score(X_test[main_feat_cols].to_numpy(), y_test))

                X_test = df_test[df_test['first_half'] == False].drop('chosen', axis=1)
                y_test = df_test[df_test['first_half'] == False]['chosen']
                accs_by_time[1].append(model.best_estimator_.score(X_test[main_feat_cols].to_numpy(), y_test))

                ests.append(best_estimator)
            
            best_estimators[mlabel] = list(ests)
            accuracies[mlabel] = (np.mean(accs), np.std(accs))
            accs_by_diff_scores[mlabel] = (np.mean(accs_by_diff[0]), np.mean(accs_by_diff[1]), np.mean(accs_by_diff[2]))
            accs_by_response_times[mlabel] = (np.mean(accs_by_time[0]), np.mean(accs_by_time[1]))
        
        user_metrics[user] = [accuracies, accs_by_diff_scores, accs_by_response_times]
        # print (user, user_metrics[user])
        user_models[user] = dict(best_estimators)

    return user_metrics, user_models


def iter_predictive_modeling(df, study="one", reps=5):
    user_metrics = {}
    user_models = {}
    user_scalers = {}
    label_to_params = estimator_params()
    model_labels = list(label_to_params.keys())
    params = list(label_to_params.values())

    idvar = "uid" if study=="one" else "id"
    users = set(df[idvar])
    
    for user in tqdm(users):
    
        df_user = df[df[idvar] == user]
        if len(df_user) == 0:
            continue
        
        accuracies, f1s, best_estimators = {}, {}, {}
        estimators = initialize_estimators()

        for i, mlabel in enumerate((model_labels)):
            estimator = estimators[mlabel]
            accs, ests = [], []
            for _ in range(reps):
    
                df_train, df_test = train_test_split(df_user, test_size=0.3)
                X_train = df_train.drop('chosen', axis=1)
                y_train = df_train['chosen']
                X_test = df_test.drop('chosen', axis=1)
                y_test = df_test['chosen']
                
                param = label_to_params[mlabel]

                accs_by_iter = []

                steps = int(len(X_train)/10)
                for N in tqdm(range(steps, len(X_train), steps)):
                    est = copy.deepcopy(estimator)
                    model = Pipeline([('scalar', StandardScaler()), ('clf', est)])
                    model = RandomizedSearchCV(estimator=model, param_distributions=param, cv=3, scoring='accuracy', n_iter=10)
                    model.fit(X_train.to_numpy()[:N], y_train[:N])
                    best_estimator = model.best_estimator_
    
                    accs_by_iter.append(model.best_estimator_.score(X_test.to_numpy(), y_test))

                accs.append(list(accs_by_iter))
                ests.append(best_estimator)
            
            best_estimators[mlabel] = list(ests)
            accuracies[mlabel] = list(accs)
    
        user_metrics[user] = [accuracies, f1s]
        user_models[user] = dict(best_estimators)

    return user_metrics, user_models



from sklearn.base import BaseEstimator, ClassifierMixin

class Life_Exp_Rule(BaseEstimator, ClassifierMixin):
    def __init__(self, col):
        self.col = col

    def fit(self, X, y):
        return self
        
    def predict(self, X):
        preds = []
        for i, row in X.iterrows():
            if row[self.col] > row[self.col]:
                preds.append(1)
            elif row[self.col] < row[self.col]:
                preds.append(0)
            else:
                preds.append(np.random.randint(2))
        
        return np.array(preds)

class Choose_Left(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        return self
        
    def predict(self, X):
        preds = np.ones(len(X))        
        return np.array(preds)

def sign(num):
    return -1 if num < 0 else 1
        
class Prominent_Feat_Rule(BaseEstimator, ClassifierMixin):
    def __init__(self, all_cols):
        self.all_cols = all_cols
        self.ord_cols = None
        self.val_cols = None
    
    def fit(self, X, y):
        Xp = X[self.all_cols].to_numpy()
        coefs = LogisticRegression().fit(Xp, list(y)).coef_[0]
        self.val_cols = {self.all_cols[i]: sign(coefs[i]) for i in range(len(coefs))}
        
        coef_dict = dict(zip(self.all_cols, np.abs(coefs)))
        coef_dict = sorted(coef_dict.items(), key=lambda item: item[1], reverse=True)
        
        self.ord_cols = [col for col, coef in coef_dict]
        
    
    def predict(self, X):
        preds = []
        for i, row in X.iterrows():
            pred = -1
            for col in self.ord_cols:
                if row[col]*self.val_cols[col] > 0:
                    pred = 1
                if row[col]*self.val_cols[col] < 0:
                    pred = 0
                
                if pred != -1:
                    break
        
            preds.append(pred)            
        return np.array(preds)

        
class Tallying_Rule(BaseEstimator, ClassifierMixin):
    def __init__(self, all_cols, tie="prom"):
        self.all_cols = all_cols
        self.ord_cols = None
        self.val_cols = None
        self.tie = tie
    
    def fit(self, X, y):
        Xp = X[self.all_cols].to_numpy()
        coefs = LogisticRegression().fit(Xp, list(y)).coef_[0]
        self.val_cols = {self.all_cols[i]: sign(coefs[i]) for i in range(len(coefs))}
        
        coef_dict = dict(zip(self.all_cols, np.abs(coefs)))
        coef_dict = sorted(coef_dict.items(), key=lambda item: item[1], reverse=True)
        
        self.ord_cols = [col for col, coef in coef_dict]
        
    
    def predict(self, X):
        preds = []
        for i, row in X.iterrows():
            pred = -1
        
            tally_pos = sum([int(row[col]*self.val_cols[col] > 0) for col in self.all_cols])
            tally_neg = sum([int(row[col]*self.val_cols[col] < 0) for col in self.all_cols])
            tally = tally_pos - tally_neg
            if tally > 0:
                pred = 1
            if tally < 0:
                pred = 0
    
            if tally == 0:
                if self.tie == "prom":
                    for col in self.ord_cols:
                        if row[col]*self.val_cols[col] > 0:
                            pred = 1
                        if row[col]*self.val_cols[col] < 0:
                            pred = 0
                        
                        if pred != -1:
                            break
                            
                elif self.tie == "random":
                    pred = np.random.randint(2)
                    
            preds.append(pred)
        return np.array(preds)
        

def initialize_heuristics(diff_cols, life_col):

    choose_left = Choose_Left()
    life_exp_rule = Life_Exp_Rule(life_col)
    prom_feat_rule = Prominent_Feat_Rule(diff_cols)
    tallying_rule_random_tiebreak = Tallying_Rule(diff_cols, tie="random")
    tallying_rule_prom_feat_tiebreak = Tallying_Rule(diff_cols, tie="prom")
    # fft = FastFrugalTreeClassifier()
    
    return {"choose_left": choose_left, 
            "life_expectancy": life_exp_rule, 
            "prom_feat_rule": prom_feat_rule,
            "tallying_rule_random_tiebreak": tallying_rule_random_tiebreak, 
            "tallying_rule_prom_feat_tiebreak": tallying_rule_prom_feat_tiebreak,
            # "fft": fft
           }

def random_split_and_test(df, estimator):
    df_train, df_test = train_test_split(df, test_size=0.7)
    X_train = df_train.drop('chosen', axis=1)
    y_train = df_train['chosen']
    X_test = df_test.drop('chosen', axis=1)
    y_test = df_test['chosen']

    estimator.fit(X_test, y_test)
    return estimator.score(X_test, y_test)

def heuristic_modeling(df, reps=5, study="one"):
    user_metrics = {}
    user_models = {}
    diff_cols = [c for c in list(df.columns) if "diff" in c]
    
    idvar = "uid" if study=="one" else "id"
    life_col = "llife" if study=="one" else "l_lifeYearsGained"
    users = set(df[idvar])
    
    for user in tqdm(users):
    
        df_user = df[df[idvar] == user]
        if len(df_user) == 0:
            continue
        
        accuracies, best_heuristics = {}, {}
        heuristics = initialize_heuristics(diff_cols, life_col)

        for i, mlabel in enumerate((heuristics.keys())):
            # print (user, mlabel)
            estimator = heuristics[mlabel]
            accs = []
            for _ in range(reps):
                acc = random_split_and_test(df_user, estimator)
                accs.append(acc)

            accuracies[mlabel] = (np.mean(accs), np.std(accs))
            best_heuristics[mlabel] = [estimator]
    
        user_metrics[user] = [accuracies]
        user_models[user] = dict(best_heuristics)

    return user_metrics, user_models


from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
import random

class DiffClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, main_estimator, reg_columns=range(13,18), alpha=0.4):
        self.alpha = alpha
        self.main_estimator = main_estimator
        self.logreg_coefs = None
        self.reg_columns = reg_columns
    
    def fit(self, X, y):
        indices = list(range(len(X)))
        random.shuffle(indices)
        N = int(self.alpha * len(indices))
        X, y = np.array(X), np.array(y)
        
        X1, X2 = X[indices[:N]], X[indices[N:]]
        y1, y2 = y[indices[:N]], y[indices[N:]]

        X1_log = X1[:, self.reg_columns]
        logreg = LogisticRegression(fit_intercept=False).fit(X1_log, y1)
        self.logreg_coefs = np.asarray(logreg.coef_).T
        
        difficulties_X2 = np.abs(np.dot(X2[:, self.reg_columns], self.logreg_coefs))

        ## add diffuclty scores to the dataframe in new column
        X2 = np.c_[X2, difficulties_X2]

        self.main_estimator.fit(X2, y2)
        
        return self

    def predict(self, X):

        # check_is_fitted(self)
        X = check_array(X)
        
        difficulties_X = np.dot(X[:, self.reg_columns], self.logreg_coefs)
        X_new = np.c_[X, difficulties_X]
        return self.main_estimator.predict(X_new)


def weighted_predictive_modeling(df, study="one", reps=5, reg_columns=range(13,18)):
    user_metrics = {}
    user_models = {}
    user_scalers = {}
    label_to_params = estimator_params()
    model_labels = list(label_to_params.keys())
    params = list(label_to_params.values())

    idvar = "uid" if study=="one" else "id"
    users = set(df[idvar])
    
    for user in tqdm(users):
    
        df_user = df[df[idvar] == user]
        if len(df_user) == 0:
            continue
        
        accuracies, f1s, best_estimators = {}, {}, {}
        estimators = initialize_estimators()

        for i, mlabel in enumerate((model_labels)):
            estimator = estimators[mlabel]
            accs, ests = [], []
            for _ in range(reps):
                df_train, df_test = train_test_split(df_user, test_size=0.3)
                X_train = df_train.drop('chosen', axis=1)
                y_train = df_train['chosen']
                X_test = df_test.drop('chosen', axis=1)
                y_test = df_test['chosen']
                # print (X_train.columns[reg_columns])
                diff_estimator = DiffClassifier(main_estimator=estimator, reg_columns=reg_columns)
        
                param = dict(label_to_params[mlabel])
                for k in list(param.keys()):
                    param['clf__main_estimator__'+k[5:]] = param.pop(k)
                param['clf__alpha'] = stats.uniform(0.1, 0.4)

                model = Pipeline([('scalar', StandardScaler()), ('clf', diff_estimator)])
                model = RandomizedSearchCV(estimator=model, param_distributions=param, cv=5, scoring='accuracy', n_iter=10)
                model.fit(X_train.to_numpy(), y_train)
                best_estimator = model.best_estimator_
    
                accs.append(model.best_estimator_.score(X_test.to_numpy(), y_test))
                ests.append(best_estimator)
            
            best_estimators[mlabel] = list(ests)
            accuracies[mlabel] = (np.mean(accs), np.std(accs))
    
        user_metrics[user] = [accuracies, f1s]
        user_models[user] = dict(best_estimators)

    return user_metrics, user_models



