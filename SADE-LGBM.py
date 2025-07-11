"""
Author: Yangdi Shen
Email: shenyd98@163.com
Coding date: 20:25 on 10 May 2024
Corresponding to paper: https://doi.org/10.1016/j.swevo.2025.102013
"""

from smt.surrogate_models import RBF
import xgboost as xgb
import pandas as pd
import numpy as np
import math
import time
import statistics
import csv
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from smt.surrogate_models import IDW
from smt.surrogate_models import RMTB
from smt.surrogate_models import LS
from smt.surrogate_models import QP
from smt.surrogate_models import KRG
from smt.surrogate_models import KPLS
from smt.surrogate_models import KPLSK
from smt.surrogate_models import SGP
from smt.surrogate_models import MGP
import lightgbm as lgb
from keras import backend as K

# Dataset loading
train = pd.read_csv('file path')
test = pd.read_csv('file path')

# Split into training and testing data
y_test = test.iloc['Rows and columns of y_test']
y_train = train.iloc['Rows and columns of y_train']
x_train = train.iloc['Rows and columns of x_train']
x_test = test.iloc['Rows and columns of x_test']

# Experiments with 30 Runs
for k in range(30):

    # Parameters initialization
    NP = 50
    HP_Dim = 10
    Ftrs_Dim = 26
    Dim = HP_Dim + Ftrs_Dim
    F = 0.5
    Cr = 0.5
    pop = []
    Total_Iterations = 100
    factor = 0.7

    # Initialize population
    for i in range(NP):
        HP = [
            np.random.randint(31, 150) # num_leaves
            , np.random.randint(1, 20) # max_depth
            , np.random.uniform(0, 1) # learning_rate
            , np.random.randint(100, 200) # n_estimators
            , np.random.uniform(0, 1) # min_split_gain
            , np.random.uniform(0, 1) # min_child_weight
            , np.random.randint(1,20) # min_child_samples
            , np.random.uniform(1e-6, 1)  # colsample_bytree
            , np.random.uniform(0, 1)  # reg_alpha
            , np.random.uniform(0, 1)  # reg_lambda
        ]

        Ftrs = [np.round(np.random.rand(1), 1)[0] for i in range(Ftrs_Dim)]
        HP_F = HP + Ftrs
        pop.append(HP_F)

    # Boundaries of LightGBM hyperparameters
    HP_bounds = [(31, 150) # num_leaves
        , (1, 20) # max_depth
        , (1e-6, 1) # learning_rate
        , (100, 200) # n_estimators
        , (0, 1)  # min_split_gain
        , (0, 1)  # min_child_weight
        , (1,20) # min_child_samples
        , (1e-6, 1)  # colsample_bytree
        , (0, 1)  # reg_alpha
        , (0, 1)  # reg_lambda
        ]

    Ftrs_bounds = [(0, 1)] * Ftrs_Dim
    bounds = HP_bounds + Ftrs_bounds

    # Object function
    def objf(HP_F):
        HP_dic = {
            'num_leaves': int(HP_F[0])
            , 'max_depth': int(HP_F[1])
            , 'learning_rate': float(HP_F[2])
            , 'n_estimators': int(HP_F[3])
            , 'min_split_gain': float(HP_F[4])
            , 'min_child_weight': float(HP_F[5])
            , 'min_child_samples': int(HP_F[6])
            , 'colsample_bytree': float(HP_F[7])
            , 'reg_alpha': float(HP_F[8])
            , 'reg_lambda': float(HP_F[9])
        }

        # Feature value transformation
        list = HP_F.copy()
        for o in range(len(list[len(HP):])):
            if list[len(HP)+o] < 0.7:
                list[len(HP)+o] = 1
            else:
                list[len(HP)+o] = 0
        random_col = np.where(np.array([round(i) for i in list[len(HP):]]) == 1)[0].tolist()

        # Calculate fitness (RMSE)
        x_train_selected = train.iloc[:, random_col]
        x_test_selected = test.iloc[:, random_col]

        LGBR = lgb.LGBMRegressor(**HP_dic,verbose=-1)

        LGBR.fit(x_train_selected, y_train)

        y_pred = LGBR.predict(x_test_selected)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        return rmse

    # SADE
    fitness = []
    for i in range(NP):
        fitness.append(objf(pop[i]))
    Evo_num = int(factor * NP)

    for n in range(Total_Iterations):
        v = [[0 for i in range(Dim)] for i in range(NP)]
        u = [[0 for i in range(Dim)] for i in range(NP)]

        for i in range(NP):

            # Mutation
            for j in range(Dim):
                r1, r2, r3 = np.random.choice(NP, 3, replace=False)
                v[i][j] = pop[r1][j] + (pop[r2][j] - pop[r3][j]) * F

                if v[i][j] < bounds[j][0]:
                    v[i][j] = bounds[j][0]
                elif v[i][j] > bounds[j][1]:
                    v[i][j] = bounds[j][1]
            # Crossover
            for j in range(Dim):
                if np.random.random() < Cr:
                    u[i][j] = v[i][j]
                else:
                    u[i][j] = pop[i][j]

        # Surrogate model assist process
        indv_train = np.array(pop)
        fitness_train = np.array(fitness)
        indv_test =  np.array(u)

        SM = RBF(d0=100,print_global=False)
        SM.set_training_values(indv_train, fitness_train)
        SM.train()

        fitness_pred = SM.predict_values(indv_test)

        real_list = np.argsort(fitness)[::-1][:Evo_num]
        fake_list = np.argsort(fitness_pred.flatten())[:Evo_num]

        for m in range(len(real_list)):
            pop[real_list[m]] = u[fake_list[m]]
            fitness[real_list[m]] = objf(u[fake_list[m]])

        print(n,np.min(fitness))
        u.clear()
        v.clear()

    # Performance metrics calculation
    HP_F = pop[np.argmin(fitness)]
    HP_dic = {
        'num_leaves': int(HP_F[0])
        , 'max_depth': int(HP_F[1])
        , 'learning_rate': float(HP_F[2])
        , 'n_estimators': int(HP_F[3])
        , 'min_split_gain': float(HP_F[4])
        , 'min_child_weight': float(HP_F[5])
        , 'min_child_samples': int(HP_F[6])
        , 'colsample_bytree': float(HP_F[7])
        , 'reg_alpha': float(HP_F[8])
        , 'reg_lambda': float(HP_F[9])
    }
    for o in range(len(HP_F[len(HP):])):
        if HP_F[len(HP) + o] < 0.7:
            HP_F[len(HP) + o] = 1
        else:
            HP_F[len(HP) + o] = 0
    random_col = np.where(np.array([round(i) for i in HP_F[len(HP):]]) == 1)[0].tolist()

    x_train_selected = train.iloc[:, random_col]
    x_test_selected = test.iloc[:, random_col]

    LGBR = lgb.LGBMRegressor(**HP_dic, verbose=-1)

    LGBR.fit(x_train_selected, y_train)

    y_pred = LGBR.predict(x_test_selected)

    y_ypred = y_test - y_pred
    var_y = statistics.variance(y_test)
    var_yy = statistics.variance(y_ypred, statistics.mean(y_ypred))
    vaf = 1 - var_yy / var_y
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    corr = np.corrcoef(y_test, y_pred)[0, 1]
    r2 = r2_score(y_test, y_pred)
    aic = np.log10(mean_squared_error(y_test, y_pred)) * len(y_train) + 2 * (len(y_test) + 1)
    bic = np.log10(mean_squared_error(y_test, y_pred)) * len(y_train) + np.log10(len(y_train)) * (len(y_test) + 1)


    print('RMSE = ', rmse)
    print('MAE = ', mae)
    print('R2:', r2)
    print('correlation:', corr)
    print('VAF:', vaf)
    print("AIC:", aic)
    print("BIC:", bic)

