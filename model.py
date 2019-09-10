import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor #GBM algorithm
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import   LinearRegression, Ridge, ElasticNet, Lasso
from sklearn.neighbors import NearestCentroid, RadiusNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.model_selection  import cross_validate, cross_val_score,GridSearchCV
from sklearn import metrics 
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer,accuracy_score,confusion_matrix, matthews_corrcoef
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils import resample

import csv
import pickle
import os

target = pd.read_csv("TARGET/od_matrix_new.csv")
chicago_tracts = target["CTID"]
scoring_method = "neg_mean_absolute_error"


# num_feature = 50

for num_feature in range(50, 310, 10):
    target_names = ["P_TRIP", "P_TRIP_W", "A_TRIP","A_TRIP_W", "P_SHARE_PCT", "P_SHARE_PCT_W"]

    for target_name in target_names:

        outputfolder = "Results/%s_%s"%(target_name,num_feature)
        if not os.path.exists("empty_trip_models"):
            os.mkdir("Results")

        if not os.path.exists(outputfolder+"/model_results.csv"):
            print target_name, num_feature
            print "-"*80
            if not os.path.exists(outputfolder):                
                os.mkdir(outputfolder)
            result_corr = []
            indepent_variables = ["ALLTRANSIT", "BUSINESS_LIC", "CRASH", "CRIME", "PARKING", "POPULATION", "RAC", "SMART_LOCATION", "WAC", "WI"]
            for _indepent_variable in indepent_variables:
                df = pd.read_csv("INDEPENDENT_V/CLEAN_%s.csv"%_indepent_variable)
                df = df.merge(chicago_tracts, left_on = "CTID", right_on = "CTID", how = "right")

                for _variable in list(df):
                    if _variable != "CTID":
                        df[_variable].astype("float")
                        corr = target[target_name].corr(df[_variable])
                        result_corr.append([_indepent_variable, _variable, corr])


            df_corr= pd.DataFrame(result_corr, columns = ["datasource", "variable", "correlation"])
            df_corr["correlation_abs"] = df_corr["correlation"].abs()
            df_corr.sort_values(by=['correlation_abs'],ascending=False,inplace = True)
            df_corr.to_csv("%s_correlation_ranking.csv"%target_name,index = False)

            # get top variables
            df_corr = df_corr[:num_feature]


            X = target[["CTID",target_name]]
            X.dropna(axis = 0, inplace = True)
            X = X[X[target_name]>0]
            
            for idx, row in df_corr.iterrows():
                df =  pd.read_csv("INDEPENDENT_V/CLEAN_%s.csv"%row["datasource"])
                df = df[["CTID", row["variable"]]]
                X = X.merge(df, left_on = "CTID", right_on = "CTID", how = "left")


            X.drop(["CTID"], axis = 1, inplace = True)


            y = X[target_name]
            y_missing = y[y.isna()]
            # print y_missing
            X.drop([target_name], axis = 1, inplace = True)

            #STANDARDIZE X
            X=(X-X.mean())/X.std()

            X.fillna(0, inplace = True)
            # train machine learning...

            modellist = [
            DecisionTreeRegressor(random_state=9),
            ExtraTreeRegressor(random_state =9),
            KNeighborsRegressor(),
            SVR(),
            LinearRegression(),
            Ridge(random_state = 9),
            ElasticNet(random_state = 9),
            Lasso(random_state = 9),
            # MLPRegressor(),
            RandomForestRegressor(random_state = 9),
            GradientBoostingRegressor()
            ]

            modelnames = [
            "DecisionTreeRegressor",
            "ExtraTreeRegressor",
            "KNeighborsRegressor",
            "SVR",
            "LinearRegression",
            "Ridge",
            "ElasticNet",
            "Lasso",
            # "MLPRegressor",
            "RandomForestRegressor",
            "GradientBoostingRegressor",
            ]

            tuning = [
            {'max_features':range(7,30,2), 
                'max_depth':range(1,11,2), 
                'min_samples_split':range(2,30,5),
                'min_samples_leaf':range(1,20,5)}, # "DecisionTreeRegressor",
            {'max_features':range(7,30,2), 
                'max_depth':range(1,11,2), 
                'min_samples_split':range(2,30,5),
                'min_samples_leaf':range(1,20,5)}, # "ExtraTreeRegressor",
            {'n_neighbors':range(1,20,2), 'p':[1,2,3]},# "KNeighborsRegressor",
            {'kernel':['linear', 'rbf', 'poly'],'gamma':[0.1, 1, 10], 'C': [0.1, 1, 10], 'degree':range(0,3)},# "SVR",
            # {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ["l1", "l2"] },# "LogisticRegression",
            {'fit_intercept':[True, False]}, #linear regression
            {'alpha':[0.01, 0.1, 1, 10]},#ridge
            {'alpha':[0.01, 0.1, 1, 10], 'l1_ratio':[0,0.2,0.4,0.6,0.8,1]},#elastic net
            {'alpha':[0.01, 0.1, 1, 10]},#lasso
            {'n_estimators':range(10, 100, 20), 
            'max_features':range(10,30,5), 
            'max_depth':range(1,10,2), 
            'min_samples_split':range(2,5),
            'min_samples_leaf':range(5,10)},# "RandomForestClassifier",
            # {'alpha':[0.001, 0.01, 1, 5 ,10] },# "RidgeClassifier",
            {'n_estimators':range(10, 100, 20), 
            'max_features':range(10,30,5), 
            'max_depth':range(1,10,2), 
            'min_samples_split':range(2,5),
            'min_samples_leaf':range(5,10)}# "GradientBoosting",

            ]



            with open(outputfolder+"/model_results.csv", "wb") as outputfile:
                csvwriter = csv.writer(outputfile)
                csvwriter.writerow(["model", "accuracy +/- std"])
                for idx, model in enumerate(modellist):
                    # print modelnames[idx]
                    param_test = tuning[idx]
                    gsearch = GridSearchCV(estimator = model, 
                                          param_grid = param_test, 
                                          scoring=scoring_method, 
                                          n_jobs=-1,
                                          iid=False, 
                                          cv=10,
                                          return_train_score=True)
                    gsearch.fit(X,y)
                    best_index = gsearch.best_index_

                    print "parameters", gsearch.best_params_

                    with open(outputfolder+'/%s_best_model.pickle'%modelnames[idx], 'wb') as handle:
                        pickle.dump(gsearch.best_estimator_,handle)

                    parameters = gsearch.best_params_
                    for parameter in parameters:
                        parameters[parameter] = [parameters[parameter]]
                    parameters = pd.DataFrame(parameters)

                    parameters.to_csv(outputfolder+"/%s_best_params.csv"%(modelnames[idx]))

                    print modelnames[idx], "%.3g +/- %.3g"%(gsearch.best_score_, gsearch.cv_results_["std_test_score"][best_index])
                    csvwriter.writerow([modelnames[idx], "%.3g +/- %.3g"%(-gsearch.best_score_, gsearch.cv_results_["std_test_score"][best_index])])


            print "="*80