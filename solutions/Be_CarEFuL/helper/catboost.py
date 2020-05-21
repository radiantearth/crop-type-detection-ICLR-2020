import catboost as cat
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time 
from config import *
from helper.utils import print_time,random_seed_cpu
import copy

def _plot_importance_(importances, features_name, fig_size=(15, 15), top=50):
    importances = [
        pd.DataFrame(
            {
                "features": importance["Feature Id"],
                "importance_" + str(i): importance["Importances"],
            }
        )
        for i, importance in enumerate(importances)
    ]

    df = pd.DataFrame()
    df["features"] = features_name
    for importance in importances:
        df = df.merge(importance, on="features", how="left")
    df["importance"] = df.drop(["features"], 1).mean(axis=1)
    df.sort_values("importance", inplace=True, ascending=False)
    plt.figure(figsize=fig_size)
    sns.barplot(data=df[:top], y="features", x="importance")

def upsampling(class_to_samples,train,columns,coff=1):
    random_seed_cpu(RANDOM_STATE)
    final_train=[train]
    for _class_ in class_to_samples : 
        train_class=train[train.label==_class_]
        for col in columns : 
            train_class[col]=np.random.permutation(train_class[col].values)
        train_class=train_class.sample(frac=coff)
        final_train.append(train_class)
    final_train=pd.concat(final_train)
    return final_train

def get_data_set(train, target_name, features, cat_features, weight,other_params,fold_num=None):
    if fold_num != None:
        X_train = train.loc[train.fold != fold_num]
        X_val = train.loc[train.fold == fold_num]
#         print(len(X_train))
#         X_train=upsampling(other_params["class_to_samples"],X_train,other_params["columns"],other_params["coff"])
#         print(len(X_train))
        dtrain = cat.Pool(
            data=X_train[features],
            label=X_train[target_name],
            feature_names=features,
            cat_features=cat_features,
            weight=weight,
        )
        dval = cat.Pool(
            data=X_val[features],
            label=X_val[target_name],
            cat_features=cat_features,
            feature_names=features,
            weight=weight,
        )
    else:
        X_train = train.loc[train.validation == 0]
        X_val = train.loc[train.validation == 1]

        dtrain = cat.Pool(
            data=X_train[features],
            label=X_train[target_name],
            feature_names=features,
            cat_features=cat_features,
            weight=weight,
        )
        dval = cat.Pool(
            data=X_val[features],
            label=X_val[target_name],
            feature_names=features,
            cat_features=cat_features,
            weight=weight,
        )

    return dtrain, dval, X_train, X_val


def cat_single(
    train,
    test,
    target_name,
    features,
    params,
    metric,
    other_params,
    fold_num=None,
    plot_importance=True,
    print_result=True,
    predict_train=True,
):

    dtrain, dval, X_train, X_val = get_data_set(
        train, target_name, features, other_params["cat_features"],other_params["weight"],other_params ,fold_num
    )
    dtest = cat.Pool(
        data=test[features],
        feature_names=features,
        cat_features=other_params["cat_features"],
    )

    model = cat.CatBoost(params)
    model.fit(dtrain, eval_set=dval, use_best_model=True)
    train_pred = (
        model.predict(dtrain, other_params["prediction_type"])
        if predict_train
        else None
    )
    val_pred = model.predict(dval, prediction_type=other_params["prediction_type"])
    test_pred = model.predict(dtest, prediction_type=other_params["prediction_type"])
    if ((other_params["prediction_type"]=="Probability")& (other_params["num_class"]==1)) : 
        if predict_train : 
            train_pred=train_pred[:,1]
        val_pred=val_pred[:,1]
        test_pred=test_pred[:,1]
        
    val_score = metric(X_val[target_name], val_pred)
    train_score = metric(X_train[target_name], train_pred) if predict_train else -1

    if print_result:
        print(
            "final train score : {} -validation score: {}".format(
                str(round(train_score, 5)), str(round(val_score, 5))
            )
        )
    importances = [model.get_feature_importance(prettified=True)]

    if plot_importance:
        _plot_importance_(importances, features)
    return train_pred, val_pred, test_pred, train_score, val_score, importances, model

def cat_kfold(
    train, test, target_name, features, params, metric, other_params,
):
    models = {}
    oof_train = np.zeros((len(train), other_params["num_class"]))
    oof_test = np.zeros((len(test), other_params["num_class"]))
    importances = []
    validation_scores = []
    train_scores = []
    for fold in np.sort(train.fold.unique()):
        
        (
            train_pred,
            val_pred,
            test_pred,
            train_score,
            val_score,
            importance,
            model,
        ) = cat_single(
            train,
            test,
            target_name,
            features,
            params,
            metric=metric,
            other_params=other_params,
            fold_num=fold,
            plot_importance=False,
            print_result=other_params["print_result"],
            predict_train=other_params["predict_train"],
        )

        models[str(fold)] = model

        oof_train[train[train.fold == fold].index, :] += np.reshape(
            val_pred, (-1, other_params["num_class"])
        )
        oof_test += np.reshape(test_pred, (-1, other_params["num_class"]))

        train_scores.append(train_score)
        validation_scores.append(val_score)
        importances.extend(importance)

        if other_params["print_kfold_eval"] : 
            print(
                "Iteration : {}  - train score : {} - CV Score : {}".format(
                    str(fold + 1), str(train_score), str(val_score)
                )
            )
            print("=" * 80)
    oof_test /= train.fold.nunique()
    final_train_score = np.mean(train_scores)
    oof_score = metric(train[target_name], oof_train)

    if other_params["print_kfold_eval"] : 
        print(
            "ending  training  : train score {} - oof Score {}".format(
                str(final_train_score), str(oof_score)
            )
        )

    if other_params["plot_importance_kfold"]:
        _plot_importance_(importances, features)

    return oof_train, oof_test, final_train_score, oof_score, models


def cat_train(train, test, other_params):
    start_time=time.time()
    if other_params["use_kfold"]:

        oof_train, oof_test, final_train_score, oof_score, models = cat_kfold(
            train,
            test,
            other_params["target_name"],
            other_params["features"],
            other_params["params"],
            other_params["metric"],
            other_params,
        )
        finish_time=time.time()
        if other_params["print_time"] : print_time(finish_time-start_time)
        return oof_train, oof_test, final_train_score, oof_score, models
    else:


        (
            train_pred,
            val_pred,
            test_pred,
            train_score,
            val_score,
            importances,
            model,
        ) = cat_single(
            train,
            test,
            other_params["target_name"],
            other_params["features"],
            other_params["params"],
            other_params["metric"],
            other_params,
            fold_num=None,
            plot_importance=other_params["plot_importance"],
            print_result=True,
            predict_train=other_params["predict_train"],
        )
        finish_time=time.time()
        if other_params["print_time"] : print_time(finish_time-start_time)
        return (train_pred, val_pred, test_pred, train_score, val_score, model)
    
    
    
    
def cat_train_add_features(train, test, other_params):
    start_time=time.time()
    other_params_copy=copy.deepcopy(other_params)
    best_score=other_params["best_score"]
    best_validation_prediction=[]
    best_test_prediction=[]
    to_keep_features=[]
    results=[]
    for feature in other_params["added_features"] :
        result=pd.Series()
        features_to_test=other_params["features"]+to_keep_features+[feature]
        other_params_copy["features"]=features_to_test
        train_score,validation_score,validation_prediction,test_prediction,models=train_model(train, test, other_params_copy) 

        result["feature"]=feature
        result["train_score"]=train_score
        result["validation_score"]=validation_score
        result["best_iteration"]=np.mean([ model.get_best_iteration() for model in models ])
            

        if other_params["maximise"] : 
            if validation_score >= best_score :
                to_keep_features.append(feature)
                result["improved"]=1
                best_score=validation_score
                best_validation_prediction=validation_prediction
                best_test_prediction=test_prediction
            else : 
                result["improved"]=0
        else  : 
            if validation_score <= best_score :
                to_keep_features.append(feature)
                result["improved"]=1
                best_score=validation_score
                best_validation_prediction=validation_prediction
                best_test_prediction=test_prediction
            else : 
                result["improved"]=0
        result["model"]=models
        result["features_len"]=len(features_to_test)
            
        display(pd.DataFrame([result]))
        results.append(result)
    results=pd.DataFrame(results)
    finish_time=time.time()
    print_time(finish_time-start_time)
    return results ,to_keep_features ,best_validation_prediction,best_test_prediction


def cat_train_remove_features(train, test, other_params):
    start_time=time.time()
    other_params_copy=copy.deepcopy(other_params)
    best_score=other_params_copy["best_score"]
    to_remove_features=[]
    results=[]
    best_validation_prediction=[]
    best_test_prediction=[]
    for feature in other_params["removed_features"] :

        result=pd.Series()
        features_to_test=copy.deepcopy(other_params["features"])
#         features_to_test.remove(feature)
#         for rm_feature in to_remove_features : 
#             features_to_test.remove(rm_feature)
#         other_params_copy["features"]=features_to_test
        features_to_test=[feat for feat in features_to_test if not feat.endswith(feature) ]
        for rm_feature in to_remove_features : 
            features_to_test=[feat for feat in features_to_test if not feat.endswith(rm_feature) ]
        other_params_copy["features"]=features_to_test
        
        train_score,validation_score,validation_prediction,test_prediction,models=train_model(train, test, other_params_copy) 
            
        result["feature"]=feature
        result["train_score"]=train_score
        result["validation_score"]=validation_score
        result["best_iteration"]=np.mean([ model.get_best_iteration() for model in models ])

        if other_params["maximise"] : 
            if validation_score >= best_score :
                to_remove_features.append(feature)
                result["improved"]=1
                best_score=validation_score
                best_validation_prediction=validation_prediction
                best_test_prediction=test_prediction
            else : 
                result["improved"]=0
        else :
            if validation_score <= best_score :
                to_remove_features.append(feature)
                result["improved"]=1
                best_score=validation_score
                best_validation_prediction=validation_prediction
                best_test_prediction=test_prediction
            else : 
                result["improved"]=0
        result["features_len"]=len(features_to_test)
        result["model"]=models


        display(pd.DataFrame([result]))
        results.append(result)
    results=pd.DataFrame(results)
    finish_time=time.time()
    print_time(finish_time-start_time)
    return results ,to_remove_features,best_validation_prediction,best_test_prediction

def train_model(train, test, other_params) :
    if other_params["use_kfold"]:
        oof_train, oof_test, final_train_score, oof_score, models = cat_kfold(
            train,
            test,
            other_params["target_name"],
            other_params["features"],
            other_params["params"],
            other_params["metric"],
            other_params,
        )
        return final_train_score ,oof_score,oof_train,oof_test,list(models.values())
        
    else:

        (
            train_pred,
            val_pred,
            test_pred,
            train_score,
            val_score,
            importances,
            model,
        ) = cat_single(
            train,
            test,
            other_params["target_name"],
            other_params["features"],
            other_params["params"],
            other_params["metric"],
            other_params,
            fold_num=None,
            plot_importance=other_params["plot_importance"],
            print_result=other_params["print_result"],
            predict_train=other_params["predict_train"],
        )
        return train_score,val_score,val_pred,test_pred,[ model]