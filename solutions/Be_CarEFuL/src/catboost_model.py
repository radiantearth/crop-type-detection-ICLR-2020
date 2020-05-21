import sys

sys.path.append("../")
from helper.utils import *

from helper.catboost import *
from config import *

random_seed_cpu(RANDOM_STATE)

is_debug = False
nrows = 1000
Print(" "*25 +"Catboost model")


if is_debug:
    train = pd.read_pickle(
        join(proc_data_path, "train_catboost.csv"), nrows=nrows
    )
    test = pd.read_pickle(
        join(proc_data_path, "test_catboost.csv"), nrows=nrows / 2
    )
else:
    train = pd.read_pickle(join(proc_data_path, "train_catboost.p"))
    test = pd.read_pickle( join(proc_data_path, "test_catboost.p"))
Print("Load data ")


target_name = "label"
Id_name = "fid"
prediction_names=["Crop_ID_"+str(i) for i in range(1,8)]
features_to_remove = [target_name, 
                      Id_name,
                      "validation", "fold","date","train"
                     ]+prediction_names
features = [
    feature for feature in train.columns.tolist() if feature not in features_to_remove
]
Print("features len "+str(len(features)))
Print("features  "+" , ".join(features))


from sklearn.metrics import log_loss
def metric(x, y):
    return log_loss(x, y)

params = {
    "loss_function": "MultiClass",
    "eval_metric": "MultiClass",
    "learning_rate": 0.01,
    "random_seed": RANDOM_STATE,
    "l2_leaf_reg": 3,
    "bagging_temperature": 1,  # 0 inf
    "rsm":1,
    "depth": 6,
    "od_type": "Iter",
    "od_wait": 50,
    "thread_count": 16,
    "iterations": 50000,
    "verbose_eval": False,
    "use_best_model": True,

}

cat_features = []
cat_features = [
    train[features].columns.get_loc(c) for c in cat_features if c in train[features]
]

other_params = {
    "prediction_type": "Probability",  # it could be RawFormulaVal ,Class,Probability
    "cat_features": cat_features,
    "print_result": False,  # print result for a single model should be False whene use_kfold==True
    "plot_importance": False,  # plot importance for single model should be false whene use_kfold==True
    "predict_train": True,  # predict train for the single model funcation False only whene  use_kfold==True
    "num_class": 7,
    "target_name": target_name,
    "features": features,
    "metric": metric,
    "params": params,
    "use_kfold": True,  # condtion to use kfold or single model
    "plot_importance_kfold": False,  # plot importance after K fold train
    "print_kfold_eval": True,  # print evalation in kfold mode
    "weight":None,
    "print_time":True,
    "class_to_samples":[0,1,2,3,4,5,6],
    "coff":0.05,
    "columns":None
}

Print("Train model")
if other_params["use_kfold"]:
    oof_train, test_pred, final_train_score, oof_score, models = cat_train(
        train, test, other_params
    )
    validation=fill_predictions_df(train,oof_train,prediction_names)
else:
    train_pred, val_pred, test_pred, train_score, val_score, model = cat_train(
        train, test, other_params
    )
    validation=fill_predictions_df(train[train.validation==1],val_pred.reshape((-1,1)),prediction_names)
    
    
def mean_min_max_loss(x):
    Series=pd.Series()
    try : 
        Series["date"]=x["date"].iloc[0]
    except :
        pass 
    Series["mean_loss"]=metric(x["label_mean"],x[[col+"_mean" for col in prediction_names]])
    Series["min_loss"]=metric(x["label_mean"],x[[col+"_min" for col in prediction_names]])    
    Series["max_loss"]=metric(x["label_mean"],x[[col+"_max" for col in prediction_names]])  
    return Series

val_fild=validation.groupby(["fid"])[prediction_names+[target_name]].agg(["mean","max","min"])
val_fild.columns=[col[0]+"_"+col[1] for col in val_fild.columns]
val_fild.reset_index(inplace=True)
loss_per_fid_df=pd.DataFrame(mean_min_max_loss(val_fild))
score=loss_per_fid_df.loc["mean_loss",0]
Print("validation score :"+str(score))

test=fill_predictions_df(test,test_pred,prediction_names)
sub=test.groupby(["fid"])[prediction_names].mean()
sub.reset_index(inplace=True)

sub_name = "catboost_sub"
Print("save sub to /data/sub/catboost_sub")
sub.to_csv(join(sub_path,sub_name),index=False)

