import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import random 
import seaborn  as sns
import gc 
import calendar
import pickle 
import os
from sklearn.preprocessing import StandardScaler
from os.path import join

from sklearn.metrics import confusion_matrix
from IPython.display import clear_output, Image, display, HTML
from datetime import datetime

from IPython.display import display


plt.style.use('fivethirtyeight')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.options.display.float_format = '{:.4f}'.format

def missing_data(data):
    total = data.isnull().sum()
    percent = (data.isnull().sum()/data.isnull().count()*100)
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    tt['Types'] = types
    return(np.transpose(tt))
#######################################################################
def read_Df(Path,name="data",format='pickle'):
    reader=eval("pd.read_"+format)
    data=reader(Path)
    print(name)
    display(data_characterization(data))
    return data
#########################################################################
def data_characterization(data):
    print("shape of data : "+str(data.shape))
    data_characterization=pd.DataFrame()
    columns=data.columns
    Type=[]
    Count=[]
    unique_values=[]
    Max=[]
    Min=[]
    Mean=[]
    Nan_counts=data.isnull().sum().tolist()
    Nan_ratio=(data.isnull().sum()/len(data)).values

    Type=data.dtypes.tolist()
    J=0
    for  i  in columns : 
        unique=list(data[i].unique())
        unique_values.append(unique)
        Count.append(len(unique))
        
        
        if (data[i].dtypes.name == 'object') :
            Max.append(0)
            Min.append(0)
            Mean.append(0)
        elif ( (data[i].dtypes == '<M8[ns]') ) : 
            Max.append(0)
            Min.append(0)
            Mean.append(0)
        elif ( (data[i].dtype.name=="category") ) : 
            Max.append(0)
            Min.append(0)
            Mean.append(0)

        else : 
            Max.append(data[i].max())
            Min.append(data[i].min())
            Mean.append(data[i].mean())
   
    data_characterization["Columns name"]=columns
    data_characterization["Type "]=data.dtypes.tolist()
    data_characterization["Count unique values"]=Count
    data_characterization["Count Nan values"]=Nan_counts
    data_characterization["Ratio Nan values"]=Nan_ratio

    data_characterization["Unique   values"]=unique_values
    data_characterization["Max"]=Max
    data_characterization["Min"]=Min
    data_characterization["Mean"]=Mean
    
    display(data_characterization)    
    return None 
#########################################################################
def Label_encoder(data): 
    data_new=data.copy()
    categoria_features=data_new.columns[data.dtypes == 'object']
    labels={}
    for col in categoria_features : 
        fact=data_new[col].factorize()
        data_new[col]= fact[0]
        labels[col]=fact[1]
        
    return data_new ,labels

#################################################################################
def visualisation_data(data,labels): 
    for  i  in data.columns : 
        data[i].plot.hist(bins=60)
        plt.title(i)
        if i  in labels.keys():
            plt.xticks(np.arange(len(labels[i])), labels[i].tolist(), rotation=90)
        plt.show()
###################################################################################
def correlation_matrix_color_bar(df):


    fig = plt.figure()
    fig.set_size_inches(18.5, 10.5)
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title('Abalone Feature Correlation')
    labels=df.columns.tolist()
    ax1.set_xticklabels(labels,fontsize=20)
    ax1.set_yticklabels(labels,fontsize=20)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ticks=[.75,.8,.85,.90,.95,1])
    plt.show()
    
    
def correlation_matrix_pandas(data):
    def magnify():
        return [dict(selector="th",
                 props=[("font-size", "9pt")]),
            dict(selector="td",
                 props=[('padding', "0em 0em")]),
            dict(selector="th:hover",
                 props=[("font-size", "12pt")]),
            dict(selector="tr:hover td:hover",
                 props=[('max-width', '200px'),
                        ('font-size', '12pt')])
    ]
    corr=data.corr()
    cmap = cmap=sns.diverging_palette(5, 250, as_cmap=True)
    return corr.style.background_gradient(cmap, axis=1)\
        .set_properties(**{'max-width': '80px', 'font-size': '10pt'})\
        .set_caption("Hover to magify")\
        .set_precision(2)\
        .set_table_styles(magnify())
###########################################################
def xl_date_to_simple_date(excel_date):

    dt = datetime.fromordinal(datetime(1900, 1, 1).toordinal() + excel_date - 2)
# tt = dt.timetuple()
    return int(dt.strftime('%Y%m%d'))
###################################################################
def get_column_ratio(data,column):
    nbrs_unique_values=data["data"].value_counts()
    nbrs_unique_values.to_dict()
    for key  in nbrs_unique_values.keys():
        print("ratio of "+str(key)+" : " +str(nbrs_unique_values[key]/float(len(data_ano))))
    return
################################################################################################
# from sklearn.model_selection import  train_test_split 
# import xgboost as xgb

# def get_importance_features(X,Y):
    
#     X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=1994 )
#     dtrain = xgb.DMatrix(X_train, y_train,feature_names=X.columns.values)
#     dval = xgb.DMatrix(X_val, y_val,feature_names=X.columns.values)
#     xgb_params = {
#                 'eta': 0.1,
#                 'max_depth': 25,
#                 'subsample': 0.9,
#                 'colsample_bytree': 0.9,
#                 'objective': 'binary:logistic',
#                 'seed' : 10,
#                 'shuffle': True,
#                 'silent':1 ,
#                 'n_jobs':-1
#                  }
# #     watchlist = [(dtrain, 'train'), (dval, 'test')]
#     model = xgb.train(xgb_params,dtrain,num_boost_round=50)

#     xgb.plot_importance(model, height=0.5)
#     plt.show()
#     return model 
    
    
########################################################################################


# def strip_consts(graph_def, max_const_size=32):
#     """Strip large constant values from graph_def."""
#     strip_def = tf.GraphDef()
#     for n0 in graph_def.node:
#         n = strip_def.node.add() 
#         n.MergeFrom(n0)
#         if n.op == 'Const':
#             tensor = n.attr['value'].tensor
#             size = len(tensor.tensor_content)
#             if size > max_const_size:
#                 tensor.tensor_content = "<stripped %d bytes>"%size
#     return strip_def

# def show_graph(graph_def, max_const_size=32):
#     """Visualize TensorFlow graph."""
#     if hasattr(graph_def, 'as_graph_def'):
#         graph_def = graph_def.as_graph_def()
#     strip_def = strip_consts(graph_def, max_const_size=max_const_size)
#     code = """
#         <script>
#           function load() {{
#             document.getElementById("{id}").pbtxt = {data};
#           }}
#         </script>
#         <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
#         <div style="height:600px">
#           <tf-graph-basic id="{id}"></tf-graph-basic>
#         </div>
#     """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))

#     iframe = """
#         <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
#     """.format(code.replace('"', '&quot;'))
#     display(HTML(iframe))   
    
    
# def factorize_features(catgo_features,list_data):
#     for c in catgo_features:
#         raw_vals = np.unique(list_data[0][c])
#         val_map = {}
#         for i in range(len(raw_vals)):
#             val_map[raw_vals[i]] = i 
#         for data in list_data :
#                 data[c]=data[c].map(val_map)
#     return list_data 


def StandardScaler_features(features_to_standar,data):
    scaler = StandardScaler()
    for c in features_to_standar:
        data[c]=scaler.fit_transform(data[c].values.reshape((-1,1)))
    return data

def StandardMax_features(features_to_standar,data):

    for c in features_to_standar:
        data[c]=data[c]/float(data[c].max())
    return data


def week_of_month(tgtdate):
    tgtdate = tgtdate.to_datetime()

    days_this_month = calendar.mdays[tgtdate.month]
    for i in range(1, days_this_month):
        d = datetime.datetime(tgtdate.year, tgtdate.month, i)
        if d.day - d.weekday() > 0:
            startdate = d
            break
    # now we canuse the modulo 7 appraoch
    return (tgtdate - startdate).days //7 + 1
    
def confusion_matrix_plot(y_true,y_pred,classs) :
    conf_arr=confusion_matrix(y_true,y_pred)
    norm_conf = []
    for i in conf_arr:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j)/float(a))
        norm_conf.append(tmp_arr)

    fig=plt.figure(figsize = (10,7))
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_title("Confusion Matrix")    


    # ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.Blues, 
                    interpolation='nearest')

    width, height = conf_arr.shape

    for x in range(width):
        for y in range(height):
            ax.annotate(str(conf_arr[x][y]), xy=(y, x), 
                        horizontalalignment='center',
                        verticalalignment='center',
                        size=25)

    cb = fig.colorbar(res)
    alphabet = classs

    plt.xticks(range(width), alphabet[:width])
    plt.yticks(range(height), alphabet[:height])
    plt.savefig('confusion_matrix.png', format='png')
    plt.show()   
    return
def info(object, spacing=5, collapse=1):
    """Print methods and doc strings.
    
    Takes module, class, list, dictionary, or string."""
    methodList = [method for method in dir(object) if callable(getattr(object, method))]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print( "\n".join(["%s %s" %
                      (method.ljust(spacing),
                       processFunc(str(getattr(object, method).__doc__)))
                     for method in methodList]))
def Save_df(data,path):
    data.to_pickle(path)
    print("DF was saved in :"+str(path))
    
    
import multiprocessing 
    
def _apply_df(args):
    dfs, func = args
    return func(dfs)

def apply(df, func,workers):

    print("is working")
    pool = multiprocessing.Pool(processes=workers,maxtasksperchild=500)
    result = pool.map(_apply_df, [(d, func)
           for d in df ])
                      
    pool.close()
    return result
def get_List_from_group(df):
    L=list(df)
    a=[data[1] for data in L]
    return a   
def multithreading(df,func ,workers=40):
    print("create  list of DataFrame")
    L=get_List_from_group(df)
    result=apply(L,func ,workers)
    del L 
    return result
   
# def factorize_features(catgo_features,data):
#     dict_map={}
#     for c in catgo_features:
#         raw_vals = np.unique(data[c])
#         val_map = {}
#         for i in range(len(raw_vals)):
#             val_map[raw_vals[i]] = i 
#         data[c]=data[c].map(val_map)  
#         dict_map[c]=val_map
#     return data ,dict_map 
def save_pickle(data , file_name):
    
    with open(file_name,"wb") as fil : 
        pickle.dump(data,fil)

def read_pickle(file_name):
     with open(file_name,"rb") as fil : 
        return pickle.load(fil)

    
def Create_year_woy_column(Data,Date_name,name=""):
    Data["date"]=pd.to_datetime(Data[Date_name],format="%Y%m%d")
    Data["year_woy"+name]=  Data["date"].dt.year*100+ Data["date"].dt.weekofyear
    del Data["date"]
############################  Noramization ########################################################""
def  Quantile_Transformer(Data,columns=None,train_test=True,random_state=1994):
    from sklearn.preprocessing import QuantileTransformer
    qt=QuantileTransformer(output_distribution="normal", random_state=0,subsample =len(Data[0]))
    if train_test :
        qt_fit=qt.fit(pd.concat(Data,axis=0)[columns])
    else :  
        qt_fit=qt.fit(Data[0][columns])
    for df in Data  : 
        df[columns]=qt_fit.transform(df[columns])
    
#####################################################################################################
def is_categorical(array_like):
    return array_like.dtype.name == 'category'
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if (col_type != object)&(col_type.name!="category")&(col_type != '<M8[ns]'):
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        elif  (col_type == object) :
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df
        
def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
        
from sklearn.model_selection import KFold
def normal_funcation(train,test,var,vars_be_agg,func,fillnan):
    agg=train.groupby(var)[vars_be_agg].agg(func)
    if isinstance(var, list):
        agg.columns = pd.Index([vars_be_agg+"_by_" + "_".join(var) + "_" + str(e) for e in agg.columns.tolist()])
    else:
        agg.columns = pd.Index([vars_be_agg+"_by_" + var + "_" + str(e) for e in agg.columns.tolist()]) 
    agg.reset_index(inplace=True)
    train = pd.merge(train,agg, on=var, how= "left")
    test = pd.merge(test,agg, on=var, how= "left")
    if fillnan : 
        for col in agg.columns  :  
            test[col].fillna(agg[col].mean(),inplace=True)
            train[col].fillna(agg[col].mean(),inplace=True)
    return train ,test
            
    
def bagging_function(train,test,var,vars_be_agg,func,ID,fillnan,n_folds,seed=2018,shuffle=False): 
    np.random.seed(seed)
    skf = KFold(n_splits=n_folds, shuffle=shuffle, random_state=seed)
    bagging_train=[]
    for train_index, test_index in skf.split(train.index.values):
        train_fold=train[train.index.isin(train_index)]
        test_fold=train[train.index.isin(test_index)]
        agg=train_fold.groupby(var)[vars_be_agg].agg(func)
        if isinstance(var, list):
            agg.columns = pd.Index([vars_be_agg+"_by_" + "_".join(var) + "_" + str(e) for e in agg.columns.tolist()])
        else:
            agg.columns = pd.Index([vars_be_agg+"_by_" + var + "_" + str(e) for e in agg.columns.tolist()]) 
        
        test_fold=test_fold.merge(agg,on=var, how= "left")    
        if fillnan : 
            for col in agg.columns  :  
                test_fold[col].fillna(agg[col].mean(),inplace=True)
                

        bagging_train.append(test_fold[agg.columns.tolist()+[ID]])
        del test_fold ,train_fold, agg
        gc.collect()
    bagging_train=pd.concat(bagging_train)
    train=train.merge(bagging_train,how="left",on=ID)
    agg=train.groupby(var)[vars_be_agg].agg(func)
    if isinstance(var, list):
        agg.columns = pd.Index([vars_be_agg+"_by_" + "_".join(var) + "_" + str(e) for e in agg.columns.tolist()])
    else:
        agg.columns = pd.Index([vars_be_agg+"_by_" + var + "_" + str(e) for e in agg.columns.tolist()]) 
    test=test.merge(agg,on=var, how= "left")    
    if fillnan : 
        for col in agg.columns  :  
            test[col].fillna(agg[col].mean(),inplace=True)
    return train ,test 
def aggr_func(train,test,vars_to_agg,vars_be_agg,
              func=["mean"],
              ID="ID",
              fillnan=True,
              bagging=True,
              n_folds=5,
             seed=1994,
             shuffle=False):

    if bagging : 
        for var in vars_to_agg : 
            train,test=bagging_function(train=train,
                                     test=test,
                                     var=var,
                                     vars_be_agg= vars_be_agg,
                                     ID=ID,
                                     func=func, 
                                     fillnan=fillnan,
                                     n_folds=n_folds,
                                     seed=seed,
                                       shuffle=shuffle)
    else : 
        for var in vars_to_agg : 
            train,test=normal_funcation(train=train,
                                     test=test,
                                     var=var,
                                     vars_be_agg= vars_be_agg,
                                    func=func, 
                                     fillnan=fillnan
                                     )
    return train ,test       
        
class  normalization() :
    def GaussianRank() :
        return None
    def Quantile_transform(Data, columns=None, train_test=False, random_state=1245):
        from sklearn.preprocessing import QuantileTransformer
        qt = QuantileTransformer(output_distribution="normal", random_state=random_state, subsample=len(Data[0]))
        if train_test:
            qt_fit = qt.fit(pd.concat(Data, axis=0)[columns])
        else:
            qt_fit = qt.fit(Data[0][columns])
        for df in Data:
            df[columns] = qt_fit.transform(df[columns])
        return Data
    def MinMax(Data, columns=None, train_test=False,feature_range=(-1,1)) :
        from sklearn.preprocessing import MinMaxScaler
        qt = MinMaxScaler(feature_range=feature_range)
        if train_test:
            qt_fit = qt.fit(pd.concat(Data, axis=0)[columns])
        else:
            qt_fit = qt.fit(Data[0][columns])
        for df in Data:
            df[columns] = qt_fit.transform(df[columns])
        
        return Data
    def StanderScaler (Data, columns=None, train_test=False) :
        from sklearn.preprocessing import StandardScaler
        qt = StandardScaler()
        if train_test:
            qt_fit = qt.fit(pd.concat(Data, axis=0)[columns])
        else:
            qt_fit = qt.fit(Data[0][columns])
        for df in Data:
            df[columns] = qt_fit.transform(df[columns])
        return Data
    
def map_categorical_feautres(columns=None,Data=[]):
    mapping_dict={}
    assert( isinstance(Data,list)) ,"data must be  List "
    for col in columns  : 
        print(col)
        unique=[]
        for data in Data : 
            unique.extend(data[col].unique().tolist()) 
        unique=list(set(unique))
        mapp_dict={}
        for i in range(len(unique))  :
            mapp_dict[unique[i]]=i
        for data in Data :  
            data[col]=data[col].map(mapp_dict)
        mapping_dict[col]=mapp_dict
    return mapping_dict 
def save_train_test(train,test,version):
    train.to_csv("../data/proc_data/train_{}.csv".format(version),index=False)
    test.to_csv("../data/proc_data/test_{}.csv".format(version),index=False) 
def load_train_test(version):
    train=pd.read_csv("../data/proc_data/train_{}.csv".format(version))
    test=pd.read_csv("../data/proc_data/test_{}.csv".format(version)) 
    return train ,test


def save_experiment(sub_name,train_score,validation_score,mean_train,note):
        try : 
            
            record=pd.read_csv("../experiments.csv")
        except : 
            record = pd.DataFrame()
        new_record=pd.DataFrame({"name":[sub_name],
                                "train_score":[train_score],
                                "val_score":[validation_score],
                                "test_score":[mean_train], 
                                "note":[note]})
        record=pd.concat([record,new_record])
        record.to_csv("../experiments.csv",index=False)
        
        
def make_sub_one_class(test, oof_test, Id_name, target_names, path):
    sub = test[[Id_name]]
    sub[target_names] = oof_test
    sub.to_csv(path, index=False)


def make_sub_multi_class(test, oof_test, Id_name, target_names, path):
    sub = test[[Id_name]]
    for i, class_name in enumerate(target_names):
        sub[class_name] = oof_test[:, i]
    sub.to_csv(path, index=False)
    
    
def save_oof_one_class(
    train,
    oof_train,
    test,
    oof_test,
    target_name,
    Id_name,
    oof_train_path,
    oof_test_path,
    score,
    model_name
):
    oof_train_df = train[[Id_name]].copy()
    oof_train_df[target_name] = oof_train

    oof_test_df = test[[Id_name]].copy()
    oof_test_df[target_name] = oof_test
    oof_train_df.to_csv(
        join(oof_train_path, "{}_{}.csv".format(model_name,str(round(score, 3)))), index=False
    )
    oof_test_df.to_csv(
        join(oof_test_path, "{}_{}.csv".format(model_name,str(round(score, 3)))), index=False
    )


def save_oof_multi_class(
    train,
    test,
    oof_train,
    oof_test,
    target_names,
    Id_name,
    oof_train_path,
    oof_test_path,
    score,
    model_name
):
    oof_train_df = train[[Id_name]].copy()
    for i, class_name in enumerate(target_names):
        oof_train_df[class_name] = oof_train[:, i]

    oof_test_df = test[[Id_name]].copy()
    for i, class_name in enumerate(target_names):
        oof_test_df[class_name] = oof_test[:, i]
    oof_train_df.to_csv(
        join(oof_train_path, "{}_{}.csv".format(model_name,str(round(score, 3)))), index=False
    )
    oof_test_df.to_csv(
        join(oof_test_path, "{}_{}.csv".format(model_name,str(round(score, 3)))), index=False
    )
    
    
def print_time(x):
    hors=x//(60*60)
    minuts=(x-(hors*60*60))//60
    seconds=(x-(hors*60*60)-(minuts*60))
    output_hors="" if hors==0 else "{} hours ".format(hors)
    output_minuts="" if minuts==0 else "{} minuts ".format(minuts)
    output_seconds="" if seconds==0 else "{} seconds ".format(round(seconds))
    print("train took: ",output_hors+output_minuts+output_seconds)

    
def fill_predictions_df(df,predections,columns) :
    for i,col in enumerate(columns) : 
        df[col]=predections[:,i]
    return df 
    
def random_seed_cpu_gpu(seed_value: int = 42) -> None:
    import torch 
    use_cuda = torch.cuda.is_available()
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    random.seed(seed_value)  # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        
def random_seed_cpu(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
    
import tifffile as tiff
def load_file(fp):
    """Takes a PosixPath object or string filepath
    and returns np array"""
    
    return tiff.imread(fp.__str__())
def Print(x):
    print("#"*70)
    print(x)
    print("#"*70)
