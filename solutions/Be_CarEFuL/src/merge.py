import sys
sys.path.append("../")
from helper.utils import *
from config import *

Print(" "*25+" Merge ")
nn_model=pd.read_csv(join(sub_path,"nn_model"))
cat_model=pd.read_csv(join(sub_path,"catboost_sub"))
cat_model.sort_values("fid",inplace=True)
nn_model.sort_values("fid",inplace=True)
columns=cat_model.drop(["fid"],1).columns
sub=cat_model[["fid"]]


nn_weight=0.6
cat_weight=0.4


sub[columns]=nn_model[columns]*nn_weight+cat_model[columns]*cat_weight

target=pd.read_pickle(join(proc_data_path,"target.p"))
target_counts=(target.label.value_counts(True)*len(sub)).round()
ratios={'Crop_ID_1': 1.19,'Crop_ID_2': 1.33,'Crop_ID_4': 1.46,'Crop_ID_5': 2.02,'Crop_ID_6': 1.31,'Crop_ID_3': 2.05,'Crop_ID_7': 2.55}
target_name=['Crop_ID_1', 'Crop_ID_2', 'Crop_ID_3', 'Crop_ID_4', 'Crop_ID_5',
       'Crop_ID_6', 'Crop_ID_7']
for label in target_counts.index : 
    column_name="Crop_ID_{}".format(int(label+1))
    other_columns=[ name for name in target_name if name != column_name]
    sub=sub.sort_values(column_name,ascending=False)
    count=int(target_counts.loc[label]/2.5)
    sub_id=sub.fid.tolist()[0:count]
    sub.loc[sub.fid.isin(sub_id),column_name]*=ratios[column_name]
Print("save final sub to "+join(sub_path,"final_sub.csv"))
sub.to_csv(join(sub_path,"final_sub.csv"),index=False)