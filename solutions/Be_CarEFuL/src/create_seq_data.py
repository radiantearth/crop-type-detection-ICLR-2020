import sys
sys.path.append("../")
from helper.utils import Print,random_seed_cpu,pd,save_pickle
from config import *
from tqdm import tqdm
from sklearn.preprocessing import QuantileTransformer

random_seed_cpu(RANDOM_STATE)
Print(" "*25+" Create sequence data ")

Print("1-read data and add NDVI NDSI ")
data = pd.read_pickle(join(proc_data_path,"data.p"))
data["NDVI"]=(data["B08"]-data["B04"])/(data["B08"]+data["B04"])
data["NDSI"]=(data["B07"]-data["B04"])/(data["B07"]+data["B04"])
bands=bands[:-1]
bands.append("NDVI")
bands.append("NDSI")
Print("bands : "+" ".join(bands))

Print("2-Apply Quantile Transformer normalization to the bands ")
qt=QuantileTransformer(output_distribution="normal",random_state=RANDOM_STATE)
data[bands]=qt.fit_transform(data[bands])

Print("3-Convert flatten data to sequence data")
data["id"]=data["fid"].astype(str)+"_"+data["row_loc"].astype(str)+"_"+data["col_loc"].astype(str)
data.sort_values(["id","date"],inplace=True)
ids=data.drop_duplicates("id").id.values
iter_numb=int(len(data)/13)
inter=range(iter_numb)
feautres=data[bands].values
features_id={}
for _id_,i in zip(ids,inter) : 
    features_id[_id_]=feautres[i*13:(i+1)*13,:]
Print("4-Save sequence data to seq_data.p")
save_pickle(features_id,join(proc_data_path,"seq_data.p"))