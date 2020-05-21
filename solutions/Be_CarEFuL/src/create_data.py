import sys
sys.path.append("../")
from config import *
import datetime
from tqdm import tqdm
from helper.utils import load_file ,random_seed_cpu , pd,Print
random_seed_cpu(RANDOM_STATE)
Print(" "*25+"create data ")


Print("1-Extract boundaries and  field ID  from tiles")
row_locs = []
col_locs = []
field_ids = []
labels = []
tiles = []
for tile in range(4):

    fid_arr = load_file(join(raw_data_path,"data","0"+str(tile),str(tile)+"_field_id.tif"))
    lab_arr = load_file(join(raw_data_path,"data","0"+str(tile),str(tile)+"_label.tif"))
    for row in range(len(fid_arr)):
        for col in range(len(fid_arr[0])):
            if fid_arr[row][col] != 0:
                row_locs.append(row)
                col_locs.append(col)
                field_ids.append(fid_arr[row][col])
                labels.append(lab_arr[row][col])
                tiles.append(tile)

data = pd.DataFrame({
    'fid':field_ids,
    'label':labels,
    'row_loc': row_locs,
    'col_loc':col_locs,
    'tile':tiles
})



dates = [datetime.datetime(2019, 6, 6, 8, 10, 7),
         datetime.datetime(2019, 7, 1, 8, 10, 4),
         datetime.datetime(2019, 7, 6, 8, 10, 8),
         datetime.datetime(2019, 7, 11, 8, 10, 4),
         datetime.datetime(2019, 7, 21, 8, 10, 4),
         datetime.datetime(2019, 8, 5, 8, 10, 7),
         datetime.datetime(2019, 8, 15, 8, 10, 6),
         datetime.datetime(2019, 8, 25, 8, 10, 4),
         datetime.datetime(2019, 9, 9, 8, 9, 58),
         datetime.datetime(2019, 9, 19, 8, 9, 59),
         datetime.datetime(2019, 9, 24, 8, 9, 59),
         datetime.datetime(2019, 10, 4, 8, 10),
         datetime.datetime(2019, 11, 3, 8, 10)]
dates=[str(d.date()).replace("-","") for d in dates]
Print("dates : "+" ".join(dates))
Print("bands : "+" ".join(bands))

Print("2-Extract pixel values for each band")
finla_data=[]
for tile in tqdm(range(4)) :
    tile_folder="0"+str(tile)
    template=data[data.tile==tile].copy()
    rows_indx,columns_indx=template["row_loc"].tolist(),template["col_loc"].tolist()
    for date in dates :
        tile_date_data=template.copy()
        folder_date=join(tile_folder,date)
        for band in bands :
            file_name=str(tile)+"_"+band+"_"+date+".tif"
            file_path=join(raw_data_path,"data",folder_date,file_name)
            image=load_file(file_path)
            tile_date_data[band]=image[rows_indx,columns_indx]
        
        tile_date_data["date"]=date
        finla_data.append(tile_date_data)
        
finla_data=pd.concat(finla_data)
finla_data["train"]=1
finla_data.loc[finla_data.label==0,"train"]=0
finla_data.loc[finla_data.train==0,"label"]=None
finla_data.label-=1
target=finla_data[["fid","label"]]
target.drop_duplicates(["fid"],inplace=True)
Print("3-Save Data to data.p and targets to target.p")
target.to_pickle(join(proc_data_path,"target.p"))
finla_data.drop(["label"],1,inplace=True)
finla_data.to_pickle(join(proc_data_path,"data.p"))
    
    