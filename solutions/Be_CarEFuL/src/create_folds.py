import  sys
sys.path.append("../")
from helper.utils import pd ,Print,random_seed_cpu
from helper.cross_validation import *
from config import *
from tqdm import tqdm
random_seed_cpu(RANDOM_STATE)

target=pd.read_pickle(join(proc_data_path,"target.p"))
target=target[~target.label.isna()]
CV = cross_validation(
    train_df=target,
    _id_="fid",
    target_name="label",
    kfold_type="skfold",
    output_dir=proc_data_path,
    split_ratio=0.1,
    nfolds=5,
    random_state=RANDOM_STATE,
    shuffle=True,
    stratify=False,
    tag="_fold_5",
    group_name="fid"
)
target = CV.split()
