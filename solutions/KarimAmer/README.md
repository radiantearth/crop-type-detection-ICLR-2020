# First Place Overall

This solution acheived the top score in the competiton. 

## Winner

Karim Amer from Egypt ([GitHub](https://github.com/karimmamer), [Linkedin](https://www.linkedin.com/in/karim-amer-42188a6b/))


## Getting Started

A summarized description of the approach can be found [here](https://zindi.africa/competitions/iclr-workshop-challenge-2-radiant-earth-computer-vision-for-crop-recognition/discussions/1147).

### Prerequisites

Firstly, you need to have 

* Ubuntu 18.04 
* Python3
* 20 GB RAM
* 11 GB GPU RAM

Secondly, you need to install the challenge data and sample submission file by the following the instructions [here](https://zindi.africa/competitions/iclr-workshop-challenge-2-radiant-earth-computer-vision-for-crop-recognition/data).

Thirdly, you need to install the dependencies by running:

```
pip3 install -r requirements.txt
```

## Running

### Dataset Preparation

```
python3 prepare_data.py --data_path ...
```

This step generates patches around each crop field in the data and saves all of them in a numpy matrix along side their ground truth labels.

### Generating a Submission File

```
python3 main.py --data_path ...
```

This step trains an ensemble of 10 instances of the same DL model on different train/valid splits then generate a submission file with results on test set. 

All augmentations are used except for Mixup augmentation. In order to use it, run

```
python3 main.py --data_path ... --mixup_augmentation True
```

However it uses a lot of RAM (~50 GB) so I wouldn't recommend using it.

