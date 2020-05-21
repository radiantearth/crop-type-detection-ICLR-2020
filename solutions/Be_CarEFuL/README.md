# Second place overall award

This solution won the **Second place overall award** and acheived the top score of 1.174099923 in the competiton. 

## Winners

Mohamed Jedidi from Tunisia ([GitHub](https://github.com/JedidiMohamed), [Linkedin](https://www.linkedin.com/in/mohamed-salam-djedidi/)) and Lawrence Moruye from Kenya ([Linkedin](https://www.linkedin.com/in/lawrence-moruye-40203715a/)).



## Solution
For the winning solution, I relied on deep learning algorithms since this task is almost a computer vision task. However, common computer vision architectures didn’t perform well. My first solution consists of training Denoing autoencoder (swap row noising ) and taking the encoded original input (the latent layer output) and passing them through a deep neural network to get the final probability of the crop type. For a second attempt, I mixed deep learning with traditional machine learning by taking the  DAE latent layer output and passing them through the LGBM model to get the probability of the crop type. Finally, I ensembled the two outcomes of both solutions to get the final score. 


## How to run 

Install the requirements:

```
pip install -r requirements.txt
./create_folders.sh
```

Put the unziped tif data per folder (00,01,02,03) in a folder called  data under data/raw_data/.
The path for each tile should be: 
```
	data/raw_data/data/00/
	data/raw_data/data/01/
	data/raw_data/data/02/
	data/raw_data/data/03/
```

Change `absolut_path="/workspace/Zindi/ICLR_2"` in config.py with the absolute path to this folder on your machine.

The trained model weights are included in the folder `data/`. If you want to re-train the model, uncomment the line `python  -m src.nn_train` in `run.sh` otherwise run the following for inference:

```
./run.sh

```
