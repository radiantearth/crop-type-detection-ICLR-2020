# First Place Award for An African Citizen

This solution won the the **First place award for an African citizen currently residing on the continent** and acheived the second top score of 1.168877091 in the competiton. 

## Winner

Femi Sotonwa from Nigeria ([GitHub](https://github.com/youngtard), [Linkedin](https://www.linkedin.com/in/femi-sotonwa//))


## Approach
I used two different approaches.  
The first approach involved training with 3 set of features:
* Image pixel values 
* About 10 vegetation/spectral indices (e.g. NDVI, AVI etc.), and their relevant statistics 
* Spatial features (e.g area of farm etc.).  

The second approach involved training with only pixel values, and their relevant statistics.  
My solution is an ensemble weighted average of the two approaches.

## Modelling
The two approaches each went through the same modelling process by using a CatboostClassifier (without class_weights), another CatboostClassifier (with class_weights to take care of class imbalance), and a LinearDiscriminant algorithm (known in sklearn as LinearDiscriminantAnalysis - LDA ). LDA is a weak learner, so in order to improve it's performance, I bagged (ensemble) it using sklearn's BaggingClassifier. The weighted Catboost and bagged LDA added some diversity to the modelling due to the highly imbalanced dataset, and in general improved performance.

## How to Run

Install the requirements:

```
pip install -r requirements.txt
```

Open jupyter notebook, and run `solution.ipynb`.