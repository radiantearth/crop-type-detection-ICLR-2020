# Crop Type Detection Competition at CV4A Workshop, ICLR 2020


The objective of this competition was to create a machine learning model to classify fields by crop type from images collected during the growing season by the Sentinel-2 satellite. The ground reference data used to generate the training dataset are from  western Kenya, and collected by the PlantVillage team.

The dataset contained a total of more than 4,000 fields. The satellite imagery includes 12 bands of observations from Sentinel-2 L2A product (observations in the ultra-blue, blue, green, red; visible and near-infrared (VNIR); and shortwave infrared (SWIR) spectra), as well as a cloud probability layer. The bands are mapped to a common 10x10m spatial resolution grid.

Western Kenya, where the data was collected is dominated by smallholder farms, which is common across Africa, and poses a challenge to build crop type classification from Sentinel-2 data. Moreover, the training dataset has a significant class imbalance.

This competition was part of the [Computer Vision for Agriculture (CV4A) Workshop](https://www.cv4gc.org/cv4a2020/) at the 2020 ICLR conference and was designed and organized by [Radiant Earth Foundation](www.radiant.earth) with support from [PlantVillage](plantvillage.psu.edu) in providing the ground reference data. Competition was run by [Zindi](https://zindi.africa/) on their platform ([competition link](https://zindi.africa/competitions/iclr-workshop-challenge-2-radiant-earth-computer-vision-for-crop-recognition/data))


## Results and Solutions

The evaluation metric for the competition was Cross Entropy with binary outcome for each crop:

![cost function](/_figures/cost_function.png)

The following table shows the competition scores of the award winners. We have also included Overall Accuracy score (%) as well as the accuracy score (%) for each crop type separately. These were not used to rank the submissions, and main criteria metric was the Cross Entropy function.  


|Team 	| Competition Score 	| Overall Accuracy 	| Maize| Cassava| Common Bean| Maize & Common Bean (intercropping)| Maize & Cassava (intercropping)| Maize & Soybean (intercropping)| Cassava & Common Bean (intercropping)|
|-------|-----------------------|-------------------|-------|-------|-------|-------|-------|-------|-------|
|KarimAmer 	| 1.102264609 		| 60.0 | 81.6 | 81.8 | 23.8 | 16.3 | 0.0 | 0.0| 0.0|
|youngtard 	| 1.168877091 		| 57.8 | 84.5 | 68.4 | 9.5 | 14.8 | 2.7 | 3.0 | 8.8|
|Be_CarEFuL 	| 1.174099923 	| 58.9 | 87.5 | 66.4 | 23.8 | 14.8 | 2.7 | 0.0 | 8.8|
|Threshold 	| 1.176934328 		| 58.5 | 85.9 | 68.7 | 9.5 | 15.8 | 0.0 | 3.0 | 8.8|
|AnsemChaieb 	| 1.177508763 	| 63.1 | 89.8 | 82.1 | 4.8 | 14.8 | 1.4 | 1.5 | 2.9|


Source code of each of the winners is included in the [solutions](/solutions/) folder. 


## About Radiant Earth Foundation

<img src="/_figures/radiantearth.png" width="305" height="88">

Founded in 2016, [Radiant Earth Foundation](www.radiant.earth) is a nonprofit organization focused on empowering organizations and individuals globally with open Machine Learning (ML) and Earth observations (EO) data, standards and tools to address the world’s most critical international development challenges. With broad experience as a neutral entity working with commercial, academic, governmental and non-governmental partners to expand EO data and information used in the global development sector, Radiant Earth Foundation recognizes the opportunity that exists today to advance new applications and products through use of ML.

To fill this need, Radiant Earth has established Radiant MLHub as an open ML commons for EO. Radiant MLHub is the first open digital data repository that allows anyone to discover and access high-quality EO training datasets. In addition to discovering others’ data, individuals and organizations can use Radiant MLHub to register or share their own training data, thereby maximizing its reach and utility. Furthermore, Radiant MLHub maps all of the training data that it hosts so stakeholders can easily pinpoint geographical areas from which more data is needed.


## About PlantVillage

<img src="/_figures/plantvillage.png" width="305">

[PlantVillage](plantvillage.psu.edu) is a research and development unit of Penn State University that empowers smallholder farmers and seeks to lift them out of poverty using cheap, affordable technology and democratizing the access to knowledge that can help them grow more food.

## About Zindi

![Zindi Logo](/_figures/zindi.png)

[Zindi](https://zindi.africa/) is Africa’s first and largest data science competition platform, growing and supporting an entire data science and AI ecosystem of scientists, engineers, academics, companies, NGOs, governments and institutions.

With roots firmly in Africa, our mission is to build a vibrant ecosystem where data scientists and society at large can create data and artificial intelligence collaborations, focused on the issues that matter most to them. We aim to be the primary source for data science and machine learning expertise on the continent, and the go-to platform for African data scientists to improve and showcase their skills, prepare themselves for the job market, and connect with each other and potential opportunities.