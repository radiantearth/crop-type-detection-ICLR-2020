# Crop Type Detection Challenge at CV4A Workshop, ICLR 2020


The objective of this competition was to create a machine learning model to classify fields by crop type from images collected during the growing season by the Sentinel-2 satellite. The ground reference data used to generate the training dataset are from  western Kenya, and collected by the PlantVillage team.

The dataset contains a total of more than 4,000 fields. The satellite imagery includes 12 bands of observations from Sentinel-2 L2A product (observations in the ultra-blue, blue, green, red; visible and near-infrared (VNIR); and short wave infrared (SWIR) spectra), as well as a cloud probability layer. The bands are mapped to a common 10x10m spatial resolution grid.

Western Kenya, where the data was collected is dominated by smallholder farms, which is common across Africa, and poses a challenge to build crop type classification from Sentinel-2 data. Moreover, the training dataset has a significant class imbalance.

This competition was part of the Computer Vision for Agriculture (CV4A) Workshop at the 2020 ICLR conference and was designed and organized by Radiant Earth Foundation with support from PlantVillage in providing the ground reference data. Competition was run by Zindi on their platfrom ([competition link](https://zindi.africa/competitions/iclr-workshop-challenge-2-radiant-earth-computer-vision-for-crop-recognition/data))


## Results

|Team 	| Competition Score 	| Overall Accuracy 	|
------------------------------------------------------
|KarimAmer 	| 1.102264609 		|  |
|youngtard 	| 1.168877091 		|  |
|Be_CarEFuL 	| 1.174099923 		|  |
|Threshold 	| 1.176934328 		|  |
|AnsemChaieb 	| 1.177508763 		|  |

## About Radiant Earth Foundation

<img src="/_figures/radiantearth.png" width="305" height="88">

Founded in 2016, [Radiant Earth Foundation](www.radiant.earth) is a nonprofit organization focused on empowering organizations and individuals globally with open Machine Learning (ML) and Earth observations (EO) data, standards and tools to address the world’s most critical international development challenges. With broad experience as a neutral entity working with commercial, academic, governmental and non-governmental partners to expand EO data and information used in the global development sector, Radiant Earth Foundation recognizes the opportunity that exists today to advance new applications and products through use of ML.

To fill this need, Radiant Earth has established Radiant MLHub as an open ML commons for EO. Radiant MLHub is the first open digital data repository that allows anyone to discover and access high-quality EO training datasets. In addition to discovering others’ data, individuals and organizations can use Radiant MLHub to register or share their own training data, thereby maximizing its reach and utility. Furthermore, Radiant MLHub maps all of the training data that it hosts so stakeholders can easily pinpoint geographical areas from which more data is needed.


## About PlantVillage

![PlantVillage Logo](/_figures/plantvillage.png)

[PlantVillage](plantvillage.psu.edu) is a research and development unit of Penn State University that empowers smallholder farmers and seeks to lift them out of poverty using cheap, affordable technology and democratizing the access to knowledge that can help them grow more food.