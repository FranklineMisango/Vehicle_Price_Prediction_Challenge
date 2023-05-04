# Vehicle price prediction challege - ResNet & CART + XGBoost

## Introduction

![](./images/cover.jpg)

This project encompasses combination of a Deep Learning model and a more accurate CART + XGBoost Model to predict a vehicle car price after a user attempts to upload it 


## Datasets

* ResNet Training : Dataset Source [Stanford Car Dataset](https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset).

* CART  + XGBoost : Dataset Source [Car Prices Dataset](https://www.kaggle.com/datasets/sidharth178/car-prices-dataset)

## Architecture

### ReNet
This model uses the ResNet PTM for Deep Learning which has an architecture as below 
![](./images/resnet.png)

### CART
The price prediction model uses Various CART models with the most accurate picked for final usage
![](./images/forests.png)

## Accuracy & Loss evaluation 

![FineTuning](images/Accuracy.png)
![FineTuning](images/loss.png)


## Installation
* Unzip the `Car_detection.zip`
* Install requirements.txt file with the command `pip install -r requirements.txt`
* Run `pipenv shell`
* Run `streamlit run app.py`
* The Model's website should open in a new Browser window 
![Mask image](images/)

