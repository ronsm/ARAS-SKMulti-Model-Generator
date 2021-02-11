# ARAS SKMulti Model Generator

This repository provides code to pre-process and train on data from the [ARAS Dataset](https://www.cmpe.boun.edu.tr/aras/), which includes 30 days of binary sensor data from two smart homes. Citation below.

```
H. Alemdar, H. Ertan, O.D. Incel, C. Ersoy, ARAS Human Activity Datasets in Multiple Homes
with Multiple Residents, Pervasive Health, Venice, May 2013. 
```

The purpose of this software is to enable real-time predictions from multiple models on streaming data from the ARAS dataset, to simulate a real smart home.

## Usage

The ```dataset_preprocessing.py``` file allows you to select which house you wish to generate a dataset for. Once run, the script will generate (non-shuffled) files: ```all.csv.```, ```train.csv```, and ```test.csv``` inside the ```datasets/house/x``` directory. The following parameters are configurable, but are set to these defaults:
* House B is the default house
* Train/test split is 70/30
* The labels for the second resident are removed, leaving 20 features and a single label for the first residents

The ```train.py``` file implements training on streaming data with various models (currently: Hoeffding Tree Classifier, Naïve Bayes, and RUS Boost). It takes command line arugments for training sample limits. Usage: ```python3 train.py model train_limit```, example usage: ```python3 skmultiflow_run.py 0```, where options are:
* ```train_limit``` : 0 to use all available data, else uses specified amount