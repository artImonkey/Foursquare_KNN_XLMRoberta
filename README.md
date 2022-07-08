# Foursquare_KNN_XLMRoberta
Using a dataset of over one-and-a-half million Places entries heavily altered to include noise, duplications, extraneous, or incorrect information, we'll produce an algorithm that predicts which Place entries represent the same point-of-interest. Each Place entry includes attributes like the name, street address, and coordinates.

## Main information
- The model from this project has been created for the ["Foursquare - Location Matching"](https://www.kaggle.com/competitions/foursquare-location-matching) competition on kaggle
- Main file is created in Ipython notebook format (.ipynb)
- Main model is transformer XLM-Roberta-base
- This project achives 0.829 [Intersection over Union](https://en.wikipedia.org/wiki/Jaccard_index) score

## How it works?
According to train dataset we saw that most of the Entries of the POI have close distance. So it's naturall to suspect that closest entries of each entry might be from the same POI. To realize that idea KNN is used. After we got neighbours of each point we have to understand wich nieghbour is close enough by distance, name and category to be related. To solve this problem I trained XLMRoberta transformer to predict if pair is match or not. XLMRoberta is chosen because we have multilingual names of entries and this model is pretrained on multilingual dataset. 
After we got probabilites of match between pairs, we can connect them, but the problem is if we have a lot of close entries, some of them can be missed, because we compare entry with just 7 closest neighbours. The idea to solve it is that we have graph with edges as probabilities of match between closest entries, so we can assume that if we have P(AB) = 0.9 and P(BC) = 0.8, we can get probability for P(AC) >= P(AB)*P(BC).

## Process in details and files
Work with preparing data, training and inferencing has been done in the following order:<br />
1) **generate_pairs.ipynb** - In this notebook we generate matching pairs according to train dataset.<br />
2) **generate_nonmatching_pairs.ipynb** - Here we generate nonmatching pairs, the idea is to generate pairs from the nighbourhood of each point to teach model making a difference between closest entries.<br />
3) **generate_folds_split.ipynb** - After the pairs are generated we need to split data to train\test parts and to avoid leakage test data cannot contain POI's from train data.<br />
4) **Foursquare_XLM-RoBERTa_train.ipynb** - Once we generated train\test split, we can train XLMRoberta.<br />
5) **Convert_train_dataset_for_scoring.ipynb** - Run this notebook to convert train dataset to more comfortable dataframe, we will use it to create dict and significantly reduce the speed of calculating IOU score.<br />
6) **Foursquare_inference_KNN_XLMRoberta.ipynb** - This notebook conatains main process of grouping data with KNN + XLMRoberta. 
