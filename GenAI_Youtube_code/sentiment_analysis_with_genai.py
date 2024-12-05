# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 13:11:22 2024

@author: PCAT
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from transformers import pipeline
from transformers import pipeline


senti_model = pipeline(task="sentiment-analysis")

senti_model("This movie is damn good. I loved it")

Senti_model_2 = pipeline(task="sentiment-analysis",
                         model="cardiffnlp/twitter-roberta-base-sentiment-latest")





import pandas as pd
user_review_data=pd.read_csv("https://raw.githubusercontent.com/venkatareddykonasani/Datasets/master/Amazon_Yelp_Reviews/Review_Data.csv")
user_review_data=user_review_data.sample(50)
user_review_data["Review"]





user_review_data["Predicted_Sentiment"] = user_review_data["Review"].apply(lambda x: Senti_model_2(x)[0]["label"])
user_review_data


Senti_model_2_gpu = pipeline(task="sentiment-analysis",
                         model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                         #device="cuda"
                         )



user_review_data["Predicted_Sentiment"] = user_review_data["Review"].apply(lambda x: Senti_model_2_gpu(x)[0]["label"])
user_review_data


