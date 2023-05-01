<p align="center">
  <b style="font-size: 36px;">FORECAST ELECTRICITY CONSUMPTION IN AUSTRALIA</b>
</p>


## Abstract
The aim of this project was to investigate whether temperature
and seasonal features could be used to accurately predict energy demand, with
and without using prior demand as a dependent variable. Several models were
assessed in this project under multiple scenarios, including linear models, random
forests, LightGBM and Long Short-Term Memory models. 


<div align="center">
  <img alt="AEMO 5 Min Set" src="https://user-images.githubusercontent.com/127566032/235395087-863b7ab4-a87d-4376-af82-c900ede0837b.jpg" width="700" />
  (https://www.aemc.gov.au/sites/default/files/content/04db4725-8144-444f-b3b9-0276e7bbaf87/6-infographic.PDF)
</div>
<br>

The findings indicated that for long-term forecasting, within the range of weeks to months, an ensemble
of four Light Gradient Boost models delivered accurate and time-efficient results
(R2=0.85, RMSE=519.63, MAPE=5.07%). 

At the same time, for short-term forecasting within the range of minutes to hours, a Unidirectional Long Short-Term
Memory model, accompanied by a 5-fold time series cross-validation, along with
dropout and regularisation, delivered optimal results (R2=0.93, RMSE=325.32,
MAPE=3.16%).

## Introduction
In the past decade, high electricity costs have negatively impacted Australians. The unstable national grid is affected by spot market fluctuations and inconsistent supply. Accurate forecasting tools can help generator firms prepare for demand spikes, easing grid pressure and reducing price surges. The project's aim was to find optimal energy demand forecasting models. Surprisingly, an ensemble of four Light Gradient Boost models provided accurate, time-efficient results for horizons up to 16 months. This enables federal and state governments to confidently forecast energy demand, aiding in planning for future energy requirements.

## Literature Review
Climate change, with weather extremes like heatwaves and droughts, significantly impacts energy supply, demand, and infrastructure. Temperature is a crucial factor in forecasting electricity demand, as shown by an 11% or more increase in India's electricity demand at temperatures of 30°C and above. Accurate forecasting of energy demand with temperature as a predictor is beneficial for managing energy supply.

Solar power adds complexity to electricity demand forecasting due to intraday variations in supply and demand. More energy is generated during the day, while the grid is more heavily relied upon in the evening. Australia's national grid could face instability by 2024 due to system overload, and electricity prices vary depending on buying strategies, leading to high costs for spot buyers.

Several statistical techniques have been used to forecast energy demand, including regression, random forests, boosting trees, and artificial neural networks. Existing models typically include prior demand as a dependent variable. This study aimed to identify optimal models for forecasting energy demand from temperature and seasonal features, both with and without using prior demand as a dependent variable.

Models were developed using linear models, random forests, LightGBM, and LSTM, and assessed under various scenarios. Coefficient of Determination (R²), Root Mean Square Error (RMSE), and Mean Absolute Percentage Error (MAPE) were used to measure accuracy.

## Material and Methods
Google Colab was chosen as the coding environment. Python was used as the programming language. Scikit-learn, Light Gradient Boosting Machine, Keras, and TensorFlow libraries were utilized to build various models. GitHub, an online platform for data and code sharing, was used to store source data and code for this project. 

The datasets consisted of electricity demand and air temperature information. After preprocessing, the data was transformed into a usable format, and the two datasets were joined. Data cleaning was performed to address issues such as different time intervals and anomalous values.

The main assumptions made were that future energy demand outcomes would be similar to the past and that temperature and seasonal features were the key variables affecting energy demand. Several modeling methods were used, including linear regression, gradient boost, random forest, light gradient boost, and LSTM (Long Short-Term Memory).

The performance of these models was evaluated using three key metrics: Coefficient of Determination (R²), Root Mean Square Error (RMSE), and Mean Absolute Percentage Error (MAPE). These metrics helped compare the accuracy of the models and determine which performed best in forecasting energy demand based on the given data.

## Exploratory Data Analysis
This study focuses on the analysis of total demand (MW) and temperature (°C) variables from merged and cleaned data sets, with a total of 1,323,266 rows. The total demand ranges from 4,286 MW to 14,649 MW, with a mean of 8,068 ± 1,308 MW. The temperature ranges from -1.3°C to 44.7°C, with a mean of 17.4 ± 5.85°C, which is reasonable for Bankstown, NSW.

<img width="1000" alt="moving-avg-tot-dem" src="https://user-images.githubusercontent.com/127566032/235382499-c2a5c7fa-bf27-40bd-84bc-c47fcf099841.png">

Seven-day and 180-day moving averages of temperature and total demand were created to analyze the data. The temperature chart demonstrates a consistent variation in temperature across recurring annual cycles over the 12-year period. The total demand chart reveals a long-term trend of reducing demand over the same period, although the variation has intensified recently, with higher peaks and deeper troughs.

<img width="1000" alt="moving-avg-temp" src="https://user-images.githubusercontent.com/127566032/235382504-f38dbed2-8d2f-4ded-9d7a-5974fffafc5d.png">

Total demand increases when the temperature is towards low or high extremes, consistent with the use of heating and air-conditioning during colder and warmer months. Histograms of temperature and total demand illustrate this relationship, with different colors for each season and mean and 95% confidence intervals depicted.

<div align="center">
  <img width="600" alt="hist-avg-temp" src="https://user-images.githubusercontent.com/127566032/235382535-a719b397-dd1b-47e3-b2fc-14f965d2b95c.png" />
</div>

Box plots of energy demand over days of the week and months of the year show expected trends. Less energy is used on weekends, likely due to reduced commercial consumption. Higher consumption is observed in summer and winter, with reduced consumption in spring and autumn. Outliers are more likely to be associated with increased energy consumption.

<div align="center">
  <img width="600" alt="hist-avg-tot-dem" src="https://user-images.githubusercontent.com/127566032/235382541-8b7f8db1-2412-4663-8746-1f84b16620a9.png" />
</div>

When analyzing temperature and total demand by time periods (night, morning, afternoon, evening), the 180-day moving averages show that demand during the night period is significantly lower and less variable than the other time periods.

<div align="center">
  <img width="600" alt="box-tot-dem-months" src="https://user-images.githubusercontent.com/127566032/235383068-d596e5f6-68a1-4301-b06a-1224b7f29c7c.png" />
</div>

## Analysis and Results
This study investigates the relationship between temperature and total energy demand in New South Wales using various machine learning models. The dataset comprises of 1,323,266 rows of data for both total demand (ranging from 4,286 MW to 14,649 MW) and air temperature (ranging from -1.3°C to 44.7°C). The data analysis begins by creating 7-day and 180-day moving averages of temperature and total demand. Seasonal temperature variations are consistent, while total energy demand shows a long-term downward trend with higher peaks and deeper troughs. Demand increases during temperature extremes due to heating and air-conditioning use.

<img width="1000" alt="simple-linear-model" src="https://user-images.githubusercontent.com/127566032/235383158-2a90a804-f938-4668-a04b-ba1226c26b4f.png">

Several machine learning models were explored, starting with simple linear regression, which performed poorly (R2 = -0.21; RMSE = 1392.96; MAPE = 15.82%). Multiple linear regression improved accuracy slightly (R2 = 0.20; RMSE = 1132.82; MAPE = 11.28%). Gradient Boost ensemble method significantly improved the performance (R2 = 0.63; RMSE = 768.83; MAPE = 7.56%), as did the Random Forest algorithm (R2 = 0.74; RMSE = 646.18; MAPE = 6.25%). The Light Gradient Boost method further improved accuracy (R2 = 0.78; RMSE = 600.05; MAPE = 5.95%).

<img width="1000" alt="random-forest" src="https://user-images.githubusercontent.com/127566032/235383007-e7612ab4-e59f-4cf9-9291-6c159e4a8358.png">

After truncating the dataset and focusing on the latter half, the Light Gradient Boost model's performance improved (R2 = 0.85; RMSE = 519.63; MAPE = 5.07%), particularly in forecasting night-time energy consumption. The Huber Loss Regressor model, however, did not perform well (R2 = 0.45; RMSE = 985.84; MAPE = 10.14%). 

<img width="1000" alt="huber-loss-regressor" src="https://user-images.githubusercontent.com/127566032/235383134-c0317890-de30-4335-97f7-bf2e3253dd4b.png">

The Quantile Gradient Boosting model showed good forecasting capability (R2 = 0.83; RMSE = 540.08; MAPE = 5.26%) but did not outperform the Light Gradient Boost model. 

<img width="1000" alt="quantile-gradient-boosting" src="https://user-images.githubusercontent.com/127566032/235392592-7dba0c53-924e-4a8f-90a3-1e8e8c0a12b7.png">

Lastly, the Grid Search Light Gradient Boost model maintained an R2 score of 0.83 (RMSE = 553.46; MAPE = 5.32%).

<img width="1000" alt="grid-search-xgboost" src="https://user-images.githubusercontent.com/127566032/235382555-df731d51-07a7-4bd2-9c2d-9e91c7bb168b.png">

In summary, the Light Gradient Boost model showed the best performance in predicting energy demand based on temperature and other time-based features. This model accounted for 85% of the variation in total energy demand, suggesting its potential for use in forecasting energy consumption, especially during night-time hours.

Long Short-Term Memory (LSTM) is a recurrent neural network structure that effectively captures long-term dependencies in time series data. LSTMs have demonstrated promise in applications such as bitcoin price forecasting and crude oil price prediction. The LSTM architecture features a memory cell with three gating mechanisms: forget, input, and output gate. These networks have shown superior performance in short-term predictions compared to traditional RNNs and other sequence learning methods, often surpassing traditional methods like ARIMA.

<div align="center">
  <img src="https://user-images.githubusercontent.com/127566032/235382607-b87862fb-a774-4d94-b548-a0c51a9a1cf5.JPG" alt="LSTM Cell" />
</div>

LSTM combined with ADAM (Adaptive Moment Estimation) optimizer has proven well-suited for electricity spot buyers and electricity price forecasting. Assuming high accuracy preference and 5-minute granularity data availability, a sequence-to-vector unidirectional LSTM model was developed. This model used a scaled two-dimensional Numpy array as input from the previous fifty timestamps and predicted a single subsequent timestamp on a rolling basis.

The deep learning model training phase ceased at the end of the training set, but recent data was continually supplied for testing purposes. This approach allowed the model to utilize the most current data for predictions based on the preceding fifty steps, promoting continuous adaptation. Despite computational complexities, extra resources were procured for training and evaluating LSTM models by subscribing to Colab Pro+.

<div align="center">
  <img width="550" alt="TSCV" src="https://user-images.githubusercontent.com/127566032/235393959-e37c20bc-bba4-4898-a76a-9feb8b941670.png" />
</div>


The preliminary LSTM model employed a slower learning rate, ADAM optimizer, L1 and L2 regularization, and set the epoch count at 20. The batch size was reduced to 2000, and Mean Absolute Error (MAE) was used as the loss function. A dropout rate of 0.1 across two layers was applied, promoting model robustness against noise.

The final LSTM model, after extensive grid search and hyperparameter optimization, achieved an R2 score of 0.93, with a significant improvement observed when the learning rate was adjusted to 0.0002 and the batch size set to 3000. The number of hidden nodes was decreased to 5, accelerating model execution and enhancing convergence. The best model was saved based on the lowest RMSE score.


<div align="center">
  <img alt="LSTM" src="https://user-images.githubusercontent.com/127566032/235394405-2f8be2e2-53b5-4a19-8ffd-311ba6708a28.JPG" />
</div>


This LSTM model's performance is comparable to the 82.5% prediction accuracy reported by Japanese researchers using LSTM networks for month-ahead electricity consumption forecasting. Overall, LSTMs hold significant potential for forecasting in various fields, demonstrating their capability for capturing long-term dependencies in time series data and outperforming traditional forecasting methods.

## Discussion
This study aimed to determine the optimal models for forecasting energy demand based on temperature and seasonal features, considering both prior demand included and excluded scenarios. The analysis is divided into two segments, each focusing on different data partitions and features.

Segment one encompasses the entire dataset, allocating 80% of the data for training and 20% for testing. The objective is to develop models using single or multiple engineered features, prioritizing high R-squared (R2) and low RMSE and MAPE values. Restricting the forecasting horizon to a 5-minute interval, the LSTM model achieves an R2 score of 0.93 when trained on 84% of the total data and tested from 20 June 2020 onwards.

Segment two involves dividing the entire dataset in half and focusing on the latter half. Four Light Gradient Boosting (LGB) models tailored to each time bracket (midnight, morning, noon, and evening) are assembled, incorporating four additional binary features to represent distinct time periods. The resulting R2 value of 0.85 implies a significant seasonal trend within the data.

If forecasting accuracy for future demand is a priority, the LSTM model is recommended. This approach can explain approximately 93% of the overall variation in the data, provided it can access the previous 50-time steps or roughly 8 hours of TOTAL DEMAND and TEMPERATURE data simultaneously.

Conversely, if the forecasting objective is focused on a longer-term horizon (spanning months and years), an ensemble model derived from the latter half of the data comprising four stacked LGB models is advised. This model can explain around 85% of the data's variation.

Another essential factor in comparison involves evaluating the suitability of these models in the context of available training resources. LGB models can be trained within seconds, while deep learning, multi-layer LSTM models may require hours. Moreover, the volume of data necessary to train the LSTM model must be considerably more extensive. Fitting LSTM solely on the latter portion of the dataset does not yield good results, stressing the significance of ample training data for LSTM models in capturing long-range dependencies effectively.

## Conclusion and Next Steps
In conclusion, the analysis consists of two segments, each focusing on different aspects of the dataset and incorporating feature engineering. Segment one allocates 80% of the data for training and 20% for testing, yielding an R2 score of 0.93 for the LSTM model. Segment two divides the dataset in half, discarding the initial portion, and emphasizes the latter half with new features engineered to enhance performance.

<div align="center">
  <img width="500" alt="Table" src="https://user-images.githubusercontent.com/127566032/235382858-f03881bb-a018-4d5f-bbf3-ae3e595d42f4.jpg" />
</div>

For companies interested in spot buying electricity, the deep learning LSTM model is recommended as it accounts for 93% of data variation. Governments and large institutions should use an ensemble of LGB models, which explains 85% of the variation over a broader horizon. When considering training resources, LGB models require seconds to execute, while LSTM models may take hours and need more extensive data.

CNNs, RNNs, and LSTM models can predict multiple time steps ahead, offering opportunities for sequence-to-sequence modeling. Bidirectional LSTMs outperform standard LSTMs by using past and future data, improving sequential task performance at the cost of increased computational resources and complexity.

Alternative models such as ARIMA offer a more straightforward approach for time series forecasting than deep learning models. The training process of ARIMA models is simpler as it relies on linear regression techniques. However, ARIMA has limitations, including the inability to model non-linear relationships, certain assumptions to be satisfied, and decreased accuracy when making long-term predictions.

Future investigations could compare the performance of the models against half-hourly forecast demand data provided for New South Wales and published by the National Electricity Market system.
