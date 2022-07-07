From Dataset to Machine Learning App - Demand Planning with Machine Learning

This project illustrates an end-to-end process culminating in a machine learning app deployed on Streamlit Cloud (https://mwale1996-demand-forecaster-demand-forecasting-app-as74gy.streamlitapp.com/).

The project uses Kaggle's Store-Item Demand Forecasting Challenge Dataset (more info at https://www.kaggle.com/c/demand-forecasting-kernels-only).

The first phase of the project is done within a Jupyter Notebook. This phase includes exploratory data analysis (EDA) with pandas and matplotlib plotting functions, as well as some initial analytical machine learning to determine the forecastability of the dataset and the models that would be best suited to achieve this. I then build a couple of forecasting models with the aim of deployment with Pycaret and Streamlit.

I use statsmodels's time series API to breakdown a sample Item-Store combination's sales by month with simple seasonal decomposition, Holt's exponential smoothing model, Hodrick Prescott filter, and seasonal and trend decomposition using Loess. I am working off the assumption that inventory planners would ideally like to hold a month's work of stock and base analysis of the assumption.

I then use pycaret's regression module to build a forecaster using a timeseries fold strategy (this is pre-Pycaret 3.0 which has its own time series module). Since the model returns a subpar model, I use Pycaret's time_series module to build another two models fitted to different store-item combinations. The results of the models prove that an exponential smoothing model is the best model for forecasting demand for these products. I then save the model and configuration of the model for deployment. (due to a Pycaret3 (likely) pre-release bug, I am not able to use load_model in the streamlit app so will have to setup the configuration and build the model again, however specifying that I want to create an exponential smoothing model).

The second phase of the project is the deployment of the model in a Streamlit app. The app layout allows the user to choose a Store and Item combination and see the unsummarized (i.e. daily, not monthly) sales data with the corresponding forecasting results using an exponential smoothing model. The user is also able to edit the unsummarized data and the forecaster will use the edited data and not the summarized data and visualize the results.

The project is intended to show the process of progressing from EDA to machine learning app.

Major kudos to the developers of the pandas, pycaret, streamlit, streamlit-aggrid, statsmodels, plotly and numpy libraries.
