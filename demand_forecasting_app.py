import streamlit as st
from st_aggrid import AgGrid, DataReturnMode, GridUpdateMode, GridOptionsBuilder
import pandas as pd
import numpy as np
import sqlite3
from sqlite3 import Error
from pycaret.time_series import load_model, predict_model, plot_model, load_config, TSForecastingExperiment, setup
from datetime import datetime
import plotly.express as px


st.set_page_config(page_title = 'Demand Forecasting', layout = 'wide')
st.title('Demand Forecasting')
st.subheader('Using Exponential Smoothing Model to predict demand')


@st.experimental_singleton #cache connection-creator function - to prevent having to reload the model for every rerun or widget interaction
def create_connection(db_file):
    """ create a database connection to a SQLite database """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(sqlite3.version)
    except Error as e:
        print(e)
    finally:
        if conn:
            conn.close()


if __name__ == '__main__':
    create_connection(r"demand_forecast_db.db")

conn = sqlite3.connect(r"demand_forecast_db.db")
cur = conn.cursor()

@st.experimental_memo #memoize the function's returned data to avoid constant data reload
def load_db():
	df1 = pd.read_csv('C:\\Users\\MXW26\\Jupyter Notebooks\\train.csv')
	df2 = pd.read_csv('C:\\Users\\MXW26\\Jupyter Notebooks\\test.csv')
	df = df1.append(df2)
	df['Month'] = [i.month for i in pd.to_datetime(df['date'], dayfirst = True)]
	df['Year'] = [i.year for i in pd.to_datetime(df['date'], dayfirst = True)]
	df['store'] = ['store_' + str(i) for i in df['store']]
	df['item'] = ['item_' + str(i) for i in df['item']]
	df['time_series'] = df[['store', 'item']].apply(lambda x: '_'.join(x), axis=1)
	df['Date1'] = pd.to_datetime(df['date'], yearfirst = True).dt.to_period('M').dt.to_timestamp()
	df.to_sql(name = 'OG', con = conn, index = False, if_exists = 'replace')

load_db()

df = pd.read_sql('SELECT * FROM OG', con = conn)

df['Date'] = pd.to_datetime(df['Date1']) 
df.drop('Date1', axis = 1, inplace = True)

store = st.selectbox('Select Store for forecast', df['store'].unique())

item = st.selectbox('Select Item for forecast', df['item'].unique())

gb = GridOptionsBuilder.from_dataframe(df)

gb.configure_grid_options(enableRangeSelection = True) 

gb.configure_default_column(editable = True, groupable = True)

gb.configure_side_bar()

gridOptions = gb.build()

st.write('Data Source - edits made in this dataframe will be carried into the forecasting algorithm')
grid_return = AgGrid(df[(df['store'] == store) & (df['item'] == item) & (df['Date'] < '2018-01-01')], 
update_mode = GridUpdateMode.SELECTION_CHANGED | GridUpdateMode.VALUE_CHANGED,
gridOptions = gridOptions,
editable = True, 
height = 600, 
theme = 'blue', 
reload_data = True)

df3 = pd.DataFrame(grid_return['data'])

st.subheader('Forecast Plot')

s2 = TSForecastingExperiment()
s2.setup(df3[['Date', 'sales']][df3['Date'] < '2018-01-01'].groupby('Date')[['sales']].sum(), target = 'sales', fold = 5)

es = s2.create_model('exp_smooth')

predictions = s2.predict_model(es, fh = 13)

predictions['Date'] = pd.date_range(start = '2017-12-01', end = '2018-12-01', freq = 'MS')

predictions.columns = ['Forecast', 'Date']

# add a date column in the dataset

# line plot
fig = px.line(pd.concat((df3[['Date', 'sales']][df3['Date'] < '2018-01-01'].groupby('Date')[['sales']].sum(), predictions.set_index('Date')), axis = 0).reset_index(), x='Date', y= ['sales', 'Forecast'], template = 'plotly_dark')
# add a vertical rectange for test-set separation
fig.add_vrect(x0="2017-12-01", x1="2018-12-01", fillcolor="grey", opacity=0.25, line_width=0)

st.plotly_chart(fig, use_container_width = True)



