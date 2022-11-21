
import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import datetime as dt
import plotly.graph_objects as go
import timeit

gamma = 1/12
population = 33942526

@st.cache(allow_output_mutation=True)
def get_model(model):
    return load_model(model)

def get_data():
   
    df = pd.read_csv('https://raw.githubusercontent.com/MoH-Malaysia/covid19-public/main/epidemic/cases_malaysia.csv')
    df['date'] = pd.to_datetime(df['date'],format= '%Y-%m-%d')

    new_df = df.loc[:,['date','cases_new','cases_active']]

    new_df['cases_recovered'] = (new_df['cases_active'].shift() + new_df['cases_new']) - new_df['cases_active']
    new_df['cases_recovered_C'] = new_df['cases_recovered'].cumsum()
    new_df['susceptible'] = population - new_df['cases_active'] - new_df['cases_recovered_C']
    growth_rate = (new_df['cases_active'] - new_df['cases_active'].shift()) / new_df['cases_active'].shift()
    new_df['reproduction_number'] = 1 + (growth_rate / gamma)
    new_df.to_csv('malaysia-country-covid-data.csv')
        
    return new_df

def set_sequence(size,data):
    x = []
    
    for count in range(len(data) - size):
        window = [[s] for s in data.loc[count : count + size,'reproduction_number']]
        x.append(window)
        
    return np.array(x)

def plot_data(x,y,legend,title):
    fig = go.Figure()
    if len(x) != len(y):
        st.write('Error')
        return
    for count in range(len(x)):
        fig.add_trace(go.Scatter(x=x[count], y=y[count], name=legend[count]))
    fig.layout.update(title_text=title, xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

def SIR(data,index,rep_num):
    idx = index - 1

    date = data.loc[idx,'date'] + dt.timedelta(1)

    case_rec = gamma * data.loc[idx,'cases_active']
    
    case_new = gamma * rep_num * data.loc[idx,'cases_active']
    
    suc = data.loc[idx,'susceptible'] - case_new
    
    case_active = data.loc[idx,'cases_active'] + case_new - case_rec
    rec_sum = data.loc[idx,'cases_recovered_C'] + case_rec
            
    
    return date,case_new,case_active,case_rec,rec_sum,suc,rep_num

st.markdown("""
<style>
.big-font {
    font-size:50px !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">Daily Cases of Covid-19 in Malaysia Predictor</p>', unsafe_allow_html=True)

df = get_data()
model = get_model('covid19-TRANSFORMER12-7.h5')

pred_df = df.loc[len(df.index) - 7 : len(df.index) - 1].reset_index(drop=True)

if 'prediction' not in st.session_state:
    st.session_state['prediction'] = df.loc[len(df.index) - 7 : len(df.index) - 1].reset_index(drop=True)

    loading = st.progress(0)

    num = 365
    for count in range(num):
        val = set_sequence(6,st.session_state['prediction'][:])
        seq_val = np.array([val[len(val) - 1]])
        pred = model.predict(seq_val)[0][0].astype(np.float64)
        idx = len(st.session_state['prediction'])

        st.session_state['prediction'].loc[idx] = SIR(st.session_state['prediction'],idx,pred)
        loading.progress(count / num)

    st.session_state['prediction']['cases_new_whole'] = st.session_state['prediction']['cases_new'].astype('int')
    loading.empty()

with st.form(key='form1'):
    a = st.slider('Month(s) to predict',1,12,1)
    st.write(a * 30)

    x = [df['date'],st.session_state['prediction']['date'][6 : 6 + (a * 30)]]
    y = [df['cases_new'],st.session_state['prediction']['cases_new_whole'][6 : 6 + (a * 30)]]
    legend = ['Actual New Cases','Predicted New Cases']

    plot_data(x,y,legend,'Daily Covid-19 Cases with Rangeslider')
    submit = st.form_submit_button(label='Submit')

if 'yesterday' not in st.session_state:
    yes = set_sequence(6,df).tolist()
    yes_pred = model.predict(yes)[len(yes) - 1][0].astype(np.float64)
    st.session_state['yesterday'] = gamma * yes_pred * df.loc[len(df.index) - 2,'cases_active']

st.subheader("Predicted Number of New Cases for Yesterday:")
st.write(str(int(st.session_state['yesterday'])) + " Case(s)")
st.subheader("Yesterday's Number of New Cases:")
st.write(str(df.loc[len(df.index) - 1,'cases_new']),'Case(s)')
st.subheader("Today's Forecasted Number of New Cases: ")
st.write(str(int(st.session_state['prediction'].loc[7,'cases_new'])) + " Case(s)")

st.subheader("Number of Days Estimated Until First Instance of 0 New Cases: ")
cases_0 = st.session_state['prediction'].loc[st.session_state['prediction']['cases_new'] < 1].reset_index(drop=True)

case_none = 'N\A' if len(cases_0) == 0 else str(cases_0.loc[0,'date'] - dt.timedelta(6))

st.write(case_none + " Day(s)")

st.subheader("Daily Comparison: ")

with st.form(key='form2'):
    d = st.date_input(
        'Pick a date',
        value = df.loc[len(df.index) - 1,'date'],
        min_value = df.loc[10,'date'],
        max_value = df.loc[len(df.index) - 1,'date'])

    enter = st.form_submit_button('Enter')

    idx = df[df['date'] == str(d)].index[0]

    st.write(len(df),idx)

    check = set_sequence(6,df)
    seq_val = np.array([check[len(check) - 1]])
    r_pred = model.predict(seq_val)[0][0].astype(np.float64)

    data = {'Actual' : [df.loc[idx,'cases_new']],
            'Moving Average' : [df.loc[idx - 7 : idx,'cases_new'].rolling(7).mean()[idx]],
            'Prediction' : [int(SIR(df,idx,r_pred)[1])]}

    focal_df = pd.DataFrame(data)

    # CSS to inject contained in a string
    hide_table_row_index = """
                <style>
                thead tr th:first-child {display:none}
                tbody th {display:none}
                </style>
                """

    # Inject CSS with Markdown
    st.markdown(hide_table_row_index, unsafe_allow_html=True)

    st.table(focal_df)