
import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import datetime as dt
import plotly.graph_objects as go
import timeit

days_to_recovery = 10
input_length = 19
gamma = 1 / days_to_recovery
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

covid_data = get_data()
model = get_model(f'Dashboard_Demo/TRANSFORMER-{str(days_to_recovery).zfill(2)}-SEQLENGTH-{input_length}.h5')

loading_text = st.empty()

if 'prediction' not in st.session_state:
    st.session_state['prediction'] = covid_data.loc[len(covid_data.index) - input_length : len(covid_data.index) - 1].reset_index(drop=True)

    loading_text.text("Do not change to a different page while this page is loading. Thank you!")
    loading = st.progress(0)

    number_of_days = 365
    for count in range(number_of_days):
        inputs = set_sequence(input_length - 1,st.session_state['prediction'][:])
        np_inputs = np.array([inputs[len(inputs) - 1]])
        pred = model.predict(np_inputs)[0][0].astype(np.float64)
        idx = len(st.session_state['prediction'])

        st.session_state['prediction'].loc[idx] = SIR(st.session_state['prediction'],idx,pred)
        loading.progress(count / number_of_days)

    st.session_state['prediction']['cases_new_whole'] = st.session_state['prediction']['cases_new'].astype('int')
    loading.empty()
    
loading_text.empty()
    
with st.form(key='form1'):
    a = st.slider('Month(s) to predict',1,12,1)
    
    x = [covid_data['date'],st.session_state['prediction']['date'][input_length - 1 : input_length - 1 + (a * 30)]]
    y = [covid_data['cases_new'],st.session_state['prediction']['cases_new_whole'][input_length - 1 : input_length - 1 + (a * 30)]]
    legend = ['Actual New Cases','Predicted New Cases']

    plot_data(x,y,legend,'Daily Covid-19 Cases with Rangeslider')
    submit = st.form_submit_button(label='Submit')

if 'yesterday' not in st.session_state:
    yesterday_cases = set_sequence(input_length - 1,covid_data).tolist()
    yesterday_r_pred = model.predict(yesterday_cases)[len(yesterday_cases) - 1][0].astype(np.float64)
    st.session_state['yesterday'] = gamma * yesterday_r_pred * covid_data.loc[len(covid_data.index) - 2,'cases_active']

with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Predicted Number of New Cases for Yesterday:")
    with col2:
        st.write('')
        st.subheader(str(int(st.session_state['yesterday'])) + " Case(s)")

with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Yesterday's Number of New Cases:")
    with col2:
        st.write('')
        st.subheader(str(covid_data.loc[len(covid_data.index) - 1,'cases_new']) + ' Case(s)')

with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Today's Forecasted Number of New Cases: ")
    with col2:
        st.write('')
        st.subheader(str(int(st.session_state['prediction'].loc[input_length,'cases_new'])) + " Case(s)")

with st.form(key='form2'):
    case_val = st.number_input('Enter number of cases:',step = 1)
    latest_date = covid_data.loc[covid_data.index[-1],'date']

    case_number_submit = st.form_submit_button('Enter')

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f'Number of Days Estimated Until First Instance of {case_val} New Cases: ')
        with col2:      
            st.write('')

            less_than_number_cases = (np.round(st.session_state['prediction']['cases_new']) <= case_val)
            greater_than_latest_date = st.session_state['prediction']['date'] > latest_date

            cases = st.session_state['prediction'].loc[less_than_number_cases & greater_than_latest_date].reset_index(drop=True)
            
            case_day = 'N\A' if len(cases) == 0 else (cases.loc[0,'date'] - latest_date)
            try:
                st.subheader(f'{case_day.days if case_day.days > 0 else 0} Day(s)')
            except:
                st.subheader(f'{case_day} Day(s)')

st.subheader("Daily Comparison: ")

with st.form(key='form3'):
    selected_date = st.date_input(
        'Pick a date',
        value = covid_data.loc[len(covid_data.index) - 1,'date'],
        min_value = covid_data.loc[15,'date'],
        max_value = covid_data.loc[len(covid_data.index) - 1,'date'])

    enter = st.form_submit_button('Enter')

    idx = covid_data[covid_data['date'] == str(selected_date)].index[0]

    check = set_sequence(input_length - 1,covid_data)
    np_inputs = np.array([check[len(check) - 1]])
    r_pred = model.predict(np_inputs)[0][0].astype(np.float64)

    data = {'Actual' : [covid_data.loc[idx,'cases_new']],
            'Moving Average' : [covid_data.loc[idx - 7 : idx,'cases_new'].rolling(7).mean()[idx]],
            'Prediction' : [int(SIR(covid_data,idx,r_pred)[1])]}

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
