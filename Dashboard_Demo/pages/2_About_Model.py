import streamlit as st

st.markdown("""
<style>
.big-font {
    font-size:50px !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">About Model</p>', unsafe_allow_html=True)

st.subheader("MODEL DETAILS")

st.markdown(
"""
- Model: Time Series Transformer + SIR Model
- Average Days for Recovery: 9 Days
- Input Sequence Length: 12
- Daily Cases RMSE: 837.52 Cases
"""
)

st.subheader("Relevant Equations")
st.write("Assumptions:")
st.markdown(
"""
- Total Population is constant.
- People who recovered from Covid-19 cannot become reinfected.
"""
)

st.markdown(
"""
(1) $S_t + I_t + R_t = N$\n
(2) $\Delta S_t = -\\beta_t I_{t-1}\\frac{S_{t-1}}{N}$\n
(3) $\Delta I_t = \\beta I_{t-1}\\frac{S_{t-1}}{N} - \gamma I_{t-1} = I_{t-1}(\\beta \\frac{S_{t-1}}{N} - \gamma)$\n
(4) $\Delta R_t = \gamma I_{t-1}$ \n
- $N$ = Total population
- $S_t$ = Population of susceptible at time t
- $I_t$ = Population of infected at time t
- $R_t$ = Population of recovered at time t
- $\Delta S_t$ = Change in S at time t
- $\Delta I_t$ = Change in I at time t
- $\Delta R_t$ = Change in R at time t
- $\\beta_t$ = Probability of transmission at time t
- $\gamma$ = Average period of recovery ($\\frac{1}{Average Days for Recovery}$)
\n
Expanding on (3),\n
(5) $\\beta \\frac{S_{t-1}}{N} - \gamma = 0$\n
(6) $R^t = \\frac{\\beta S_{t-1}}{\gamma N}$\n
- $R^t$ = Effective reproduction number at time t
The value of $\Delta I_t$ is dependent on (5) which can be rewritten into (6) to obtain the effective reproduction number.\n
$R^t$ can be used to identify the expected trend of disease growth.\n
\n
(7) $\\beta \\frac{S_{t-1}}{N}  = \gamma R^t$\n
\n
If we substitue (7) into (3),\n
(8) $\Delta I_t =  \gamma R^t I_{t-1} - \gamma I_{t-1} = \gamma I_{t-1} (R^t - 1) $\n
(9) $I_t = I_{t-1} + \gamma I_{t-1} (R^t - 1) $\n
(10) $R^t = 1 + \\frac{I_t - I_{t-1}}{\gamma I_{t-1}}$\n
(10) is then used to get the rough estimation of the effective reproduction number for all available data.
If we substitute (7) into (2),\n
(11) $\Delta S_t = -\gamma R^t I_{t-1}$\n
we can forgo calculating the probability of transmission. The absolute value of (11) is equal to the number of new cases at time t 
which is the main prediction focus of the model and used as part of the metric for scoring the model. 
"""
)

st.subheader("REFERENCE")
st.write("[Tracking R of COVID-19: A new real-time estimation using the Kalman filter](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0244474)")
st.write("[Timeseries classification with a Transformer model](https://keras.io/examples/timeseries/timeseries_transformer_classification/)")