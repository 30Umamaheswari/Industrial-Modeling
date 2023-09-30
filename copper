import streamlit as st
import pickle
import numpy as np
import pickle
from streamlit_option_menu import option_menu
from sklearn.preprocessing import StandardScaler, OneHotEncoder

st.set_page_config(page_title="Predictive Analysis",
                   layout="wide", )
app_title = """
    color: lightseagreen;
    text-align: center;
    font-family: Serif;
    font-size: 50px;
"""
st.markdown(
    f'<h1 style="{app_title}">INDUSTRIAL COPPER MODELING</h1>',
    unsafe_allow_html=True
)
c1, c2, c3 = st.columns([5, 7, 3])
c1.image('ana.jpeg')

# import joblib
#
# with open('model.joblib', 'rb') as file:
#     loaded_model = joblib.load(file)

# Pickle File open and load

with open(r"model.pkl", 'rb') as file:
    loaded_model = pickle.load(file)

with open(r'scaler.pkl', 'rb') as f:
    scaler_loaded = pickle.load(f)

with open(r"type.pkl", 'rb') as f:
    type_loaded = pickle.load(f)

with open(r"status.pkl", 'rb') as f:
    status_loaded = pickle.load(f)

status = [1, 0]
item_type = [0, 1, 2, 3, 4, 5, 6]
application = [10., 41., 28., 59., 15., 4., 38., 56., 42., 26., 27., 19., 20., 66., 29., 22., 40., 25., 67., 79., 3.,
               99., 2., 5., 39., 69., 70., 65., 58., 68.]
country = [28., 25., 30., 32., 38., 78., 27., 77., 113., 79., 26., 39., 40., 84., 80., 107., 89.]
# st.write(type(country)
# country = [int(x) for x in country]

product = [611112, 611728, 628112, 628117, 628377, 640400, 640405, 640665, 611993, 929423819, 1282007633, 1332077137,
           164141591, 164336407, 164337175, 1665572032, 1665572374, 1665584320, 1665584642, 1665584662,
           1668701376, 1668701698, 1668701718, 1668701725, 1670798778, 1671863738,
           1671876026, 1690738206, 1690738219, 1693867550, 1693867563, 1721130331, 1722207579]

tab1, tab2 = c2.tabs(["$\\huge Price Prediction $", "$\\huge Status Prediction $"])

with tab1:
    col1, col2 = c2.columns([4,4], gap='large')
    quantity_tons = col1.slider("Enter quality tons", min_value=611728, max_value=1722207579)
    col1.write('')

    col1.text('1 - Won')
    col1.text('2 - Lost')
    selected_status = col1.radio("Select status", status, key=1)

    col1.write('NOTE: 0 - W, 1 - WI, 2 - S, 3 - Others, 4 - PL, 5 - IPL, 6 - SLAWR')
    selected_item_type = col1.selectbox("Select item Type", item_type, key=2)

    selected_application = col1.selectbox(" Select Application", sorted(application), key=3)

    thickness = col2.slider('Select Thickness', min_value=0.18, max_value=400.0)
    width = col2.slider('Select Width', min_value=1, max_value=2990)

    selected_country = col2.selectbox('Select country', country)
    selected_product = col2.selectbox("Product Reference", product, key=5)

    if col2.button('Predict Price'):
        new_sample = np.array([[(float(quantity_tons)), float(selected_status), float(selected_item_type),
                                np.log(float(selected_application)), np.log(float(thickness)), np.sqrt(float(width)),
                                int(selected_country), int(selected_product)]])
        new_sample_ohe = type_loaded.transform(new_sample[:, [6]]).toarray()
        new_sample_be = status_loaded.transform(new_sample[:, [7]]).toarray()
        new_sample = np.concatenate((new_sample[:, [0, 1, 2, 3, 4, 5, ]], new_sample_ohe, new_sample_be), axis=1)
        new_sample1 = scaler_loaded.transform(new_sample)
        new_pred = loaded_model.predict(new_sample1)[0]
        st.write('## :red[Predicted selling price:] ', (new_pred))
