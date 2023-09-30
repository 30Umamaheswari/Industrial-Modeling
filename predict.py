import streamlit as st
import numpy as np
import pickle
# from streamlit_option_menu import option_menu
# from sklearn.preprocessing import StandardScaler, OneHotEncoder

st.set_page_config(page_title="Predictive Analysis",
                   layout="wide", )
# with st.sidebar:
#     select = option_menu('Menu', ["Price", "Status"],
#                          default_index=0,
#                          orientation="vertical",
#                          styles={"nav-link": {"font-size": "15px", "text-align": "left",
#                                               "margin": "0px", "--hover-color": "#BCADD0"}})
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

# Pickle File for Status

with open(r"model_status.pkl", 'rb') as file:
    m_status = pickle.load(file)

with open(r'scaler_status.pkl', 'rb') as f:
    scaler_s = pickle.load(f)

with open(r"c_status.pkl", 'rb') as f:
    Status = pickle.load(f)

status = ['Won', 'Lost']
item_type = ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']
application = [10., 41., 28., 59., 15., 4., 38., 56., 42., 26., 27., 19., 20., 66., 29., 22., 40., 25., 67., 79., 3.,
               99., 2., 5., 39., 69., 70., 65., 58., 68.]
country = [28., 25., 30., 32., 38., 78., 27., 77., 113., 79., 26., 39., 40., 84., 80., 107., 89.]
# st.write(type(country)
# country = [int(x) for x in country]

product = [611112, 611728, 628112, 628117, 628377, 640400, 640405, 640665, 611993, 929423819, 1282007633, 1332077137,
           164141591, 164336407, 164337175, 1665572032, 1665572374, 1665584320, 1665584642, 1665584662,
           1668701376, 1668701698, 1668701718, 1668701725, 1670798778, 1671863738,
           1671876026, 1690738206, 1690738219, 1693867550, 1693867563, 1721130331, 1722207579]

select = c2.radio('Select Prediction', ['Price', 'Status'])

# tab1, tab2 = c2.tabs(["$\\huge Price Prediction $", "$\\huge Status Prediction $"])

if select == 'Price':
    c2.markdown(f'<div style="display: flex; align-items: center;"><div style="color: mediumaquamarine; font-weight: '
                f'bold; font-size: 50px;">Price Prediction</div></div>', unsafe_allow_html=True)

    col1, col2 = c2.columns([4, 4], gap='large')
    quantity_tons = col1.slider("Enter quality tons", min_value=1000, max_value=1000000000, key=10)
    col1.write('')

    selected_status = col1.radio("Select status", status, key=9)

    # col1.write('NOTE: 0 - W, 1 - WI, 2 - S, 3 - Others, 4 - PL, 5 - IPL, 6 - SLAWR')
    selected_item_type = col1.selectbox("Select item Type", item_type, key=12)

    selected_application = col1.selectbox(" Select Application", sorted(application), key=13)

    thickness = col2.slider('Select Thickness', min_value=0.18, max_value=400.0, key=14)
    width = col2.slider('Select Width', min_value=1, max_value=2990, key=15)

    selected_country = col2.selectbox('Select country', country, key=21)
    selected_product = col2.selectbox("Product Reference", product, key=16)

    if col2.button('Predict Price'):
        new_sample = np.array([[int(selected_country), np.log(float(selected_application)), int(selected_product),
                                float(quantity_tons), np.log(float(thickness)), np.sqrt(float(width)),
                                selected_item_type, selected_status]])
        # col2.write(new_sample)
        new_sample_ohe = type_loaded.transform(new_sample[:, [6]]).toarray()
        # col2.write(new_sample_ohe)
        new_sample_be = status_loaded.transform(new_sample[:, [7]]).toarray()
        # col2.write(new_sample_be)

        new_sample = np.concatenate((new_sample[:, [0, 1, 2, 3, 4, 5, ]], new_sample_ohe, new_sample_be), axis=1)
        # col2.write(new_sample)
        new_sample1 = scaler_loaded.transform(new_sample)

        new_pred = loaded_model.predict(new_sample1)[0]
        price = np.exp(new_pred)

        c2.markdown(
            f'<div style="display: flex; align-items: center;"><div style="color: mediumaquamarine; font-weight: '
            f'bold; font-size: 30px;">Price Prediction - </div><div style="margin-left: 10px;">{price}</div></div>',
            unsafe_allow_html=True)

if select == 'Status':
    c2.markdown(f'<div style="display: flex; align-items: center;"><div style="color: mediumaquamarine; font-weight: '
                f'bold; font-size: 50px;">Status Prediction</div></div>', unsafe_allow_html=True)

    cl1, cl2 = c2.columns([4, 4], gap='large')
    quantity_tons = cl1.slider("Enter quality tons", min_value=0, max_value=1000000000)
    cl1.write('')

    selected_price = cl1.slider("Select Price", min_value=500, max_value=10000)

    selected_item_type = cl1.selectbox("Select item Type", item_type, key=2)

    selected_application = cl1.selectbox(" Select Application", sorted(application), key=3)

    thickness = cl2.slider('Select Thickness', min_value=0.18, max_value=400.0)
    width = cl2.slider('Select Width', min_value=1, max_value=2990)

    selected_country = cl2.selectbox('Select country', country)
    selected_product = cl2.selectbox("Product Reference", product, key=5)

    if cl2.button('Predict Status'):
        new_sample = np.array([[(float(quantity_tons)), selected_country,
                                float(selected_application), (float(thickness)), float(width),
                                int(selected_product),
                                int(selected_price), selected_item_type]])
        new_sample_ohe = Status.transform(new_sample[:, [7]]).toarray()
        new_sample_for_prediction = new_sample[:, [0, 1, 2, 3, 4, 5, 6]]

        new_sample = np.concatenate((new_sample_for_prediction, new_sample_ohe), axis=1)
        new_sample = scaler_s.transform(new_sample)

        predict = m_status.predict(new_sample)
        # cl2.write(predict)
        if (predict == 1).any():
            cl2.write('<p style="color: teal; font-size: 30px">STATUS IS WON</p>', unsafe_allow_html=True)
        else:
            cl2.write('<p style="color: teal; font-size: 30px">STATUS IS LOST</p>', unsafe_allow_html=True)
