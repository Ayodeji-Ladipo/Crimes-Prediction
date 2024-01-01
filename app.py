import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

# loading in the saved model
with open('random_forest_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)




# Load label encoders using pickle
with open('location_group_label_encoder.pkl', 'rb') as file:
    location_group_encoder = pickle.load(file)

with open('day_label_encoder.pkl', 'rb') as file:
    day_encoder = pickle.load(file)

with open('month_label_encoder.pkl', 'rb') as file:
    month_encoder = pickle.load(file)

with open('status_category_label_encoder.pkl', 'rb') as file:
    status_category_encoder = pickle.load(file)

# Set the page title and favicon
st.set_page_config(page_title="Crime Prediction", page_icon="🔫")

st.sidebar.title("Crime Prediction App 🔫")
st.sidebar.header("Navigation")
selected_option = st.sidebar.radio("Select an option", ["Project Description", "Developer Info", "Crime Prediction App"])


# About the Project section
if selected_option == "Project Description":

    st.title("About the Project")
    st.write("Welcome to the Crime Prediction App! This project aims to predict crime categories based on various features, providing valuable insights into potential criminal activities. Leveraging machine learning models and advanced data preprocessing techniques, the app processes user-inputted data to make predictions and enhance understanding of crime patterns.")

    st.write("#### Key Features:")
    st.write("1. **Predictive Analytics:** The app utilizes a trained machine learning model to predict crime categories, offering users a glimpse into possible outcomes based on provided information.")
    st.write("2. **User-Friendly Interface:** With a streamlined and intuitive design, the app makes it easy for users to input data, receive predictions, and explore the project's findings.")
    st.write("3. **Data Preprocessing:** Advanced data preprocessing techniques, including feature engineering and encoding, ensure accurate and meaningful predictions by the machine learning model.")
    st.write("4. **Crime Categories:** The model categorizes crimes into various types, providing insights into the nature of incidents and aiding in proactive decision-making.")
    st.write("5. **Privacy and Security:** The app prioritizes user privacy, handling data securely, and maintaining confidentiality.")

    st.write("#### How to Use:")
    st.write("1. **Input Data:** Enter relevant details such as victim's sex, location, day, month, and more to receive predictions about potential crime categories.")
    st.write("2. **Explore Predictions:** Once the input is provided, the app will generate predictions based on the trained model, helping users understand potential outcomes.")
    st.write("3. **Gain Insights:** Dive into the results and gain insights into crime patterns, contributing to a safer and more informed community.")

    st.write("Explore the app and empower yourself with the knowledge to make informed decisions regarding crime prevention and public safety. Thank you for being a part of this innovative project!")

        # Add more information about the project here

# Developer Info section
elif selected_option == "Developer Info":
    st.title("Meet the Developer 🚀")
    st.write("# Ladipo Ayodeji")
    st.write("Passionate Developer | Machine Learning Enthusiast")

    # Add more student information here
    

elif selected_option == "Crime Prediction App":

    # Streamlit app
    st.title("Crime Prediction App")

    # Creating form for input fields
    with st.form(key='crime_input_form'):
        # User input
        vict_sex = st.selectbox("Select victim's sex", ['Female', 'Male', 'Non Binary', 'H', 'Unknown'])
        lat = st.number_input("Enter latitude", min_value=0.0, max_value=34.3343)
        lon = st.number_input("Enter longitude", min_value=-118.6676, max_value=0.0)

        descent = st.selectbox("Select victim's descent", ['Unknown', 'Black', 'White', 'Asian', 'Hispanic'])
        has_victim = st.checkbox("Does the incident have a victim?")
        age_bins = st.selectbox("Select age group", ['No Victim', '0-20', '21-30', '31-45', '46+'])
        weapon_group = st.selectbox("Select weapon group", ['Guns', 'Knife and Cutting Instruments', 'Blunt and Impact Weapons',
                                            'STRONG-ARM (HANDS, FIST, FEET OR BODILY FORCE)', 'Unclassified', 'No Weapon Used'])
        location_group = st.selectbox("Select location group", ['Other', 'Street', 'Boulevard', 'Avenue'])
        day = st.selectbox("Select day", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        month = st.selectbox("Select month", ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
        status_category = st.selectbox("Select status category", ['Investigation', 'Other'])
        # submit button

        # Custom CSS to inject larger button styles
        button_style = """
        <style>
        div.stButton > button:first-child {
            font-size: 90px;  # Change font size as needed
            height: 3em;      # Adjust height
            width: 100% !important;      # Adjust width
            # Add other styling if needed
        }
        </style>
        """

        st.markdown(button_style, unsafe_allow_html=True)

        
        submit_button = st.form_submit_button(label='Predict',help="Click to make a prediction")

        # Preprocess the input features
        vict_sex_encoded = {'Female':2, 'Male':3, 'Non Binary':0,'H':1, 'Unknown':1}[vict_sex]
        descent_encoded = {'Unknown': 0, 'Black': 1, 'White':2, 'Asain':3, 'Hispanic': 4}[descent]
        has_victim_encoded = 1 if has_victim else 0
        age_bins_encoded = {'No Victim':0, '0-20':1, '21-30':2, '31-45':3, '46+':4}[age_bins]  # Assuming this is already encoded
        weapon_group_encoded = {'Guns': 100, 'Knife and Cutting Instruments': 200, 'Blunt and Impact Weapons': 300,
                                            'STRONG-ARM (HANDS, FIST, FEET OR BODILY FORCE)': 400, 'Unclassified': 500, 'No Weapon Used': 0}[weapon_group]
        location_group_mapping = {'Other':'Other', 'Street':'ST', 'Boulevard':'BL', 'Avenue':'AV'}
        location_group_encoded = location_group_encoder.transform([location_group_mapping[location_group]])[0]
        day_encoded = day_encoder.transform([day])[0]
        month_encoded = month_encoder.transform([month])[0]
        status_category_encoded = status_category_encoder.transform([status_category])[0]

        # Create a DataFrame with the preprocessed features
        input_data = pd.DataFrame({
            'vict sex': [vict_sex_encoded],
            'lat': [lat],
            'lon': [lon],
            'modified_descent': [descent_encoded],
            'has_victim': [has_victim_encoded],
            'age_bins': [age_bins_encoded],
            'weapon_group': [weapon_group_encoded],
            'location_group': [location_group_encoded],
            'day': [day_encoded],
            'month': [month_encoded],
            'status_category': [status_category_encoded]
        })

        

        if submit_button:
            # Make the prediction
            prediction = model.predict(input_data)

            # Display the prediction
            st.subheader("Prediction:")
            st.markdown(f"<p style='font-size: 00px !important; font-weight: bold;'>Predicted Crime Category: {prediction[0]}</p>", unsafe_allow_html=True)

            

