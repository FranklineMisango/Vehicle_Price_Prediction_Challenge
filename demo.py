import streamlit as st
import pickle


model = pickle.load(open('/home/misango/code/Vehicle_Price_Prediction_Challenge/models/model1_RMSE.pkl', 'rb')) # load ml model

def main():
    st.title("Car Price Predictor")

    Airbags = st.number_input("The Number of airbags", min_value=0, max_value=1000000, step=1)
    Mileage = st.number_input("Input the Mileage", min_value=0, max_value=100, step=1)
    Production_year = st.number_input("The Year of production", min_value=1950, max_value=2050, step=1)
    ID= st.number_input("The ID of the Vehicle", min_value=0, max_value=1000000, step=1)
    transmission_type = st.selectbox("What type of Gear Box does the car have", ['Tiptronic', 'Automatic', 'Variator'])
    leather = st.selectbox("Does the Car have a leather interior", ['Yes', 'No'])
    levy = st.number_input("What is the current levy of the car is USD", min_value=0, max_value=10000000, step=1)
    fuel_type = st.selectbox("Fuel Type", ['Hybrid', 'Petrol', 'Diesel', 'CNG', 'Plug-in Hybrid', 'LPG', 'Hydrogen'])
    engine_volume = st.number_input("What is the Engine volume", min_value= 100,max_value=1000000, step=1 )
    Manufacturer = st.text_input("Who has manufactured this car")
    doors = st.selectbox("How many doors are in this car", ['4', '2'])
    wheels = st.selectbox("Is this car left or hand drive", ['Right-hand drive', 'Left-hand drive'])
    color = st.text_input("What is the color of the car")
    drive_wheel = st.selectbox("Where are the drive wheels located", ['Front', 'Rear'])
    car_model = st.text_input("What is the model of this car")
    current_price = st.text_input("Estimate the current price for this car")
    
    if st.button("Predict"):
        prediction = model.predict([[Airbags, Mileage,Production_year,ID,transmission_type,leather,levy,fuel_type,engine_volume,Manufacturer,doors,wheels,color,drive_wheel,car_model]])
        output = round(prediction[0], 2)
        st.success("Predicted car price: {} USD ".format(output))

if __name__ == '__main__':
    main()
