from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained Random Forest model
random_forest_model = joblib.load("random_forest_model.joblib")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        
        # Get input values from the form
        crop = request.form['crop']
        location = request.form['location']
        rainfall = float(request.form['rainfall'])
        temperature = float(request.form['temperature'])
        humidity = int(request.form['humidity'])
        price = float(request.form['price'])

        # Map crop and city to their respective label encoded values
        crop_mapping = {'Chilli': 0, 'Groundnut': 1, 'Maize':2, 'Rice': 3, 'Sugarcane': 4}
        location_mapping = {'Chittoor':0, 'Guntur': 1, 'Kadapa': 2, 'Nellore': 3, 'Vijayawada': 4}

        crop_encoded = crop_mapping.get(crop, -1)
        location_encoded = location_mapping.get(location, -1)

        # Ensure valid input
        if crop_encoded == -1 or location_encoded == -1:
            return "Invalid crop or city selected. Please try again."
        
# Generate prices for the next 12 months starting from the current month
        today = pd.Timestamp.today()
        next_12_months = pd.date_range(today, periods=12, freq='MS').strftime("%m/%y").tolist()

        # Prepare the DataFrame for prediction
        next_12_months_df = pd.DataFrame({
            'Location': [location_encoded] * 12,
            'Crop': [crop_encoded] * 12,
            'Rainfall': [rainfall] * 12,
            'Temperature': [temperature] * 12,
            'Humidity': [humidity] * 12,
            'Month': [int(month.split('/')[0]) for month in next_12_months],
            'Year': [int(month.split('/')[1]) for month in next_12_months]  # Convert 'yy' to 'yyyy'
        })

        # Make predictions using the trained model
        predicted_prices_next_12_months = random_forest_model.predict(next_12_months_df)

        # Prepare the result to display in the template
        price_data = [
            {'month': pd.to_datetime(f"{month}/20{year}").strftime('%B %Y'), 'price': round(price, 2)}
            for month, year, price in zip(next_12_months_df['Month'], next_12_months_df['Year'], predicted_prices_next_12_months)
        ]

        # Result dictionary
        result = {
            'crop': crop,
            'price_data': price_data
        }

        # Sample crop info data 
        crop_info = [
            {'crop': 'Rice', 
            'info': 'Rice (paddy) is a significant crop in Andhra Pradesh, especially in the coastal and delta regions. Major rice-producing districts include East Godavari, West Godavari, Krishna, and Guntur. The state’s extensive canal irrigation system supports the cultivation of multiple rice varieties, including BPT, Swarna, and Samba. The Kharif season (June to October) is key for rice farming, with planting starting with the onset of the monsoon.'},
            {'crop': 'Groundnut', 
            'info': 'Groundnut cultivation is widespread in Andhra Pradesh, with major production in districts like Anantapur, Kurnool, and Chittoor. The crop is well-suited to the states arid and semi-arid regions. Groundnuts are grown during the Kharif season, with sowing usually taking place in June and harvesting in October. The states sandy loam soils and warm climate contribute to successful groundnut cultivation.'},
            {'crop': 'Sugarcane', 
            'info': 'Sugarcane is a vital cash crop in Andhra Pradesh, with cultivation in districts like Krishna, Guntur, and East Godavari. The tropical climate and fertile alluvial soils contribute to successful sugarcane farming. The crop is grown throughout the year, with the peak crushing season typically from November to April. Sugarcane plays a pivotal role in the states sugar and ethanol production.' },
            {'crop': 'Maize', 
            'info': 'Maize is cultivated across various regions of Andhra Pradesh, with notable production in districts like Anantapur and Prakasam. The crop is grown both during the Kharif and Rabi seasons, benefiting from the state’s diverse soil types and climate. Maize serves as an important fodder crop and is also used in food industries, making it a key agricultural product.'},
            {'crop': 'Chilli', 
            'info': 'Mirchi, or chili peppers, are prominently cultivated in Andhra Pradesh, known for its spicy cuisine. Guntur district, often referred to as the "Chili Capital of India," is a major producer. The Rabi season, from October to March, is crucial for mirchi cultivation. The red loamy soils and warm temperatures favor the growth of high-quality mirchi varieties, making it a significant cash crop.'},
        ]

        return render_template('result.html', result=result, crop_info=crop_info)

    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
