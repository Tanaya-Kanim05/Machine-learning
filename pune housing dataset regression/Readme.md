## Pune House Price Prediction

## Project Overview
The Pune House Price Prediction project aims to build a machine learning model that predicts house prices in Pune based on various factors such as location, size, number of bedrooms, amenities, and more. The project utilizes data preprocessing, feature engineering, and various regression models to provide accurate price estimations.

## Dataset
The dataset used in this project contains real estate listings from Pune, including details such as:
- Location
- Number of bedrooms, bathrooms, and balconies
- Square footage
- Amenities (parking, swimming pool, gym, etc.)
- Age of the property
- Price (target variable)

## Project Workflow
1. *Data Collection:* The dataset is collected from various sources such as Kaggle, real estate websites, or government records.
2. *Data Cleaning:* Handling missing values, duplicate entries, and inconsistent data.
3. *Exploratory Data Analysis (EDA):* Understanding the dataset through visualizations and statistical summaries.
4. *Feature Engineering:* Creating new features, encoding categorical variables, and scaling numerical features.
5. *Model Selection:* Testing different regression models like Linear Regression, Decision Trees, Random Forest, and Gradient Boosting.
6. *Model Training & Evaluation:* Training the model on historical data and evaluating its performance using metrics like RMSE, R^2, and MAE.
7. *Hyperparameter Tuning:* Optimizing model parameters to improve accuracy.
8. *Deployment:* Deploying the trained model using Flask, FastAPI, or a cloud service for real-time predictions.

## Dependencies
To run this project, install the following Python libraries:
bash
pip install pandas numpy scikit-learn matplotlib seaborn flask


## Usage
1. Clone the repository:
   bash
   git clone https://github.com/Tanaya-Kanim05/Machine-learning/tree/main/pune%20housing%20dataset%20regression
   cd pune-house-price-prediction
   
2. Run the Jupyter notebook or Python script to train the model:
   bash
   python train_model.py
   
3. Start the Flask API to serve predictions:
   bash
   pune_house_app.py
   
4. Send a request to the API to get predictions:
   bash
   curl -X POST -H "Content-Type: application/json" -d '{"location": "Kothrud", "sqft": 1200, "bhk": 2, "bath": 2}' http://127.0.0.1:5000/predict
   

## Model Performance
- *RMSE:* X.XX
- *R² Score:* X.XX

## Future Enhancements
- Use advanced models like XGBoost or Neural Networks.
- Deploy the model using cloud services like AWS, GCP, or Azure.
- Develop a web-based interface for user interaction.
- Incorporate more real-time data for better predictions.

## Contributors
- Tanaya Kanim

## License
This project is licensed under the MIT License.

  

