import numpy as np
import pandas as pd
import pickle
import streamlit as st

def predict_crop(model, input_data):
    """
    Make crop prediction using the provided model and input data
    
    Parameters:
    -----------
    model : object
        Trained machine learning model loaded from pickle file
    input_data : numpy.ndarray
        Array of input features for prediction
        
    Returns:
    --------
    str
        Predicted crop name
    """
    try:
        # Check if the model has a predict method (most sklearn models do)
        if hasattr(model, 'predict'):
            prediction = model.predict(input_data)
            
            # If prediction is an array or list, get the first element
            if isinstance(prediction, (np.ndarray, list)):
                prediction = prediction[0]
                
            # Check if the model has a classes_ attribute (for classification models)
            if hasattr(model, 'classes_'):
                # If prediction is a class index, convert to class name
                if isinstance(prediction, (int, np.integer)):
                    prediction = model.classes_[prediction]
            
            return prediction
        else:
            # Try to use the model as a function
            prediction = model(input_data)
            return prediction
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return "Prediction failed"

def get_crop_recommendations(predicted_crop):
    """
    Get recommendations for the predicted crop
    
    Parameters:
    -----------
    predicted_crop : str
        Name of the predicted crop
        
    Returns:
    --------
    dict
        Dictionary containing recommendations for the crop
    """
    # Dictionary of recommendations for different crops
    recommendations = {
        "rice": {
            "fertilizer": "NPK 15-15-15, Urea",
            "soil_prep": "Puddle soil, maintain water level",
            "irrigation": "Keep soil saturated during growing season",
            "best_seasons": "Summer, Monsoon"
        },
        "wheat": {
            "fertilizer": "NPK 20-20-0, Urea",
            "soil_prep": "Fine tilth, well-drained soil",
            "irrigation": "Critical at crown root initiation, flowering, and grain filling",
            "best_seasons": "Winter"
        },
        "maize": {
            "fertilizer": "NPK 20-10-10, Urea",
            "soil_prep": "Deep plowing, well-drained soil",
            "irrigation": "Critical during tasseling and grain filling",
            "best_seasons": "Spring, Summer"
        },
        "chickpea": {
            "fertilizer": "NPK 10-20-20, Rhizobium inoculant",
            "soil_prep": "Well-drained soil, neutral pH",
            "irrigation": "Light irrigation, avoid waterlogging",
            "best_seasons": "Winter"
        },
        "cotton": {
            "fertilizer": "NPK 20-10-10, Urea",
            "soil_prep": "Deep plowing, well-drained soil",
            "irrigation": "Regular irrigation, avoid waterlogging",
            "best_seasons": "Summer"
        },
        "tea": {
            "fertilizer": "NPK 10-5-10, Organic matter",
            "soil_prep": "Well-drained acidic soil",
            "irrigation": "Regular irrigation, high humidity",
            "best_seasons": "Year-round (in suitable climate)"
        },
        "millet": {
            "fertilizer": "NPK 15-15-15, low fertilizer requirement",
            "soil_prep": "Light soil, drought resistant",
            "irrigation": "Minimal irrigation needed",
            "best_seasons": "Summer"
        }
    }
    
    # Convert predicted crop to lowercase for case-insensitive matching
    predicted_crop = predicted_crop.lower()
    
    # Return recommendations if crop is in dictionary, else return a default message
    if predicted_crop in recommendations:
        return recommendations[predicted_crop]
    else:
        return {
            "fertilizer": "Based on soil test",
            "soil_prep": "Ensure well-drained soil",
            "irrigation": "Regular irrigation as needed",
            "best_seasons": "Varies by region"
        }
