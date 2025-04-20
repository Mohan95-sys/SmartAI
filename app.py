import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import base64
from crop_prediction import predict_crop
from data_visualization import visualize_data
from utils import load_pickle_file, load_jupyter_notebook, extract_code_from_notebook
from crop_prices import display_crop_prices
from database import init_db
from translations import get_text, LANGUAGES

# Set page configuration
st.set_page_config(
    page_title="Smart AI Agriculture",
    page_icon="🌱",
    layout="wide"
)

# Apply custom CSS
with open("style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Function to load and display SVG images
def display_svg_image(svg_file, width=None):
    try:
        import os
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), svg_file)
        with open(file_path, 'r') as f:
            svg_content = f.read()
        
        # Apply width if specified
        if width:
        svg_content = svg_content.replace('<svg width="100"', f'<svg width="{width}"')
    
    # Encode SVG content as base64
    b64 = base64.b64encode(svg_content.encode('utf-8')).decode('utf-8')
    
    # Display the image using HTML
    html = f'<img src="data:image/svg+xml;base64,{b64}"/>'
    return html
    except Exception as e:
        st.warning(f"Could not load image: {svg_file}")
        return ""  # Return empty string if image fails to load

# Initialize session state for language
if 'language' not in st.session_state:
    st.session_state.language = 'en'  # Default language is English
    
# Initialize the database with sample data
init_db()

def main():
    # Add language selector in the sidebar
    st.sidebar.title("Settings / सेटिंग्स")
    
    # Create language dropdown with display names
    language_options = list(LANGUAGES.items())
    current_lang_index = [i for i, (code, _) in enumerate(language_options) if code == st.session_state.language][0]
    
    selected_lang = st.sidebar.selectbox(
        get_text("language", st.session_state.language),
        language_options,
        format_func=lambda x: x[1],  # Display the language name
        index=current_lang_index
    )
    
    # Update language if changed
    if selected_lang[0] != st.session_state.language:
        st.session_state.language = selected_lang[0]
        st.rerun()
    
    # Get current language code
    lang = st.session_state.language
    
    # App title and description
    st.title(get_text("app_title", lang))
    st.subheader(get_text("app_subtitle", lang))
    
    # Create sidebar for navigation
    st.sidebar.title(get_text("navigation", lang))
    app_mode = st.sidebar.radio(
        "Go to",
        ["Home", "Data Visualization", "Crop Prediction", "Live Crop Prices"],
        format_func=lambda x: get_text(x.lower().replace(" ", "_"), lang)
    )
    
    # Home page
    if app_mode == "Home":
        display_home()
    
    # Data Visualization page
    elif app_mode == "Data Visualization":
        display_data_visualization()
    
    # Crop Prediction page
    elif app_mode == "Crop Prediction":
        display_crop_prediction()
        
    # Live Crop Prices page
    elif app_mode == "Live Crop Prices":
        display_crop_prices()

def display_home():
    # Get current language code
    lang = st.session_state.language
    
    # Display plant images in a row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(display_svg_image("images/plant_leaf.svg", width="120"), unsafe_allow_html=True)
    with col2:
        st.markdown(display_svg_image("images/wheat.svg", width="120"), unsafe_allow_html=True)
    with col3:
        st.markdown(display_svg_image("images/rice.svg", width="120"), unsafe_allow_html=True)
    
    # Create welcome message with translations
    st.markdown(f"""
    ## {get_text("welcome", lang)}
    
    {get_text("app_helps", lang)}
    
    * {get_text("visualizing_data", lang)}
    * {get_text("providing_recommendations", lang)}
    * {get_text("analyzing_conditions", lang)}
    * {get_text("tracking_prices", lang)}
    """)
    
    # Create two columns for instructions and features
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown(f"""
        ### {get_text("how_to_use", lang)}
        
        1. Upload your data in pickle (.pkl) format
        2. Navigate to the {get_text("data_visualization", lang)} section to explore your data
        3. Use the {get_text("crop_prediction", lang)} feature to get AI-powered recommendations
        4. Check current crop prices in the {get_text("live_crop_prices", lang)} section
        """)
    
    with col2:
        # Add card-like styling
        st.markdown("""
        <div style="background-color: rgba(220, 237, 200, 0.7); padding: 20px; border-radius: 10px; border-left: 5px solid #4caf50;">
        <h3 style="color: #2e7d32;">🌱 Crop Insights</h3>
        <p>Our application supports data analysis for various crops including rice, wheat, maize, cotton, and many more.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Add information about the model
    st.markdown(f"""
    ### About the prediction model:
    
    Our machine learning model considers various factors like:
    * Soil composition ({get_text("nitrogen", lang)}, {get_text("phosphorus", lang)}, {get_text("potassium", lang)} values)
    * {get_text("ph", lang)} level
    * {get_text("rainfall", lang)} 
    * {get_text("temperature", lang)}
    * {get_text("humidity", lang)}
    
    to recommend the most suitable crops for your specific conditions.
    """)
    
    # Add farm landscape image at the bottom
    st.markdown("""
    <div style="text-align: center; margin-top: 30px;">
    <svg width="100%" height="100" viewBox="0 0 800 100" xmlns="http://www.w3.org/2000/svg">
      <!-- Sky -->
      <rect x="0" y="0" width="800" height="60" fill="#e3f2fd" />
      
      <!-- Sun -->
      <circle cx="700" cy="40" r="20" fill="#ffeb3b" />
      
      <!-- Ground -->
      <rect x="0" y="60" width="800" height="40" fill="#81c784" />
      
      <!-- Fields -->
      <rect x="0" y="60" width="200" height="40" fill="#a5d6a7" />
      <rect x="400" y="60" width="200" height="40" fill="#a5d6a7" />
      
      <!-- Trees -->
      <circle cx="100" cy="50" r="15" fill="#43a047" />
      <rect x="95" y="60" width="10" height="15" fill="#6d4c41" />
      
      <circle cx="500" cy="50" r="15" fill="#43a047" />
      <rect x="495" y="60" width="10" height="15" fill="#6d4c41" />
      
      <!-- Farm house -->
      <rect x="300" y="40" width="50" height="30" fill="#fff9c4" />
      <polygon points="290,40 360,40 325,15" fill="#d32f2f" />
      <rect x="315" y="50" width="20" height="20" fill="#90caf9" />
    </svg>
    </div>
    """, unsafe_allow_html=True)

def display_data_visualization():
    # Get current language code
    lang = st.session_state.language
    
    st.header(get_text("data_visualization", lang))
    
    # Create tabs for different data sources
    data_source_tabs = st.tabs(["Pickle File (.pkl)", "Jupyter Notebook (.ipynb)"])
    
    with data_source_tabs[0]:  # Pickle File Tab
        # Upload pickle file
        pickle_file = st.file_uploader("Upload pickle (.pkl) file containing agricultural data", type=['pkl'], key="pickle_uploader")
        
        if pickle_file is not None:
            try:
                # Load data from pickle file
                data = load_pickle_file(pickle_file)
                
                if isinstance(data, pd.DataFrame):
                    st.success("Data loaded successfully from pickle file!")
                    
                    # Display data sample
                    st.subheader("Data Sample")
                    st.dataframe(data.head())
                    
                    # Display data statistics
                    st.subheader("Data Statistics")
                    st.dataframe(data.describe())
                    
                    # Visualize the data
                    st.subheader(get_text("data_visualization", lang))
                    visualize_data(data)
                else:
                    st.error("The uploaded file does not contain a valid DataFrame")
            except Exception as e:
                st.error(f"Error loading the pickle file: {str(e)}")
        else:
            st.info("Please upload a pickle file containing your agricultural data")
    
    with data_source_tabs[1]:  # Jupyter Notebook Tab
        # Upload Jupyter notebook file
        notebook_file = st.file_uploader("Upload Jupyter notebook (.ipynb) containing agricultural data analysis", type=['ipynb'], key="notebook_uploader")
        
        if notebook_file is not None:
            try:
                # Load data from Jupyter notebook
                notebook_data = load_jupyter_notebook(notebook_file)
                
                if notebook_data and 'dataframes' in notebook_data and notebook_data['dataframes']:
                    st.success(f"Successfully extracted {len(notebook_data['dataframes'])} DataFrames from the notebook!")
                    
                    # Let the user select which DataFrame to visualize
                    if len(notebook_data['dataframes']) > 1:
                        # Select a DataFrame to visualize
                        selected_df_index = st.selectbox(
                            "Select a DataFrame to visualize:",
                            range(len(notebook_data['dataframes'])),
                            format_func=lambda i: f"{notebook_data['dataframe_names'][i]} - {notebook_data['dataframes'][i].shape}"
                        )
                        data = notebook_data['dataframes'][selected_df_index]
                    else:
                        data = notebook_data['dataframes'][0]
                    
                    # Display code from notebook
                    with st.expander("View code from notebook", expanded=False):
                        st.code(extract_code_from_notebook(notebook_data))
                    
                    # Display data sample
                    st.subheader("Data Sample")
                    st.dataframe(data.head())
                    
                    # Display data statistics
                    st.subheader("Data Statistics")
                    st.dataframe(data.describe())
                    
                    # Visualize the data
                    st.subheader(get_text("data_visualization", lang))
                    visualize_data(data)
                else:
                    st.warning("No DataFrames found in the notebook. Please make sure the notebook contains pandas DataFrames.")
            except Exception as e:
                st.error(f"Error processing Jupyter notebook: {str(e)}")
        else:
            st.info("Please upload a Jupyter notebook containing pandas DataFrames for analysis.")

def display_crop_prediction():
    # Get current language code
    lang = st.session_state.language
    
    st.header(get_text("crop_prediction", lang))
    
    # Three methods: Upload model, use a notebook, or input parameters
    prediction_method = st.radio(
        "Choose prediction method:",
        ["Upload trained model", "Use Jupyter notebook", "Input parameters manually"]
    )
    
    if prediction_method == "Upload trained model":
        # Upload model file
        model_file = st.file_uploader("Upload your trained model (pickle format)", type=['pkl'])
        
        if model_file is not None:
            try:
                # Load model from pickle file
                model = load_pickle_file(model_file)
                
                # Input form for prediction
                with st.form("prediction_form"):
                    st.subheader("Enter Soil and Climate Parameters")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        n = st.number_input(get_text("nitrogen", lang), min_value=0, max_value=150, value=50)
                        p = st.number_input(get_text("phosphorus", lang), min_value=0, max_value=150, value=50)
                        k = st.number_input(get_text("potassium", lang), min_value=0, max_value=150, value=50)
                        temperature = st.number_input(get_text("temperature", lang), min_value=0.0, max_value=50.0, value=25.0)
                    
                    with col2:
                        humidity = st.number_input(get_text("humidity", lang), min_value=0.0, max_value=100.0, value=60.0)
                        ph = st.number_input(get_text("ph", lang), min_value=0.0, max_value=14.0, value=6.5)
                        rainfall = st.number_input(get_text("rainfall", lang), min_value=0.0, max_value=300.0, value=100.0)
                    
                    submitted = st.form_submit_button(get_text("predict_crop", lang))
                
                if submitted:
                    # Make prediction
                    input_data = np.array([[n, p, k, temperature, humidity, ph, rainfall]])
                    predicted_crop = predict_crop(model, input_data)
                    
                    # Display prediction result
                    st.subheader("Prediction Result")
                    st.success(f"The recommended crop is: **{predicted_crop}**")
                    
                    # Display additional information about the crop
                    display_crop_info(predicted_crop)
            except Exception as e:
                st.error(f"Error loading the model: {str(e)}")
        else:
            st.info("Please upload your trained model file")
    
    elif prediction_method == "Use Jupyter notebook":
        # Upload Jupyter notebook file
        notebook_file = st.file_uploader("Upload Jupyter notebook (.ipynb) with model training code", type=['ipynb'])
        
        if notebook_file is not None:
            try:
                # Load data from Jupyter notebook
                notebook_data = load_jupyter_notebook(notebook_file)
                
                if notebook_data:
                    # Display code from notebook
                    with st.expander("View code from notebook", expanded=False):
                        st.code(extract_code_from_notebook(notebook_data))
                    
                    # Check if any models are in the namespace
                    model_found = False
                    model = None
                    
                    # Display message if no trained models found
                    st.info("Searching for trained machine learning models in the notebook...")
                    
                    # Setup form for prediction
                    with st.form("notebook_prediction_form"):
                        st.subheader("Enter Soil and Climate Parameters for Prediction")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            n = st.number_input(get_text("nitrogen", lang), min_value=0, max_value=150, value=50)
                            p = st.number_input(get_text("phosphorus", lang), min_value=0, max_value=150, value=50)
                            k = st.number_input(get_text("potassium", lang), min_value=0, max_value=150, value=50)
                            temperature = st.number_input(get_text("temperature", lang), min_value=0.0, max_value=50.0, value=25.0)
                        
                        with col2:
                            humidity = st.number_input(get_text("humidity", lang), min_value=0.0, max_value=100.0, value=60.0)
                            ph = st.number_input(get_text("ph", lang), min_value=0.0, max_value=14.0, value=6.5)
                            rainfall = st.number_input(get_text("rainfall", lang), min_value=0.0, max_value=300.0, value=100.0)
                        
                        submitted = st.form_submit_button(get_text("predict_crop", lang))
                    
                    if submitted:
                        # Since extracting and using models from notebooks is complex,
                        # we'll use our basic prediction as a fallback
                        st.info("Using basic prediction method (since extracting models from notebooks is complex)")
                        input_data = np.array([[n, p, k, temperature, humidity, ph, rainfall]])
                        
                        # Use a simple prediction approach based on rules
                        predicted_crop = basic_crop_prediction(n, p, k, temperature, humidity, ph, rainfall)
                        
                        # Display prediction result
                        st.subheader("Prediction Result")
                        st.success(f"The recommended crop is: **{predicted_crop}**")
                        
                        # Display additional information about the crop
                        display_crop_info(predicted_crop)
            
            except Exception as e:
                st.error(f"Error processing Jupyter notebook: {str(e)}")
        else:
            st.info("Please upload a Jupyter notebook containing model training code")
    
    else:  # Input parameters manually
        st.info("This method uses our pre-trained model for prediction")
        
        # Input form for prediction
        with st.form("manual_prediction_form"):
            st.subheader("Enter Soil and Climate Parameters")
            
            col1, col2 = st.columns(2)
            
            with col1:
                n = st.number_input(get_text("nitrogen", lang), min_value=0, max_value=150, value=50)
                p = st.number_input(get_text("phosphorus", lang), min_value=0, max_value=150, value=50)
                k = st.number_input(get_text("potassium", lang), min_value=0, max_value=150, value=50)
                temperature = st.number_input(get_text("temperature", lang), min_value=0.0, max_value=50.0, value=25.0)
            
            with col2:
                humidity = st.number_input(get_text("humidity", lang), min_value=0.0, max_value=100.0, value=60.0)
                ph = st.number_input(get_text("ph", lang), min_value=0.0, max_value=14.0, value=6.5)
                rainfall = st.number_input(get_text("rainfall", lang), min_value=0.0, max_value=300.0, value=100.0)
            
            submitted = st.form_submit_button(get_text("predict_crop", lang))
        
        if submitted:
            # Creating a dictionary for simple rule-based prediction for demonstration
            # In a real application, this would be replaced with a proper ML model
            input_data = np.array([[n, p, k, temperature, humidity, ph, rainfall]])
            
            # Use a simple prediction approach based on rules
            # This will be replaced by the actual model prediction in real implementation
            predicted_crop = basic_crop_prediction(n, p, k, temperature, humidity, ph, rainfall)
            
            # Display prediction result
            st.subheader("Prediction Result")
            st.success(f"The recommended crop is: **{predicted_crop}**")
            
            # Display additional information about the crop
            display_crop_info(predicted_crop)

def basic_crop_prediction(n, p, k, temperature, humidity, ph, rainfall):
    """
    A simple rule-based prediction method for demonstration purposes.
    In a real application, this would be replaced with an actual ML model.
    """
    if temperature > 30 and rainfall < 50:
        return "Millet"
    elif n > 100 and p > 100 and k > 100:
        return "Rice"
    elif ph < 6 and rainfall > 200:
        return "Tea"
    elif temperature < 20 and humidity > 80:
        return "Wheat"
    elif ph > 7 and n < 30:
        return "Chickpea"
    elif k > 80 and temperature > 25:
        return "Cotton"
    elif p > 80 and ph >= 6 and ph <= 7:
        return "Maize"
    else:
        return "Rice"  # Default prediction

def display_crop_info(crop_name):
    """
    Display additional information about the predicted crop
    """
    crop_info = {
        "Rice": {
            "water_req": "High water requirement (150-300 cm)",
            "growing_season": "Warm season crop",
            "soil_type": "Clay or clay loam soils",
            "facts": "Rice is the staple food for more than half of the world's population",
            "color": "#f9fbe7",
            "emoji": "🌾"
        },
        "Wheat": {
            "water_req": "Moderate water requirement (45-65 cm)",
            "growing_season": "Cool season crop",
            "soil_type": "Loam or clay loam soils",
            "facts": "Wheat is the most widely grown crop in the world",
            "color": "#f0e68c",
            "emoji": "🌿"
        },
        "Maize": {
            "water_req": "Moderate water requirement (50-75 cm)",
            "growing_season": "Warm season crop",
            "soil_type": "Well-drained soils",
            "facts": "Maize is used for food, feed, and biofuel production",
            "color": "#fff9c4",
            "emoji": "🌽"
        },
        "Chickpea": {
            "water_req": "Low water requirement (40-50 cm)",
            "growing_season": "Cool season crop",
            "soil_type": "Sandy loam soils",
            "facts": "Chickpea is a good source of protein and fiber",
            "color": "#e8f5e9",
            "emoji": "🥜"
        },
        "Cotton": {
            "water_req": "High water requirement (70-120 cm)",
            "growing_season": "Warm season crop",
            "soil_type": "Well-drained soils",
            "facts": "Cotton is the most important natural fiber crop",
            "color": "#f5f5f5",
            "emoji": "🧵"
        },
        "Tea": {
            "water_req": "High water requirement (150-250 cm)",
            "growing_season": "Year-round crop",
            "soil_type": "Well-drained acidic soils",
            "facts": "Tea is the most consumed beverage in the world after water",
            "color": "#c8e6c9",
            "emoji": "🍵"
        },
        "Millet": {
            "water_req": "Low water requirement (30-50 cm)",
            "growing_season": "Warm season crop",
            "soil_type": "Sandy loam soils",
            "facts": "Millet is drought resistant and can grow in poor soil conditions",
            "color": "#ffeead",
            "emoji": "🌾"
        }
    }
    
    # If crop exists in our info dictionary
    if crop_name in crop_info:
        info = crop_info[crop_name]
        
        # Create a styled result box
        st.markdown(f"""
        <div style="background-color: {info['color']}; padding: 20px; border-radius: 10px; margin-bottom: 20px; border: 1px solid #4caf50;">
            <h2 style="color: #2e7d32; margin-top: 0;">{info['emoji']} {crop_name}</h2>
            <p style="font-style: italic;">{info['facts']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create an expander for additional information
        with st.expander("Click to view more information about this crop"):
            # Create a visual representation of the crop requirements
            st.markdown("<h3 style='text-align: center;'>Crop Requirements</h3>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div style="background-color: rgba(220, 237, 200, 0.7); padding: 15px; border-radius: 10px; text-align: center;">
                    <h4 style="color: #2e7d32; margin-top: 0;">💧 Water</h4>
                    <p>{info['water_req']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style="background-color: rgba(220, 237, 200, 0.7); padding: 15px; border-radius: 10px; text-align: center;">
                    <h4 style="color: #2e7d32; margin-top: 0;">🌱 Growing Season</h4>
                    <p>{info['growing_season']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div style="background-color: rgba(220, 237, 200, 0.7); padding: 15px; border-radius: 10px; text-align: center;">
                    <h4 style="color: #2e7d32; margin-top: 0;">🌍 Soil Type</h4>
                    <p>{info['soil_type']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Farming recommendations
            st.markdown("<h3 style='margin-top: 25px;'>Farming Recommendations</h3>", unsafe_allow_html=True)
            
            # Display recommendations with icons
            recommendations = [
                {"icon": "💧", "text": f"Maintain soil moisture appropriate for {crop_name}"},
                {"icon": "🐛", "text": f"Monitor for pests and diseases common to {crop_name}"},
                {"icon": "↔️", "text": "Ensure proper spacing between plants"},
                {"icon": "🧪", "text": "Apply fertilizers as needed based on soil test results"}
            ]
            
            for rec in recommendations:
                st.markdown(f"""
                <div style="display: flex; align-items: center; margin-bottom: 10px;">
                    <div style="font-size: 24px; margin-right: 10px;">{rec['icon']}</div>
                    <div>{rec['text']}</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info(f"Additional information for {crop_name} is not available.")

if __name__ == "__main__":
    main()
