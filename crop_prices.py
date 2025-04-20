import streamlit as st
import pandas as pd
import numpy as np
import time
import trafilatura
import re
from database import get_crop_prices_from_db
from translations import get_text

def get_crop_prices_data():
    """
    Get crop price data from the database or external sources.
    
    Returns:
    --------
    tuple
        (global_prices, india_prices) containing dataframes with price data
    """
    try:
        # Get data from database
        global_df, india_df = get_crop_prices_from_db()
        
        # Attempt to fetch real-time data from websites if needed
        try:
            # This is a placeholder for actual web scraping functionality
            # In a real application, this would fetch live data from reliable sources
            scrape_successful = False
            
            if scrape_successful:
                # Process and update the dataframes with real data
                pass
                
        except Exception as e:
            st.warning(f"Could not fetch real-time data. Using database data. Error: {str(e)}")
        
        return global_df, india_df
    
    except Exception as e:
        st.error(f"Error fetching crop prices: {str(e)}")
        # Return empty dataframes if there's an error
        return pd.DataFrame(), pd.DataFrame()

def display_crop_prices():
    """
    Display crop prices from global and Indian markets
    """
    # Get current language code
    lang = st.session_state.language
    
    st.header(get_text("live_crop_prices", lang))
    
    # Get crop price data
    global_prices, india_prices = get_crop_prices_data()
    
    if global_prices.empty or india_prices.empty:
        st.error("Could not load crop price data. Please try again later.")
        return
    
    # Create tabs for Global and India prices
    tab1, tab2 = st.tabs([get_text("global_market", lang), get_text("india_market", lang)])
    
    with tab1:
        st.subheader("Global Crop Prices")
        st.markdown("*Prices in USD per Metric Ton*")
        
        # Format data for display
        formatted_global = global_prices.copy()
        formatted_global['Change (%)'] = formatted_global['Change (%)'].apply(
            lambda x: f"+{x:.2f}%" if x > 0 else f"{x:.2f}%"
        )
        
        # Display as a table with conditional formatting
        st.dataframe(
            formatted_global,
            column_config={
                "Price (USD/MT)": st.column_config.NumberColumn(format="$%.2f"),
                "Change (%)": st.column_config.TextColumn()
            },
            hide_index=True
        )
        
        # Add a note about data source
        st.caption("Data sources: International commodity exchanges, USDA, and FAO")
        
        # Allow filtering by crop
        selected_global_crop = st.selectbox(
            "Select a crop for detailed information",
            global_prices['Crop'].tolist(),
            key="global_crop_selector"
        )
        
        # Display detailed information for selected crop
        if selected_global_crop:
            crop_data = global_prices[global_prices['Crop'] == selected_global_crop].iloc[0]
            
            st.subheader(f"{selected_global_crop} - Global Market Analysis")
            
            # Create columns for information
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    label="Current Price (USD/MT)",
                    value=f"${crop_data['Price (USD/MT)']:.2f}",
                    delta=f"{crop_data['Change (%)']}%"
                )
            
            with col2:
                st.metric(
                    label="Last Updated",
                    value=crop_data['Last Updated']
                )
            
            # Add some analysis text based on the price trend
            if crop_data['Change (%)'] > 0:
                st.info(f"{selected_global_crop} prices are trending upward in the global market, indicating increased demand or reduced supply.")
            elif crop_data['Change (%)'] < 0:
                st.info(f"{selected_global_crop} prices are trending downward in the global market, potentially due to increased supply or reduced demand.")
            else:
                st.info(f"{selected_global_crop} prices are stable in the global market.")
    
    with tab2:
        st.subheader("India Crop Prices")
        st.markdown("*Prices in INR per Quintal*")
        
        # Format data for display
        formatted_india = india_prices.copy()
        formatted_india['Change (%)'] = formatted_india['Change (%)'].apply(
            lambda x: f"+{x:.2f}%" if x > 0 else f"{x:.2f}%"
        )
        
        # Display as a table with conditional formatting
        st.dataframe(
            formatted_india,
            column_config={
                "Price (INR/Quintal)": st.column_config.NumberColumn(format="₹%d"),
                "Change (%)": st.column_config.TextColumn()
            },
            hide_index=True
        )
        
        # Add a note about data source
        st.caption("Data sources: Indian agricultural markets, Ministry of Agriculture & Farmers Welfare")
        
        # Allow filtering by crop
        selected_india_crop = st.selectbox(
            "Select a crop for detailed information",
            india_prices['Crop'].tolist(),
            key="india_crop_selector"
        )
        
        # Display detailed information for selected crop
        if selected_india_crop:
            crop_data = india_prices[india_prices['Crop'] == selected_india_crop].iloc[0]
            
            st.subheader(f"{selected_india_crop} - Indian Market Analysis")
            
            # Create columns for information
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    label="Current Price (INR/Quintal)",
                    value=f"₹{crop_data['Price (INR/Quintal)']:,}",
                    delta=f"{crop_data['Change (%)']}%"
                )
            
            with col2:
                st.metric(
                    label="Last Updated",
                    value=crop_data['Last Updated']
                )
            
            # Add MSP information for relevant crops
            msp_crops = {
                'Rice': 2183,
                'Wheat': 2275,
                'Maize': 2090,
                'Cotton': 6620,
                'Millet': 2350
            }
            
            if selected_india_crop in msp_crops:
                st.info(f"Minimum Support Price (MSP) for {selected_india_crop}: ₹{msp_crops[selected_india_crop]} per quintal")
            
            # Add some analysis text based on the price trend
            if crop_data['Change (%)'] > 1.5:
                st.info(f"{selected_india_crop} prices are trending significantly upward in the Indian market.")
            elif crop_data['Change (%)'] > 0:
                st.info(f"{selected_india_crop} prices are showing a slight upward trend in the Indian market.")
            elif crop_data['Change (%)'] < -1.5:
                st.info(f"{selected_india_crop} prices are trending significantly downward in the Indian market.")
            elif crop_data['Change (%)'] < 0:
                st.info(f"{selected_india_crop} prices are showing a slight downward trend in the Indian market.")
            else:
                st.info(f"{selected_india_crop} prices are stable in the Indian market.")
    
    # Add a refresh button
    if st.button(get_text("refresh_prices", lang)):
        st.rerun()