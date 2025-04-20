import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

def visualize_data(data):
    """
    Visualize agricultural data using various plots
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing agricultural data
    """
    # Check if DataFrame is empty
    if data.empty:
        st.warning("The dataset is empty. Please upload a different file.")
        return
    
    # Get column names for selection
    columns = data.columns.tolist()
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    # Display data exploration options
    st.subheader("Explore Your Data")
    
    # Filter data if needed
    if st.checkbox("Enable Data Filtering"):
        filtered_data = filter_data(data)
    else:
        filtered_data = data
    
    # Select visualization type
    viz_type = st.selectbox(
        "Select Visualization Type",
        ["Distribution Plots", "Correlation Analysis", "Time Series Analysis (if applicable)", "Custom Plot"]
    )
    
    if viz_type == "Distribution Plots":
        display_distribution_plots(filtered_data, numeric_columns)
    
    elif viz_type == "Correlation Analysis":
        display_correlation_analysis(filtered_data, numeric_columns)
    
    elif viz_type == "Time Series Analysis (if applicable)":
        display_time_series_analysis(filtered_data, columns)
    
    elif viz_type == "Custom Plot":
        display_custom_plot(filtered_data, columns, numeric_columns)

def filter_data(data):
    """
    Create UI elements for filtering data
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Original DataFrame
        
    Returns:
    --------
    pandas.DataFrame
        Filtered DataFrame
    """
    st.subheader("Filter Data")
    
    # Create a copy of the data to avoid modifying the original
    filtered_data = data.copy()
    
    # Get all column names
    columns = data.columns.tolist()
    
    # Select columns to filter on
    filter_columns = st.multiselect("Select columns to filter", columns)
    
    # Create filters for each selected column
    for column in filter_columns:
        # Check data type of column
        if pd.api.types.is_numeric_dtype(data[column]):
            # For numeric columns, create a range slider
            min_val, max_val = float(data[column].min()), float(data[column].max())
            filter_values = st.slider(
                f"Range for {column}",
                min_val, max_val, (min_val, max_val)
            )
            filtered_data = filtered_data[
                (filtered_data[column] >= filter_values[0]) & 
                (filtered_data[column] <= filter_values[1])
            ]
        else:
            # For categorical columns, create a multiselect
            unique_values = data[column].unique().tolist()
            selected_values = st.multiselect(
                f"Values for {column}",
                unique_values,
                default=unique_values
            )
            filtered_data = filtered_data[filtered_data[column].isin(selected_values)]
    
    # Show number of records after filtering
    st.write(f"Filtered data contains {len(filtered_data)} records")
    
    return filtered_data

def display_distribution_plots(data, numeric_columns):
    """
    Display distribution plots for numeric columns
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing data
    numeric_columns : list
        List of numeric column names
    """
    st.subheader("Distribution Plots")
    
    # Select columns for distribution plot
    plot_columns = st.multiselect(
        "Select columns to plot distribution",
        numeric_columns,
        default=numeric_columns[:min(3, len(numeric_columns))]
    )
    
    if not plot_columns:
        st.info("Please select at least one column to plot")
        return
    
    # Plot type selection
    plot_type = st.radio(
        "Select plot type",
        ["Histogram", "Box Plot", "Violin Plot"]
    )
    
    # Create the selected plot
    if plot_type == "Histogram":
        for column in plot_columns:
            fig = px.histogram(
                data, x=column,
                title=f"Histogram of {column}",
                labels={column: column},
                opacity=0.7
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif plot_type == "Box Plot":
        fig = px.box(
            data,
            y=plot_columns,
            title="Box Plot"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif plot_type == "Violin Plot":
        fig = px.violin(
            data,
            y=plot_columns,
            title="Violin Plot",
            box=True
        )
        st.plotly_chart(fig, use_container_width=True)

def display_correlation_analysis(data, numeric_columns):
    """
    Display correlation analysis for numeric columns
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing data
    numeric_columns : list
        List of numeric column names
    """
    st.subheader("Correlation Analysis")
    
    if len(numeric_columns) < 2:
        st.warning("Need at least two numeric columns for correlation analysis")
        return
    
    # Calculate correlation matrix
    corr_matrix = data[numeric_columns].corr()
    
    # Display correlation matrix as a table
    st.write("Correlation Matrix:")
    st.dataframe(corr_matrix.style.highlight_abs(axis=None))
    
    # Plot correlation heatmap
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        title="Correlation Heatmap",
        labels=dict(color="Correlation"),
        color_continuous_scale="RdBu_r"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature pairs correlation
    st.subheader("Explore Correlation Between Features")
    
    # Select columns for scatter plot
    x_column = st.selectbox("Select X-axis column", numeric_columns, index=0)
    y_column = st.selectbox("Select Y-axis column", numeric_columns, index=min(1, len(numeric_columns)-1))
    
    if x_column and y_column:
        # Create scatter plot
        fig = px.scatter(
            data,
            x=x_column,
            y=y_column,
            title=f"Correlation between {x_column} and {y_column}",
            trendline="ols"  # Add trend line
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display correlation value
        correlation = data[x_column].corr(data[y_column])
        st.write(f"Correlation coefficient: {correlation:.2f}")

def display_time_series_analysis(data, columns):
    """
    Display time series analysis if applicable
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing data
    columns : list
        List of column names
    """
    st.subheader("Time Series Analysis")
    
    # Check if there are date columns in the data
    date_columns = [col for col in columns if 'date' in col.lower() or 'time' in col.lower()]
    
    if not date_columns:
        st.info("No date/time columns detected. Please select a date column if one exists.")
        date_column = st.selectbox("Select a date column (if applicable)", columns)
    else:
        date_column = st.selectbox("Select a date column", date_columns)
    
    if not date_column:
        st.warning("No date column selected or available. Cannot perform time series analysis.")
        return
    
    # Try to convert the selected column to datetime
    try:
        data[date_column] = pd.to_datetime(data[date_column])
    except Exception as e:
        st.error(f"Error converting column to date: {str(e)}")
        return
    
    # Select a value column to analyze over time
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_columns:
        st.warning("No numeric columns available for time series analysis.")
        return
    
    value_column = st.selectbox("Select value to analyze over time", numeric_columns)
    
    if value_column:
        # Group by date and aggregate
        date_part = st.selectbox(
            "Group by",
            ["Day", "Month", "Year"],
            index=0
        )
        
        if date_part == "Day":
            grouped_data = data.groupby(data[date_column].dt.date)[value_column].mean().reset_index()
        elif date_part == "Month":
            grouped_data = data.groupby(data[date_column].dt.to_period("M").astype(str))[value_column].mean().reset_index()
        else:  # Year
            grouped_data = data.groupby(data[date_column].dt.year)[value_column].mean().reset_index()
        
        # Create time series plot
        fig = px.line(
            grouped_data,
            x=date_column,
            y=value_column,
            title=f"{value_column} over time ({date_part})",
            labels={date_column: "Date", value_column: value_column}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show moving average if there are enough data points
        if len(grouped_data) > 5:
            show_ma = st.checkbox("Show moving average")
            if show_ma:
                window_size = st.slider(
                    "Moving average window size",
                    min_value=2,
                    max_value=min(10, len(grouped_data) - 1),
                    value=3
                )
                
                # Calculate moving average
                grouped_data['MA'] = grouped_data[value_column].rolling(window=window_size).mean()
                
                # Create plot with moving average
                fig = px.line(
                    grouped_data,
                    x=date_column,
                    y=[value_column, 'MA'],
                    title=f"{value_column} with {window_size}-point Moving Average",
                    labels={date_column: "Date", value_column: value_column, 'MA': f'{window_size}-point MA'}
                )
                st.plotly_chart(fig, use_container_width=True)

def display_custom_plot(data, columns, numeric_columns):
    """
    Display custom plot based on user selection
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing data
    columns : list
        List of all column names
    numeric_columns : list
        List of numeric column names
    """
    st.subheader("Custom Plot")
    
    # Plot type selection
    plot_type = st.selectbox(
        "Select plot type",
        ["Scatter Plot", "Line Plot", "Bar Chart", "Pie Chart", "Bubble Chart"]
    )
    
    if plot_type == "Scatter Plot":
        # For scatter plot, we need x and y columns
        x_col = st.selectbox("Select X-axis column", numeric_columns, index=0)
        y_col = st.selectbox("Select Y-axis column", numeric_columns, index=min(1, len(numeric_columns)-1))
        
        # Optional color column
        color_col = st.selectbox("Color points by (optional)", ["None"] + columns, index=0)
        color_col = None if color_col == "None" else color_col
        
        # Create scatter plot
        fig = px.scatter(
            data,
            x=x_col,
            y=y_col,
            color=color_col,
            title=f"Scatter Plot: {y_col} vs {x_col}",
            labels={x_col: x_col, y_col: y_col}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif plot_type == "Line Plot":
        # For line plot, we need x and y columns
        x_col = st.selectbox("Select X-axis column", columns, index=0)
        y_cols = st.multiselect("Select Y-axis column(s)", numeric_columns, default=[numeric_columns[0]])
        
        if not y_cols:
            st.warning("Please select at least one Y-axis column")
            return
        
        # Create line plot
        fig = px.line(
            data,
            x=x_col,
            y=y_cols,
            title=f"Line Plot",
            labels={x_col: x_col}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif plot_type == "Bar Chart":
        # For bar chart
        x_col = st.selectbox("Select X-axis column", columns, index=0)
        y_col = st.selectbox("Select Y-axis column (value to plot)", numeric_columns, index=0)
        
        # Aggregation method
        agg_method = st.selectbox(
            "Aggregation method",
            ["Sum", "Mean", "Count", "Min", "Max"],
            index=0
        )
        
        # Prepare data for bar chart
        if agg_method == "Sum":
            agg_data = data.groupby(x_col)[y_col].sum().reset_index()
        elif agg_method == "Mean":
            agg_data = data.groupby(x_col)[y_col].mean().reset_index()
        elif agg_method == "Count":
            agg_data = data.groupby(x_col)[y_col].count().reset_index()
        elif agg_method == "Min":
            agg_data = data.groupby(x_col)[y_col].min().reset_index()
        else:  # Max
            agg_data = data.groupby(x_col)[y_col].max().reset_index()
        
        # Create bar chart
        fig = px.bar(
            agg_data,
            x=x_col,
            y=y_col,
            title=f"Bar Chart: {agg_method} of {y_col} by {x_col}",
            labels={x_col: x_col, y_col: f"{agg_method} of {y_col}"}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif plot_type == "Pie Chart":
        # For pie chart
        cat_col = st.selectbox("Select category column", columns, index=0)
        value_col = st.selectbox("Select value column", numeric_columns, index=0)
        
        # Prepare data for pie chart
        pie_data = data.groupby(cat_col)[value_col].sum().reset_index()
        
        # Create pie chart
        fig = px.pie(
            pie_data,
            names=cat_col,
            values=value_col,
            title=f"Pie Chart: Distribution of {value_col} by {cat_col}"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif plot_type == "Bubble Chart":
        # For bubble chart, we need x, y, and size columns
        x_col = st.selectbox("Select X-axis column", numeric_columns, index=0)
        y_col = st.selectbox("Select Y-axis column", numeric_columns, index=min(1, len(numeric_columns)-1))
        size_col = st.selectbox("Select size column", numeric_columns, index=min(2, len(numeric_columns)-1))
        
        # Optional color column
        color_col = st.selectbox("Color bubbles by (optional)", ["None"] + columns, index=0)
        color_col = None if color_col == "None" else color_col
        
        # Create bubble chart
        fig = px.scatter(
            data,
            x=x_col,
            y=y_col,
            size=size_col,
            color=color_col,
            title=f"Bubble Chart: {y_col} vs {x_col} (size: {size_col})",
            labels={x_col: x_col, y_col: y_col, size_col: size_col}
        )
        st.plotly_chart(fig, use_container_width=True)
