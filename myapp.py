import pandas as pd
import plotly.express as px
import streamlit as st
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle
import pyodbc

st.set_page_config(page_title="my analytics app", page_icon='📊')

st.title(':rainbow[DATA ANALYTICS PORTAL]')
st.subheader(':grey[Explore data with ease.]', divider='rainbow')

file = st.file_uploader('Drop CSV or Excel file', type=['csv', 'xlsx'])
if file is not None:
    if file.name.endswith('xlsx'):
        data = pd.read_excel(file)
    else:
        data = pd.read_csv(file)
    
    st.dataframe(data)
    st.info("File is successfully uploaded", icon="🔥")
    
    st.subheader(':rainbow[Basic information of the Dataset]', divider='rainbow')
    tab1, tab2, tab3, tab4 = st.tabs(['Summary', 'Top and Bottom Rows', 'Data Types', 'Columns'])
    
    with tab1:
        st.write(f'There are {data.shape[0]} rows in the dataset and {data.shape[1]} columns.')
        st.subheader(':gray[Statistical Summary]')
        st.dataframe(data.describe())
    
    with tab2:
        st.subheader(':gray[Top Rows]')
        toprows = st.slider('Number of top rows you want', 1, data.shape[0], key='topslider')
        st.dataframe(data.head(toprows))
        st.subheader(':gray[Bottom Rows]')
        bottomrows = st.slider('Number of bottom rows you want', 1, data.shape[0], key='bottomslider')
        st.dataframe(data.tail(bottomrows))
    
    with tab3:
        st.subheader(':grey[Data types of columns]')
        st.dataframe(data.dtypes)
    
    with tab4:
        st.subheader(':grey[Column names in dataset]')
        st.write(list(data.columns))
    
    # Handle Missing Data
    st.subheader(':rainbow[Handle Missing Data]', divider='rainbow')
    if st.checkbox('Show missing values per column'):
        missing_data = data.isnull().sum()
        st.dataframe(missing_data)
    
    if st.checkbox('Fill Missing Values'):
        fill_value = st.selectbox('Choose a value to fill missing data', ['Mean', 'Median', 'Mode', 'Zero'])
        if fill_value == 'Mean':
            data = data.fillna(data.mean())
        elif fill_value == 'Median':
            data = data.fillna(data.median())
        elif fill_value == 'Mode':
            data = data.fillna(data.mode().iloc[0])
        elif fill_value == 'Zero':
            data = data.fillna(0)
        st.success("Missing values filled")
        st.dataframe(data)

    # Outlier Detection
    st.subheader(':rainbow[Detect Outliers]', divider='rainbow')
    outlier_method = st.selectbox('Select outlier detection method', ['IQR', 'Z-score'])
    column_to_check = st.selectbox('Select column for outlier detection', options=data.select_dtypes(include=['float64', 'int64']).columns)

    if outlier_method == 'IQR':
        Q1 = data[column_to_check].quantile(0.25)
        Q3 = data[column_to_check].quantile(0.75)
        IQR = Q3 - Q1
        outliers = data[(data[column_to_check] < (Q1 - 1.5 * IQR)) | (data[column_to_check] > (Q3 + 1.5 * IQR))]
        st.dataframe(outliers)
    elif outlier_method == 'Z-score':
        z_scores = stats.zscore(data[column_to_check])
        outliers = data[(z_scores > 3) | (z_scores < -3)]
        st.dataframe(outliers)

    # Feature Scaling
    st.subheader(':rainbow[Feature Scaling]', divider='rainbow')
    if st.checkbox('Scale Features'):
        scale_method = st.selectbox('Choose scaling method', ['Standard', 'Min-Max'])
        columns_to_scale = st.multiselect('Select columns to scale', options=data.select_dtypes(include=['float64', 'int64']).columns)
        if scale_method == 'Standard':
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
        data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])
        st.dataframe(data)

    # Correlation Matrix
    st.subheader(':rainbow[Correlation Matrix]', divider='rainbow')
    if st.checkbox('Show Correlation Matrix'):
        corr_matrix = data.corr()
        st.dataframe(corr_matrix)
        fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r')
        st.plotly_chart(fig)

    # Data Transformation
    st.subheader(':rainbow[Data Transformation]', divider='rainbow')
    
    # Encoding Categorical Columns
    if st.checkbox('Encode Categorical Columns'):
        columns_to_encode = st.multiselect('Select columns to encode', data.select_dtypes(include='object').columns)
        data = pd.get_dummies(data, columns=columns_to_encode)
        st.dataframe(data)
    
    # Extract Features from Date
    if st.checkbox('Extract Features from Date Columns'):
        date_col = st.selectbox('Select Date Column', data.select_dtypes(include='datetime').columns)
        data['Year'] = data[date_col].dt.year
        data['Month'] = data[date_col].dt.month
        data['Day'] = data[date_col].dt.day
        st.dataframe(data)

    # Value Count
    st.subheader(':rainbow[Column Values To Count]', divider='rainbow')
    with st.expander('Value Count'):
        col1, col2 = st.columns(2)
        with col1:
            column = st.selectbox('Choose Column Name', options=list(data.columns))
        with col2:
            toprows = st.number_input('Top Rows', min_value=1, step=1)

        count = st.button('Count')
        if count:
            result = data[column].value_counts().reset_index().head(toprows)
            st.dataframe(result)
            st.subheader('Visualization', divider='gray')
            fig = px.bar(data_frame=result, x='index', y=column, text=column, template='plotly_white')
            st.plotly_chart(fig)
            fig = px.pie(data_frame=result, names='index', values=column)
            st.plotly_chart(fig)

    # Groupby
    st.subheader(':rainbow[Groupby : Simplify Your Data Analysis]', divider='rainbow')
    with st.expander('Group By your columns'):
        col1, col2, col3 = st.columns(3)
        with col1:
            groupby_cols = st.multiselect('Choose columns to groupby', options=list(data.columns))
        with col2:
            operation_cols = st.selectbox('Choose column for operation', options=list(data.columns))
        with col3:
            operation = st.selectbox('Choose operation', options=['sum', 'max', 'min', 'mean', 'median', 'count'])

        if groupby_cols:
            result = data.groupby(groupby_cols).agg(
                newcol=(operation_cols, operation)
            ).reset_index()
            st.dataframe(result)
            st.subheader(':gray[Data Visualization]', divider='gray')
            graphs = st.selectbox('Choose your graphs', options=['line', 'bar', 'scatter', 'pie', 'sunburst'])
            if graphs == 'line':
                x_axis = st.selectbox('Choose X axis', options=list(result.columns))
                y_axis = st.selectbox('Choose Y axis', options=list(result.columns))
                fig = px.line(data_frame=result, x=x_axis, y=y_axis, markers=True)
                st.plotly_chart(fig)
            elif graphs == 'bar':
                x_axis = st.selectbox('Choose X axis', options=list(result.columns))
                y_axis = st.selectbox('Choose Y axis', options=list(result.columns))
                fig = px.bar(data_frame=result, x=x_axis, y=y_axis)
                st.plotly_chart(fig)

    # Machine Learning Model
    st.subheader(':rainbow[Build a Regression Model]', divider='rainbow')
    if st.button('Train Linear Regression Model'):
        target = st.selectbox('Choose Target Variable', options=list(data.columns))
        features = st.multiselect('Choose Features', options=list(data.columns))

        X = data[features]
        y = data[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        st.write(f'Mean Squared Error: {mean_squared_error(y_test, predictions)}')

    # Load and Predict with Pre-trained Model
    st.subheader(':rainbow[Predict Using Pre-Trained Model]')
    model_file = st.file_uploader('Upload Your Pre-trained Model', type=['pkl'])
    if model_file:
        model = pickle.load(model_file)
        prediction_data = data.drop(columns=[target])
        predictions = model.predict(prediction_data)
        st
