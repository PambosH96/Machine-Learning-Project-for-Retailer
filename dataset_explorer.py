#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import streamlit as st

# Exploratory Data analysis packages

import pandas as pd

# Visualizations packages

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import eda_summary as eda
import graphs


def get_raw_cleaned_df():
    raw = pd.read_csv('./src/data/data.csv', encoding="ISO-8859-1")
    cleaned = pd.read_csv('./src/data/data_cleaned_2.csv')
    return (raw, cleaned)


def app():
    """ Common ML Dataset Explorer """

    st.title('Common ML Dataset Explorer')
    st.subheader('Datasets For ML Explorer with Streamlit')
    st.markdown("""##### This page has 2 sections: 

	1. Dataset Overview: Understanding the dataset
	2. Dataset Explorer (Interactive): Interact dynamically with the dataset
    
    """)

    type_of_data = st.radio(
        "Type of Data",
        ('Raw Data', 'Cleaned Data'),
        help='Data source that will be displayed in the charts'
    )

    raw_df, clean_df = get_raw_cleaned_df()

    if type_of_data == 'Raw Data':
        data = raw_df.copy()
    else:
        data = clean_df.copy()

    with st.container():
        st.header('Descriptive Statistics\n')
        st.dataframe(eda.summary_table(data))
        st.dataframe(data.describe())

    st.header('Data Visualization')

    height, width, margin = 450, 700, 10

    # st.subheader('Country Transactions Distribution')

    # TODO: GET UNIQUE TRANSACTIONS
    st.write("##### Transactions per country")
    temp = data.groupby(by=['Country', 'InvoiceNo'], as_index=False)['InvoiceDate'].count()
    temp.rename(columns={"InvoiceDate": "Transactions"}, inplace=True)
    fig = graphs.plot_histogram(data=temp, x='Country', nbins=50, height=height, width=width, margin=margin)
    st.plotly_chart(fig)

    st.header('Dataframe info: ')
    columns_info=pd.DataFrame(data.dtypes.astype(str)).T.rename(index={0:'Column Type'})
    columns_info=columns_info.append(pd.DataFrame(data.isnull().sum()).T.rename(index={0:'Null Values (NB)'}).astype(str))
    columns_info=columns_info.append(pd.DataFrame(data.isnull().sum()/data.shape[0]*100).T.rename(index={0:'Null Values (%)'}).astype(str))
    st.write(columns_info)

    st.subheader('Correlation Matrix')
    corr_matrix = data.corr()
    fig = graphs.plot_heatmap(corr_matrix=corr_matrix, height=height, margin=margin)
    st.plotly_chart(fig)

    st.markdown("""
    ### To Do:
    ##### From the Descriptive statistics we need to do in the next steps:**
    **Data Cleaning and preparation**
    * 1. Remove the rows without Customer ID
    * 2. Remove the duplicate rows
    * 3. Remove the canceled orders / returns (Negative Quantities)
    * 4. Deal with Zero Unit Prices
    * 5. Explore the DataFrame Again and continue with Feature Engineering and Data Transformation
    """)

    # # DATASET EXPLORER ##

    st.markdown("# Explore The DataFrame Interactively")

    html_temp = \
        """
	<div style="background-color:tomato;"><p style="color:white;font-size:30px;padding:10px">NOTE: Take Care of The columns Selecting due to the large amount of the dataset</p></div>
	"""
    st.markdown(html_temp, unsafe_allow_html=True)

    st.markdown("""
    Select the dataframe below:
    * `data.csv` is the raw dataset
    * `data_cleaned_2.csv` is the cleaned dataset (see following steps)
    """)

    def file_selector(folder_path='./src/data_to_select'):
        filenames = os.listdir(folder_path)
        selected_filename = st.selectbox('Select A file', filenames)
        return os.path.join(folder_path, selected_filename)

    filename = file_selector()
    st.info('You Selected {}'.format(filename))

    # Read Data

    df = pd.read_csv(filename, encoding='ISO-8859-1',
                     dtype={'CustomerID': str, 'InvoiceNo': str})

    # Show Dataset

    if st.checkbox('Show Dataset'):
        st.markdown('**Showing the first 5 rows of the dataset**')
        st.dataframe(df.head(5))

    # Show Columns

    if st.button('Column Names'):
        st.write(df.columns)

    # Show Shape

    if st.checkbox('Shape of Dataset'):
        data_dim = st.radio('Show Dimension By ', ('Rows', 'Columns'))
        if data_dim == 'Rows':
            st.text('Number of Rows')
            st.write(df.shape[0])
        elif data_dim == 'Columns':
            st.text('Number of Columns')
            st.write(df.shape[1])
        else:
            st.write(df.shape)

    # Select Columns

    if st.checkbox('Select Columns To Show'):
        all_columns = df.columns.tolist()
        selected_columns = st.multiselect('Select', all_columns)
        new_df = df[selected_columns]
        st.dataframe(new_df.head())

    # Show Summary

    if st.checkbox('Summary'):
        st.write(df.describe().T)

    # # Plot and Visualization

    st.subheader('Data Visualization')

    # Correlation
    # Seaborn Plot

    if st.checkbox('Correlation Plot[Seaborn]'):
        st.write(sns.heatmap(df.corr(), annot=True))
        st.pyplot()