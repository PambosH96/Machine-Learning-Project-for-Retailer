import pandas as pd
import streamlit as st
import numpy as np
import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import datetime, nltk, warnings
import matplotlib.cm as cm
import itertools
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import preprocessing, model_selection, metrics, feature_selection
from sklearn.model_selection import GridSearchCV, learning_curve, KFold
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn import neighbors, linear_model, svm, tree, ensemble
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_validate 
from sklearn.neural_network import MLPClassifier
from IPython.display import display, HTML
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode,iplot
import os
warnings.filterwarnings("ignore")
plt.rcParams["patch.force_edgecolor"] = True
plt.style.use('fivethirtyeight')
mpl.rc('patch', edgecolor = 'dimgray', linewidth=1)


# Title of the main page
def get_data():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    url = str(dir_path) + "/data/data.csv"
    return pd.read_csv(url, encoding="ISO-8859-1")

def app():
  st.title("Data Preparation: Data Overview & Cleaning")
  df = get_data()
  ### DATA PREPARATION SECTION ###
  st.title("Data Preparation Section")
  st.markdown("""
      ### Data Cleaning Section
      Here we will briefly explain what we did and how we did it in terms of data cleansing

      Now we know that almost 25% of the transactions are not assigned to a particular client and arround 0.27% of the transactions descriptions are not specified, 
      there are several ways to deal with missing values:

      Impute values for the CustomerID and Description, in this case it is impossible, does not make sense.
      
      Once these patterns are detected we can assign a labels for them and use these labels as a generic CustomerID and Description.
      Delete the rows where these missing values are found.

      Our goal is to apply clustering analysis and see patterns in those unknown clients and unknown description of products. 
  """)

  st.write("##### By utilizing the code below we can drop the rows where the customer is NaN and thus we cannot utilize them in our project")

  st.code("df.dropna(axis = 0, subset = ['CustomerID'], inplace = True")

  df.dropna(axis = 0, subset = ['CustomerID'], inplace = True)

  st.write("Now we have cleared the missing values of Customer ID lets see again whats happening")

  columns_info=pd.DataFrame(df.dtypes.astype(str)).T.rename(index={0:'Column Type'})
  columns_info=columns_info.append(pd.DataFrame(df.isnull().sum()).T.rename(index={0:'Null Values (NB)'}).astype(str))
  columns_info=columns_info.append(pd.DataFrame(df.isnull().sum()/df.shape[0]*100).T.rename(index={0:'Null Values (%)'}).astype(str))
  st.write(columns_info)

  duplicates = df.duplicated().sum()
  st.write("We have now to check if there are duplicated values in our dataset: {} rows in total.".format(duplicates))

  st.write("Lets see a sample of them: ")
  st.dataframe(df[df.duplicated()].head(5))

  st.write("Lets take the first one to filter the original dataframe to see if they are actual duplicates: ")
  st.markdown("""
      **So we can clearly see that we have duplicates with the row bellow:**
  """)
  st.dataframe(df[(df['InvoiceNo'] == '536409') & (df['StockCode'] == '21866')])

  st.write("We can remove them using the code bellow: ")
  st.code("df.drop_duplicates(inplace=True)")
  df.drop_duplicates(inplace=True)

  ## SEE THE MAX AND MINIMUM DATE ##
  df_date = pd.DataFrame()
  df_date['InvoiceDate'] = df['InvoiceDate'].astype('datetime64[ns]')
  st.write("##### We will check the max and minimum date:")
  st.code(df.InvoiceDate.max(), df.InvoiceDate.min()) 
  st.write(df_date.InvoiceDate.max(), df_date.InvoiceDate.min())
  st.write("**It seems like we have Online Retail Data and Transactions for one whole year**")

  ## FEATURE EXPLORATION ##
  st.markdown("#### Explore Columns and Possible Features")
  st.markdown("Lets investigate the orders per country")

  ### Let's create a dataframe where we will group the countries to see from which country the most of the sales come from
  temp = df[['CustomerID', 'InvoiceNo', 'Country']].groupby(['CustomerID', 'InvoiceNo', 'Country']).count()
  temp = temp.reset_index(drop=False)
  countries = temp['Country'].value_counts()

  COUNTRY, COUNTRY_ORDERS = np.unique(temp['Country'], return_counts=True)
  data_country_purchases = pd.DataFrame({'Country': COUNTRY, 'Orders': COUNTRY_ORDERS})
  data_country_purchases.sort_values(by='Orders', ascending=False, inplace=True, ignore_index=True)
  st.dataframe(data_country_purchases.head(10))

  fig, ax = plt.subplots()
  plt.xticks(rotation=90)
  ax.bar(data_country_purchases['Country'], data_country_purchases['Orders'])
  ax.set(title="Countries orders", ylabel="Count")
  st.pyplot(fig=fig)

  ### Now lets see the products the transactions and the customers
  st.markdown("""
  ##### Customers, Products and Transaction
  Lets count the values
  """)
  st.dataframe(pd.DataFrame([{'Products': len(df['StockCode'].value_counts()), 'Transactions': len(df['InvoiceNo'].value_counts()), 'Customers': len(df['CustomerID'].value_counts())}], columns = ['Products', 'Transactions', 'Customers'], index = ['Quantity']))


  ### Now lets see the Products sold 
  st.markdown("""
  ##### Now lets visualize how many times a product was sold
  Lets count the values
  """)
  PRODUCTS_ID, QUANTITY_SOLD = np.unique(df['StockCode'], return_counts=True)
  products_purchases = pd.DataFrame({'Product ID': PRODUCTS_ID, 'Quantity Sold': QUANTITY_SOLD})
  products_purchases.sort_values(by='Quantity Sold', ascending=False, inplace=True, ignore_index=True)
  st.dataframe(products_purchases.head(10))

  st.markdown("""
  #### Now lets see / check the number of products bought in every transaction
  """)
  temp = df.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['InvoiceDate'].count()
  products_per_transaction = temp.rename(columns = {'InvoiceDate': 'Number of Products'})
  products_per_transaction.sort_values('CustomerID', ascending=True, inplace=True, ignore_index=True)
  st.dataframe(products_per_transaction.head(10))

  st.write("As we can see from the above we gained some insights: ")
  st.markdown("""
  1. We have customers that bought once and customers that made more than one purchase
  2. We have transactions that start with the 'C' letter and this imply that they were canceled
  """)

  #### CANCELED ORDERS IN DATA CLEANING 

  st.markdown("""
  #### Tackle with Canceled Orders
  Run a describe to remember what we have
  """)
  st.dataframe(df.describe())

  st.write("As we remember from the first section, we have to tackle negative quantities.")
  # lets find the canceled orders and add them in a temporary dataframe
  products_per_transaction['Order_Canceled'] = products_per_transaction['InvoiceNo'].apply(lambda x:int('C' in x))
  st.dataframe(products_per_transaction[['InvoiceNo', 'Order_Canceled']])

  st.markdown("##### We need to check the proportion of canceled and completed order: ")
  canceled_transactions = products_per_transaction['Order_Canceled'].sum()
  total_transactions = products_per_transaction.shape[0]
  percent_canceled = canceled_transactions/total_transactions*100
  percent_completed = 100-(canceled_transactions/total_transactions*100)

  st.write('Number of orders canceled: {}'.format(canceled_transactions))
  st.write('Number of orders completed: {}'.format(total_transactions-canceled_transactions))

  fig = plt.figure()
  ax = fig.add_axes([0,0,1,1])
  x = ['Orders Canceled', 'Orders Completed']
  y = [percent_canceled, percent_completed]
  ax.bar(x,y)
  ax.set_ylim([0, 100])
  st.pyplot(fig=fig)

  st.write("""
  ##### As we can see the proportion of canceled orders are big so we cannot just drop them
  - We will find the cancel orders if they have counterparts and drop them
  - We will drop also the negative quantities
  - We will recheck the df if any negative quantity remains
  \n 
  """)

  st.markdown("""
  #### Lets take a Cancelation Order and visualize the quantities to understand what is happening.
  """)
  st.code("df[(df['StockCode'] == '35004C') & (df['CustomerID'] == 15311)]")
  st.dataframe(df[(df['StockCode'] == '35004C') & (df['CustomerID'] == 15311)])

  st.markdown("""
  From the table above we can see that the castomer can return only a part of the transaction so we need to try something else

  Moreover we have 3 possibel cases:
  * Negative quantities that have counterparts
  * Negative quantities without counterparts (which simply means that the custimer returned only a portion of the items back)
  * Negative quantities due to discound ("D")
  """)


  st.markdown("""
  Some canceled order may have an identical order with positive quantity but some of them may not. Lets check it:

  ##### Algorithm for finding and tackle with Canceled Orders 

  Loop through the dataframe
  * Select each row and check whether there is an identical in the dataframe but the only different should be the quantity and opposite (at this step we will check if there is the exact same date and time)
  * If there is then add the index of the row to a list to delete this positive entry
  * Additionally, add the negative quantity to another list to remove it
  * After this finish we need to delete StockCodes that do not correspond to items and thus we will completely discard all the negative quantities (i assume)

  **Lets wrap up the plan:**
  * First find the positive quantities that have counterparts
  * After check the dataframe and see if there are negative quantities
  * If there are we will remove the StockCodes refer to special codes such as discount etc. and then we will recheck the dataframe.
  * If they still exist in the dataframe (negative quantities) we need to somehow remove them from other products because they may are returns of a portion of a product.
  """)

  st.code("""
  # Mark the negative entries to subset them out of the data later
  # create a list of negative quantities that must be removed and they have identical
  # Create a list of positive quantities that have identical and remove them
  neg_quantity = []
  risk_pos_quantity = []
  for index in df.index:
    quantity = df.loc[index,'Quantity']
    cust_id = df.loc[index,'CustomerID']
    desc = df.loc[index,'Description']
    stockcode = df.loc[index,'StockCode']
    if (quantity < 0) & (stockcode != 'D'):
      neg_quantity.append(index)
      subset = df[(df['Quantity'] == -quantity) & 
                          (df['CustomerID'] == cust_id ) &
                          (df['Description'] == desc) &
                          (df['StockCode'] == stockcode )]

      #take only the first of it to remove it
      if subset.shape[0] >= 1:
        risk_pos_quantity.append(subset.index.values[0])
  """)

  ## DONT RUN THE CODE BECAUSE IS VERY TIME CONSUMING ##
  ## LOAD THE CLEANED DATASET STRAIGHTFORWARDLY ## 

  # # Mark the negative entries to subset them out of the data later
  # # create a list of negative quantities that must be removed and they have identical
  # # Create a list of positive quantities that have identical and remove them
  # neg_quantity = []
  # risk_pos_quantity = []
  # for index in df.index:
  #   quantity = df.loc[index,'Quantity']
  #   cust_id = df.loc[index,'CustomerID']
  #   desc = df.loc[index,'Description']
  #   stockcode = df.loc[index,'StockCode']
  #   if (quantity < 0) & (stockcode != 'D'):
  #     neg_quantity.append(index)
  #     subset = df[(df['Quantity'] == -quantity) & 
  #                         (df['CustomerID'] == cust_id ) &
  #                         (df['Description'] == desc) &
  #                         (df['StockCode'] == stockcode )]

  #     #take only the first of it to remove it
  #     if subset.shape[0] >= 1:
  #       risk_pos_quantity.append(subset.index.values[0])

  # st.code("len(neg_quantity), len(risk_pos_quantity)")
  # st.write("negative quantities: {}, positive quantities that were canceled: {}".format(len(neg_quantity), len(risk_pos_quantity)))

  st.code("len(neg_quantity), len(risk_pos_quantity)")
  st.write("negative quantities: {}, positive quantities that were canceled: {}".format(8795, 3224))

  st.markdown("Run the bellow to drop them")
  st.code("""
  df.drop(neg_quantity, axis = 0, inplace = True)
  df.drop(risk_pos_quantity, axis = 0, inplace = True)
  df.describe()
  """)

  # also the drop is commented because of the above comments

  # df.drop(neg_quantity, axis = 0, inplace = True)
  # df.drop(risk_pos_quantity, axis = 0, inplace = True)

  #### get_data that are cleaned
  def get_data_cleaned():
      dir_path = os.path.dirname(os.path.realpath(__file__))
      url = str(dir_path) + "/data/data_cleaned_2.csv"
      return pd.read_csv(url, encoding="ISO-8859-1", dtype={'CustomerID': str, 'InvoiceNo': str})

  df = get_data_cleaned()

  st.markdown("""
  From the describe() above we can see that we still have negative values. Lets remove the stockcodes that are not casual transactions and then come back again to recheck everything.
  \n
  **Now lets remove rows that dont have to do with customers (like discounts or manual added entries).**
  We can find them by getting all the stockcodes that are different We need a way to filter stockcodes that differ from the casual.
  """)

  st.write("Remove also the special stockcodes: ".format(df[df['StockCode'].str.contains('^[a-zA-Z]+', regex=True)]['StockCode'].unique()))

  st.code("""
    df = df[df['StockCode']!= 'POST']
    df = df[df['StockCode']!= 'D']
    df = df[df['StockCode']!= 'C2']
    df = df[df['StockCode']!= 'M']
    df = df[df['StockCode']!= 'BANK CHARGES']
    df = df[df['StockCode']!= 'PADS']
    df = df[df['StockCode']!= 'DOT']
  """)

  df = df[df['StockCode']!= 'POST']
  df = df[df['StockCode']!= 'D']
  df = df[df['StockCode']!= 'C2']
  df = df[df['StockCode']!= 'M']
  df = df[df['StockCode']!= 'BANK CHARGES']
  df = df[df['StockCode']!= 'PADS']
  df = df[df['StockCode']!= 'DOT']

  st.write("Lets run describe to see if we still have negative quantities: ")
  st.dataframe(df.describe())

  st.markdown("""
  **After Cleaning**: Seems Like we have a final dataset with 393374 rows
  """)

  df_cleaned = get_data_cleaned()