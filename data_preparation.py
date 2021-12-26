import pandas as pd
import streamlit as st
import numpy as np
import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import datetime, nltk, warnings
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
from kmodes.kmodes import KModes
from sklearn.metrics import silhouette_samples, silhouette_score
from IPython.display import display, HTML
from sklearn import preprocessing, model_selection, metrics, feature_selection
from sklearn.cluster import KMeans
import pickle
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode,iplot
import os
warnings.filterwarnings("ignore")
plt.rcParams["patch.force_edgecolor"] = True
plt.style.use('fivethirtyeight')
mpl.rc('patch', edgecolor = 'dimgray', linewidth=1)

def get_data():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    url = str(dir_path) + "/data/data_cleaned.csv"
    return pd.read_csv(url, encoding="ISO-8859-1")

def return_list_of_products(count_keywords, keywords_select):
    list_product_keywords = []
    for k,v in count_keywords.items():
        word = keywords_select[k]
        if word in ['pink', 'blue', 'tag', 'green', 'orange']: continue
        if len(word) < 3 or v < 15: continue
        if ('+' in word) or ('/' in word): continue
        list_product_keywords.append([word, v])
    return list_product_keywords

def app():
    st.title("Data Preparation: Feature Engineering")

    ########### DATA PREPARATION: FEATURE ENGINEERING #################
    def get_data_cleaned():
        dir_path = os.path.dirname(os.path.realpath(__file__))
        url = str(dir_path) + "/data/data_cleaned_2.csv"
        return pd.read_csv(url, encoding="ISO-8859-1", dtype={'CustomerID': str, 'InvoiceNo': str})

    df_cleaned = get_data_cleaned()

    st.title("Data Preparation: Feature Engineering")

    st.markdown("""
        ### Data Preparation section consits of some steps that include:
        * Data Transformation
        * Feature Engineering

        ### In this Page we will create the following categories:
        1. Product Categories
        2. Customer Categories
    """)

    st.dataframe(df_cleaned.describe())

    st.write("""
    Lets create a dataframe CartPrice which will contain the data for each transaction (group by invoice number)
    """)
    
    temp = df_cleaned.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['TotalPrice'].sum()
    cart_price = temp.rename(columns = {'TotalPrice':'Cart Price'})

    st.code("""
    temp_df = df_cleaned.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['TotalPrice'].sum()
    cart_price = temp_df.rename(columns = {'TotalPrice':'Cart Price'})
    cart_price
    """)

    st.dataframe(cart_price)

    st.markdown("""
    #### Add the date to the CartPrice DataFrame
    """)

    # with the code bellow we extract the date from the original df_cleaned, then we convert it to int and then assign it to the cart_price by reconstructing it to date
    df_cleaned['InvoiceDate_int'] = df_cleaned['InvoiceDate'].astype(np.datetime64).astype(np.int64)
    temp_df_2 = df_cleaned.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['InvoiceDate_int'].mean()
    df_cleaned.drop('InvoiceDate_int', axis = 1, inplace = True)  #now drop it because we dont actually need it
    cart_price.loc[:, 'InvoiceDate'] = pd.to_datetime(temp_df_2['InvoiceDate_int']) #set the value of date int to date time for the entire column
    st.dataframe(cart_price)

    st.markdown("""
    #### Now lets utilize CartPrice Dataframe for visualizations
    * We have to create a barchart by grouping transactions to bins (lets say of 10 bins).
    * We exclude the transactions that costs more than 5000 Pounds because as we can see bellow they are very few
    """)

    # We can see tha the portion of transactions bigger than >5000 is very low (0.5%)
    cart_price_bigger5000 = cart_price[cart_price['Cart Price'] > 5000]  
    perc = len(cart_price_bigger5000) / len(cart_price)
    st.markdown("Percentage of CartPrice values bigger than 5000 Pounds are: `{}`".format(perc))

    # lets see the max and the minimum
    max = cart_price['Cart Price'].max()
    min = cart_price['Cart Price'].min()
    step = 5000/10

    bin_range = np.arange(0, 5000+step, step)
    out, bins  = pd.cut(cart_price['Cart Price'], bins=bin_range, include_lowest=True, right=False, retbins=True)
    st.bar_chart(data=out.value_counts(sort=False), width=300, height=400)

    # Lets plot a barchart

    price_range = [0, 50, 100, 200, 500, 1000, 5000, 50000]
    count_price = []
    for i, price in enumerate(price_range):
        if i == 0: continue
        val = cart_price[(cart_price['Cart Price'] < price) &
                        (cart_price['Cart Price'] > price_range[i-1])]['Cart Price'].count()
        count_price.append(val)

    plt.rc('font', weight='bold')
    f, ax = plt.subplots(figsize=(11, 6))
    colors = ['yellow', 'red', 'blue', 'green', 'magenta', 'cyan','black']
    labels = [ '{} < {}'.format(price_range[i-1], s) for i,s in enumerate(price_range) if i != 0]
    sizes  = count_price
    explode = [0.0 if sizes[i] < 100 else 0.0 for i in range(len(sizes))]
    ax.pie(sizes, explode = explode, labels=labels, colors = colors,
        autopct = lambda x:'{:1.0f}%'.format(x) if x > 1 else '',
        shadow = False, startangle=0)
    ax.axis('equal')
    f.text(0.5, 1.01, "Distribution of Orders", ha='center', fontsize = 18)

    st.pyplot(fig=f)

    st.markdown("""
    **From the Charts above we can see:**
    * The majority of CartPrice values are between 0-500 Pounds
    * There are a lot less CartPrice values above 500
    """)

    st.markdown("""
    # Understand Products: Create Product Categories
    **What do we already know about the products?**
    * Each product has a unique stockcode
    * Each product has a description which describes each product

    **We will use basic NLP and bag of words to create a DataFrame based on the one_hot_encoding and create cluster of products**

    What does the dataset tell us of its products?
    What we are going to do is to explore the content of the column Description in order to group the products into different categories
    This is going to be very excited and tricky. 
    - We will use one hot representation to create the `word_X_matrix` which sill contain for each description the one hot representation using bag of words technique.
    - we are going to understand what this does and what its purpose is for our object of understanding the `Description` column.
    """)

    is_noun = lambda pos: pos[:2] == 'NN'

    def bags_of_keywords(dataframe, column = 'Description'):
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        stemmer = nltk.stem.SnowballStemmer("english")
        keywords_roots  = dict()  # collect the roots of words
        keywords_select = dict()  # associates the root and keyword
        count_keywords  = dict()
        category_keys   = []
        
        icount = 0
        for s in dataframe[column]:
            if pd.isnull(s): continue
            lines = s.lower()
            tokenized = nltk.word_tokenize(lines)
            nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)] 
            
            for t in nouns:
                t = t.lower() ; root = stemmer.stem(t)
                if root in keywords_roots:                
                    keywords_roots[root].add(t)
                    count_keywords[root] += 1                
                else:
                    keywords_roots[root] = {t}
                    count_keywords[root] = 1
        
        for s in keywords_roots.keys():
            if len(keywords_roots[s]) > 1:  
                min_length = 1000
                for k in keywords_roots[s]:
                    if len(k) < min_length:
                        category_key = k ; min_length = len(k)            
                category_keys.append(category_key)
                keywords_select[s] = category_key
            else:
                category_keys.append(list(keywords_roots[s])[0])
                keywords_select[s] = list(keywords_roots[s])[0]
                
        print("Number of keywords in variable '{}': {}".format(column, len(category_keys)))
        return category_keys, keywords_roots, keywords_select, count_keywords


    st.markdown("""
    Now we will create a new DF object in which we will have the unique values from the column Description and these values are obtained by the df['Description'].unique
    """)

    df_products = pd.DataFrame(df_cleaned.Description.unique())
    df_products.rename(columns = {0: 'Description'}, inplace=True)
    st.write(df_products)

    st.write("Now lets get the keywords by running the `bag_of_keywords` function: ")
    st.code("keywords, keywords_roots, keywords_select, count_keywords = bags_of_keywords(df_products)")
    keywords, keywords_roots, keywords_select, count_keywords = bags_of_keywords(df_products)

    st.markdown(""""

    Great! We now have 1473 keywords. Our function returned the next:
    * keywords: The list of extracted keywords.
    * keywords_roots: A dictionary where its keys are the keywords roots and the values are the lists of words associated with these roots.
    * keywords_select: A dictionary that has the keywords that where selected for categories.
    * count_keywords: A dictionary with the numbers of times every word has been used.

    Now we create a list `list_keywords` and we iterate with a `for` loop the items in `count_keyword` dictionary with `k` as iterator index and `v` as object, we then append to <code>list_products` with `[keywords_select[k], v]`, this appends the selected keywords and its values.
    """)

    list_keywords = []
    for k,v in count_keywords.items():
        list_keywords.append([keywords_select[k],v])
    len(list_keywords)
    st.write(list_keywords[:5])

    product_list = sorted(list_keywords, key = lambda x:x[1], reverse = True)
    st.dataframe(product_list)

    st.markdown("""
    ####### Plot the top 50 keywords
    """)
    ## Now lets plot the product list sorted by the top 50 keywords
    plt.rc('font', weight='normal')

    fig, ax = plt.subplots(figsize=(10, 10))

    # get only the first/top num_of_words keywords
    y_axis = [i[1] for i in product_list[:50]]

    # get only the first/top num_of_words keywords
    x_axis = [k for k,i in enumerate(product_list[:50])]
    x_label = [i[0] for i in product_list[:50]]
    plt.yticks(x_axis, x_label)
    plt.xlabel("Word Frequency")
    ax.barh(x_axis, y_axis, align = 'center', color=['black', 'red', 'green', 'blue', 'cyan', 'yellow', 'magenta'])
    ax = plt.gca()
    ax.invert_yaxis()
    st.pyplot(fig)


    st.markdown("""
    # CREATE AND DEFINE PRODUCT CATEGORIES BASED ON THE DESCRIPTION
    """)
    st.markdown("""
    ## Defining Product Categories

    The keywords list contains 1473 keywords and the most frequent ones appear in more than 200 products. 
    When examinating the content of this list, we can notice that some names are useless, do not carry information. We will drop them such as stopwords, color keywords etc. 
    Therefore we should discard these words from the analysis that follows and also let's consider only the words that appear more than 15 times.
    """)

    list_product_keywords = return_list_of_products(count_keywords=count_keywords, keywords_select=keywords_select)
    list_product_keywords.sort(key = lambda x:x[1], reverse = True)
    st.write('Number of words that were kept: `165`')

    st.markdown("""
    Now we will use the onehot encoding principle:     
    * Create a matrix of the unique Descriptions and put 0 and 1 where the description has a keyword or not
    """)
    list_descriptions = df_cleaned['Description'].unique()
    Word_X_matrix = pd.DataFrame()
    for key, frequency in list_product_keywords:
        # we create one column for each keyword so thats why we use the key as column
        key_UPPER = key.upper()
        list_to_append = list(map(lambda x:int(key_UPPER in x), list_descriptions))
        Word_X_matrix.loc[:, key_UPPER] = list_to_append

    st.dataframe(Word_X_matrix.head(5))


    st.markdown("""
    ## Now lets add the price range to the matrix X for each description to augment the dataset: 
    Shortly, we will create 5 price ranges (see the visualization of the price above `(pie chart and barchart))` to augment the dataset associated each one_hot_description with the price range
    """)
    with st.spinner('One-hot-encoding matrix creation in progress: Wait for it...'):
        threshold = [0, 1, 2, 3, 5, 10]
        label_col = []
        for i in range(len(threshold)):
            if i == len(threshold)-1:
                col = ' > {}'.format(threshold[i])
            else:
                col = '{} < {}'.format(threshold[i],threshold[i+1])
            label_col.append(col)
            Word_X_matrix.loc[:, col] = 0

        for i, prod in enumerate(list_descriptions):
            prix = df_cleaned[ df_cleaned['Description'] == prod]['UnitPrice'].mean()
            j = 0
            while prix > threshold[j]:
                j+=1
                if j == len(threshold): break
            Word_X_matrix.loc[i, label_col[j-1]] = 1

        st.write("{:<8} {:<20}\n".format('Range and', 'Count of Products in each price range'))

        for i in range(len(threshold)):
            if i == len(threshold)-1:
                col = ' > {}'.format(threshold[i])
            else:
                col = '{} < {}'.format(threshold[i],threshold[i+1])    
            st.write("{:<10}  {:<20}".format(col, Word_X_matrix.loc[:, col].sum()))

    st.write("adding the price range from 1-10 as columns to augment the dataset information (and enhance clustering)")
    st.dataframe(Word_X_matrix.head(5))

    st.markdown("## Now we are going to use KModes to create clusters of products")
    st.markdown("###### By running the code bellow we conclude to use 5 clusters for products:")
    st.markdown("""
    We use silhouette analysis to determine how many cluster we will use: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
    """)
    st.code("""
    for n_clusters in range(3,10):
        kmodes = KModes(init='Huang', max_iter=100, n_clusters=n_clusters, n_init=30, n_jobs=-1, random_state=42)
        clusters = kmodes.fit_predict(Word_X_matrix)
        silhouette_avg = silhouette_score(Word_X_matrix, clusters)
        print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)
    """)
    st.write("""
    Result for n_clusters = 3 The average silhouette_score is : 0.11142050517146847
    * For n_clusters = 4 The average silhouette_score is : 0.11369426930609004
    * For n_clusters = 5 The average silhouette_score is : 0.15605111618663936
    * For n_clusters = 6 The average silhouette_score is : 0.15591614204472466
    * For n_clusters = 7 The average silhouette_score is : 0.15497509597519985
    * For n_clusters = 8 The average silhouette_score is : 0.12549958008303969
    * For n_clusters = 9 The average silhouette_score is : 0.13722223664399463
    """)
    
    st.write("Through the code above we conclude to use `K=5` (we dont run it because of the long runtime required)")
    st.write("The code bellow will create clusters of Products utilizing the word_X_matrix we created utilizing the one-hot-encoding principle and adding the price range scaled from 1-10")
    st.code("""
    n_clusters = 5
    silhouette_avg = -1
    while silhouette_avg < 0.15:
        kmodes = KModes(init='Huang', max_iter=75, n_clusters=n_clusters, n_init=30, n_jobs=-1, random_state=42)
        clusters = kmodes.fit_predict(Word_X_matrix)
        silhouette_avg = silhouette_score(Word_X_matrix, clusters)
        print('For n_clusters = ', n_clusters, ' The average silhouette_score is : ', silhouette_avg)
    """)

    st.write("Clustering in progres...")
    n_clusters = 5
    silhouette_avg = -1
    with st.spinner('Clustering in progress: Wait for it...'):
        while silhouette_avg < 0.15:
            kmodes = KModes(init='Huang', max_iter=75, n_clusters=n_clusters, n_init=30, n_jobs=-1, random_state=42)
            clusters = kmodes.fit_predict(Word_X_matrix)
            silhouette_avg = silhouette_score(Word_X_matrix, clusters)
            st.write('For n_clusters = {} The average silhouette_score is : {}'.format(n_clusters,silhouette_avg))

    st.markdown("## Now we need to evaluate the content of clusters and check the distribution")

    def graph_component_silhouette(n_clusters, lim_x, mat_size, sample_silhouette_values, clusters):
        plt.rcParams["patch.force_edgecolor"] = True
        plt.style.use('fivethirtyeight')
        mpl.rc('patch', edgecolor = 'dimgray', linewidth=1)
        
        fig, ax1 = plt.subplots(1, 1)
        fig.set_size_inches(8, 8)
        ax1.set_xlim([lim_x[0], lim_x[1]])
        ax1.set_ylim([0, mat_size + (n_clusters + 1) * 10])
        y_lower = 10
        for i in range(n_clusters):
            
            # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[clusters == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            cmap = cm.get_cmap("Spectral")
            color = cmap(float(i) / n_clusters)        
            ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                            facecolor=color, edgecolor=color, alpha=0.8)
            
            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.03, y_lower + 0.5 * size_cluster_i, str(i), color = 'red', fontweight = 'bold',
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round, pad=0.3'))
            
            # Compute the new y_lower for next plot
            y_lower = y_upper + 10

        st.pyplot(fig)

    sample_silhouette_values = silhouette_samples(Word_X_matrix, clusters)

    graph_component_silhouette(n_clusters, [-0.07, 0.35], len(Word_X_matrix), sample_silhouette_values, clusters)


    st.markdown("""
    ## Now we have clusters of products Lets create customer categories:
    - We will create the categories for our customers, but first we need to give some proper format to some data.
    - As we already grouped our products into five different clusters we must incorporate this information into the dataframe, we are going to create a new column/feature called Product_Category and it will hold the cluster of each product.
    - We create product_category dictionary, we iterate trough zipping list_descriptions and clusters as key for descriptions from list_descriptions and val for the number of cluster from clusters, then we assign to product_category[key] the value of val.
    - product_category will have the descriptions and to what cluster they belong.
    """)

    product_category = dict()
    for key, val in zip (list_descriptions, clusters):
        product_category[key] = val

    st.markdown("""
    We create the column Product_Category and we assign it the categories by mapping with
    df_cleaned.loc[:, 'Description'].map(product_category).
    """)

    df_cleaned['Product_Category'] = df_cleaned.loc[:, 'Description'].map(product_category)

    st.dataframe(df_cleaned.sample(5))

    st.markdown("""
    We now have every transaction and its category.
    - `Grouping the Products`
    - Let's create a `Cat_N` variables (with  $N$ $∈$ $[0:4]$ ) that contains the amount spent in each product category.
    """)

    st.markdown("""
    Create a column in which we will have how much money each customer spent in each category.
    """)

    for i in range(5):
        col = 'Cat_{}'.format(i)        
        df_temp = df_cleaned[df_cleaned['Product_Category'] == i]
        price_temp = df_temp['TotalPrice']
        price_temp = price_temp.apply(lambda x:x if x > 0 else 0)
        df_cleaned.loc[:, col] = price_temp
        df_cleaned[col].fillna(0, inplace = True)

    st.dataframe(df_cleaned.sample(5))

    st.markdown("""
    Now we create a temporal DataFrame object temp, in this new temporal dataframe we are going to hold the TotalPrice sum grouped by CustomerID and InvoiceNo, then we are going to assign to cart_price the values from temp.
    """)
    st.code("""
    temp = df_cleaned.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['TotalPrice'].sum()
    cart_price = temp.rename(columns = {'TotalPrice':'Cart Price'})
    """)
    temp = df_cleaned.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['TotalPrice'].sum()
    cart_price = temp.rename(columns = {'TotalPrice':'Cart Price'})

    st.markdown("""
    Then we iterate in a `for` loop a range of `5` iterations with `i` as iterator index, inside this loop we first assign to the var `col` the name of the column `Cat_{i}`, then we assign to `temp` the result from grouping `CustomerID` and `InvoiceNo` and the sum of `col`, then we assign to `cart_price` in the column `col` the values in `temp`. 
    """)

    st.code("""
        for i in range(5):
        col = 'Cat_{}'.format(i) 
        temp = df_cleaned.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)[col].sum()
        cart_price.loc[:, col] = temp
    """)

    for i in range(5):
        col = 'Cat_{}'.format(i) 
        temp = df_cleaned.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)[col].sum()
        cart_price.loc[:, col] = temp

    st.dataframe(cart_price.head(5))

    st.markdown("####### Now what are we going to do is to add the dates of the transactions to the dataframe cart_price")

    df_cleaned['InvoiceDate_int'] = df_cleaned['InvoiceDate'].astype(np.datetime64).astype(np.int64)
    temp = df_cleaned.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['InvoiceDate_int'].mean()
    df_cleaned.drop('InvoiceDate_int', axis = 1, inplace = True)
    cart_price.loc[:, 'InvoiceDate'] = pd.to_datetime(temp['InvoiceDate_int'])

    st.markdown("""
    Now we are going to filter in cart_price the values from the column Cart Price that are bigger than 0, then we sort its values in ascending order according to CustomerID column and display the first 5 samples in this dataframe.
    """)

    cart_price = cart_price[cart_price['Cart Price'] > 0]
    cart_price.sort_values('CustomerID', ascending = True).head(5)

    st.write("Oldest transactions")

    st.write("min invoice date: {}".format(cart_price['InvoiceDate'].min()))
    st.write("max invoice date: {}".format(cart_price['InvoiceDate'].max()))

    st.markdown("""
    ### Taking Care of Data Over Time
    We have to remember (see code bellow) that we have data from 1 exact year. 
    * The main objectives of this notebook is to develop a model to characterizing and anticipating the habits of the customers visiting the online retail from their first visit. 
    * How can we test the model in a realistic way?, We can split the dataset by keeping the first 10 months for training, development and testing of the model and do the last two months the same way in order to see if there are not any differences among the evaluations scores it means that we are good. 
    * Now let's define a var date_limit which is going to work as the limit day for comparison.
    * The reasons is because we cannot train the model in the first ten months and test it with the last 2 months because we will lose seasonality. 
    * Additionally we have to mention that we experiment to test with the last 2 months the model trained with the first 10 months and it was preforming very bad
    """)

    st.code("""
    # Original start dataframe
    df['InvoiceDate'] = df['InvoiceDate'].astype('datetime64[ns]')
    df.InvoiceDate.max(), df.InvoiceDate.min()
    """)
    st.markdown("""
    **Output:** (Timestamp('2011-12-09 12:50:00'), Timestamp('2010-12-01 08:26:00'))
    """)

    import datetime
    date_limit = np.datetime64(datetime.date(2011, 10, 1))
    date_limit

    st.markdown("""
    ### As we mentioned above we will train 
    train_set: data from cart_price that was registered before the date 2011-10-1 and
    test_set: data from cart_price that was registered during and after the date 2011-10-1.
    """)

    st.code("""
    train_set = cart_price[cart_price['InvoiceDate'] < date_limit]
    test_set = cart_price[cart_price['InvoiceDate'] >= date_limit]
    """)

    train_set = cart_price[cart_price['InvoiceDate'] < date_limit]
    test_set = cart_price[cart_price['InvoiceDate'] >= date_limit]

    cart_price = train_set.copy(deep = True)

    st.write("Train Set DF:")
    st.dataframe(train_set.tail(5))
    st.write("Test Set DF: ")
    st.dataframe(test_set.head(5))

    st.markdown("""
    Then we create the DataFrame object `transactions_per_user`, in this new dataframe we assign the values of count, min, max, mean and sum from gruping by CustomerID and Cart Price. The information on transactions_per_user is just basic statistics of the values found in the Cart Price of each customer.
    """)

    transactions_per_user=cart_price.groupby(by=['CustomerID'])['Cart Price'].agg(['count','min','max','mean','sum'])
    st.dataframe(transactions_per_user)

    for i in range(5):
        col = 'Cat_{}'.format(i)
        transactions_per_user.loc[:,col] = cart_price.groupby(by=['CustomerID'])[col].sum() / transactions_per_user['sum']*100

    st.dataframe(transactions_per_user.head(5))

    st.markdown("""
    * We group cart_price dataframe by CustomerID and sum the values from Category_0 column.
    * Therefore cart_price will have how much each customer has bought in Category_0.
    """)

    transactions_per_user.reset_index(drop=False, inplace=True)
    cart_price.groupby(by=['CustomerID'])['Cat_0'].sum()
    st.dataframe(transactions_per_user.sort_values('CustomerID', ascending=True).head(5))

    st.markdown("""
    Lets define two additional columns for the number of days elapsed since the first purchase ( FirstPurchase ) and the number of days since the last purchase ( LastPurchase ):

    We take in last_date the maximun date on InvoiceDate from cart_price.
    """)

    last_date = cart_price['InvoiceDate'].max().date()
    st.write("Last date: {}".format(last_date))

    st.markdown("""
    Let's create the next dataframes `first_registration` for the first date that a customer made a transaction, this is done through grouping by CustomerID and taking the minimun date from InvoiceDate and for last_purchase dataframe we take the last date that a customer made a transaction, this is done through grouping by CustomerID and taking the maximum date from InvoiceDate.
    """)

    st.code("""
    first_registration = pd.DataFrame(cart_price.groupby(by=['CustomerID'])['InvoiceDate'].min())
    last_purchase      = pd.DataFrame(cart_price.groupby(by=['CustomerID'])['InvoiceDate'].max())
    """)

    first_registration = pd.DataFrame(cart_price.groupby(by=['CustomerID'])['InvoiceDate'].min())
    last_purchase      = pd.DataFrame(cart_price.groupby(by=['CustomerID'])['InvoiceDate'].max())
    st.dataframe(first_registration.head())
    st.dataframe(last_purchase.head())

    st.markdown("""
    We have seen what info do first_registration and last_purchase are holding, now we are going to calculate how many days have passed, this is done by creating two separete dataframe, one for first_registration and last_purchase.

    Let's create test_fp a dataframe, where we are going to apply a lambda function that calculates the days that have to first_registration with the function applymap(lambda x:(last_date - x.date()).days).

    Now we are going to create test_lp a dataframe, where we are going to apply a lambda function that calculates the days that have to last_purchase with the function applymap(lambda x:(last_date - x.date()).days).
    """)

    st.code("""
    test_fp  = first_registration.applymap(lambda x:(last_date - x.date()).days)
    test_lp = last_purchase.applymap(lambda x:(last_date - x.date()).days)
    """)
    test_fp  = first_registration.applymap(lambda x:(last_date - x.date()).days)
    test_lp = last_purchase.applymap(lambda x:(last_date - x.date()).days)

    st.markdown("""
    We are going to create new columns for transactions_per_user, one column called `FirstPurchase` and other column named `LastPurchase`.

    `FirstPurchase`: is going to take the values from test_fp, we do not reset its index, this will match the CustomerID in transactions_per_user. `LastPurchase`: is going to take the values from test_lp, we do not reset its index, this will match the CustomerID in transactions_per_user.
    """)

    st.code("""
    transactions_per_user.loc[:, 'FirstPurchase'] = test_fp.reset_index(drop=False)['InvoiceDate']
    transactions_per_user.loc[:, 'LastPurchase'] = test_lp.reset_index(drop=False)['InvoiceDate']
    """)

    transactions_per_user.loc[:, 'FirstPurchase'] = test_fp.reset_index(drop=False)['InvoiceDate']
    transactions_per_user.loc[:, 'LastPurchase'] = test_lp.reset_index(drop=False)['InvoiceDate']


    st.dataframe(transactions_per_user.head(5))


    st.markdown("""
    ## Creating Customers Categories
    #### Data Encoding
    * `transactions_per_user` is a DF that contains a summary of all the transaction that were made by each client. 
    * This information will be used to characterize the different types of customers and only keep a subset of variables:
    * Let's create a list call list_cols that will hold the features that are going to be used for the model to learn patterns in order to define the clusters.
    """)
    list_cols = ['count','min','max','mean','Cat_0','Cat_1','Cat_2','Cat_3','Cat_4', 'LastPurchase', 'FirstPurchase']

    selected_customers = transactions_per_user.copy(deep=True)
    matrix = selected_customers[list_cols].to_numpy()

    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    minmax_scaler = MinMaxScaler()
    minmax_scaler.fit(matrix)
    minmaxscaled_matrix = minmax_scaler.transform(matrix)

    st.markdown("""
    #### Creation of Customer Categories

    Well it is the time we all have been waiting, the creation of this clusters will be done by using KMeans, it is a very similar process as the one we did with creating the clusters for words with KModes.
    \n
    This may take a while!
    \n
    The best number of clusters will be defined by the technique Elbow Method.
    """)

    st.write("Run the code bellow to find the best K using k-means (we cannot run it in streamlit because we used the go Figure)")
    st.code("""
    for n_clusters in range(1, 21):
        kmeans = KMeans(init='k-means++', max_iter=100, n_clusters=n_clusters, n_init=100, random_state=42).fit(minmaxscaled_matrix)
        inertia.append(kmeans.inertia_)
    clusters_history['inertia'] = inertia
    clusters_history

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=clusters_history.cluster_range,
                            y=clusters_history.inertia,
                            name='Clusters',
                            text='Quantity of Clusters and Inertia Value'))
    fig.update_layout(
        title_text='Clusters vs Inertia',
        title_x=0.5,
        xaxis = dict(
            title='Quantity Clusters'),
        yaxis = dict(title='Inertia')
    )
    fig.show()
    """)
    
    st.write("""
    ##### After experimenting with the clusters we conclude that 14 clusters is the right value for n_clusters
    """)

    with st.spinner('Clustering - Grouping Cutomers: Wait for it...'):
        n_clusters = 14
        kmeans = KMeans(init='k-means++', max_iter=100, n_clusters=n_clusters, n_init=100, random_state=42)
        kmeans.fit(minmaxscaled_matrix)

    if 'kmeans' not in st.session_state:
        st.session_state['kmeans'] = kmeans
    
    # we ran the code bellow to store kmeans model once
    # with open("./src/models/kmeans.pkl", "wb") as f:
    #     pickle.dump(kmeans, f)
        
    clients_clusters = kmeans.predict(minmaxscaled_matrix)
    silhouette_avg = silhouette_score(minmaxscaled_matrix, clients_clusters)
    print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)

    st.write("print the frequencies of the clusters")
    st.dataframe(pd.DataFrame(pd.Series(clients_clusters).value_counts(), columns = ['Quantity of Clients in Cluster']).T)

    st.write("Plot the silhouette graph and evaluate")
    sample_silhouette_values = silhouette_samples(minmaxscaled_matrix, clients_clusters)
    graph_component_silhouette(n_clusters, [-0.15, 0.55], len(minmaxscaled_matrix), sample_silhouette_values, clients_clusters)

    st.markdown("""
    * Some of the clusters are indeed disjoint. It remains to understand the habits of the customers in each cluster. 
    * To do so, we start by adding to the selected_customers dataframe a variable that defines the cluster to which each client belongs.
    """)

    st.code("selected_customers.loc[:, 'cluster'] = clients_clusters")
    selected_customers.loc[:, 'cluster'] = clients_clusters

    st.markdown("""
    * Then, We average the contents of this dataframe by first selecting the different groups of clients. 
    * This gives access to, for example, the average cart price, the number of visits or the total sums spent by the clients of the different clusters. 
    * We also determine the number of clients in each group (variable size ):
    """)

    merged_df = pd.DataFrame()
    for i in range(n_clusters):
        test_fp = pd.DataFrame(selected_customers[selected_customers['cluster'] == i].mean())
        test_fp = test_fp.T.set_index('cluster', drop = True)
        test_fp['size'] = selected_customers[selected_customers['cluster'] == i].shape[0]
        merged_df = pd.concat([merged_df, test_fp])
    st.dataframe(merged_df)

    # merged_df.drop('CustomerID', axis = 1, inplace = True)
    print('number of customers:', merged_df['size'].sum())
    merged_df = merged_df.sort_values('sum')

    st.markdown("""
    * Finally, We re-organize the content of the dataframe by ordering the different clusters: 
    * First, in relation to the amount spent in each product category and then, according to the total amount spent:
    """)

    st.code("""
    list_index = []
    for i in range(5):
        column = 'Cat_{}'.format(i)
        list_index.append(merged_df[merged_df[column] > 45].index.values[0])

    list_index_reordered = list_index
    list_index_reordered += [ s for s in merged_df.index if s not in list_index]

    merged_df = merged_df.reindex(index = list_index_reordered)
    merged_df = merged_df.reset_index(drop = False)
    merged_df[['cluster', 'count', 'min', 'max', 'mean', 'sum', 'Cat_0', 'Cat_1', 'Cat_2', 'Cat_3', 'Cat_4', 'size']]

    columns = ['count','min', 'max', 'mean', 'Cat_0', 'Cat_1', 'Cat_2', 'Cat_3', 'Cat_4', 'LastPurchase', 'FirstPurchase']
    X_fp = selected_customers[columns]
    Y_fp = selected_customers['cluster']
    """)

    list_index = []
    for i in range(5):
        column = 'Cat_{}'.format(i)
        list_index.append(merged_df[merged_df[column] > 45].index.values[0])

    list_index_reordered = list_index
    list_index_reordered += [ s for s in merged_df.index if s not in list_index]

    merged_df = merged_df.reindex(index = list_index_reordered)
    merged_df = merged_df.reset_index(drop = False)
    merged_df[['cluster', 'count', 'min', 'max', 'mean', 'sum', 'Cat_0', 'Cat_1', 'Cat_2', 'Cat_3', 'Cat_4', 'size']]

    columns = ['count','min', 'max', 'mean', 'Cat_0', 'Cat_1', 'Cat_2', 'Cat_3', 'Cat_4', 'LastPurchase', 'FirstPurchase']
    X_fp = selected_customers[columns]
    Y_fp = selected_customers['cluster']

    st.write("""
    ##### Now lets see some products of each category to understand what each category buys in combination with the table above
    """)
    with st.spinner('Product Lists creation in progress: Wait for it...'):

        list_products_cat_0 = []
        list_products_cat_1 = []
        list_products_cat_2 = []
        list_products_cat_3 = []
        list_products_cat_4 = []
        for index, row in df_cleaned.iterrows():
            if row['Product_Category'] == 0:
                list_products_cat_0.append([row['Description'], row['StockCode']])
            if row['Product_Category'] == 1:
                list_products_cat_1.append([row['Description'], row['StockCode']])
            if row['Product_Category'] == 2:
                list_products_cat_2.append([row['Description'], row['StockCode']])
            if row['Product_Category'] == 3:
                list_products_cat_3.append([row['Description'], row['StockCode']])
            if row['Product_Category'] == 4:
                list_products_cat_4.append([row['Description'], row['StockCode']])

        st.write("**Cat_0 Sample of 3 products:**")
        st.write(list_products_cat_0[:3])
        st.write("**Cat_1 Sample of 3 products:**")
        st.write(list_products_cat_1[:3])
        st.write("**Cat_2 Sample of 3 products:**")
        st.write(list_products_cat_2[:3])
        st.write("**Cat_3 Sample of 3 products:**")
        st.write(list_products_cat_3[:3])
        st.write("**Cat_4 Sample of 3 products:**")
        st.write(list_products_cat_4[:3])