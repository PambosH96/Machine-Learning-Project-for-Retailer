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
st.set_option('deprecation.showPyplotGlobalUse', False)
plt.style.use('fivethirtyeight')
mpl.rc('patch', edgecolor = 'dimgray', linewidth=1)

# Custom imports for pages
from multipage import MultiPage
import data_preparation # the data cleaning file
import dataset_overview # the dataset overview File
import data_modeling # the data modeling File
import dataset_explorer # dataset explorer
import intro # introductory page

# Create an instance of the app 
app = MultiPage()
# st.set_page_config(layout="wide")
st.sidebar.markdown("# CEI523 - FINAL ASSIGNMENT")
st.sidebar.markdown("""
## In the present web application there are 5 steps:
* Explore the DataFrame: Understand what data and the problem
* Data Preparation: Data Overview & Cleaning
* Data Preparation: Feature Engineering & Data Transformation
* Data Modeling: Train the ML Models, Evaluate them and Interpret the Results

***Note**: Follow the steps 1 by one to get the results at the end
""")

# Add pages to the sidebar and associate them with each app function with the corresponding file.
app.add_page("1. Introduction to the Problem and the Domain", intro.app)
app.add_page("2. Explore The Dataframe", dataset_explorer.app)
app.add_page("3. Data Preparation: Dataset Overview & Cleaning", dataset_overview.app)
app.add_page("4. Data Preparation: Feature Engineering", data_preparation.app)
app.add_page("5. Data Modelling, Evaluation and Results", data_modeling.app)

# The main app
app.run()