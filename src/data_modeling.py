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
import pickle
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
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
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


def get_x_y_train():
    X = pd.read_csv('./src/data/x_data.csv')
    y = pd.read_csv('./src/data/y_data.csv')
    return X, y['cluster']


def clusters_histogram(y_cluster, title ):
    x_unique, y_counts = np.unique(y_cluster, return_counts=True)

    x = x_unique
    y = y_counts
    # Use the hovertext kw argument for hover text
    fig = go.Figure(data=[go.Bar(x=x,
                                 y=y,
                                 text='Clusters and Quantity of Clients')])
    # Customize aspect
    fig.update_traces(marker_color='rgb(158,202,225)',
                      marker_line_color='rgb(8,48,107)',
                      marker_line_width=1.5, opacity=0.6)

    fig.update_layout(
        title_text='Histogram of Clusters Distribution and Quantity Of Clients in set {}'.format(title),
        title_x=0.5,
        xaxis = dict(
                type='category',
                title='Clusters'),
        yaxis = dict(title='Quantity of Clients')
    )
    # fig.show()
    st.plotly_chart(fig)

def clusters_predictions_histogram(y_cluster, y_predictions, title):
    # Calculate y_cluster unique values and their frequency
    x_unique, y_counts = np.unique(y_cluster, return_counts=True)

    x = x_unique
    y = y_counts
    # Use the hovertext kw argument for hover text
    fig = go.Figure()
    
    fig.add_trace(go.Bar(x=x,
                         y=y,
                         name='Y true Clusters',
                         text='Cluster and Quantity of Clients',
                         marker_color='rgb(158,202,225)',
                         marker_line_color='rgb(8,48,107)',
                         marker_line_width=1.5,
                         opacity=0.6))
    
    # Calculate y_predictions unique values and their frequency
    x_unique, y_counts = np.unique(y_predictions, return_counts=True)

    x = x_unique
    y = y_counts
    
    fig.add_trace(go.Bar(x=x,
                         y=y,
                         name='Y Predictions Clusters',
                         marker_color='lightsalmon',
                         marker_line_color='rgb(18,38,207)',
                         marker_line_width=1.5,
                         opacity=0.6))

    fig.update_layout(barmode='group',
                      bargroupgap=0.2,
                      title_text='Histogram of Clusters Distribution and Quantity Of Clients in {}'.format(title),
                      title_x=0.5,
                      xaxis = dict(
                          type='category',
                          title='Clusters'),
                      yaxis = dict(title='Quantity of Clients')
    )
    # fig.show()
    st.plotly_chart(fig, use_container_width=True)



models = {"svm": svm.SVC(),
          "KNN": KNeighborsClassifier(),
          "Random Forest": RandomForestClassifier(),
          "Gauusian": GaussianNB(),
          "BaggingClassifier": BaggingClassifier(),
          "ExtraTreesClassifier": ExtraTreesClassifier(),
          "DecisionTreeClassifier": DecisionTreeClassifier()}

# Create a function to fit and score models
def fit_and_score(models, X_train, X_test, y_train, y_test):
    """
    Fits and evaluates given machine learning models.
    models : a dict of differetn Scikit-Learn machine learning models
    X_train : training data (no labels)
    X_test : testing data (no labels)
    y_train : training labels
    y_test : test labels
    """
    # Set random seed
    np.random.seed(42)
    # Make a dictionary to keep model scores
    model_scores = {}
    # Loop through models
    for name, model in models.items():
        # Fit the model to the data
        model.fit(X_train, y_train)
        # Evaluate the model and append its score to model_scores
        model_scores[name] = model.score(X_test, y_test)
    return model_scores

def MLP_Classifier():
    mlp_clf = MLPClassifier(hidden_layer_sizes=(30, 30, 30),
                            activation='relu',
                            solver='adam',
                            alpha=0.000099,
                            batch_size=32,
                            learning_rate='adaptive',
                            learning_rate_init=0.001,
                            power_t=0.5,
                            max_iter=246,
                            shuffle=True,
                            random_state=42,
                            tol=0.0001,
                            verbose=False,
                            warm_start=True,
                            momentum=0.9,
                            nesterovs_momentum=True,
                            early_stopping=True,
                            validation_fraction=0.05,
                            beta_1=0.9,
                            beta_2=0.999,
                            epsilon=1e-08,
                            n_iter_no_change=100)
    return mlp_clf

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Greens):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
        
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

def RandomForestTraining(X_train, y_train, X_test, y_test):
    clf_rf = RandomForestClassifier()
    clf_rf.fit(X_train, y_train)
    predictions = clf_rf.predict(X_test)
    score = clf_rf.score(X_test, y_test)
    report = metrics.classification_report(y_test, predictions, output_dict=True)
    df = pd.DataFrame(report).transpose()
    st.markdown("#### Classification Report: Random Forest Classifier")
    st.dataframe(df)
    class_names = [i for i in range(14)]
    cnf_matrix = confusion_matrix(y_test, predictions) 
    np.set_printoptions(precision=2)
    plt.figure(figsize = (10,10))
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize = False, title='Confusion Matrix')
    return clf_rf, predictions, score

def ExtraTreesTraining(X_train, y_train, X_test, y_test):
    clf_et = ExtraTreesClassifier()
    clf_et.fit(X_train, y_train)
    predictions = clf_et.predict(X_test)
    score = clf_et.score(X_test, y_test)
    report = metrics.classification_report(y_test, predictions, output_dict=True)
    df = pd.DataFrame(report).transpose()
    st.markdown("#### Classification Report: Extra Trees Classifier")
    st.dataframe(df)
    class_names = [i for i in range(14)]
    cnf_matrix = confusion_matrix(y_test, predictions) 
    np.set_printoptions(precision=2)
    plt.figure(figsize = (10,10))
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize = False, title='Confusion Matrix')
    return clf_et, predictions, score

def BaggingTraining(X_train, y_train, X_test, y_test):
    clf_bg = ExtraTreesClassifier()
    clf_bg.fit(X_train, y_train)
    predictions = clf_bg.predict(X_test)
    score = clf_bg.score(X_test, y_test)
    report = metrics.classification_report(y_test, predictions, output_dict=True)
    df = pd.DataFrame(report).transpose()
    st.markdown("#### Classification Report: Bagging Classifier")
    st.dataframe(df)
    class_names = [i for i in range(14)]
    cnf_matrix = confusion_matrix(y_test, predictions) 
    np.set_printoptions(precision=2)
    plt.figure(figsize = (10,10))
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize = False, title='Confusion Matrix')
    return clf_bg, predictions, score

def prepare_input(cart_price_test, kmeans_trained):
    cart_price_test['InvoiceDate'] = cart_price_test['InvoiceDate'].astype('datetime64[ns]')
    transactions_per_user=cart_price_test.groupby(by=['CustomerID'])['Cart Price'].agg(['count','min','max','mean','sum'])
    for i in range(5):
        col = 'Cat_{}'.format(i)
        transactions_per_user.loc[:,col] = cart_price_test.groupby(by=['CustomerID'])[col].sum() / transactions_per_user['sum']*100

    transactions_per_user.reset_index(drop = False, inplace = True)
    cart_price_test.groupby(by=['CustomerID'])['Cat_0'].sum()

    # Correcting time range
    transactions_per_user['count'] = 5 * transactions_per_user['count']
    transactions_per_user['sum']   = transactions_per_user['count'] * transactions_per_user['mean']

    transactions_per_user.sort_values('CustomerID', ascending = True).head(5)
    last_date = cart_price_test['InvoiceDate'].max().date()

    first_registration = pd.DataFrame(cart_price_test.groupby(by=['CustomerID'])['InvoiceDate'].min())
    last_purchase      = pd.DataFrame(cart_price_test.groupby(by=['CustomerID'])['InvoiceDate'].max())

    test_fp  = first_registration.applymap(lambda x:(last_date - x.date()).days)
    test_lp = last_purchase.applymap(lambda x:(last_date - x.date()).days)

    transactions_per_user.loc[:, 'FirstPurchase'] = test_fp.reset_index(drop=False)['InvoiceDate']
    transactions_per_user.loc[:, 'LastPurchase'] = test_lp.reset_index(drop=False)['InvoiceDate']

    list_cols = ['count','min','max','mean', 'Cat_0','Cat_1','Cat_2','Cat_3','Cat_4', 'LastPurchase', 'FirstPurchase']
    test_matrix = transactions_per_user[list_cols].to_numpy()

    minmax_scaler = MinMaxScaler()
    minmax_scaler.fit(test_matrix)
    minmaxscaled_test_matrix = minmax_scaler.transform(test_matrix)
    X = transactions_per_user[list_cols]
    Y = kmeans_trained.predict(minmaxscaled_test_matrix)

    return X, Y
  
def app():
    st.title("DATA MODELING SECTION")
    st.write("""
    ### Training, evaluating the models and interpreting the results:
    Sections of this page:
    - Training and Evaluating the models
    - Evaluation Insights
    \n
    To train the 4 models we will follow the process bellow (as explained in FE section):
    - The Train and test Datasets are splitted into two parts: `First Period Dataset`, `Last Period Dataset`
    - The `Last Period Dataset` will contain the data of the last two months of the year and the `First Period Dataset` will contain the first 10 months of the year.
    - The reason is because of seasonality. We have to remember that we have data for exact on year
    \n
    Algorithms:
    - Multi Layer Perceptron (MLP)
    - Random Forest Regression
    - Extra Trees Classifier
    - Baggin Classifier

    """)

    st.write("# TRAINING AND EVALUATING THE MODELS (CLASSIFIERS)")
    X_first_period, y_first_period = get_x_y_train()

    st.markdown("""
    #### We have to split our dataset to X_train, X_test, y_train, y_test 
    """)

    st.code("X_train, X_test, y_train, y_test = model_selection.train_test_split(X_first_period, y_first_period, train_size = 0.8, random_state=42)")
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X_first_period, y_first_period, train_size = 0.8, random_state=42)

    st.markdown("""
    #### We need to scale our data
    """)
    st.code("""
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    """)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    st.code("""
    cv_results_fp = cross_validate(MLP_Classifier(),
                        X_train,
                        y_train,
                        cv=10, 
                        return_train_score=True, 
                        scoring='accuracy')
    """)

    with st.spinner('MLP 10-FOLD VALIDATION CLASSIFIER TRAINING: Wait for it...'):
        cv_results_fp = cross_validate(MLP_Classifier(),
                                X_train,
                                y_train,
                                cv=10, 
                                return_train_score=True, 
                                scoring='accuracy')

        for k_fp, s_fp in enumerate(cv_results_fp['test_score']):
            st.write("Fold {} with Test Accuracy Score: {}".format(k_fp, s_fp))

    st.write("Average Test Accuracy Score: {}".format(np.sum(cv_results_fp['test_score'])/10))

    st.markdown("""
    ##### Now lets train MLP properly
    From the K-fold validation we can see that the algorithm performs quite well with `cv=10`
    """)

    st.code("""
    mlp_clf = MLP_Classifier()
    predictions = mlp_clf.predict(X_test)
    metrics.accuracy_score(y_test, predictions)
    mlp_clf.fit(X_train, y_train)
    """)

    with st.spinner('MLP CLASSIFIER TRAINING with First Period Data: Wait for it...'):
        mlp_clf = MLP_Classifier()
        mlp_clf.fit(X_train, y_train)
        predictions = mlp_clf.predict(X_test)
        report = metrics.classification_report(y_test, predictions, output_dict=True)
        df = pd.DataFrame(report).transpose()
        st.dataframe(df)
        class_names = [i for i in range(14)]
        cnf_matrix = confusion_matrix(y_test, predictions) 
        np.set_printoptions(precision=2)
        plt.figure(figsize = (10, 10))
        plot_confusion_matrix(cnf_matrix, classes=class_names, normalize = False, title='Confusion matrix')
        st.write("#### Clusters Prediction Histogram: MLP Classifier")
        clusters_predictions_histogram(y_test, predictions, 'Y Sets')

    st.markdown("""
    ## Find some more classifiers that perform well utilizing the custom made function bellow:
    """)
    
    st.code("""
    models = {"svm": svm.SVC(),
          "KNN": KNeighborsClassifier(),
          "Random Forest": RandomForestClassifier(),
          "Gauusian": GaussianNB(),
          "BaggingClassifier": BaggingClassifier(),
          "ExtraTreesClassifier": ExtraTreesClassifier(),
          "DecisionTreeClassifier": DecisionTreeClassifier()}

    # Create a function to fit and score models
    def fit_and_score(models, X_train, X_test, y_train, y_test):
        Fits and evaluates given machine learning models.
        models : a dict of differetn Scikit-Learn machine learning models
        X_train : training data (no labels)
        X_test : testing data (no labels)
        y_train : training labels
        y_test : test labels
        # Set random seed
        np.random.seed(42)
        # Make a dictionary to keep model scores
        model_scores = {}
        # Loop through models
        for name, model in models.items():
            # Fit the model to the data
            model.fit(X_train, y_train)
            # Evaluate the model and append its score to model_scores
            model_scores[name] = model.score(X_test, y_test)
        return model_scores
    """)

    with st.spinner('Fast Training of 6 models: Wait for it...'):
        results = fit_and_score(models, X_train, X_test, y_train, y_test)

    st.markdown("Results for a fast training of the models:\n")
    st.markdown(results)

    st.markdown("""
    #### From the results above we can see the three best performing algorithms are the ensemble algorithms:
    * Ranndom Forest
    * Extra Trees Classifier
    * Baggin Classifier
    So, we decided to train also these three algorithms and evaluate them.
    """)

    st.markdown("### Random Forest Classifier")

    with st.spinner('Training Random Forest: Wait for it...'):
        clf_rf, rf_predictions, rf_score = RandomForestTraining(X_train, y_train, X_test, y_test)

        st.write("SCORE: {}".format(rf_score))

        st.write("#### Clusters Prediction Histogram: Random Forest Classifier")
        clusters_predictions_histogram(y_test, rf_predictions, 'Y Sets')

    with st.spinner('Training Extra Trees Classifier: Wait for it...'):
        clf_et, et_predictions, et_score = ExtraTreesTraining(X_train, y_train, X_test, y_test)

        st.write("SCORE: {}".format(et_score))

        st.write("#### Clusters Prediction Histogram: Extra Trees Classifier")
        clusters_predictions_histogram(y_test, et_predictions, 'Y Sets')

    with st.spinner('Training Bagging Classifier: Wait for it...'):
        clf_bg, bg_predictions, bg_score = BaggingTraining(X_train, y_train, X_test, y_test)

        st.write("SCORE: {}".format(bg_score))

        st.write("#### Clusters Prediction Histogram: Baggin Classifier")
        clusters_predictions_histogram(y_test, bg_predictions, 'Y Sets')


    st.markdown("""
    ## TRAINING WITH THE LAST PERIOD DATA
    #### Now lets train MLP Algorithm with the test set we saved in the previous step and thus check if the classifiers are going to perform the same as the first_period
    """)

    st.markdown("""
    #### MLP Classifier with the Last period data
    """)

    # import the kmeans from the previous stage (Feature Engineering) because we need it to reconstruct the data
    with open("./src/models/kmeans.pkl", "rb") as f:
        kmeans_trained = pickle.load(f)

    # kmeans_trained = st.session_state.kmeans

    cart_price_test = pd.read_csv('./src/data/test_set.csv')
    X_last_period, y_last_period = prepare_input(cart_price_test, kmeans_trained)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X_last_period, y_last_period, train_size = 0.8, random_state=42)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    with st.spinner('10-fold training and validation with the Last Date: Wait for it...'):
        cv_results_lp = cross_validate(mlp_clf,
                                X_train,
                                y_train,
                                cv=10, 
                                return_train_score=True, 
                                scoring='accuracy')

        for k_lp, s_lp in enumerate(cv_results_lp['test_score']):
            print("Fold {} with Test Accuracy Score: {}".format(k_lp, s_lp))

    print("Average Test Accuracy Score: {}".format(np.sum(cv_results_lp['test_score'])/10))
    st.write("Average Test Accuracy Score: {}".format(np.sum(cv_results_lp['test_score'])/10))

    mlp_clf = MLP_Classifier()
    mlp_clf.fit(X_train, y_train)
    predictions = mlp_clf.predict(X_test)

    metrics.accuracy_score(y_test, predictions)
    report = metrics.classification_report(y_test, predictions, output_dict=True)
    df = pd.DataFrame(report).transpose()
    st.dataframe(df)
    class_names = [i for i in range(14)]
    cnf_matrix = confusion_matrix(y_test, predictions) 
    np.set_printoptions(precision=2)
    plt.figure(figsize = (10, 10))
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize = False, title='Confusion matrix')


    st.markdown("### Random Forest Classifier with Last Period Data")
    with st.spinner('Training Random Forest with Last Period Data: Wait for it...'):
        clf_rf_lp, predictions, rf_score = RandomForestTraining(X_train, y_train, X_test, y_test)

        st.write("SCORE: {}".format(rf_score))

        st.write("#### Clusters Prediction Histogram: Random Forest Classifier")
        clusters_predictions_histogram(y_test, rf_predictions, 'Y Sets')

    with st.spinner('Training Extra Trees Classifier: Wait for it...'):
        clf_et_lp, et_predictions, et_score = ExtraTreesTraining(X_train, y_train, X_test, y_test)

        st.write("SCORE: {}".format(et_score))

        st.write("#### Clusters Prediction Histogram: Extra Trees Classifier")
        clusters_predictions_histogram(y_test, et_predictions, 'Y Sets')

    with st.spinner('Training Bagging Classifier: Wait for it...'):
        clf_bg_lp, bg_predictions, bg_score = BaggingTraining(X_train, y_train, X_test, y_test)

        st.write("SCORE: {}".format(bg_score))

        st.write("#### Clusters Prediction Histogram: Baggin Classifier")
        clusters_predictions_histogram(y_test, bg_predictions, 'Y Sets')

    
    st.markdown("""
    ## TRAINING WITH THE WHOLE DATASET
    #### Now lets train The same Algorithms with the test set we saved in the previous step and thus check if the classifiers are going to perform the same as the first_period
    * As the results suggest from the results above (First period and last period training), we can generalize our results for the whole year
    * The reason for selecting two seperate periods to separately train and test it is because we cannot train a model with the first 10 months of the year and then test it to the last 2.
    """)

    st.markdown("""
    #### MLP Classifier with the whole dataset
    """)
    X = np.concatenate((X_first_period, X_last_period))
    Y = np.concatenate((y_first_period, y_last_period))
    st.write("X shape")
    st.write(X.shape)
    st.write("Y shape")
    st.write(Y.shape)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, train_size = 0.8, random_state=42)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    with st.spinner('10-fold validation with Full Period Data: Wait for it...'):
        cv_results = cross_validate(MLP_Classifier(),
                                X_train,
                                y_train,
                                cv=10, 
                                return_train_score=True, 
                                scoring='accuracy')
        
        for k_fullData, s_fullData in enumerate(cv_results['test_score']):
            print("Fold {} with Test Accuracy Score: {}".format(k_fullData, s_fullData))

        st.write("Average Test Accuracy Score: {}".format(np.sum(cv_results['test_score'])/10))
    
    st.write("### MLP Classifier Training with Full Period Data")
    with st.spinner('MLP Classifier Training with Full Period Data: Wait for it...'):
    
        mlp_clf = MLP_Classifier()
        mlp_clf.fit(X_train, y_train)
        predictions = mlp_clf.predict(X_test)
        report = metrics.classification_report(y_test, predictions, output_dict=True)
        df = pd.DataFrame(report).transpose()
        st.markdown("##### Classification Report: MLP Classifier")
        st.dataframe(df)
        class_names = [i for i in range(14)]
        cnf_matrix = confusion_matrix(y_test, predictions) 
        np.set_printoptions(precision=2)
        plt.figure(figsize = (10, 10))
        plot_confusion_matrix(cnf_matrix, classes=class_names, normalize = False, title='Confusion matrix')

        st.write("#### Clusters Prediction Histogram: MLP Classifier")
        clusters_predictions_histogram(y_test, predictions, 'Y Sets')

    #save MLP
    # with open('./src/models/kmeans.pkl','wb') as f:
    #     pickle.dump(mlp_clf, f)

    st.markdown("### Random Forest Classifier with Last Period Data")
    with st.spinner('Training Random Forest with Whole Dataset: Wait for it...'):
        clf_rf_gen, rf_predictions, rf_score = RandomForestTraining(X_train, y_train, X_test, y_test)

        st.write("SCORE: {}".format(rf_score))

        st.write("#### Clusters Prediction Histogram: Random Forest Classifier")
        clusters_predictions_histogram(y_test, rf_predictions, 'Y Sets')

        # # save Random Forest
        # with open('./src/models/random_forest.pkl','wb') as f:
        #     pickle.dump(clf_rf_gen, f)

    with st.spinner('Training Extra Trees Classifier Whole Dataset: Wait for it...'):
        clf_et_gen, et_predictions, et_score = ExtraTreesTraining(X_train, y_train, X_test, y_test)

        st.write("SCORE: {}".format(et_score))

        st.write("#### Clusters Prediction Histogram: Extra Trees Classifier")
        clusters_predictions_histogram(y_test, et_predictions, 'Y Sets')

        # # save Extra Trees
        # with open('./src/models/extra_trees.pkl','wb') as f:
        #     pickle.dump(clf_et_gen, f)

    with st.spinner('Training Bagging Classifier Whole Dataset: Wait for it...'):
        clf_bg_gen, bg_predictions, bg_score = BaggingTraining(X_train, y_train, X_test, y_test)

        st.write("SCORE: {}".format(bg_score))

        st.write("#### Clusters Prediction Histogram: Baggin Classifier")
        clusters_predictions_histogram(y_test, bg_predictions, 'Y Sets')

        # # save Bagging Classifier
        # with open('./src/models/bagging.pkl','wb') as f:
        #     pickle.dump(clf_bg_gen, f)


## SOME RESULTS
    st.write("# EVALUATION INSIGHTS")
    st.write("Here we will write the results and our conclusions")

    st.markdown("""
    The classifiers trained are from scikit-learn:
    - MultiLayer Perceptron
    - Bagging Classifier
    - Extra Trees Classifier
    - Random Forest Classifier

    We can clearly see that the all the classifiers perform well in all the cases:
    - First 10 months data
    - Last 2 months data
    - Full Year data

    The accuracy/precision/recall F1 metrics seems to perform approximately to 90% and this let us conclude that our classifiers perform well in all the cases.

    Additionally, we could not use the first period data for training and the last two months for testing because we clearly lose our seasonality.

    Finally, our models seems that can be used in production on real world retail stores to cluster the products and the customers in order to predict what a customer is going to buy

    """)