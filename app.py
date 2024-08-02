import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import os

# Prevent warnings
#st.set_option('deprecation.showPyplotGlobalUse', False)


# Determine the directory of the current script
current_dir = os.path.dirname(__file__)
data_path = os.path.join(current_dir, 'Input', 'social_media_train', 'social_media_train.csv')

# Print out the resolved path for debugging
print(f"Resolved data path: {data_path}")

# Check if the file exists
if os.path.exists(data_path):
    data_ = pd.read_csv(data_path)
else:
    st.error(f"CSV file '{data_path}' not found.")

# Load your models
model_svm = pickle.load(open('model_svm.pkl', 'rb'))
model_dt = pickle.load(open('model_dt.pkl', 'rb'))
model_rf = pickle.load(open('model_rf.pkl', 'rb'))
tuned_knn = pickle.load(open('tuned_knn.pkl', 'rb'))
model_dnn = load_model('model_dnn.h5')
model_lstm = load_model('model_lstm.h5')
classifier = load_model('classifier.h5')
model_resnet = load_model('model_resnet.h5')


# Load label-encoded DataFrame from pickle file
with open('df_train.pkl', 'rb') as f:
    df_train = pickle.load(f)

# Function to perform data visualization
def perform_data_visualization(data):
    st.subheader('Data Visualization')

    # Pairplot
    st.write('### Pairplot')
    
    pairplot = sns.pairplot(data, hue='fake')  # Replace 'target_variable' with your actual target variable
    st.pyplot(pairplot)

    # Heatmap
    st.write('### Heatmap')
    plt.figure(figsize=(10, 8))  # Add this line to create a new figure for the heatmap
    corr = df_train.corr()
    heatmap = sns.heatmap(corr, annot=True, cmap='coolwarm')
    st.pyplot(heatmap.figure)


    # Histograms
    st.write('### Histograms')
    numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
    for col in numeric_columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(data[col], kde=True)
        plt.title(f'Histogram of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        st.pyplot()

    # Class imbalance representation (assuming binary classification)
    st.write('### Class Imbalance Representation')
    sns.countplot(x='fake', data=data)
    plt.title('Class Distribution')
    plt.xlabel('Target Variable')
    plt.ylabel('Count')
    st.pyplot()

def main():
    st.title('Fake Social Media Profile Detection App')
    st.write('Please use the Sidepanel to interact with the App!')

    # Sidebar panel for navigation
    st.sidebar.title('Navigation')
    app_mode = st.sidebar.selectbox('Select Option', ['Select an Option', 'Select Model (ML or DL)', 'Perform Data Visualization'])

    if app_mode == 'Select Model (ML or DL)':
        st.subheader('Select Model')

        model_type = st.radio('Select Model Type', ('Machine Learning (ML)', 'Deep Learning (DL)'))

        if model_type == 'Machine Learning (ML)':
            st.subheader('Machine Learning Models')

            ml_model = st.selectbox('Select Model', ('SVM', 'Decision Tree', 'Random Forest', 'KNN'))

            if ml_model == 'SVM':
                # Input fields for SVM prediction
                st.write('You selected SVM model.')
                profile_pic = st.text_input("Profile picture (1 if present, 0 if not)", key='svm_profile_pic')
                ratio_numlen_username = st.text_input("Ratio of numeric characters in username to length", key='svm_ratio_numlen_username')
                len_fullname = st.text_input("Number of characters in full name", key='svm_len_fullname')
                ratio_numlen_fullname = st.text_input("Ratio of numeric characters in full name to length", key='svm_ratio_numlen_fullname')
                len_desc = st.text_input("Number of characters in account description", key='svm_len_desc')
                extern_url = st.text_input("Whether the account description includes a URL (1) or not (2)", key='svm_extern_url')
                private = st.text_input("Whether the account is private (1) or not (2).", key='svm_private')
                num_posts = st.text_input("Number of posts", key='svm_num_posts')
                num_followers = st.text_input("Number of followers", key='svm_num_followers')
                num_following = st.text_input("Number of following", key='svm_num_following')
                if st.button('Predict'):
                    data = np.array([profile_pic, ratio_numlen_username, len_fullname, ratio_numlen_fullname, len_desc, extern_url, private, num_posts, num_followers, num_following]).reshape(1, -1).astype(float)
                    output = model_svm.predict(data)
                    if output == 0:
                        st.success('This account is not Spam!')
                    else:
                        st.error('This account is Spam!')

            elif ml_model == 'Decision Tree':
                # Input fields for Decision Tree prediction
                st.write('You selected Decision Tree model.')
                profile_pic = st.text_input("Profile picture (1 if present, 0 if not)", key='svm_profile_pic')
                ratio_numlen_username = st.text_input("Ratio of numeric characters in username to length", key='svm_ratio_numlen_username')
                len_fullname = st.text_input("Number of characters in full name", key='svm_len_fullname')
                ratio_numlen_fullname = st.text_input("Ratio of numeric characters in full name to length", key='svm_ratio_numlen_fullname')
                len_desc = st.text_input("Number of characters in account description", key='svm_len_desc')
                extern_url = st.text_input("Whether the account description includes a URL (1) or not (2)", key='svm_extern_url')
                private = st.text_input("Whether the account is private (1) or not (2).", key='svm_private')
                num_posts = st.text_input("Number of posts", key='svm_num_posts')
                num_followers = st.text_input("Number of followers", key='svm_num_followers')
                num_following = st.text_input("Number of following", key='svm_num_following')
                if st.button('Predict'):
                    data = np.array([profile_pic, ratio_numlen_username, len_fullname, ratio_numlen_fullname, len_desc, extern_url, private, num_posts, num_followers, num_following]).reshape(1, -1).astype(float)
                    output = model_dt.predict(data)
                    if output == 0:
                        st.success('This account is not Spam!')
                    else:
                        st.error('This account is Spam!')

            elif ml_model == 'Random Forest':
                # Input fields for Random Forest prediction
                st.write('You selected Random Forest model.')
                profile_pic = st.text_input("Profile picture (1 if present, 0 if not)", key='svm_profile_pic')
                ratio_numlen_username = st.text_input("Ratio of numeric characters in username to length", key='svm_ratio_numlen_username')
                len_fullname = st.text_input("Number of characters in full name", key='svm_len_fullname')
                ratio_numlen_fullname = st.text_input("Ratio of numeric characters in full name to length", key='svm_ratio_numlen_fullname')
                len_desc = st.text_input("Number of characters in account description", key='svm_len_desc')
                extern_url = st.text_input("Whether the account description includes a URL (1) or not (2)", key='svm_extern_url')
                private = st.text_input("Whether the account is private (1) or not (2).", key='svm_private')
                num_posts = st.text_input("Number of posts", key='svm_num_posts')
                num_followers = st.text_input("Number of followers", key='svm_num_followers')
                num_following = st.text_input("Number of following", key='svm_num_following')
                if st.button('Predict'):
                    data = np.array([profile_pic, ratio_numlen_username, len_fullname, ratio_numlen_fullname, len_desc, extern_url, private, num_posts, num_followers, num_following]).reshape(1, -1).astype(float)
                    output = model_rf.predict(data)
                    if output == 0:
                        st.success('This account is not Spam!')
                    else:
                        st.error('This account is Spam!')

            elif ml_model == 'KNN':
                # Input fields for KNN prediction
                st.write('You selected KNN model.')
                profile_pic = st.text_input("Profile picture (1 if present, 0 if not)", key='svm_profile_pic')
                ratio_numlen_username = st.text_input("Ratio of numeric characters in username to length", key='svm_ratio_numlen_username')
                len_fullname = st.text_input("Number of characters in full name", key='svm_len_fullname')
                ratio_numlen_fullname = st.text_input("Ratio of numeric characters in full name to length", key='svm_ratio_numlen_fullname')
                len_desc = st.text_input("Number of characters in account description", key='svm_len_desc')
                extern_url = st.text_input("Whether the account description includes a URL (1) or not (2)", key='svm_extern_url')
                private = st.text_input("Whether the account is private (1) or not (2).", key='svm_private')
                num_posts = st.text_input("Number of posts", key='svm_num_posts')
                num_followers = st.text_input("Number of followers", key='svm_num_followers')
                num_following = st.text_input("Number of following", key='svm_num_following')
                if st.button('Predict'):
                    data = np.array([profile_pic, ratio_numlen_username, len_fullname, ratio_numlen_fullname, len_desc, extern_url, private, num_posts, num_followers, num_following]).reshape(1, -1).astype(float)
                    output = tuned_knn.predict(data)
                    if output == 0:
                        st.success('This account is not Spam!')
                    else:
                        st.error('This account is Spam!')

        elif model_type == 'Deep Learning (DL)':
            st.subheader('Deep Learning Models')

            dl_model = st.selectbox('Select Model', ('Dense Neural Network (DNN)', 'LSTM', 'ResNet'))

            if dl_model == 'Dense Neural Network (DNN)':
                # Input fields for DNN prediction
                st.write('You selected Dense Neural Network (DNN) model.')
                profile_pic = st.text_input("Profile picture (1 if present, 0 if not)", key='dnn_profile_pic')
                ratio_numlen_username = st.text_input("Ratio of numeric characters in username to length", key='dnn_ratio_numlen_username')
                len_fullname = st.text_input("Number of characters in full name", key='dnn_len_fullname')
                ratio_numlen_fullname = st.text_input("Ratio of numeric characters in full name to length", key='dnn_ratio_numlen_fullname')
                len_desc = st.text_input("Number of characters in account description", key='dnn_len_desc')
                extern_url = st.text_input("Whether the account description includes a URL (1) or not (2)", key='dnn_extern_url')
                private = st.text_input("Whether the account is private (1) or not (2).", key='dnn_private')
                num_posts = st.text_input("Number of posts", key='dnn_num_posts')
                num_followers = st.text_input("Number of followers", key='dnn_num_followers')
                num_following = st.text_input("Number of following", key='dnn_num_following')
                if st.button('Predict'):
                    data = np.array([profile_pic, ratio_numlen_username, len_fullname, ratio_numlen_fullname, len_desc, extern_url, private, num_posts, num_followers, num_following]).reshape(1, -1).astype(float)
                    output = model_resnet.predict(data)
                    output = (output > 0.5).astype(int)
                    if output == 0:
                        st.success('This account is not Spam!')
                    else:
                        st.error('This account is Spam!')

            elif dl_model == 'LSTM':
                # Input fields for LSTM prediction
                st.write('You selected LSTM model.')
                profile_pic = st.text_input("Profile picture (1 if present, 0 if not)", key='lstm_profile_pic')
                ratio_numlen_username = st.text_input("Ratio of numeric characters in username to length", key='lstm_ratio_numlen_username')
                len_fullname = st.text_input("Number of characters in full name", key='lstm_len_fullname')
                ratio_numlen_fullname = st.text_input("Ratio of numeric characters in full name to length", key='lstm_ratio_numlen_fullname')
                len_desc = st.text_input("Number of characters in account description", key='lstm_len_desc')
                extern_url = st.text_input("Whether the account description includes a URL (1) or not (2)", key='lstm_extern_url')
                private = st.text_input("Whether the account is private (1) or not (2).", key='lstm_private')
                num_posts = st.text_input("Number of posts", key='lstm_num_posts')
                num_followers = st.text_input("Number of followers", key='lstm_num_followers')
                num_following = st.text_input("Number of following", key='lstm_num_following')
                if st.button('Predict'):
                    data = np.array([profile_pic, ratio_numlen_username, len_fullname, ratio_numlen_fullname, len_desc, extern_url, private, num_posts, num_followers, num_following]).reshape(1, -1).astype(float)
                    output = model_resnet.predict(data)
                    output = (output > 0.5).astype(int)
                    if output == 0:
                        st.success('This account is not Spam!')
                    else:
                        st.error('This account is Spam!')

            elif dl_model == 'ResNet':
                # Input fields for ResNet prediction
                st.write('You selected ResNet model.')
                profile_pic = st.text_input("Profile picture (1 if present, 0 if not)", key='resnet_profile_pic')
                ratio_numlen_username = st.text_input("Ratio of numeric characters in username to length", key='resnet_ratio_numlen_username')
                len_fullname = st.text_input("Number of characters in full name", key='resnet_len_fullname')
                ratio_numlen_fullname = st.text_input("Ratio of numeric characters in full name to length", key='resnet_ratio_numlen_fullname')
                len_desc = st.text_input("Number of characters in account description", key='resnet_len_desc')
                extern_url = st.text_input("Whether the account description includes a URL (1) or not (2)", key='resnet_extern_url')
                private = st.text_input("Whether the account is private (1) or not (2).", key='resnet_private')
                num_posts = st.text_input("Number of posts", key='resnet_num_posts')
                num_followers = st.text_input("Number of followers", key='resnet_num_followers')
                num_following = st.text_input("Number of following", key='resnet_num_following')
                if st.button('Predict'):
                    data = np.array([profile_pic, ratio_numlen_username, len_fullname, ratio_numlen_fullname, len_desc, extern_url, private, num_posts, num_followers, num_following]).reshape(1, -1).astype(float)
                    # Replace predict_classes with predict and apply a threshold for binary classification
                    output = model_resnet.predict(data)
                    output = (output > 0.5).astype(int)

                    if output == 0:
                        st.success('This account is not Spam!')
                    else:
                        st.error('This account is Spam!')

    elif app_mode == 'Perform Data Visualization':
        perform_data_visualization(data_)  # Pass your loaded dataset here

if __name__ == '__main__':
    main()
