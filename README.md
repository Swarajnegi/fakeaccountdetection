# Fake Social Media Account Detection

## Software and Tools Required

1. [Github Account](https://github.com)
2. [heroku Account](https://heroku.com)
3. [VSCodeIDE](https://cpde.visualstudio.com/)
4. [GitCLI](htps://git-scm.com/book/en/v2/Getting-Started-The-Command-Line)
5. [Streamlit Account](https://streamlit.io/)

## Installation

Clone the repository and install dependencies:

### Run the following commands in your command line
git clone https://github.com/yourusername/fake-account-detection.git

cd fake-account-detection

pip install -r requirements.txt

## Running the App

1. Streamlit:

To run the Streamlit app locally, navigate to the directory where app.py is located and execute the following command:

streamlit run app.py

2. Deploying to Heroku
   
Ensure you have the Heroku CLI installed.
execute the following command:

heroku login

heroku create your-app-name

git add .
git commit -m "Add Procfile for Heroku deployment"

git push heroku main

heroku open

## Project Structure

fake-account-detection/
│
├── fakeaccdet/
│   ├── Input/
│   │   ├── fake_account__data_dict/
│   │   ├── Instagram fake spammer/
│   │   ├── social_media_aim/
│   │   ├── social_media_test/
│   │   └── social_media_train/
│   │       └── social_media_train.csv
│   ├── my_dir/
│   ├── templates/
│   │   └── home.html
│   ├── .gitignore
│   ├── app.py
│   ├── classifier.h5
│   ├── df_train.pkl
│   ├── Dockerfile
│   ├── fake-social-media-account-detection.ipynb
│   ├── LICENSE
│   ├── model_dnn.h5
│   ├── model_dt.pkl
│   ├── model_lstm.h5
│   ├── model_resnet.h5
│   ├── model_rf.pkl
│   ├── model_svm.pkl
│   └── requirements.txt
└── README.md

## License
This project is licensed under the Apache 2.0 License. See the LICENSE file for details.

Feel free to modify the git clone URL and other specifics as needed for your repository and setup.
