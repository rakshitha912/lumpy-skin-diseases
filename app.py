import pandas as pd 
import os
from sklearn.tree import ExtraTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
from odeflask import Flask, render_template, request
from io import BytesIO
import pickle
import joblib


app=Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/viewdata', methods=["GET", "POST"])
def viewdata():
    # Load the dataset
    dataset = pd.read_csv(r'Lumpy skin disease data.csv')
    
    # Get the top 10 rows of the dataset
    top_10_rows = dataset.head(15)
    
    # Print dataset info for debugging
    print(top_10_rows)
    print(dataset.columns)

    # Render the HTML table with top 10 rows
    return render_template("viewdata.html", columns=dataset.columns.values, rows=top_10_rows.values.tolist())


@app.route('/preprocess', methods=['POST', 'GET'])
def preprocess():
    global x, y, x_train, x_test, y_train, y_test,  df
    if request.method == "POST":
        size = int(request.form['split'])
        size = size / 100
        df=pd.read_csv(r'correlation.csv')

        ##splitting
        x=df.drop('lumpy',axis=1)
        y=df['lumpy']

        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

        # describes info about train and test set
        print("Number transactions X_train dataset: ", x_train.shape)
        print("Number transactions y_train dataset: ", y_train.shape)
        print("Number transactions X_test dataset: ", x_test.shape)
        print("Number transactions y_test dataset: ", y_test.shape)

    
        print(x_train,x_test)

        return render_template('preprocess.html', msg='Data Preprocessed and It Splits Successfully')
    return render_template('preprocess.html')

@app.route('/model', methods=["POST","GET"])
def model():
    if request.method=="POST":
        global model
        s=int(request.form['algo'])
        if s==0:
            return render_template('model.html',msg="Choose an algorithm")
        elif s==1:
            
            mlp=MLPClassifier()
            mlp.fit(x_train,y_train)
            y_pred=mlp.predict(x_test)
            ac_mlp=accuracy_score(y_pred,y_test)
            ac_mlp=ac_mlp*100
            msg="The Accuracy  obtained by MLPClassifier is "+str(ac_mlp) + str('%')
            return render_template("model.html",msg=msg)
        elif s==2:
            
            nb = GaussianNB()
            nb.fit(x_train,y_train)
            y_pred=nb.predict(x_test)
            ac_nb=accuracy_score(y_pred,y_test)
            ac_nb=ac_nb*100
            msg="The Accuracy  obtained by GaussianNB "+str(ac_nb) +str('%')
            return render_template("model.html",msg=msg)
        elif s==3:
            
            ex = ExtraTreeClassifier(ccp_alpha=0.7)
            ex.fit(x_train,y_train)
            y_pred=ex.predict(x_test)
            ac_dt=accuracy_score(y_pred,y_test)
            ac_dt=ac_dt*100
            msg="The Accuracy obtained by ExtraTreeClassifier is "+str(ac_dt) +str('%')
            return render_template("model.html",msg=msg)
        
    return render_template("model.html")

#=====================================================================================================

# Load the models and encoders
nb_model = joblib.load('naive_bayes_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/prediction', methods=["POST", "GET"])
def prediction():
    if request.method == "POST":
        try:
            # Getting input values from the form
            f1 = float(request.form['y'])
            f2 = request.form['region']
            f3 = request.form['country']
            f4 = float(request.form['cld'])
            f5 = float(request.form['dtr'])
            f6 = float(request.form['frs'])
            f7 = float(request.form['pet'])
            f8 = float(request.form['pre'])
            f9 = float(request.form['tmn'])
            f10 = float(request.form['tmp'])
            f11 = float(request.form['tmx'])
            f12 = float(request.form['vap'])
            f13 = float(request.form['wet'])
            f14 = float(request.form['elevation'])
            f15 = float(request.form['X5_Ct_2010_Da'])

            # Encode categorical features using LabelEncoders
            f2_encoded = label_encoders['region'].transform([f2])  # Encode 'region'
            f3_encoded = label_encoders['country'].transform([f3])  # Encode 'country'

            # Convert encoded values to standard Python integers
            f2_encoded_int = int(f2_encoded[0])  # Convert np.int64 to regular int
            f3_encoded_int = int(f3_encoded)  # Convert np.int64 to regular int

            # Create the feature vector with encoded values
            feature_vector = [
                f1, f2_encoded_int, f3_encoded_int, f4, f5, f6, f7,
                f8, f9, f10, f11, f12, f13, f14, f15
            ]

            # Convert feature vector to a 2D array (1 row, 15 columns)
            feature_vector = [feature_vector]

            print(f"Feature vector: {feature_vector}")

            # Make a prediction using the pre-trained Naive Bayes model
            result = nb_model.predict(feature_vector)

            # Output message based on prediction result
            if result[0] == 0:
                msg = 'The predicted result is No Lumpy Skin Disease'
            else:
                msg = 'The predicted result is Lumpy Skin Disease'

            return render_template('prediction.html', msg=msg)

        except Exception as e:
            # If any error occurs, display a generic error message
            msg = f"An error occurred: {str(e)}"
            return render_template('prediction.html', msg=msg)

    return render_template("prediction.html")

#=====================================================================================================

 
# Load your saved MobileNet model (only once at startup)
model_path = "my_mobilenet_model.h5"
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
else:
    model = None

@app.route('/classification', methods=["GET", "POST"])
def classification():
    msg = ""
    if request.method == "POST":
        # Check if an image was uploaded
        if 'skin_image' not in request.files:
            msg = "No file part in the request."
            return render_template("classification.html", msg=msg)

        file = request.files['skin_image']
        if file.filename == '':
            msg = "No image selected for uploading."
            return render_template("classification.html", msg=msg)


        if not model:
            msg = "Model not found on the server."
            return render_template("classification.html", msg=msg)

        try:
            file_bytes = file.read()
            img_stream = BytesIO(file_bytes)
            # Read the image file and prepare for prediction
            img = image.load_img(img_stream, target_size=(150, 150))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = x / 255.0  

            # Get prediction
            predictions = model.predict(x)
            
            if predictions[0][0] >= 0.5:
                msg = "The model predicts: Healthy Skin"
            else:
                msg = "The model predicts: Lumpy Skin"
        except Exception as e:
            msg = f"Error in processing the image: {str(e)}"

    return render_template("classification.html", msg=msg)


if __name__=="__main__":
    app.run(debug=True)