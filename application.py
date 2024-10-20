from flask import Flask , Response ,request , render_template
import pickle 
import numpy as np
import pandas as pd



with open('Model/scaler.pkl' , "rb") as f:  # we did not use ../Model b/c my application is in main folder
    scaler_model = pickle.load(f)

with open("Model/log_regressor.pkl" , 'rb') as f:
    logistic_model = pickle.load(f)



application = Flask( __name__)
app = application



# Route for homepage
@app.route( "/" )
def index():

    return render_template( "index.html")


# Route for single datapoint prediction
@app.route("/predictdata" , methods = ["GET" , "POST"])
def  predict_datapoint():
    result =""

    if request.method == "POST":

        Pregnancies = int( request.form.get("Pregnancies"))
        Glucose = float(request.form.get( "Glucose"))
        BloodPressure = float(request.form.get("BloodPressure"))
        SkinThickness = float(request.form.get("SkinThickness"))
        Insulin      = float(request.form.get("Insulin"))
        BMI = float( request.form.get("BMI"))
        DiabetesPedigreeFunction = float( request.form.get("DiabetesPedigreeFunction"))
        Age = float( request.form.get("Age"))

        new_scaled_data = scaler_model.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        predict = logistic_model.predict( new_scaled_data)


        if predict[0] == 1:
            result = "Diabetic"
        else:
            result = "No-Diabetic"

        return render_template("single_prediction.html", result = result)

    else:
        return render_template("home.html")




if __name__ == "__main__":
    app.run( host= "0.0.0.0")