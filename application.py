import pickle
from flask import Flask, request, jsonify, render_template # for result in json format 
import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler #cuz we've to load the pickle file, 

application = Flask(__name__) # Flask..
app = application

## import ridge regressor model and standard scaler pickle 
ridge_model = pickle.load(open("models/ridge.pkl","rb"))
standard_scaler = pickle.load(open("models/scaler.pkl","rb")) 

## route for homepage 
@app.route("/")
def index() : #with every route we write its function to execute.
    return render_template("index.html") # mere home page mein jab bhi apan jaaye, index.html page aajaana chahiye
# http://127.0.0.1:5000 -- ip address jaha bhi machine mein chal ra hain, but cannot access publicly
# http://172.18.0.12:5000 -- machine ip address, jo publicly available hain 

@app.route("/predictdata", methods= ["GET","POST"])  # thru ridge pkl
def predict_datapoint():
    # saare values ko retrive krke model file se predict karna hain..
    if request.method == "POST" : # agar post method hain, toh interact with pkl file and then post
        ## saara data padna hain, from the form 
        Temperature = float(request.form.get("Temperature")) #Field ka value idhar..
        RH = float(request.form.get("RH"))
        Ws = float(request.form.get("Ws"))
        Rain = float(request.form.get("Rain"))
        FFMC = float(request.form.get("FFMC"))
        DMC = float(request.form.get("DMC"))
        ISI = float(request.form.get("ISI"))
        Classes = float(request.form.get("Classes"))
        Region = float(request.form.get("Region"))

    ## ab scaling karna hain.. puure values mil gaye.. ab
    # new data ko aap hamesha transform hi toh karte hain.. uske baad Feature Scaling kardenge..

        new_data_scaled = standard_scaler.transform([[Temperature,RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]]) # 2d array, nested list 
        result = ridge_model.predict(new_data_scaled)
    
        return render_template("home.html", result = result[0])   # result = {{}}

    else :
        return render_template("home.html") # get ka functionality 

    

if __name__=="__main__":
    app.run(host="0.0.0.0") # host address..run bhi run karra, will map with the local address 127.0.0.1
