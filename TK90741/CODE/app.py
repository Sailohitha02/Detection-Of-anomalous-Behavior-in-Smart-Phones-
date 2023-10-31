import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sqlalchemy import true
from xgboost import XGBClassifier
from sklearn.svm import SVC 
from flask import *

app =Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/load',methods=["GET","POST"])
def load():
    global df, dataset
    if request.method == "POST":
        data = request.files['data']
        df = pd.read_csv(data)
        dataset = pd.read_csv("drebin-215-dataset-5560malware-9476-benign_bhavishya_(1).csv")
        dataset = df.head(100)
        msg = 'Data Loaded Successfully'
        return render_template('load.html', msg=msg)
    return render_template('load.html')

@app.route('/view')
def view():
    dataset = pd.read_csv("drebin-215-dataset-5560malware-9476-benign_bhavishya_(1).csv")
    dataset = dataset.head(100)
    return render_template('view.html', columns=dataset.columns.values, rows=dataset.values.tolist())

@app.route('/preprocess', methods=['POST', 'GET'])
def preprocess():
    global x, y, x_train, x_test, y_train, y_test,  countvectorizer,size
    if request.method == "POST":
        size = int(request.form['split'])
        size = size / 100

        print('#########################################')

        df = pd.read_csv("drebin-215-dataset-5560malware-9476-benign_bhavishya_(1).csv")
        from sklearn.preprocessing import LabelEncoder
        le=LabelEncoder()
        df['class']=le.fit_transform(df['class'])
        df=df[['SEND_SMS','android.telephony.SmsManager','READ_PHONE_STATE','RECEIVE_SMS','READ_SMS','android.intent.action.BOOT_COMPLETED','TelephonyManager.getLine1Number','WRITE_SMS','WRITE_HISTORY_BOOKMARKS','TelephonyManager.getSubscriberId','android.telephony.gsm.SmsManager','INSTALL_PACKAGES','READ_HISTORY_BOOKMARKS','INTERNET','ACCESS_LOCATION_EXTRA_COMMANDS','WRITE_APN_SETTINGS','class']]
        print(df)
        df.head()
        y = df['class']
        x = df.drop(['class'], axis = 1) 

        
        

        x_train, x_test, y_train, y_test = train_test_split(x,y, stratify=y, test_size=size, random_state=42)

        # describes info about train and test set
        print("Number transactions X_train dataset: ", x_train.shape)
        print("Number transactions y_train dataset: ", y_train.shape)
        print("Number transactions X_test dataset: ", x_test.shape)
        print("Number transactions y_test dataset: ", y_test.shape)

    
        print(x_train,x_test)

        return render_template('preprocess.html', msg='Data Preprocessed and It Splits Successfully')
    return render_template('preprocess.html')
@app.route('/model',methods=['POST','GET'])
def model():

    if request.method=="POST":
       
        print('ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc')
        s=int(request.form['algo'])
        if s==0:
            return render_template('model.html',msg='Please Choose an Algorithm to Train')
        elif s==1:
            print('aaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb')
            dt = DecisionTreeClassifier()
            dt.fit(x_train,y_train)
            # Predicting the Test set results
            y_pred = dt.predict(x_test)
            acc_dt = accuracy_score(y_pred, y_test)*100
            print('aaaaaaaaaaaaaaaaaaaaaaaaa')
            msg = 'The accuracy obtained by Decision Tree Classifier is ' + str(acc_dt) + str('%')
            return render_template('model.html', msg=msg)
        elif s==2:
            svc = SVC(kernel='linear')
            svc.fit(x_train,y_train)
            acc_svc = svc.score(x_test, y_test)*100
            msg = 'The accuracy obtained by Support Vector Classifier is ' + str(acc_svc) + str('%')
            return render_template('model.html', msg=msg)
        elif s==3:
            xgb = XGBClassifier()
            xgb.fit(x_train,y_train)
            acc_xgb = xgb.score(x_test, y_test)*100
            msg = 'The accuracy obtained by XGBoost Classifier is ' + str(acc_xgb) + str('%')
            return render_template('model.html', msg=msg)
        elif s==4:
            adb = AdaBoostClassifier()
            adb.fit(x_train,y_train)
            acc_adb = adb.score(x_test, y_test)*100
            msg = 'The accuracy obtained by AdaBoost Classifier is ' + str(acc_adb) + str('%')
            return render_template('model.html', msg=msg)
        
    return render_template('model.html')

@app.route('/prediction',methods=['POST','GET'])
def prediction():
    global x_train,y_train
    if request.method == "POST":
        f1 = request.form['f1']
        f2 = request.form['f2']
        f3 = request.form['f3']
        f4 = request.form['f4']
        f5 = request.form['f5']
        f6 = request.form['f6']
        f7 = request.form['f7']
        f8 = request.form['f8']
        f9 = request.form['f9']
        f10 = request.form['f10']
        f11 = request.form['f11']
        f12 = request.form['f12']
        f13 = request.form['f13']
        f14 = request.form['f14']
        f15 = request.form['f15']
        f16 = request.form['f16']
        print(f1)
        
        li = [f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16]
        print(li)
        dt = DecisionTreeClassifier()
        dt.fit(x_train,y_train)
        result = dt.predict([li])
        print(result)
             
        if result == 0:
            msg = ' The Smart Phone Device Has Anomalous Behaviour'
        else:
            msg = 'The Smart Phone Device Has no Anomalous Behaviour'
        return render_template('prediction.html',msg=msg)    

    return render_template('prediction.html')



if __name__ == '__main__':
    app.run(debug=True)