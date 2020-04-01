from flask import Flask,render_template,request
app = Flask(__name__)
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle

def data_split(data,ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    te_size=int(len(data) * ratio)
    te_indices=shuffled[:te_size]
    tr_indices=shuffled[te_size:]
    return data.iloc[tr_indices],data.iloc[te_indices]


file1=open('model.pkl','rb')
clf = pickle.load(file1)
file1.close()

@app.route('/',methods=["GET","POST"])
def prob():
    if request.method=="POST":
        # try:
            fe=request.form
            fever2=float(fe['fever'])
            fever=round(fever2)
            age=int(fe['age'])
            pain=int(fe['pain'])
            nose=int(fe['nose'])
            breath=int(fe['breath'])
            df = pd.read_csv("coivd.csv")
            train,test = data_split(df,0.2)

            x_train = train[['fever','age','bodypain','cold','breath']].to_numpy()
            x_test = test[['fever','age','bodypain','cold','breath']].to_numpy()

            y_train = train[['prob']].to_numpy().reshape(1600,)
            y_test = test[['prob']].to_numpy().reshape(400,)
            

            clf = LogisticRegression(solver='lbfgs',multi_class='auto')

            clf.fit(x_train,y_train)
            user_input=[[fever,age,pain,nose,breath]]
            prob3=clf.predict(user_input)
            if prob3==1:
                o_inf=1
            else:
                o_inf=0
            print(o_inf)
            prob=(breath*12500)+(nose*1000)+(pain*1000)+(fever*202)+(age*3)
            
            if prob <21000:
                inf=0
            elif prob>=21000 and prob < 25000:
                inf=1
            elif prob >= 25000 and prob < 27000:
                inf=2
            elif prob>=27000:
                inf=3
            print(request.form)
            print(prob)
            print(prob3)
            print(inf)
            print(o_inf)
            return render_template('result.html',inf=inf,o_inf=o_inf)
        # except:
        #     return render_template('index.html')
            
    return render_template('index.html')



if __name__ == "__main__":
    app.run(debug=True)
