from flask import Flask,render_template,request
app = Flask(__name__)
import pickle




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
