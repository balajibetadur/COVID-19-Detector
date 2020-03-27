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


if __name__=="__main__":

   
    df = pd.read_csv("coivd.csv")
    train,test = data_split(df,0.2)

    x_train = train[['fever','age','bodypain','cold','breath']].to_numpy()
    x_test = test[['fever','age','bodypain','cold','breath']].to_numpy()

    y_train = train[['prob']].to_numpy().reshape(1600,)
    y_test = test[['prob']].to_numpy().reshape(400,)
    

    clf = LogisticRegression(solver='lbfgs',multi_class='auto')

    clf.fit(x_train,y_train)
    # user_input=[[97,19,0,0,0]]
    # prob2=clf.predict(user_input)
    # print(prob2)
    
    file1=open ('model.pkl','wb')
    pickle.dump(clf,file1)
    file1.close()
