import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns
def CarFraud():
    df=pd.read_csv("creditcard.csv")
    
    print(df.head())
    print("__________________________________________________________________________")
    print(df.shape)
    print("__________________________________________________________________________")
    print("Columns Names Are:- ",df.columns)
    print("__________________________________________________________________________")
    fruad=df[df['Class']==1]
    valid=df[df['Class']==0]
    print("TOTAL LENGTH OF FRAUD CASES ARE:- ",len(fruad))
    print("TOTAL LENGTH OF VALID CASES ARE:- ",len(valid))

    print("----CORE_FORMAT-----")
    correation=df.corr()
    sns.heatmap(correation,vmax=.8,square=True)
    plt.show()

    print("__________________________________________________________________________")
    df.drop(['Time'],axis=1,inplace=True)
    print("After Removing 'Class' Column from datset Size will Be:- ",df.shape)

    print("__________________________________________________________________________")
    x=df.drop(['Class'],axis=1)
    #print(x)
    y=df['Class']
    #print(y)
    XData=x.values
    #print(XData)
    YData=y.values
    data_train,data_test,target_train,target_test=train_test_split(XData,YData,test_size=0.9,random_state=42)
    classifire=DecisionTreeClassifier()
    classifire.fit(data_train,target_train)
    prediction=classifire.predict(data_test)

    print(prediction)
    accuracy=accuracy_score(target_test,prediction)
    return accuracy
def main():
   ret= CarFraud()
   print("Your accuracy score is:- ",ret*100)
if __name__=="__main__":
    main()