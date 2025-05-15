import numpy as np

class DecisionStump:
    def __init__(self,T,W=None):
        self.T=T
        if W==None:
            self.W=[1/len(self.T) for i in range(len(self.T))]
        else:
            self.W=W

        self.bj=None
        self.bv=None
        self.bop=None
        self.min=None
        self.Z=None
        self.a=None
        
    def predict(self,x,j,v,ch):
        if ch=="<":
            if x[j]<v:
                return 1
            else:
                return -1
        elif ch==">":
            if x[j]>v:
                return 1
            else:
                return -1
            
    def predict_trained(self,x):
       return self.predict(x,self.bj,self.bv,self.bop)

    def train(self):   
        self.min=1.0
        for op in [">","<"]:
            for j in range(len(self.T[0][0])):
                X=[x[j] for x,y in self.T]
                
                temp=list(np.unique(X))
                temp.append(np.max(X)+1)
                temp.append(np.min(X)-1)
                for v in temp:
                    e=0.0
                    for i in range(len(self.T)):
                        x=self.T[i][0]
                        y=self.T[i][1]
                        if self.predict(x,j,v,op)!=y:
                            e+=self.W[i]
                    if e<self.min:
                        self.min=e
                        self.bv=v
                        self.bj=j
                        self.bop=op    
        print("bj:",self.bj)
        print("bv:",self.bv)
        print("bop:",self.bop)
        print("min:",self.min)

    def calculate_a(self):
        self.a=np.log((1-self.min)/self.min)/2 
        print("self.a:",self.a) 
        
    def calculate_Z(self):
        self.Z=0.0
        E_a=np.exp(-self.a)
        Ea=np.exp(self.a)
        for i in range(len(train_data)):
            x_i=self.T[i][0]
            y_i=self.T[i][1]
            w_i=self.W[i]
            if(self.predict_trained(x_i)==y_i):
                self.Z+=w_i*E_a
            else:
                self.Z+=w_i*Ea
        print("self.Z:",self.Z)

    def updateW(self):
        for i in range(len(self.T)):
            self.W[i]=self.W[i]*np.exp(-self.a*train_data[i][1]*self.predict_trained(train_data[i][0]))/self.Z;
        print("new W:",self.W);
        return self.W

    def get_a(self):
        return self.a

def f(x,stumplis):
    mySum=0.0
    for stump in stumplis:
        mySum+=stump.get_a()*stump.predict_trained(x)
    return np.sign(mySum)

if __name__=="__main__":
    train_data=[[(0,1,3),-1],
       [(0,3,1),-1],
       [(1,2,2),-1],
       [(1,1,3),-1],
       [(1,2,3),-1],
       [(0,1,2),-1],
       [(1,1,2),1],
       [(1,1,1),1],
       [(1,3,1),-1],
       [(0,2,1),-1]]
    

    stumplis=list()
    result=[False]
    W=[1/len(train_data) for i in range(len(train_data))]
    while not all(result):
        stump=DecisionStump(train_data,W)
        stump.train()
        stump.calculate_a()
        stump.calculate_Z()
        W=stump.updateW()
        stumplis.append(stump)
        result=[f(train_data[i][0],stumplis)==train_data[i][1] for i in range(len(train_data))]
        print("误分类点个数：",result.count(False))
       


