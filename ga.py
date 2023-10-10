import numpy as np
import mido
import glob
from mido import MidiFile
import pickle
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import scipy.stats as stats
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

r = 50
c = 20
score = []

def ga(song,ind):
    pop = np.random.uniform(low=0, high=np.max(song), size=(r,c)).astype('int')
    glbvar = np.std(song)
    for i in range(r):
        song[ind:ind+20] = pop[i]
        var = np.std(song[(ind-20):(ind+40)])
        score.append(glbvar-var)
    for j in range(1000):
        while(1):
            rand = np.random.randint(r,size=2)
            if rand[0]!=rand[1] :
                break
        crom1= pop[rand[0]]
        crom2= pop[rand[1]]
        rr = np.random.randint(c)
        crom1[0:rr], crom2[0:rr] = crom2[0:rr], crom1[0:rr]
        #crom2[0:rr], crom1[0:rr] = crom1[0:rr], crom2[0:rr]
        song[ind:ind+20] = crom1
        var1 = np.std(song[(ind-20):(ind+40)])
        song[ind:ind+20] = crom2
        var2 = np.std(song[(ind-20):(ind+40)])
        temps=glbvar-var1
        if temps >= score[rand[0]]:
            pop[rand[0]]= crom1
            score[rand[0]] = temps
        temps=glbvar-var2
        if temps >= score[rand[1]]:
            pop[rand[1]]= crom2
            score[rand[1]] = temps   
    max = np.max(score)
    mres = np.where(score == max)
    mpos = mres[0][0]
    song[ind:ind+20] = song[mpos:mpos+20]

    #print(pop)

train_ds= pickle.load(open('train_data.pkl' , 'rb'))
train_lb= pickle.load(open('train_labels.pkl' , 'rb'))

test_ds = pickle.load(open('test_data.pkl' , 'rb'))
test_lb = pickle.load(open('test_labels.pkl' , 'rb'))

 

train_data=[]
train_labels=[]

for i in range(len(train_ds)):
    a = np.array(train_ds[i])
    l = np.array(train_lb[i])
    s = int(len(a)/4)
    r = int(len(a)/10)
    x = a[0:s-r-1]
    train_data.append(x)
    train_labels.append(l)
    y = a[s-r:(2*s)-1]
    train_data.append(y)
    train_labels.append(l)
    z = a[(2*s)-r:(3*s)-1]
    train_data.append(z)
    train_labels.append(l)
    t = a[(3*s)-r:4*(s)]
    train_data.append(t)
    train_labels.append(l)

train_data = np.array(train_data)
train_labels = np.array(train_labels)


x = []
for e in train_data:
    xx = []
    xx.append(np.mean(e))
    xx.append(np.max(e))
    xx.append(np.min(e))
    xx.append(np.std(e))
    xx.append(np.median(e))
    xx.append(stats.skew(e))
    xx.append(stats.kurtosis(e))
    xx.append(stats.mode(e)[0])
    xx.append(stats.mode(e)[1])
    xx.append(len(np.unique(e)))
    q = []
    for r in np.unique(e):
        qr = np.array(np.where(e == r))
        we = np.delete(qr , 0)
        qr = np.delete(qr , -1)
        if(len(we) > 0):
            we -= qr
            q.append(np.mean(we))
    xx.append(np.mean(q))
    xx.extend(np.histogram(e , bins = 8)[0])
    xx.extend(np.histogram(q , bins = 8)[0])
    x.append(xx)

x = np.array(x)
#x = np.reshape(x , (-1 ,  1))
sc = MinMaxScaler()
x = sc.fit_transform(x)

k=0
y = []

# for p in test_ds:
#     p = np.array(p)
#     result = np.where(p == 0)
#     pos = result[0][0]
#     ga(p,pos)
#     test_ds[k]=p.tolist()
#     k = k+1

for d in test_ds:
    yy = []
    yy.append(np.mean(d))
    yy.append(np.max(d))
    yy.append(np.min(d))
    yy.append(np.std(d))
    yy.append(np.median(d))
    yy.append(stats.skew(d))
    yy.append(stats.kurtosis(d))
    yy.append(stats.mode(d)[0])
    yy.append(stats.mode(d)[1])
    yy.append(len(np.unique(d)))
    qq = []
    for rr in np.unique(d):
        qrr = np.array(np.where(d == rr))
        wee = np.delete(qrr , 0)
        qrr = np.delete(qrr , -1)
        if(len(wee) > 0):
            wee -= qrr
            qq.append(np.mean(wee))
    yy.append(np.mean(qq))
    yy.extend(np.histogram(d , bins = 8)[0])
    yy.extend(np.histogram(qq , bins = 8)[0])

    y.append(yy)

y = np.array(y)

classifiers = [
    #KNeighborsClassifier(3),
    #SVC(kernel="linear", C=0.025),
    #SVC(gamma=2, C=1),
    #DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=10,random_state=5,min_samples_split=2,n_estimators=350),
    #MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(base_estimator=RandomForestClassifier(max_depth=10)),
    #GaussianNB(),
    #QuadraticDiscriminantAnalysis()
    ]
x_train , x_test , y_train , y_test = train_test_split(x , train_labels , test_size=0.3)
for clf in classifiers:
    x_train , x_test , y_train , y_test = train_test_split(x , train_labels , test_size=0.3)
    clf.fit(x_train , y_train)
    print(clf.__class__.__name__)
    print(accuracy_score(y_test , clf.predict(x_test)),"train")
    print(accuracy_score(y_train , clf.predict(x_train)),"overfit")
    print(accuracy_score(test_lb , clf.predict(y)),"test")