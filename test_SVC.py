import pickle
import random
from os import path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

def get_Data(filename):
    file = open(filename,'rb')
    log = pickle.load(file)
    Frames = []
    Balls = []
    Ball_speed=[]
    Blocker=[]
    pos=[]
    Block_speed=[]
    p1=[]
    p2=[]
    cmd=[]
    for a in range(len(log)):
        pos.append(-2)
        Block_speed.append(0)
        cmd.append(-1)
    for a in range(len(log)):
        if log[a]['ball'][1]>=415:
            pos[a]=log[a]['ball'][0]
        Frames.append(log[a]['frame'])
        Balls.append(log[a]['ball'])
        Ball_speed.append(log[a]['ball_speed'])
        Blocker.append(log[a]['blocker'])
        p1.append(log[a]["platform_1P"])
        p2.append(log[a]["platform_2P"])
    Block_speed[0]=Blocker[1][0]-Blocker[0][0]
    vector=Block_speed[0]
    for i in range(len(log)):
        
        if Blocker[i][0]==0:
            vector=-vector
        elif Blocker[i][0]==200:
            vector=-vector
        Block_speed[i]=vector 
               
    b=-1
    for a in range(0,len(log)):
        if pos[len(log)-1-a]!=-2:
            b=pos[len(log)-1-a]
        else:
            pos[len(log)-1-a]=b
    
    for a in range(0,len(log)):
        if p1[a][0]+20<pos[a]:
            cmd[a]=2
        elif p1[a][0]+20>pos[a]:
            cmd[a]=1
        else:
            cmd[a]=0
   
    #print(len(Balls),len(Ball_speed),len(Blocker),len(pos),len(cmd))
    pos=np.array(pos)
    pos=pos.reshape(len(pos),1)
    cmd=np.array(cmd)
    cmd=cmd.reshape(len(cmd),1)
    Block_speed=np.array(Block_speed)
    Block_speed=Block_speed.reshape(len(Block_speed),1)
    frame_ary = np.array(Frames)
    frame_ary = frame_ary.reshape((len(Frames), 1))
    data = np.hstack((Balls, Ball_speed,Blocker,Block_speed,p1,cmd))
    return data
def get_Data_rand(filename):  #for data of none platform information
    file = open(filename,'rb')
    log = pickle.load(file)
    Frames = []
    Balls = []
    Ball_speed=[]
    Blocker=[]
    pos=[]
    Block_speed=[]
    p1=[]
    p1y=[]
    cmd=[]
    for a in range(len(log)):
        pos.append(-2)
        Block_speed.append(0)
        cmd.append(-1)
        p1.append((random.randint(0,160),420))
    for a in range(len(log)):
        if log[a]['ball'][1]>=415:
            pos[a]=log[a]['ball'][0]
        Frames.append(log[a]['frame'])
        Balls.append(log[a]['ball'])
        Ball_speed.append(log[a]['ball_speed'])
        Blocker.append(log[a]['blocker'])
    Block_speed[0]=Blocker[1][0]-Blocker[0][0]
    vector=Block_speed[0]
    #print(p1)
    for i in range(len(log)):
        
        if Blocker[i][0]==0:
            vector=-vector
        elif Blocker[i][0]==200:
            vector=-vector
        Block_speed[i]=vector 
               
    b=-1
    for a in range(0,len(log)):
        if pos[len(log)-1-a]!=-2:
            b=pos[len(log)-1-a]
        else:
            pos[len(log)-1-a]=b
    
    for a in range(0,len(log)):
        if p1[a][0]+25<pos[a]:
            cmd[a]=2
        elif p1[a][0]+15>pos[a]:
            cmd[a]=1
        else:
            cmd[a]=0

    #print(len(Balls),len(Ball_speed),len(Blocker),len(pos),len(cmd))
    pos=np.array(pos)
    pos=pos.reshape(len(pos),1)
    cmd=np.array(cmd)
    cmd=cmd.reshape(len(cmd),1)
    p1=np.array(p1)
    p1=p1.reshape(len(p1),2)
    Block_speed=np.array(Block_speed)
    Block_speed=Block_speed.reshape(len(Block_speed),1)
    frame_ary = np.array(Frames)
    frame_ary = frame_ary.reshape((len(Frames), 1))
    data = np.hstack((Balls, Ball_speed,Blocker,Block_speed,p1,cmd))
    return data
if __name__ == '__main__':
    #反覆讀取檔案
    filename = path.join(path.dirname(__file__), 'ml_HARD_2020-05-08_12-56-53_rule.pickle')
    data = get_Data(filename)
    #print(data)

    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-09_10-07-42.pickle')
    data1 = get_Data(filename)
    data=np.append(data,data1,axis=0)
    
    #--no plat
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-04-30_17-02-27.pickle')
    data1 = get_Data_rand(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-07_15-19-46.pickle')
    data1 = get_Data_rand(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-08_12-06-47.pickle')
    data1 = get_Data_rand(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-08_12-11-29.pickle')
    data1 = get_Data_rand(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-08_12-19-04.pickle')
    data1 = get_Data_rand(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-08_12-29-36.pickle')
    data1 = get_Data_rand(filename1)
    data=np.append(data,data1,axis=0)
    #11
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-08_12-38-26.pickle')
    data1 = get_Data_rand(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-08_12-44-42.pickle')
    data1 = get_Data_rand(filename1)
    data=np.append(data,data1,axis=0)
    #------
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-09_14-41-31.pickle')
    data1 = get_Data(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-09_14-41-39.pickle')
    data1 = get_Data(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-09_15-08-29.pickle')
    data1 = get_Data(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-09_15-08-39.pickle')
    data1 = get_Data(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-09_15-08-43.pickle')
    data1 = get_Data(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-09_15-18-23.pickle')
    data1 = get_Data(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-09_15-18-27.pickle')
    data1 = get_Data(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-09_15-18-31.pickle')
    data1 = get_Data(filename1)
    data=np.append(data,data1,axis=0)
    #---
    for a in range(2):
        filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-09_15-29-14.pickle')
        data1 = get_Data(filename1)
        data=np.append(data,data1,axis=0)
        
        filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-09_15-29-19.pickle')
        data1 = get_Data(filename1)
        data=np.append(data,data1,axis=0)
        
        filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-09_15-29-23.pickle')
        data1 = get_Data(filename1)
        data=np.append(data,data1,axis=0)
    
    #---
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-09_16-27-02.pickle')
    data1 = get_Data(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-09_16-27-11.pickle')
    data1 = get_Data(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-09_16-27-40.pickle')
    data1 = get_Data(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-09_17-20-44.pickle')
    data1 = get_Data(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-09_17-20-47.pickle')
    data1 = get_Data(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-09_17-42-17.pickle')
    data1 = get_Data(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-09_17-42-25.pickle')
    data1 = get_Data(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-09_17-42-28.pickle')
    data1 = get_Data(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-09_17-50-42.pickle')
    data1 = get_Data(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-09_17-50-50.pickle')
    data1 = get_Data(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-09_17-50-58.pickle')
    data1 = get_Data(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-09_17-51-01.pickle')
    data1 = get_Data(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-09_17-51-06.pickle')
    data1 = get_Data(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-09_19-30-30.pickle')
    data1 = get_Data(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-09_19-30-33.pickle')
    data1 = get_Data(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-09_19-30-36.pickle')
    data1 = get_Data(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-09_19-58-28.pickle')
    data1 = get_Data(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-09_20-16-56.pickle')
    data1 = get_Data(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-09_20-18-17.pickle')
    data1 = get_Data(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-09_20-19-34.pickle')
    data1 = get_Data(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-10_08-11-39.pickle')
    data1 = get_Data(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-10_08-11-51.pickle')
    data1 = get_Data(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-10_08-12-15.pickle')
    data1 = get_Data(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-10_08-12-35.pickle')
    data1 = get_Data(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-10_08-27-03.pickle')
    data1 = get_Data(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-10_08-27-05.pickle')
    data1 = get_Data(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-10_08-27-11.pickle')
    data1 = get_Data(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-10_08-27-18.pickle')
    data1 = get_Data(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-10_08-41-01.pickle')
    data1 = get_Data(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-10_08-41-11.pickle')
    data1 = get_Data(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-10_08-41-16.pickle')
    data1 = get_Data(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-10_08-41-38.pickle')
    data1 = get_Data(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-10_09-21-20.pickle')
    data1 = get_Data(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-10_09-35-32.pickle')
    data1 = get_Data(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-10_10-06-05.pickle')
    data1 = get_Data(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-10_10-06-05.pickle')
    data1 = get_Data(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-10_10-12-12.pickle')
    data1 = get_Data(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-10_10-32-31.pickle')
    data1 = get_Data(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-10_10-35-08.pickle')
    data1 = get_Data(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-10_10-33-51.pickle')
    data1 = get_Data(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-10_10-36-20.pickle')
    data1 = get_Data(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-10_10-36-29.pickle')
    data1 = get_Data(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-10_10-36-29.pickle')
    data1 = get_Data_rand(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-10_10-36-39.pickle')
    data1 = get_Data(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-10_10-53-40.pickle')
    data1 = get_Data(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-10_10-54-05.pickle')
    data1 = get_Data(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-10_10-54-12.pickle')
    data1 = get_Data(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-10_10-54-12.pickle')
    data1 = get_Data_rand(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-10_10-54-20.pickle')
    data1 = get_Data(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-10_10-54-20.pickle')
    data1 = get_Data_rand(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-10_10-54-35.pickle')
    data1 = get_Data(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-10_10-58-16.pickle')
    data1 = get_Data(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-10_10-58-48.pickle')
    data1 = get_Data(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-10_11-19-57.pickle')
    data1 = get_Data(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-10_11-20-16.pickle')
    data1 = get_Data(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-10_11-20-32.pickle')
    data1 = get_Data(filename1)
    data=np.append(data,data1,axis=0)
    
    filename1 = path.join(path.dirname(__file__), 'ml_HARD_2020-05-10_11-20-45.pickle')
    data1 = get_Data(filename1)
    data=np.append(data,data1,axis=0)

    print(len(data))
    mask=[0,1,2,3,4,5,6,7,8]
    X=data[:,mask]
    Y=data[:,-1]
    
    #訓練model
    x_train , x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2)
    
    platform_predict_clf = svm.SVC(gamma=0.001, decision_function_shape='ovo')
    
    platform_predict_clf.fit(x_train,y_train)        
    
    y_predict = platform_predict_clf.predict(x_test)
    print(y_predict)
    
    accuracy = metrics.accuracy_score(y_test, y_predict)
    print("Accuracy(正確率) ={:8.3f}%".format(accuracy*100))
    
    #儲存model
    with open('save/SVM_C.pickle', 'wb') as f:
        pickle.dump(platform_predict_clf, f)
    
    