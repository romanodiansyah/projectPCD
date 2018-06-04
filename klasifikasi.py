import numpy as np
import cv2
import glob
import csv
import pandas
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import metodePCD as mp
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import shutil
import os
import uuid

# fixed-sizes for image
fixed_size = tuple((1000,1000)) #Resize pixel menjadi px x px
train_path = "./data_train/" #Path untuk data training
train_labels = os.listdir(train_path) #Mendapatkan label dari data train, ambilnya dari nama folder
train_labels.sort() #Mengurutkan  nama folder
#=====================================================================
filename = open('data.csv', 'r') #Membaca data hasil training
dataframe = pandas.read_csv(filename) #membuat data menjadi data frame

kelas = dataframe.drop(dataframe.columns[:-1], axis=1)
data = dataframe.drop(dataframe.columns[-1:], axis=1)

# Inisiasi vektor dan label serta atribut yang akan digunakan nantinya
local_features = []
global_features = []
labels = []

i, j = 0, 0
k = 0

# Membuat model machine learning
models = []
models.append(('Random Forest',RandomForestClassifier(max_depth=None, random_state=0)))
#Inisiasi atribut yang akan digunakan di proses selanjutnya
results = []
names = []
scoring="accuracy"

import warnings
warnings.filterwarnings('ignore')
for name, model in models:
    kfold = KFold(n_splits=10, random_state=7)
    cv_results = cross_val_score(model, data, kelas, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# Membuat model - Random Forests
clf  =  RandomForestClassifier(max_depth=None, random_state=0)
# Memasukan data training kedalam model
clf.fit(data,kelas)
# Path dari data test
test_path = "./data_test/"

for file in glob.glob(test_path + "/*.jpg"): #Melakukan loop ke semua file di dalam folder dan menambahkan nama filenya yang berekstensi .jpg
    image = cv2.imread(file) #Membaca image
    image = cv2.resize(image, fixed_size) #Resize ukuran gambar
    #Ekstrasi informasi dari gambar
    humoments = mp.hu_moments(image)
    cannywhite = mp.canny(image)
    morphsum = mp.morph(image)
    H,S,V = mp.rataHSV(image)
    diamA, diamB = mp.diameterDetect(image)
    #Menggabungkan informasi atribut
    global_feature = np.hstack([humoments, cannywhite, morphsum, H, S, V, diamA, diamB])
    # Melakukan prediksi gambar
    prediction = clf.predict(global_feature.reshape(1,-1))[0]
    print(prediction)
    # Jika ingin melihat hasil output gambar, hapus # pada kode dibawah
    #cv2.putText(image,prediction, (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)
    #plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    #plt.show()
    ask = input('Tolong bantu kami meningkatkan akurasi! (Y/N)')
    if(ask=='Y' or ask=='y'):
        tanya = input("Apakah klasifikasi benar? Y/N")
        if(tanya == 'Y' or tanya == 'y'):
            local_feature = np.hstack([humoments, cannywhite, morphsum, H, S, V, diamA, diamB, prediction])
            local_features.append(local_feature)
            shutil.move(file,'./data_train/'+prediction+'/'+str(uuid.uuid1())+'.jpg')
        elif(tanya == 'N' or tanya == 'n'):
            print('(1) Chinese Tallow')
            print('(2) Euphorbia Mili')
            print('(3) Excoecaria')
            print('(4) Gordon Croton')
            print('(5) Hevea Brasilinsis')
            print('(6) Tidak Tahu')
            print('Jawab dengan angka 1/2/3/4/5/6')
            lagi = input('Termasuk apakah? ')
            if(lagi=='1'):
                local_feature = np.hstack([humoments, cannywhite, morphsum, H, S, V, diamA, diamB, 'Chinese Tallow'])
                local_features.append(local_feature)
                shutil.move(file,'./data_train/Chinese Tallow/'+str(uuid.uuid1())+'.jpg')
            elif(lagi=='2'):
                local_feature = np.hstack([humoments, cannywhite, morphsum, H, S, V, diamA, diamB, 'Euphorbia Mili'])
                local_features.append(local_feature)
                shutil.move(file,'./data_train/Euphorbia Mili/'+str(uuid.uuid1())+'.jpg')
            elif(lagi=='3'):
                local_feature = np.hstack([humoments, cannywhite, morphsum, H, S, V, diamA, diamB, 'Excoecaria'])
                local_features.append(local_feature)
                shutil.move(file,'./data_train/Excoecaria/'+str(uuid.uuid1())+'.jpg')
            elif(lagi=='4'):
                local_feature = np.hstack([humoments, cannywhite, morphsum, H, S, V, diamA, diamB, 'Gordon Croton'])
                local_features.append(local_feature)
                shutil.move(file,'./data_train/Gordon Croton/'+str(uuid.uuid1())+'.jpg')
            elif(lagi=='5'):
                local_feature = np.hstack([humoments, cannywhite, morphsum, H, S, V, diamA, diamB, 'Hevea Brasilinsis'])
                local_features.append(local_feature)
                shutil.move(file,'./data_train/Hevea Brasilinsis/'+str(uuid.uuid1())+'.jpg')
            else:
                print()

if(ask=='Y' or ask=='y'):
    with open('data.csv', 'a') as myDaun: #Mengekstrak informasi untuk di expor menjadi .csv
        daun = csv.writer(myDaun, dialect='excel')
        daun.writerows(local_features)
    myDaun.close()

print('Terimakasih sudah mencoba Deteksi Daun!')


