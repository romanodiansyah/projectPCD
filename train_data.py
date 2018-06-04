from sklearn.preprocessing import LabelEncoder
import numpy as np
import cv2
import os
import glob
import csv
import metodePCD as mp

#Inisiasi awal
fixed_size = tuple((1000,1000)) #Untuk mengatur ukuran gambar nantinya, px x px
train_path = "./data_train/" #Alamat untuk data yang mau di training
train_labels = os.listdir(train_path) #Mendapatkan label dari data train, ambilnya dari nama folder
train_labels.sort() #Mengurutkan  nama folder
print(train_labels)

# Inisiasi vektor dan label serta atribut yang akan digunakan nantinya
global_features = []
labels = []

i, j = 0, 0
k = 0

#==================================================================

for training_name in train_labels: #Loop folder
    dir = os.path.join(train_path, training_name) #Menggabungkan path training dengan nama foldernya
    current_label = training_name #Mendapatkan nama folder yang sedang diakses
    k = 1
    for file in glob.glob(dir + "/*.jpg"): #Melakukan loop ke semua file di dalam folder dan menambahkan nama filenya yang berekstensi .jpg
        image = cv2.imread(file) #Read image
        image = cv2.resize(image, fixed_size) #Resize gambar
        #Melakukan ekstrasi informasi dari gambar
        humoments = mp.hu_moments(image)
        cannywhite = mp.canny(image)
        morphsum = mp.morph(image)
        H,S,V = mp.rataHSV(image)
        diamA, diamB = mp.diameterDetect(image)
        global_feature = np.hstack([humoments, cannywhite, morphsum, H, S, V, diamA,diamB, current_label]) #Menggabungkan berbagai atribut kedalam satu atribut

        global_features.append(global_feature) #Menggabungkan hasil penggabungan atribut ke dalam gabungan yang lebih besar lagi

        i += 1
        k += 1
    print("[STATUS] processed folder: {}".format(current_label))
    j += 1

#==========================================================================================

targetNames = np.unique(labels) #Mengambil nama untuk dijadikan label
le = LabelEncoder()
target = le.fit_transform(labels)

x = input("Apakah anda yakin mau mencetak data? Data akan di overwrite jika sudah ada sebelumnya (Y/N)")
if (x=='Y' or x=='y'):
    with open('data.csv', 'w') as myDaun: #Mengekstrak informasi untuk di expor menjadi .csv
        daun = csv.writer(myDaun, dialect='excel')
        daun.writerows(global_features)
    myDaun.close()
    print("Ekstrasi data berhasil!")
else:
    print("Ekstrasi data gagal!")

