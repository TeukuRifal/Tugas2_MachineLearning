## Tugas 2 - Pembelajaran Mesin

Artikel berikut menjadi referensi dalam mengerjakan Tugas 2 - Pembelajaran Mesin:

<a herf>https://medium.com/@youness.habach/support-vector-machines-svm-explanation-mini-project-9d4b4962be52</href>

<p>Anda diminta untuk menggunakan SVM dalam menyelesaikan 2 kasus yaitu Klasifikasi dan Regresi dengan menggunakan dataset yang tidak sama dengan dataset yang ada pada referensi.<p></p>

<p>Nama: T.Rifal AUlia</p>
<p></p>NPM : 2108107010064</p>

**Libary Python yang dipakai**

1. Numpy
2. Scikit-Learn
3. Matplotlib
4. Seaborn

## Cara menjalankan Program SVM (SVC dan SVR)

## 1. Klasifikasi (SVC) Loan Prediction

  <p>Dataset yang digunakan berasal dari kaggle : https://www.kaggle.com/datasets/muratkokludataset/pumpkin-seeds-dataset/data</p>
  <p>Biji labu sering dikonsumsi sebagai penganan di seluruh dunia karena kandungan protein, lemak, karbohidrat, dan mineralnya yang memadai. Penelitian ini dilakukan pada dua jenis biji labu yang paling penting dan berkualitas, ''Urgup_Sivrisi'' dan ''Cercevelik'', yang umumnya ditanam di daerah Urgup dan Karacaoren di Turki. Namun, pengukuran morfologi dari 2500 biji labu dari kedua varietas tersebut dapat dilakukan dengan menggunakan teknik ambang batas abu-abu dan biner. Dengan mempertimbangkan fitur morfologi, semua data dimodelkan dengan lima metode pembelajaran mesin yang berbeda: Regresi Logistik (LR), Multilayer Perceptron (MLP), Support Vector Machine (SVM) dan Random Forest (RF), dan k-Nearest Neighbor (k-NN), yang selanjutnya menentukan metode yang paling berhasil untuk mengklasifikasikan varietas biji labu. Namun, kinerja model-model tersebut ditentukan dengan bantuan metode validasi silang 10 kfold.SVM 88,64 persen

</p>
<p>Berikut adalah fiturnya, terdapat 13 fitur dan 1 class</p>
   
    1   Perimeter          2500 non-null   float64
    2   Major_Axis_Length  2500 non-null   float64
    3   Minor_Axis_Length  2500 non-null   float64
    4   Convex_Area        2500 non-null   int64  
    5   Equiv_Diameter     2500 non-null   float64
    6   Eccentricity       2500 non-null   float64
    7   Solidity           2500 non-null   float64
    8   Extent             2500 non-null   float64
    9   Roundness          2500 non-null   float64
    10  Aspect_Ration      2500 non-null   float64
    11  Compactness        2500 non-null   float64
    12  Class              2500 non-null   object

<p>
      ## Support Vector Machine (SVM) Model for Pumpkin Seed Classification

### Introduction
This repository contains code for building a Support Vector Machine (SVM) model to classify different varieties of pumpkin seeds based on morphological features. The dataset used in this project consists of morphological measurements of 'Urgup_Sivrisi' and 'Cercevelik' pumpkin seed varieties, obtained using thresholding techniques.

### Steps for Modeling SVM
1. **Import Libraries**: Load necessary libraries including numpy, pandas, matplotlib, seaborn, and sklearn.
   
2. **Load Data**: Load the dataset from a CSV file into a pandas DataFrame.
   
3. **Split Features and Labels**: Separate features (morphological measurements) and labels (seed varieties) from the dataset.
   
4. **Split Training and Test Data**: Split the data into training and test sets for model evaluation.
   
5. **Feature Scaling**: Scale the features using StandardScaler from sklearn to ensure uniformity of scale.
   
6. **SVM Modeling**: Train an SVM model using the linear kernel on the training data.
   
7. **Evaluate Model**: Evaluate the trained model using the test data to calculate accuracy, confusion matrix, and classification report.
   
8. **Visualize Data and Decision Boundaries**: If possible, visualize the data and decision boundaries. However, due to high-dimensional data, direct visualization may not be feasible.

### Dataset Source
The dataset used in this project is sourced from [here](https://www.muratkoklu.com/datasets/). It contains morphological measurements of pumpkin seeds and their corresponding varieties.

### Results
The SVM model achieved an accuracy of 88.64% in classifying pumpkin seed varieties.

For more details, refer to the [research paper](https://link.springer.com/content/pdf/10.1007/s10722-021-01226-0.pdf) associated with this dataset.

vscode</p>

## 2. Regresi (SVR) Boston House

   <p>Dataset yang digunakan berasal dari kaggle : https://www.kaggle.com/datasets/saquib7hussain/experience-salary-dataset</p>

   <p>
    Dataset ini berisi informasi tentang hubungan antara pengalaman kerja (dalam bulan) dan gaji bulanan yang sesuai (dalam ribuan dolar) karyawan di berbagai industri. Dataset ini dirancang untuk membantu para penggemar data dan calon ilmuwan data mempraktikkan teknik regresi linier dengan menganalisis dan memodelkan prediksi gaji berdasarkan pengalaman.
   </p>

<p>   
   Kolom:

<br> 0 exp(in months) 1000 non-null float64
<br> 1 salary(in thousands) 1000 non-null float64

</p>

   ## Langkah-Langkah untuk Prediksi Gaji berdasarkan Pengalaman Menggunakan SVR

### 1. Import Library
Impor library yang diperlukan untuk pemrosesan data dan pembuatan model, termasuk numpy, pandas, matplotlib.pyplot, StandardScaler, train_test_split, dan SVR dari scikit-learn.

### 2. Import Dataset
Impor dataset 'Experience-Salary.csv' menggunakan pandas dan pisahkan fitur (pengalaman) dan label (gaji) dari dataset.

### 3. Standarisasi Fitur
Standarisasi fitur pengalaman menggunakan StandardScaler untuk memastikan bahwa setiap fitur memiliki skala yang serupa.

### 4. Membagi Data
Bagi dataset menjadi data latih dan data uji menggunakan train_test_split dari scikit-learn.

### 5. Membuat dan Melatih Model SVM
Buat model SVR (Support Vector Regression) dengan kernel linear dan latih model menggunakan data latih.

### 6. Evaluasi Model
Hitung skor model untuk data latih dan data uji menggunakan metode score() dari model SVR.

### 7. Visualisasi Data dan Garis Regresi
Visualisasikan data latih dan data uji, serta garis regresi yang dihasilkan oleh model SVR pada data yang telah distandarisasi. Plot scatter plot untuk data latih dan data uji, serta plot garis regresi pada skala data yang telah distandarisasi.

