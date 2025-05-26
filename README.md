# Machine Learning Project Report - Rahman Ilyas Al Kahfi

## Project Overview
Pertanian menjadi sektor yang krusial dalam perekonomian dan sebagai sumber penghidupan bagi masyarakat di seluruh dunia. Meskipun begitu, masih banyak kasus di beberapa petani di berbagai negara yang memiliki tingkat produktivitas pertanian masih tergolong rendah. salah satu permasalahan utama yang dihadapi petani yaitu pemilihan jenis tanaman yang tidak sesuai dengan karakteristik lahan pertanian, sehingga menyebabkan rendahnya kualitas dan kuantitas hasil panen. Hal ini akan berdampak pada pendapatan petani yang rendah, ketahanan pangan yang terancam, dan ekonomi yang tidak berkembang secara maksimal.

Menurut (Rajak, R. K., Pawar, A., Pendke, M., Shinde, P., Rathod, S., & Devare, A. , 2017) dalam Jurnal *Crop recommendation system to maximize crop yield using machine learning technique*, pendekatan Pertanian Presisi dapat menjadi solusi untuk permasalahan ketidaksesuaian pemilihan jenis tanaman berdasarkan karakteristik lahan. Pertanian Presisi (*Precision Agriculture*) adalah penggunaan teknologi dan manajemen berbasis data untuk pengoptimalan sumber daya dengan tujuan menghasilkan panen yang berkualitas dalam pertanian. Dalam pendekatan ini, secara umum terdapat sensor yang mengumpulkan data karakteristik lahan pertanian. Pengumpulan data  ini membuka peluang untuk membuat sebuah sistem klasifikasi dengan teknologi Machine Learning untuk merekomendasikan jenis tanaman yang cocok berdasarkan karakteristik lahan pertanian seperti kandungan unsur hara, iklim, dan kondisi tanaman.

Proyek ini bertujuan untuk mengembangkan model pembelajaran mesin yang dapat mengklasifikasikan karakteistik lahan petanian yang dapat membantu petani dalam memilih jenis tanaman yang tepat. Dengan bantuan teknologi Machine Learning, model diharapkan dapat mempelajari pola hubungan antara kondisi tanah dan tanaman yang optimal dari data historis, kemudian menggeneralisasi pengetahuannya untuk memberikan rekomendasi pada lahan-lahan baru. Pendekatan ini tidak hanya meningkatkan efisiensi dan hasil produksi, tetapi juga mendorong penerapan pertanian presisi yang berkelanjutan dan ramah lingkungan.

Referensi : Rajak, R. K., Pawar, A., Pendke, M., Shinde, P., Rathod, S., & Devare, A. (2017). [Crop recommendation system to maximize crop yield using machine learning technique](https://d1wqtxts1xzle7.cloudfront.net/55495855/IRJET-V4I12179-libre.pdf?1515565970=&response-content-disposition=inline%3B+filename%3DCrop_Recommendation_System_to_Maximize_C.pdf&Expires=1748185704&Signature=J92lMDQdx~3Cq8QgwyNeigJbRX8b6pfpmNgRNCqOXumLFTNXGT1~B928gQN1mvEE~HpwF8RBzLJmyyU5ep2wqpX4EXZbY~Xa-QQhUpD2X~5Sqxhe-OFQiIiXYNNqLHEZyioNSn~S9Vx7d~OkItxGDxZ1X6qkamR7oz-tcB1WBWx8fLZnd5aBjZYgpDtNjRP6hepjovb1oo8Us88u90QXFMH5baO9lBzxWhJJAHBvmro1diNWbVVPf~3piWK~8A1DT9yw0HfFkwiikDxQZxKLw~U7jQpgFMN7fonUsWJs0TwloJgEVWH2Avrx9GZ5GdCIkFhrBfyx6GZFmxia-yjKIA__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA). International Research Journal of Engineering and Technology, 4(12), 950-953.


## Business Understanding

### Problem Statements

- Bagaimana memastikan pemilihan jenis tanaman oleh petani sesuai dengan karakteristik lahan?
- Bagaimana pendekatan Pertanian Presisi dapat diimplementasikan secara efektif untuk mengatasi ketidaksesuaian antara tanaman dan lahan pertanian?
- Bagaimana pendekatan Pertanian Presisi dapat membantu mengklasifikasikan karakteristik lahan pertanian secara efisien?

### Goals

- Mengembangkan sistem yang dapat membantu petani dalam memilih jenis tanaman yang sesuai berdasarkan karakteristik lahan, guna meningkatkan kualitas dan kuantitas hasil panen.
- Menerapkan pendekatan Pertanian Presisi dengan memanfaatkan data karakteristik lahan (seperti unsur hara, iklim, dan kondisi tanah) untuk mengoptimalkan pemanfaatan sumber daya pertanian.
- Membangun model pembelajaran mesin yang mampu mengklasifikasikan karakteristik lahan dan merekomendasikan jenis tanaman yang paling cocok secara otomatis dan akurat.

### Solution statements

- Menggunakan algoritma Random Forest dan K-Nearest Neighbors (KNN) untuk mengklasifikasikan jenis tanaman yang sesuai berdasarkan karakteristik lahan pertanian.
- Melakukan tuning hyperparameter menggunakan GridSearchCV untuk menentukan parameter terbaik bagi masing-masing model klasifikasi berdasarkan nilai akurasi tertinggi dan serta hasil evaluasi model yang optimal.


## Data Understanding

Dataset yang digunakan merupakan Crop Recommendation Dataset yang berisi data terkait kondisi tanah dan lingkungan yang memengaruhi pemilihan jenis tanaman yang optimal. Dataset ini terdiri dari 2200 baris data dan digunakan untuk membangun sistem rekomendasi tanaman berdasarkan karakteristik lahan. Sumber Dataset Kaggle: [ Crop Recommendation Dataset](https://www.kaggle.com/datasets/madhuraatmarambhagat/crop-recommendation-dataset). 

### Variabel-variabel pada dataset Crop Recommendation Dataset yaitu :

- N : Kandungan Nitrogen dalam Tanah (dalam mg/kg).
- P : Kandungan Fosfor dalam Tanah (dalam mg/kg).
- K : Kandungan Kalium dalam Tanah (dalam mg/kg).
- temperature : Suhu rata-rata dalam °C.
- humidity: Kelembaban relatif rata-rata dalam %.
- ph: Nilai pH Tanah.
- rainfall: Curah Hujan pada lahan (dalam mm).
- label (Target Variable): Tanaman yang paling sesuai dengan kondisi lahan tertentu. terdapat 22 jenis tanaman dalam dataset (rice, maize, jute, cotton, coconut, papaya, orange, apple, muskmelon, watermelon, grapes, mango, banana, pomegranate, lentil, blackgram, mungbean, mothbeans, pigeonpeas, kidneybeans, chickpea, coffee
).

**Exploratory Data Analysis (EDA) for Variable Description**

![image](assets\df_description.png)

![image](assets\df_stat_description.png)

![image](assets\df_label_counts.png)

Tahap ini dilakukan deskripsi dari tiap fitur mulai dari tipe data, statistik deskriptif, dan jumlah tipe label. Hal ini untuk memberikan informasi awal mengenai dataset.

**Exploratory Data Analysis (EDA) to handle Missing Value and Outliers**

![image](assets\EDA_MissingValue.png)

Tahap ini dilakukan pengecekan Missing Value dalam dataset. Missing Value adalah representasi dari data yang tidak ada, tidak diketahui, atau tidak relevan. Missing Value Perlu dihilangkan agar tidak merusak performa model.

![image](assets\EDA_Boxplot.png)

Boxplot merupakan visualisasi statistik yang menampilkan distribusi data secara ringkas melalui lima nilai statistik utama seperti kuartil pertama (Q1), median (Q2), kuartil ketiga (Q3), batas atas, dan batas bawah yang membantu mengidentifikasi nilai-nilai ekstrem di luar rentang interkuartil (IQR). Nilai-nilai yang ekstrem pada dataset berpotensi sebagai outliers.

**Exploratory Data Analysis (EDA) with Univariate Analysis**

![image](assets\EDA_univariate.png)

Exploratory Data Analysis (EDA) with Univariate Analysis merupakan eksplorasi data yang berfokus menganalisis satu variabel pada satu waktu dengan tujuan memahami karakteristik dan distribusi masing-masing variabel dalam dataset secara individual.

**Exploratory Data Analysis (EDA) with Multivariate Analysis**

![image](assets\EDA_multivariate.png)
![image](assets\EDA_multivariate2.png)

Exploratory Data Analysis (EDA) with Univariate Analysis adalah teknik eksplorasi data yang menganalisis hubungan dan interaksi antara dua atau lebih variabel secara bersamaan untuk mengidentifikasi pola, korelasi, dan dependensi dalam dataset.

**EDA results:**
- Distribusi data yang ditampilkan pada boxplot menunjukan bahwa terdapat data berpotensi menjadi outlier karena nilai yang ekstrem.
- Dari hasil Exploratory Data Analysis (EDA) with Univariate Analysis, menunjukan bahwa sebagian besar fitur (N, P, K, humidity, rainfall) memiliki distribusi yang bimodal/skewed. Hanya fitur pH dan temperature yang mendekati distribusi normal. Dapat disimpulkan bahwa diperlukan normalisasi pada data.
- Dari hasil Exploratory Data Analysis (EDA) with Multivariate Analysis, menunjukkan bahwa terdapat struktur dependensi yang kompleks antar fitur sehingga dataset bisa dimanfaatkan model machine learning untuk membuat prediksi.

## Data Preparation

**Removing Outliers**

![image](assets\zscore.png)

![image](assets\AfterRemoveOutlier.png)

Tahap ini menggunakan z score untuk menghilangkan outliers. Nilai-nilai yang ekstrem pada dataset berpotensi sebagai outliers perlu dihilangkan agar tidak merusak performa model.

**Label Encoding**

![image](assets\LE.png)

![image](assets\LabelEncoding.png)

Label Encoding akan mengubah fitur kategorikal yaitu data berupa teks atau kategori menjadi bentuk numerik. setiap jenis nilai kategori yang ada pada fitur yang diencoding akan diubah menjadi bilangan bulat (integer) yang unik. Label Encoding ini digunakan agar algoritma Machine Learning bisa memproses informasi fitur kategorikal.

**Train Test Split**

![image](assets\split_data.png)

Train-test split adalah metode untuk membagi dataset menjadi dua bagian: subset pelatihan (trainset) dan subset pengujian (testset). Dalam skenario ini, pembagian dilakukan dengan mengalokasikan 80% data untuk trainset yang digunakan untuk melatih model machine learning. Sementara itu, 20% data sisanya dialokasikan untuk testset, yang berfungsi untuk menguji kinerja model pada data yang belum pernah dilihat sebelumnya.

**Standardisation with StandardScaler**

![image](assets\StdScaler.png)

Standard Scaler berfungsi untuk menyesuaikan setiap fitur dalam dataset agar memiliki rata-rata nol dan varians satu. Tujuannya adalah untuk menyamakan rentang nilai antar variabel, sehingga tidak ada fitur yang mendominasi model yang disebabkan oleh skala yang lebih besar.


## Modeling

1. **Random Forest Classifier**: Random Forest adalah algoritma Machine Learning yang termasuk Supervised Learning yang menggunakan beberapa pohon keputusan untuk membuat prediksi. Setiap pohon keputusan dihasilkan dari subsampel kumpulan data dan menggunakan pemisahan berdasarkan fitur yang dipilih secara acak.

**Kelebihan :** 
- Tahan terhadap overfitting karena melakukan rata-rata beberapa pohon. 
- Mampu menangani data yang tidak seimbang.

**Kekurangan :** 
- Cenderung lambat dalam fase pelatihan dan prediksi pada kumpulan data besar.

**Parameter optimalisasi hyperparameter yang digunakan dalam GridSearchCV :**
- ‘n_estimators’: Jumlah pohon keputusan (decision trees) yang akan dibangun dalam ensemble (kumpulan pohon). [100, 200, 300]
- ‘max_depth’: Kedalaman maksimum setiap pohon keputusan dalam ensemble. [None, 10, 20, 30]
- ‘min_samples_split’: Jumlah minimum sampel yang diperlukan untuk membagi node internal. [2, 5, 10] 
- ‘min_samples_leaf’: Jumlah minimum sampel yang diperlukan untuk menjadi node leaf (daun). [1, 2, 4] 
- ‘max_features’: Menentukan jumlah fitur yang akan dipertimbangkan saat mencari pemisahan terbaik. 'sqrt' Mengambil akar kuadrat dari total fitur, 'log2' Mengambil logaritma basis 2 dari total fitur, 'None' Mempertimbangkan semua fitur. ['sqrt', 'log2', None]

2. **K-Nearest Neighbors Classifier**: K-Nearest Neighbors (KNN) adalah algoritma Machine Learning yang termasuk Supervised Learning untuk memprediksi label suatu data baru, KNN mencari sejumlah K tetangga terdekatnya berdasarkan jarak dalam data pelatihan, lalu memilih label yang paling umum di antara tetangga tersebut.

**Kelebihan :** 
- Sederhana dan mudah dipahami. 
- Cocok untuk data dengan distribusi tidak linear.

**Kekurangan :**
- Lambat dalam fase pelatihan pada dataset besar karena perlu hitung jarak ke semua titik.
- Sangat sensitif terhadap skala fitur.

**Parameter optimalisasi hyperparameter yang digunakan dalam GridSearchCV :**
- ‘n_neighbors’: Jumlah tetangga terdekat yang akan dipertimbangkan. [3, 5, 7, 9, 11]
- ‘weights’: bobot diberikan kepada tetangga saat membuat prediksi. 'uniform' Semua tetangga memiliki bobot yang sama sehingga memiliki pengaruh yang sama. 'distance' Tetangga yang lebih dekat memiliki bobot yang lebih besar sehingga tetangga terdekat memiliki pengaruh lebih besar. ['uniform', 'distance']
- ‘metric’: metrik jarak yang akan digunakan untuk menghitung jarak antara titik data. 'euclidean' jarak garis lurus yang paling umum. 'manhattan' mengukur jumlah perbedaan absolut antara koordinat. 'minkowski'  jarak umum yang dapat mencakup Euclidean dan Manhattan. ['euclidean', 'manhattan', 'minkowski']

Setelah melakukan pelatihan model dan optimalisasi hyperparameter yang berpengaruh pada model menggunakan GridSearchCV, model Random Forest Classifier menunjukan performa yang menungguli model K-Nearest Neighbors Classifier berdasarkan parameter berikut:

**Random Forest Classifier**

- Best Parameters: {'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 100}
- Best Score (Cross-Validation): 0.9962981956315289
- Train Accuracy (Best Model): 1.0
- Test Accuracy (Best Model): 0.9950738916256158


**K-Nearest Neighbors Classifier**

- Best Parameters (KNN): {'metric': 'manhattan', 'n_neighbors': 5, 'weights': 'distance'}
- Best Score (Cross-Validation): 0.9753523266856601
- Train Accuracy (Best KNN): 1.0
- Test Accuracy (Best KNN): 0.9827586206896551


## Evaluation

Metrik evaluasi yang digunakan dalam proyek ini adalah :

1. Accuracy : mengukur seberapa banyak prediksi model yang benar dibandingkan dengan seluruh jumlah prediksi yang dilakukan. metrik yang paling mudah dipahami, namun kurang informatif jika data tidak seimbang

Formula Accuracy = (TP + TN) / (TP + TN + FP + FN)

2. Precision : mengukur seberapa akurat prediksi positif model. Dari semua data yang diprediksi sebagai positif, berapa banyak yang benar-benar positif.

Formula Precision: TP / (TP + FP)

3. Recall : mengukur seberapa baik model menangkap semua data positif. Dari semua data yang benar-benar positif, berapa banyak yang berhasil dikenali oleh model.

Formula Recall : TP / (TP + FN)

4. F1 Score : rata-rata harmonik dari Precision dan Recall. Metrik ini berguna untuk mendapatkan keseimbangan antara Precision dan Recall, terutama ketika distribusi kelas tidak seimbang.

Formula F1 Score : 2 * (Precision * Recall) / (Precision + Recall)

**Accuracy Model Random Forest Classifier dan K-Nearest Neighbors Classifier**

![image](assets\compare_acc.png)

**Classification Report Model K-Nearest Neighbors Classifier**

| Kelas       | presisi | recall | f1-score | support |
|-------------|---------|--------|----------|---------|
| apple       | 1.00    | 1.00   | 1.00     | 11      |
| banana      | 1.00    | 1.00   | 1.00     | 20      |
| blackgram   | 1.00    | 1.00   | 1.00     | 17      |
| chickpea    | 1.00    | 1.00   | 1.00     | 20      |
| coconut     | 1.00    | 1.00   | 1.00     | 22      |
| coffee      | 1.00    | 1.00   | 1.00     | 26      |
| cotton      | 1.00    | 1.00   | 1.00     | 20      |
| grapes      | 1.00    | 1.00   | 1.00     | 6       |
| jute        | 0.88    | 1.00   | 0.94     | 15      |
| kidneybeans | 1.00    | 1.00   | 1.00     | 27      |
| lentil      | 1.00    | 1.00   | 1.00     | 17      |
| maize       | 1.00    | 1.00   | 1.00     | 16      |
| mango       | 1.00    | 1.00   | 1.00     | 15      |
| mothbeans   | 1.00    | 1.00   | 1.00     | 19      |
| mungbean    | 1.00    | 1.00   | 1.00     | 21      |
| muskmelon   | 1.00    | 1.00   | 1.00     | 25      |
| orange      | 1.00    | 1.00   | 1.00     | 16      |
| papaya      | 1.00    | 1.00   | 1.00     | 24      |
| pigeonpeas  | 1.00    | 1.00   | 1.00     | 19      |
| pomegranate | 1.00    | 1.00   | 1.00     | 18      |
| rice        | 1.00    | 0.88   | 0.94     | 17      |
| watermelon  | 1.00    | 1.00   | 1.00     | 15      |

| Metrik Agregat (RF) | presisi | recall | f1-score | support |
|---------------------|---------|--------|----------|---------|
| accuracy            |         |        | 1.00     | 406     |
| macro avg           | 0.99    | 0.99   | 0.99     | 406     |
| weighted avg        | 1.00    | 1.00   | 1.00     | 406     |

**Classification Report Model K-Nearest Neighbors Classifier**


| Kelas       | presisi | recall | f1-score | support |
|-------------|---------|--------|----------|---------|
| apple       | 1.00    | 1.00   | 1.00     | 11      |
| banana      | 1.00    | 1.00   | 1.00     | 20      |
| blackgram   | 1.00    | 0.94   | 0.97     | 17      |
| chickpea    | 1.00    | 1.00   | 1.00     | 20      |
| coconut     | 1.00    | 1.00   | 1.00     | 22      |
| coffee      | 1.00    | 1.00   | 1.00     | 26      |
| cotton      | 1.00    | 1.00   | 1.00     | 20      |
| grapes      | 1.00    | 1.00   | 1.00     | 6       |
| jute        | 0.88    | 0.93   | 0.90     | 15      |
| kidneybeans | 0.96    | 1.00   | 0.98     | 27      |
| lentil      | 0.85    | 1.00   | 0.92     | 17      |
| maize       | 1.00    | 1.00   | 1.00     | 16      |
| mango       | 1.00    | 1.00   | 1.00     | 15      |
| mothbeans   | 1.00    | 0.89   | 0.94     | 19      |
| mungbean    | 1.00    | 1.00   | 1.00     | 21      |
| muskmelon   | 1.00    | 1.00   | 1.00     | 25      |
| orange      | 1.00    | 1.00   | 1.00     | 16      |
| papaya      | 1.00    | 1.00   | 1.00     | 24      |
| pigeonpeas  | 1.00    | 0.95   | 0.97     | 19      |
| pomegranate | 1.00    | 1.00   | 1.00     | 18      |
| rice        | 0.94    | 0.88   | 0.91     | 17      |
| watermelon  | 1.00    | 1.00   | 1.00     | 15      |

| Metrik Agregat (KNN) | presisi | recall | f1-score | support |
|----------------------|---------|--------|----------|---------|
| accuracy             |         |        | 0.98     | 406     |
| macro avg            | 0.98    | 0.98   | 0.98     | 406     |
| weighted avg         | 0.98    | 0.98   | 0.98     | 406     |

**Hasil proyek :**

Berdasarkan 4 buah metrik evaluasi yaitu Accuracy, Precision, Recall, and F1-Score model Random Forest Classifier menjadi yang terbaik jika dibandingkan dengan model K-Nearest Neighbors Classifier. Hal tersebut karena model menunjukkan kinerja yang secara signifikan lebih baik, bahkan mendekati sempurna. Model Random Forest Classifier menunjukan metrik evaluasi Precision, Recall, and F1-Score yang lebih seimbang pada setiap kelas label jika dibandingkan dengan model K-Nearest Neighbors (KNN).

Model Machine Learning dapat dijadikan solusi dalam penerapan Pertanian Presisi (*Precision Agriculture*) secara efektif untuk menyelesaikan permasalahan petani yaitu pemilihan jenis tanaman yang tidak sesuai dengan karakteristik lahan pertanian. Pada proyek ini, secara umum Model Machine Learning Random Forest Classifier dan K-Nearest Neighbors Classifier dapat mengklasifikasikan data karakteristik lahan pertanian seperti kandungan unsur hara, suhu, kelembaban, PH, dan curah hujan untuk menghasilkan rekomendasi jenis tanaman yang cocok ditanam dengan Accuracy tinggi yaitu secara berurut 1.00 dan 0.98. Model dapat ini dapat membantu pendekatan Pertanian Presisi (*Precision Agriculture*) secara efektif karena cepat dan akurat sehingga dapat meningkatkan pendapatan petani, memperkuat ketahanan pangan, dan mendorong ekonomi berkembang secara maksimal.