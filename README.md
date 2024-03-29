# Laporan Proyek Machine Learning - Andre Yulius Sinambela M-01

## Domain Proyek
Ford *Motor Company* (atau hanya Ford atau FoMoCo, NYSE: F) adalah sebuah produsen mobil asal Amerika Serikat yang didirikan oleh Henry Ford di Dearborn, dekat Detroit,Michigan. Perusahaan ini didirikan pada 16 Juni 1903. Seiring  dengan berkembangnya jaman dan semakin meningkatnya kebutuhan alat transportasi membawa peluang bagi perusahaan otomotif roda empat, yang sangat dibutuhkan oleh banyak khalayak publik sebagai sarana transportasi sehari hari yang lebih efisien dan dinamis. Saat ini banyak sekali bermunculan merk mobil dengan berbagai model,desain dengan pilihan kualitas dan harga yang cukup bersaing. Ditengah situasi ekonomi yang belum stabil seperti sekarang ini, membuat para produsen mobil berpikir ulang untuk melakukan inovasi-inovasi agar dapat terus eksi dalam dunia otomotif dengan memproduksi mobil yang sesuai dengan minat konsumen indonesia. 

Masyarakat yang demikian membuat konsumen akan lebih mempertimbangkan untuk membeli mobil yang terjangkau dari sisi harganya namun dengan kapasitas daya angkut penumpang yang lebih besar. Harga mobil dengan budget 100-150 juta untuk mobil baru merupakan range harga yang ideal untuk masyarakat dengan tingkat ekonomi menengah kebawah. Dari sekitar 20 brand mobil baru yang ada di Tanah Air, sekarang tertinggal enam merek yang bisa diperoleh dengan range dana 100-150 juta, misalnya pada merek Toyota, Daihatsu, Suzuki, Kia, Hyundai, dan produsen otomotif baru dari Negara Malaysia yaitu Proton, dan itu hanya ada di beberapa varian dan tipe mobilnya saja, padahal di range harga 100-150 juta bisa dikatakan sebagai potensial buyer di Indonesia. Range harga 100-150 juta tidak hanya sekedar untuk konsumen first entry car (pembelian mobil pertama), tapi juga bagi pembeli mobil kedua, ketiga, dan seterusnya dalam artian range harga 100-150 juta tersebut tidak hanya berlaku pada konsumen yang membeli mobil dengan kondisi baru, namun berlaku juga bagi konsumen yang membeli mobil bekas pakai atau second hand, karena rata-rata mobil berjenis MPV atau mobil niaga banyak digemari para konsumen, khususnya di Indonesia.

## *Business Understanding*

### *Problem Statements*
Berdasarkan kondisi yang telah diuraikan, agen dan perusahaan akan mengembankan sistem prediksi harga rumah untuk menjawab Pertanyaan berikut:
- Apa fitur yang berpengaruh terhadap harga mobil Ford?
- Berapa harga mobil dengan model dan penggunaan bahan bakar tertentu?
- Apakah tahun produksi atau pembelian mobil berpengaruh terhadap harga?

### *Goals*
Berdasarkan pernyataan di atas,saya akan membuat sebuah predictive modelling dengan tujuan sebagai berikut :
- Mengetahui fitur yang berkolerasi dengan harga mobil ford.
- Membuat Model Machine learning untuk memprediksi harga mobil ford dengan akurat.
- Mengetahui tahun produksi untuk setiap mobil serta memilah kenaikan atau penurunan harga setiap tahunannya.

### *Solution Statement*
- Untuk melakukan prediksi harga mobil ford yang ingin dicapai, Maka seperti yang sudah diketahui bahwa harga merupakan variabel kontinu dalam predictive analiytics, saat membuat prediksi variabel kontinu maka kita akan menyelesaikan masalah regresi.

- Untuk mengevaluasi seberapa baik model dalam meprediksi harga untuk kasus regresi biasanya menggunakan beberapa metrik, Beberapa metrik yang biasa digunakan adalah Mean Squared Error(MSE) atau Root Mean Square(RMSE). Metrik ini digunakan untuk mengetahui seberapa jauh hasil prediksi dengan nilai yang sebenarnya.

- Untuk melakukan prediksi harga kita akan menggunakan pengembangan model yang menggunakan algoritma Machine learning yaitu K-Nearest Neighbor, Random Forest, dan Boosting Algorithm, Dari ketiga Model ini akan kita akan memilih salah satu model yang memungkin model tersebut memiliki nilai prediksi terbesar atau dengan kata lain model yang memiliki nilai kesalahan terkecil.

## *Data Understanding*

Dataset yang saya gunakan adalah dataset Ford Car Price, Dataset ini memiliki  17.966 Sampel data dengan berbagai karakteristik harga yang ada. Karakteristik pada mobil ford yang berpengaruh pada harga adalah model, tahun pembuatan (year), transmisi (transmission), jarak tempuh (mileage), jenis bahan bakar (fuelType), pajak (tax), mpg (penggunaan bahan bakar), Ukuran mesin (engineSize). jadi 8 fitur ini akan saya digunakan untuk menemukan pola pada data sedangkan harga merupakan fitur target.

* DataSet dapat Diunduh pada Tautan Berikut
https://www.kaggle.com/datasets/adhurimquku/ford-car-price-prediction

### Variabel-variabel yang terdapat pada dataset ford price adalah sebagai berikut:
  * Model -> Merek/brands mobil Ford
  * Year -> Tahun Produksi mobil
  * Price -> Harga Mobil dalam Dollar Amerika ($)
  * Transmission -> transmisi yang digunakan seperti Automatic,Manual,dan Semi-Auto
  * Mileage -> Jarak tempuh mobil
  * Fuel Type -> Jenis bahan bakar yang digunakan seperti Petrol, Diesel, Hybrid,Electric dan lainnya
  * Tax -> Pajak tahunan sebuah mobil
  * Mpg -> Konsumsi bahan bakar yang digunakan 
  * Engine Size -> Ukuran Mesin pada Mobil

**Tahapan yang dilakukan untuk memahami data**:

Ada beberapa Tahapan Yang di lakukan untuk memahami dataset dengan Exploratory Data Analysis antara lain :
* Data Loading
  Data loading merupakan proses untuk membaca data set agar kita bisa memahami dataset dan mengetahui seluruh jumlah dataset.
  Terdapat 17.966 baris dalam dataset dan terdapat 9 kolom yaitu : model, year, price, transmission, mileage, fuelType, tax, mpg, engineSize .

* Analisis Data (Mendeskripsikan Variabel)
  Data analisis merupakan proses menganalisa karakteristik, menemukan pola, anomali dan memeriksa asumsi pada data, Yaitu untuk mengetahui beberapa hal berikut : 
    * Variabel pada data
      Untuk mengetahui variabel apa saja yang ada pada data bisa dengan menuliskan kode nama_variabel.info() maka akan tampil variabel beserta type datanya
    * Terdapat 4 kolom dengan tipe data int64 yaitu kolom year, price, mileage, dan tax .
    * Terdapat 3 kolom dengan tipe data object yaitu kolom model, transmission dan fuelType.
    * Kemudian Terdapat 2 kolom dengan tipe data float64 yaitu kolom mpg dan engineSize .
    Terdapat 17.966 baris data pada dataset yang digunakan terdiri dari 9 kolom .

|index|model|year|price|transmission|mileage|fuelType|tax|mpg|engineSize|
|---|---|---|---|---|---|---|---|---|---|
|0| Fiesta|2017|12000|Automatic|15944|Petrol|150|57\.7|1\.0|
|1| Focus|2018|14000|Manual|9083|Petrol|150|57\.7|1\.0|
|2| Focus|2017|13000|Manual|12456|Petrol|150|57\.7|1\.0|
|3| Fiesta|2019|17500|Manual|10460|Petrol|145|40\.3|1\.5|
|4| Fiesta|2019|16500|Automatic|1482|Petrol|145|48\.7|1\.0|

* Bagaimana distribusi variabel dalam dataset
Untuk mengetahui distribusi variabel kita perlu menggunakan baris kode nama_variabel.describe() maka akan menampilkan informasi statistik pada masing-masing kolom, Seperti : 

|index|year|price|mileage|tax|mpg|engineSize|
|---|---|---|---|---|---|---|
|count|17915\.0|17915\.0|17915\.0|17915\.0|17915\.0|17915\.0|
|mean|2016\.8651967624896|12280\.966117778398|23373\.34641361987|113\.34200390734021|57\.90984649734859|1\.3546525258163549|
|std|2\.051495289522369|4743\.706538648912|19476\.44928760471|62\.016557887097015|10\.127366082501418|0\.42692403875290175|
|min|1996\.0|495\.0|1\.0|0\.0|20\.8|1\.0|
|25%|2016\.0|8999\.0|9987\.5|30\.0|52\.3|1\.0|
|50%|2017\.0|11291\.0|18250\.0|145\.0|58\.9|1\.2|
|75%|2018\.0|15299\.0|31083\.0|145\.0|65\.7|1\.5|
|max|2060\.0|54995\.0|177644\.0|580\.0|201\.8|5\.0|

      * Count  adalah jumlah sampel pada data.
      * Mean adalah nilai rata-rata.
      * Std adalah standar deviasi.
      * Min yaitu nilai minimum setiap kolom. 
      * 25% adalah kuartil pertama. Kuartil adalah nilai yang menandai batas               interval dalam empat bagian sebaran yang sama. 
      * 50% adalah kuartil kedua, atau biasa juga disebut median (nilai tengah).
      * 75% adalah kuartil ketiga.
      * Max adalah nilai maksimum.

* Apakah ada missing value
Dari hasil fungsi describe, Nilai standar deviasi untuk kolom tax dan engineSize memiliki nilai 0,jadi di kolom ini terdapat data yang tidak valid atau di sebut dengan missing value.

                nilai 0 kolom model  0
                nilai 0 kolom transmission 0
                nilai 0 kolom fuelType 0
                nilai 0 kolom tax 2153
                nilai 0 kolom engine Size 51    

* Apakah ada fitur yang tidak berguna
Tidak ada fitur yang tidak berguna atau tidak digunakan dalam dataset semua fitur memiliki kolerasi yang cukup kuat dan semua fitur kategori mempengaruhi harga dengan cukup kuat. 
    
* Menangani Missing Value
Setiap teknik tentu yang akan digunakan memiliki kelebihan dan kekurangan, Selain itu, penanganan missing value juga bersifat unik tergantung kasusnya. Pada kasus kita yang saya angkat terdapat 2203 sampel missing value merupakan jumlah yang kecil jika dibandingkan dengan jumlah total sampel yaitu 17.966. apabila sampel ini dihapus, maka tentu kita akan kehilangan beberapa informasi. Akan tetapi, ini tidak akan jadi masalah sebab kita masih memiliki 15.769 sampel lainnya. Oleh karena itu, mising value ini bisa dihapus saja.

* Menangani Outliers
   pengamatan dalam satu set data kadang berada di luar lingkungan pengamatan lainnya, Pengamatan seperti itu disebut outlier.
   Ada beberapa teknik untuk menangani outliers, antara lain:
  * Hypothesis Testing
  * Z-score method
  * IQR Method
  Pada kasus ini, kita akan mendeteksi outliers dengan teknik visualisasi data (boxplot), Kemudian untuk menangani outliers kita akan menggunakan teknik IQR method. IQR adalah singkatan dari Inter Quartile Range, yaitu Kuartil dari suatu populasi adalah tiga nilai yang membagi distribusi data menjadi empat sebaran. Seperempat dari data berada di bawah kuartil pertama (Q1), setengah dari data berada di bawah kuartil kedua (Q2), dan tiga perempat dari data berada di kuartil ketiga (Q3). Dengan demikian interquartile range atau IQR = Q3 - Q1.

Pada kasus ini terdapat satu kolom yang mempunyai data data oulier yaitu kolom mileage. Untuk menanggani hal tersebut saya menngunakan metode IQR untuk membersihkan tabel/dataset dari outlier.


![outlier](https://user-images.githubusercontent.com/64059031/202162425-c68a133c-266b-43a6-b0da-4b8ebf4bd0ff.png)


Setelah melakukan pembersihan outliers maka output yang dihasilkan adalah :
  * DataSet sudah bersih dari Outliers dan memiliki 8.401 sampel
  
 ### EDA (*Exploratory Data Analysis*)
*Univariate Analysis*

Melakukan proses analisis data dengan teknik Univariate EDA, yang pertama kita perlu membagi 2 fitur, antara fitur numerik dan fitur category . Hasil dari fitur category sebagai berikut : 

- Model mobil paling banyak dalam dataset yaitu Fiesta dengan jumlah 3.122.

![model](https://user-images.githubusercontent.com/64059031/202242223-7eb93159-0f73-4627-baae-9e8ef67feff9.jpg)

- Transmisi mobil yang paling banyak adalah Manual.

![trans](https://user-images.githubusercontent.com/64059031/202242369-cef940a9-0abb-4cdd-ac49-2a232e48c308.jpg)


- Jenis bahan bakar yang paling banyak digunakan adalah Petrol.

![fuel](https://user-images.githubusercontent.com/64059031/202242428-f2c47919-661c-4052-bcff-75eb7a9f9195.jpg)


Hasil dari fitur numerik sebagai berikut :

![numerical](https://user-images.githubusercontent.com/64059031/202242980-8ee7a2fa-91a1-47f2-a308-588197696d2f.jpg)

- Pada kolom tahun yang paling besar adalah Tahun 2018.
- Semakin tinggi mileage semakin rendah jumlah nya.
- Engine size 1.0 jumlah nya paling banyak. 
  
 Selanjutnya Melakukan Analisis pada fitur Numerik,karena Harga merupakan fitur target (label) pada data, dari histogram harga kita memperoleh beberapa informasi dari data fitur numerikal yaitu:

 - Rentang harga mobil cukup stabil tidak terlalu tinggi dan telalu rendah
 - distribusi harga agak berada pada tengah yang berati cukup stabil
 
*Multivariate Analysis*

Multivariate EDA menunjukkan hubungan antara dua atau lebih variabel pada data. Multivariate EDA yang menunjukkan hubungan antara dua variabel biasa disebut sebagai bivariate EDA. Selanjutnya, kita akan melakukan analisis data pada fitur kategori dan numerik.
Hasil Plot dari Multivariate EDA :

* categorical

![multi_cat](https://user-images.githubusercontent.com/64059031/202241748-1425b23d-f06a-4a9d-9453-2813d19c10a5.jpg)

![multi_cat2](https://user-images.githubusercontent.com/64059031/202241799-f818572a-82d6-4d59-9d8c-7ae968fcf35d.jpg)

* Numerik

![multi_num](https://user-images.githubusercontent.com/64059031/202242111-3b3306ef-25fe-492a-9403-0951c6a3b17a.jpg)

* Bagaimana korelasi antar fitur

Koefisien korelasi berkisar antara -1 dan +1. Ia mengukur kekuatan hubungan antara dua variabel serta arahnya (positif atau negatif). Mengenai kekuatan hubungan antar variabel, semakin dekat nilainya ke 1 atau -1, korelasinya semakin kuat. Sedangkan, semakin dekat nilainya ke 0, korelasinya semakin lemah,
fitur ‘mpg’, ‘mileage', dan ‘year’ memiliki skor korelasi yang cukup lemah (rata-rata di 0.3-0.48) dengan fitur target ‘harga’.

Gambar tabel korelasi

![korelasi](https://user-images.githubusercontent.com/64059031/202210493-3fc7dc8c-806a-44c3-8cef-f18319128c61.jpg)



## *Data Preparation*
Proses data preparation merupakan suatu tahapan yang cukup penting dalam proses pengembangan model machine learning, Pada bagian ini kita akan melakukan empat tahap persiapan data, yaitu:

* Encoding fitur kategori.
Encoding merupakan proses merubah tipe data category menjadi tipe numerik.
Fitur yang mengalami proses encoding yaitu fitur model, year, transmission dan fuel type.
  
* Reduksi dimensi dengan *Principal Component Analysis* (PCA)
Teknik reduksi (pengurangan) dimensi adalah prosedur yang mengurangi jumlah fitur dengan tetap mempertahankan informasi pada data. Teknik pengurangan dimensi yang paling populer adalah Principal Component Analysis atau disingkat menjadi PCA. Ia adalah teknik untuk mereduksi dimensi, mengekstraksi fitur, dan mentransformasikan data dari “n-dimensional space” ke dalam sistem berkoordinat baru dengan dimensi m, di mana m lebih kecil dari n.
  
- Parameter n-components merupakan jumlah komponen atau dimensi, dalam kasus kita jumlahnya ada 3, yaitu 'mileage', 'mpg', dan 'enginesize', Sedangkan parameter random-state berfungsi untuk mengontrol random number generator yang digunakan, Parameter ini berupa sebuah bilangan integer(angka) dan nilainya bebas. Pada kasus ini,kita menerapkan random_state = 123, Berapa pun nilai integer yang kita tentukan selama itu bilangan integer, ia akan memberikan hasil yang sama setiap kali dilakukan pemanggilan fungsi (dalam kasus kita, class PCA).
  
- Selanjutnya kita membuat itur baru bernama (dimensi) dan hanya mempertahankan PCA (komponen) pertama saja. PCA pertama ini akan menjadi fitur dimensi atau ukuran sebuah mobil ford menggantikan tiga fitur lainnya ('mileage','mpg' dan 'enginesize'),Kita beri nama fitur ini 'dimension',Parameter yang digunakan n-component = 1, karena kali ini jumlah komponen kita hanya satu,random-state = 123, Fit model dengan data masukan,Tambahkan fitur baru ke dataset dengan nama 'dimension' dan lakukan proses transformasi,Drop kolom ‘mileage’,'mpg' dan ‘enginesize’. 

* Pembagian dataset dengan fungsi train_test_split dari library sklearn.
 Membagi dataset menjadi data latih (train) dan data uji (test) merupakan hal yang harus kita lakukan sebelum membuat model, Kita perlu mempertahankan sebagian data yang ada untuk menguji seberapa baik generalisasi model terhadap data baru, Misalnya kita memiliki  100 ribu sampel. Dengan proporsi pembagian 80:20 atau 80% data training dan 20% data testing, Dalam kasus proses pengujian ini sebenarnya kita cukup menggunakan 1-2% data atau sebanyak 100.000 hingga 200.000 sampel saja.
 
* Standarisasi.
Standardisasi adalah teknik transformasi yang paling umum digunakan dalam tahap persiapan pemodelan, Untuk fitur numerik, kita tidak akan melakukan transformasi dengan one-hot-encoding,StandardScaler melakukan proses standarisasi fitur dengan mengurangkan mean (nilai rata-rata) kemudian membaginya dengan standar deviasi untuk menggeser distribusi,  StandardScaler menghasilkan distribusi dengan standar deviasi sama dengan 1 dan mean sama dengan 0, Sekitar 68% dari nilai akan berada di antara -1 dan 1, Fitur Standarisasi biasanya digunakan untuk menghindasi kebocoran data pada data uji  proses standarisasi mengubah nilai rata-rata (mean) menjadi 0 dan nilai standar deviasi menjadi 1.
  

## *Modeling*
* Model random forest adalah salah satu Algoritma supervised learning karena            algoritma ini disusun dari banyak algoritma pohon(decision tree) yang pembagian data dan fiturnya dipilih secara acak.
    * Parameter yang digunakan :
      - n-estimator: jumlah trees (pohon) di forest.Di sini kita setn_estimator=50.
      - max_depth: kedalaman atau panjang pohon.Ia merupakan ukuran seberapa banyak pohon dapat membelah (splitting) untuk membagi setiap node ke dalam jumlah                  pengamatan yang diinginkan.
      - random_state: digunakan untuk mengontrol random number generator yang digunakan. 
      - n-jobs: jumlah job (pekerjaan) yang digunakan secara paralel. Ia merupakan komponen untuk mengontrol thread atau proses yang berjalan secara paralel. n_jobs=-1 artinya semua proses berjalan secara paralel.
      
* Model KNN(K-Nearest Neighbor) Meroakan algoritma yang sederhana dibandigkan dengan algoritma lain, Algoritma KNN menggunakan kesamaan Fitur. untuk memprediksi nilai dari setiap data yang baru, seperti setiap data baru diberikan nilai berdasarkan seberapa mirip titik tersebut dalam set tiap pelatihan.
    * parameter yang digunakan :
      - n-neighbors : tetangga dan metric Euclidean untuk mengukur jarak antara titik, n-neighbors = 10.
       
* Boosting Algorithm merupakan algoritma yang menggunakan teknik boosting bekerja dengan membangun model dari data latih, Kemudian membuat model kedua yang               bertugas untuk memperbaiki kesalahan dari model pertama.
    * Parameter yang digunakan :
      - learning_rate: bobot yang diterapkan pada setiap regressor di masing-masing proses iterasi boosting.
      - random_state: digunakan untuk mengontrol random number generator yang digunakan.
     
**Kelebihan dan Kekurangan Algoritma Random Forest, KNN, dan Boosting algorithm : ** 

***Random Forest***
  - Kelebihan dari Model Random Forest, dapat digunakan untuk menyelesaikan masalah klasifikasi dan regresi, Random forest juga merupakan algoritma yang sering             digunakan karena cukup sederhana tetapi memiliki stabilitas yang mumpuni.
  - Kekurangan algoritma random forest adalah pembelajaran bisa berjalan lambat, tergantung pada parameter yang digunakan dan tidak bisa memperbaiki model yang      dihasilkan secara berulang.
  
***KNN(K-Nearest Neighbors)***
  - Kelebihan Algoritma KNN yaitu kuat dalam mentraining data yang noisy, Algoritma ini juga sangat efektif jika datanya besar serta mudah diImplementasikan
  - Kekurangan Algoritma KNN perlu menentukan nilai parameter K,serta sangat sensitif pada data pencilan dan rentan pada variabel yang non-informatif

***Mengapa kita menggunakan Model KNN dan Boosting Agorithm***
Untuk dapat memilih atau melakukan penyeleksian terhadap beberapa algoritma, saya menggunakan ketiga algoritma tersebut sebagai acuan untuk proses seleksi memilih algoritma model yang terbaik yang dapat digunakan pada data set yang car price.

## Evaluation
Mengevaluasi model regresi secara/umum semua metrik adalah sama,jika prediksi yang dilakukan sesuai atau mendekati nilai sebenarnya,maka itu adalah metrik dengan performa terbaik. Secara teknis,selisih antara nilai sebenarnya yang di prediksi adalah nilai eror, Maka semua metrik mengukur seberapa kecil nilai eror tersebut.

Hasil dari train dan test MSE sebagai berikut :
|index|train|test|
|---|---|---|
|KNN|2743\.6632502187495|4033\.1179996787628|
|RF|1546\.6100186673173|3798\.795424871027|
|Boosting|5866\.5807833042045|6126\.817427970641|
* plot bar evaluasi MSE *

![download1](https://user-images.githubusercontent.com/64059031/202170171-898fece2-31d8-48a0-80c7-d04ddd06554d.png)


**MSE Atau Mean Squared Error**

* Metrik MSE Atau Mean Squared Error yang menghitung jumlah selisih rata-rata nilai  sebenarnya dengan nilai prediksi, Sebelum menghitung Nilai MSE dalam model,sebaiknya melakukan proses scalling fitur numerik terlebih dahulu pada data uji baru melakukan proses scalling terhadap data latih untuk menghindari kebocoran data.

|index|y\_true|prediksi\_KNN|prediksi\_RF|prediksi\_Boosting|
|---|---|---|---|---|
|4679|9750|9885\.4|10027\.9|10838\.1|
|773|11498|13020\.9|13075\.4|14326\.7|
|13893|8700|12428\.8|10793\.0|14819\.0|
|2542|16298|15215\.6|15155\.9|14674\.0|
|1612|15400|16984\.8|17023\.9|17312\.6|

## *Conclusion*

Pada kasus ini saya menggunakan metrik MSE Atau Mean Squared Error dan hasil yang di dapatkan yaitu model Random Forest(RF) memberikan nilai eror yang paling kecil sedangkan model dengan Boosting Algorthm Memberikan nilai eror yang cukup besar (berdasarkan grafik angkanya diatas 9000 . Jadi untuk prediksi harga kali ini saya memilih Model Random Forest sebagai model terbaik yang dapat digunakan untuk saat ini.




