# Proyek Machine Learning - Safiira Rahmah Linisa

## Project Overview

Membaca adalah salah satu keterampilan yang penting bagi manusia. Membaca memiliki banyak manfaat diantaranya memperluas wawasan, menambah pengetahuan, meningkatkan daya ingat dan kecerdasan otak. Salah satu bahan bacaan yang banyak digunakan adalah buku. "Buku adalah jendela dunia" mungkin kita sering mendengar slogan tersebut. Hal ini wajar karena begitu banyak manfaat yang bisa kita dapatkan dengan membaca buku. Buku merupakan sumber ilmu. Oleh karenanya membaca buku sangat penting dan sangat dibutuhkan.

Namun minat baca di Indonesia masih sangat rendah.Hal ini tentunya sangat disayangkan dam perlu menjadi perhatian penting. Kendala yang muncul bisa dari tidak terbiasa dan rasa malas untuk membaca.Ada pula pembaca yang hanya menyukai jenis jenis buku tertentu saja. Sedangkan Jenis buku yang tersedia sangat banyak. Menemukan buku yang sesuai dengan keinginan pembaca juga menjadi kendala. Karena terlalu banyak jenis buku yang ada juga menyulitkan pembaca dalam memilah buku yang diinginkan dan dapat menambah rasa malas saat harus menghabiskan waktu untuk mencari buku bacaan yang diinginkan.

Disisi lain kemajuan teknologi yang sangat pesat mendorong banyak perubahan besar. Terjadi banyak pergeseran aktivitas ke media internet. Ditambah dengan kondisi pandemi kemarin mengharuskan kita beraktivitas, belajar serta bekerja dari rumah. Hal ini menimbulkan tantangan tetapi disisi lain juga membuka peluang baru. Salah satunya dalam hal kemudahan.

Dengan melihat peluang dan permasalahan yang ada, saya berinisiatif membuat model machine learning yang dapat merekomendasikan buku-buku yang serupa dengan buku yang sebelumnya pernah dibaca dan disukai oleh pembaca. Tujuannya agar pembaca akan lebih mudah dalam mencari buku bacaan yang sesuai dengan yang disukainya. Dengan sistem rekomendasi ini diharapkan lebih memudahkan pembaca menemukan buku bacaan yang sesuai dan turut mendorong perkembangan minat baca terutama Di Indonesia.

## Business Understanding

### Problem Statements

- Menggunakan data yang ada, bagaimana cara membuat sistem untuk merekomendasikan buku buku lain yang serupa dengan buku yang pernah dibaca pembaca yang mungkin juga disukai oleh pembaca?

### Goals

- Membuat model machine learning yang mampu menghasilkan rekomendasi untuk pembaca dalam menemukan buku - buku yang serupa dengan buku yang disukai pembaca di masa lalu. Menggunakan pendekatan algoritma _Content Based Filtering_.

### Solution approach

- Solusi yang diterapkan untuk dalam proyek ini adalah dengan menerapkan ALgoritma yaitu _Content Based Filtering_ dengan ketentuan sebagai berikut :
  _Content Based Filtering_ dalam pengembangan model machine learning ini digunakan untuk merekomendasikan buku-buku berdasarkan buku yang mirip dengan buku yang disukai pembaca di masa lalu. Dengan kata lain Algoritma ini akan membuat rekomendasi berdasarkan aktivitas yang telah dilakukan pengguna, Sehingga semakin banyak interaksi yang dilakukan oleh pengguna akan semakin baik.

## Data Understanding

Tabel 1. Informasi dataset

| Informasi    | Keterangan                                                                              |
| ------------ | --------------------------------------------------------------------------------------- |
| Link         | https://www.kaggle.com/datasets/justinnguyen0x0x/best-books-of-the-21st-century-dataset |
| Lisensi      | CC0: Public Domain                                                                      |
| Nama Dataset | Best Books of The 21st Century Dataset                                                  |
| Usability    | 10.00                                                                                   |
| Jumlah Data  | 10018                                                                                   |
| Jumlah Kolom | 14                                                                                      |

Dataset yang saya gunakan merupakan Dataset Best Books of The 21st Century Dataset. Dataset ini berisi lebih dari 9000 buku berbeda dengan berbagai genre, diambil dari daftar Buku Terbaik abad ke-21 di goodreads.
Berikut adalah isi atau variabel-variable yang terdapat dalam dataset:

- id: id buku
- title: judul buku
- series: seri buku. Jika buku tersebut bukan milik seri mana pun, nilainya akan menjadi null.
- author: penulis buku
- book_link: URL buku di GoodReads
- genre: genre buku (diurutkan berdasarkan jumlah suara genre)
- date_published: tanggal publikasi
- publisher: penerbit buku
- num_of_page: jumlah halaman
- lang: bahasa pada buku
- review_count: jumlah review/ulasan
- rating_count: jumlah ratings
- rate: rating


Mengecek informasi pada data

Tabel 2. Informasi pada Data

| #   | Column         | Non-Null Count | Dtype   |
| --- | -------------- | -------------- | ------- |
| 0   | id             | 18249 non-null | int64   |
| 1   | title          | 18249 non-null | object  |
| 2   | series         | 18249 non-null | object  |
| 3   | author         | 18249 non-null | object  |
| 4   | book_link      | 18249 non-null | object  |
| 5   | genre          | 18249 non-null | object  |
| 6   | date_published | 18249 non-null | object  |
| 7   | publisher      | 18249 non-null | object  |
| 8   | num_of_page    | 18249 non-null | float64 |
| 9   | lang           | 18249 non-null | object  |
| 10  | review_count   | 18249 non-null | object  |
| 11  | rating_count   | 18249 non-null | object  |
| 12  | rate           | 18249 non-null | float64 |
| 13  | award          | 18249 non-null | object  |

dtypes: float64(2), int64(1), object(11)
memory usage: 1.1+ MB

- Dataset terdiri:
  - 2 variabel bertipe data float
  - 1 variabel bertipe data integer
  - 11 variable bertipe data object



## Data Preparation

### Mengatasi Missing Value

Tujuan dari proses _missing value_ yaitu agar dataset bersih dari fitur yang tidak dibutuhkan, maupun yang valuenya kosong. Hl ini karena fitur dengan value kosong dapat menimbulkan hasil akurasi yang kurang baik ataupun hasil. Pengecekan missing value ini dengan menggunakan fitur '.isnull().sum()'. Pada proyek ini pertama dilakukan pengecekan _missing value_ pada dataset karena terlalu banyak data yang memiliki _missing value_ kemudian dilakukan penghapusan fitur yang tidak digunakan menggunakan fungsi drop. setelah itu kembali dilakukan pengecekan pada data dan karena masih ditemukan banyak _missing value_ digunakan fungsi dropna() untuk membersihkan data yang masih mengandung _missing value_.

### Menghapus Data Duplikat

Pada data sangat memungkinkan terdapat judul dari buku yang sama yang muncul berulang-ulang. Hal tersebut akan mengganggu prose modeling karena dapat membuat model tidak berjalan dengan baik. Oleh karenanya data yang duplikat atau muncul berulang seperti ini harus dihapus. Penghapusan data duplikat pada proyek ini yaitu menggunakan fungsi drop_duplicates()pada fitur title.

### Membuat Dictionary Data

Dictionary dibuat dengan tujuan agar data yang dihasilkan hanya memprediksi hasil dari fitur fitur yang telah ambil untuk digunakan sebagai fitur untuk melakukan proses rekomendasi. Pada proyek ini saya menggunakan fitur 'id', 'title', 'author', 'genre', 'review_count', dan 'rate'.



## Modeling

Dalam tahap Development model pada proyek ini, Algoritma machine learning yang saya gunakan sebagai solusi yaitu dengan pendekatan _content based filtering_ menggunakan _TfidfVectorizer_.
_Content Based Filtering_ merupakan algoritma yang akan membuat sistem rekomendasi berdasarkan item yang disukai pengguna di masa lalu.

- Kelebihan
  - Tidak memerlukan data user lain
  - Dapat merekomendasikan item khusus

- Kelemahan
  - Tidak mampu menetukan profil pengguna baru
  - Hanya digunakan untuk fitur yang sesuai.

Berikut adalah proses dalam tahap development model pada kasus rekomendasi buku :
* _TF-IDF Vectorizer_
  TF-IDF atau _term Frequency-Inverse Document Frequency_ digunakan untuk mengukur frekuensi seberapa sering suatu kata atau term muncul dalam teks tertentu. Proyek ini menggunakan fungsi TfidfVectorizer() dari _library sklearn_. fungsi tersebut diinisialisasi terlebih dahulu, kemudian dilakukan perhitungan idf pada data genre. kemudian dilanjutkan dengan mapping array dari fitur index ke fitur nama.
  Setelah itu fit dan transformasikan data kedalam bentuk matriks. untuk mengubah vektor tf-idf dalam bentuk matriks gunakan fungsi todense().

* _Cosine Similarity_
  _Cosine similarity_ adalah sebuah teknik untuk menghitung derajat kesamaan (_similarity_) antar buku. Dengan menggunakan fungsi cosine_similarity dari _library sklearn_. Output yang dihasilkan berupa matriks kesamaan dalam bentuk array.

* Mendapatkan Rekomendasi
  Setelah mendapatkan data _similarity_ tahap selanjutnya adalah menghasilkan rekomendasi buku untuk pembaca. Disini saya membuat fungsi book_recommendations dengan parameter :
   - title : judul buku (index kemiripan      dataframe).
   - Similarity_data : Dataframe mengenai similarity yang telah kita definisikan sebelumnya.
   - items : Nama dan fitur yang digunakan untuk mendefinisikan kemiripan, dalam hal ini 'title' dan 'genre'.
   - k : Banyak rekomendasi yang ingin diberikan.

 Prose rekomendasi ini akan mencari buku dengan genre yang mirip dengan buku yang sesuai keinginan pembaca.
 Rekomendasi buku yang mirip dengan buku berjudul 'The Great Good Thing (The Sylvie Cycle, #1)' dengan genre 'Fantasy,Fiction,Young Adult,Childrens,Children..' diperoleh hasil top N recommendation sebagai berikut.

 Tabel 3. Hasil Rekomendasi Buku.

 |   | title                                              | genre                                             |
 | - | -------------------------------------------------- | ------------------------------------------------- |
 | 0 | Tilly and the Lost Fairytales (Pages & Co., #2)    | Fantasy,Childrens,Middle Grade,Fiction,Childre... |
 | 1 | Story Thieves (Story Thieves, #1)	                | Fantasy,Childrens,Middle Grade,Adventure,Ficti... |
 | 2 | Steinbeck's Ghost                                  | Young Adult,Fiction,Childrens,Middle Grade,Mys... |
 | 3 | Tilly and the Bookwanderers (Pages & Co. #1)	      | Childrens,Middle Grade,Fantasy,Writing,Books A... |
 | 4 | Inkheart (Inkworld, #1)	                          | Fantasy,Young Adult,Fiction,Childrens,Middle G... |



## Evaluation

Evaluasi untuk sistem rekomendasi dengan pendekatan _content based filtering_ dapat menggunakan matric _precision_. Matric _precision_ yaitu mengukur tingkat ketepatan antara informasi yang diminta dengan jawaban yang diberikan sistem.

$$ precision = {TP \over (TP+FP)} $$

- dimana : 
   - TP = True Positive (data positif yang diprediksi benar)
   - FP = False Positive (data negatif namun diprediksi sebagai data positif)

Sistem menampilakan rekomendasi berdasarkan genre 'Fantasy,Fiction,Young Adult,Childrens,Children..'
TP(0) = 4     TP(1) = 3     TP(2) = 3      TP(3) = 2     TP(4) = 4
FP(0) = 1     FP(1) = 2     FP(2) = 2      FP(3) = 3     FP(4) = 1

Precission : 
* pada indeks ke 0
  precision = 4/4+1 = 0.8

* pada indeks ke 1
  precision = 3/3+2 = 0.6

* pada indeks ke 2
  pprecision = 3/3+2 = 0.6

* pada indeks ke 3
  precision = 2/2+3 = 0.4

* pada indeks ke 4
  precision = 4/4+1 = 0.8

total precision = (0.8 + 0.6 + 0.6 + 0.4 + 0.8)/5 = 0.64 = 64%
karena nilai precision menunjukkan nilai diatas 50% yaitu sebesar 64% maka dapat dikatakan sistem sudah mampu menghasilkan rekomendasi yang sesuai dengan yang diinginkan pengguna.

Berdasarkan data hasil rekomendasi buku dan buku yang dicari dapat dilihat bahwa Jumlah data menunjukan true positif yang sesuai dan hanya sedikit saja mengandung False Positif. Sehingga sistem tergolong memiliki presisi yang cukup baik. Jadi, dapat disimpulkan bahwa sistem sudah berhasil menampilkan rekomendasi yang sesuai.




 
