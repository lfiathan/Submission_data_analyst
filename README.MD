# Proyek Analisis Data: E-Commerce Public Dataset

## Deskripsi Proyek
Proyek ini merupakan submission analisis data menggunakan **E-Commerce Public Dataset** dengan tujuan memahami performa bisnis e-commerce dari beberapa sudut pandang utama: kontribusi kategori produk, kualitas pengiriman, perilaku pelanggan, dan distribusi geografis pelanggan.

Analisis dilakukan secara end-to-end, dimulai dari pemahaman data mentah, proses data wrangling, exploratory data analysis (EDA), pembuatan visualisasi, hingga penyusunan dashboard interaktif menggunakan **Streamlit**. Karena dataset tersusun dalam beberapa tabel yang saling berelasi, salah satu fokus utama proyek ini adalah membangun dataset analitis yang lebih siap pakai agar proses analisis menjadi lebih konsisten dan mudah diinterpretasikan.

Selain menjawab dua pertanyaan bisnis utama, proyek ini juga dilengkapi dengan analisis lanjutan untuk memberikan gambaran yang lebih kaya mengenai perilaku pelanggan dan distribusi pasar. Analisis lanjutan tersebut mencakup:
- **RFM Analysis**
- **Customer Clustering (Manual Grouping / Binning)**
- **Geographic Analysis**

Melalui proyek ini, analisis tidak hanya berhenti pada deskripsi angka penjualan, tetapi juga berupaya menerjemahkan data menjadi insight yang relevan untuk kebutuhan bisnis, operasional, dan strategi pemasaran.

---

## Pertanyaan Bisnis
1. **Kategori produk apa yang memberikan kontribusi terbesar terhadap pendapatan dan jumlah pesanan?**
2. **Bagaimana performa pengiriman memengaruhi tingkat kepuasan pelanggan?**

Kedua pertanyaan tersebut dipilih karena mewakili dua area penting dalam bisnis e-commerce. Pertanyaan pertama berhubungan dengan performa komersial, yaitu kategori produk mana yang benar-benar menjadi penggerak revenue dan volume transaksi. Pertanyaan kedua berhubungan dengan pengalaman pelanggan, khususnya bagaimana kualitas pengiriman berkaitan dengan review yang diberikan pelanggan setelah transaksi selesai.

---

## Insight Utama
Berdasarkan hasil analisis pada notebook dan dashboard, terdapat beberapa temuan yang menonjol.

Kategori produk tidak memberikan kontribusi yang merata terhadap performa bisnis. Sejumlah kategori tertentu menjadi penyumbang utama pendapatan dan jumlah pesanan, sehingga kategori-kategori tersebut dapat dipandang sebagai pendorong utama performa e-commerce. Temuan ini penting karena dapat membantu menentukan prioritas promosi, pengelolaan stok, dan strategi pengembangan produk.

Analisis juga menunjukkan adanya hubungan yang cukup jelas antara performa pengiriman dan tingkat kepuasan pelanggan. Pesanan yang tiba lebih cepat atau tepat waktu cenderung memperoleh review score yang lebih baik. Sebaliknya, keterlambatan pengiriman berasosiasi dengan penurunan skor ulasan. Hal ini menunjukkan bahwa kualitas logistik memiliki pengaruh langsung terhadap pengalaman pelanggan, dan bukan sekadar aspek operasional di belakang layar.

Dari sisi pelanggan, analisis RFM menunjukkan bahwa pelanggan tidak bersifat homogen. Terdapat kelompok pelanggan bernilai tinggi yang perlu dipertahankan, pelanggan loyal yang layak diprioritaskan dalam program retensi, pelanggan baru yang memiliki potensi berkembang, serta pelanggan yang berisiko menurun aktivitasnya. Dengan segmentasi seperti ini, strategi bisnis dapat menjadi lebih terarah dan tidak bergantung pada pendekatan yang seragam untuk semua pelanggan.

Pada sisi geografis, distribusi pelanggan dan pendapatan terlihat lebih terkonsentrasi di state tertentu. Temuan ini dapat dimanfaatkan untuk mendukung strategi pemasaran regional, penentuan prioritas distribusi, dan optimasi logistik di wilayah yang memberikan kontribusi terbesar.

---

## Ruang Lingkup Analisis
Proyek ini mencakup beberapa tahapan utama yang saling terhubung.

### 1. Data Gathering
Pada tahap awal, seluruh file dataset dimuat dan diperiksa untuk memahami struktur data yang tersedia. Dataset terdiri dari beberapa tabel yang mewakili entitas berbeda, seperti pelanggan, pesanan, item pesanan, review, pembayaran, produk, seller, serta geolokasi.

Karena data tidak tersedia dalam satu tabel tunggal, proses analisis membutuhkan penggabungan antartabel berdasarkan key yang sesuai, seperti `order_id`, `customer_id`, dan `product_id`.

### 2. Data Assessment
Tahap assessment dilakukan untuk memeriksa kualitas data sebelum digunakan lebih jauh. Beberapa aspek yang diperiksa meliputi:
- struktur kolom dan tipe data,
- missing values,
- duplikasi data,
- serta konsistensi relasi antarfile.

Tahap ini penting karena kualitas hasil analisis sangat bergantung pada kualitas data yang digunakan. Beberapa tabel memiliki nilai kosong pada kolom tertentu yang masih masuk akal secara konteks, misalnya kolom tanggal pengiriman yang tidak terisi untuk pesanan yang belum selesai.

### 3. Data Cleaning
Setelah assessment, dilakukan proses pembersihan data untuk memastikan dataset siap digunakan. Tahapan cleaning meliputi:
- konversi tipe data menjadi format yang sesuai, terutama kolom tanggal,
- penanganan missing values pada kolom yang relevan,
- normalisasi beberapa nama kolom,
- pembersihan data numerik dan geolokasi,
- serta penggabungan tabel menjadi satu dataset analitis utama.

Pada tahap ini juga dilakukan filtering agar analisis utama lebih fokus, khususnya pada pesanan yang relevan untuk evaluasi penjualan dan pengiriman.

### 4. Feature Engineering
Beberapa fitur turunan dibuat untuk memudahkan analisis, antara lain:
- `sales` untuk merepresentasikan nilai transaksi,
- `purchase_month` untuk analisis tren waktu,
- `delivery_time_days` untuk menghitung lama pengiriman,
- `delivery_delay_days` untuk mengukur keterlambatan pengiriman,
- `is_late` untuk menandai apakah pengiriman terlambat,
- `delay_category` untuk mengelompokkan kondisi pengiriman menjadi kategori yang lebih mudah dibaca.

Hasil akhir dari proses ini adalah dataset analitis seperti `main_data.csv` yang digunakan kembali di dashboard Streamlit.

---

## Exploratory Data Analysis (EDA)
Tahap EDA dilakukan untuk memahami pola utama dalam data sebelum membangun visualisasi yang lebih terfokus.

Analisis dimulai dari ringkasan metrik dasar, seperti jumlah order, jumlah pelanggan unik, total revenue, rata-rata review score, dan persentase keterlambatan. Setelah itu, eksplorasi dilanjutkan ke distribusi kategori produk, tren transaksi bulanan, hubungan pengiriman dengan review, serta distribusi revenue berdasarkan state pelanggan.

Melalui EDA, dapat terlihat bahwa performa bisnis tidak tersebar merata. Ada kategori produk yang jauh lebih dominan dibanding kategori lain, ada wilayah yang berkontribusi lebih besar terhadap pendapatan, serta ada pola yang cukup konsisten antara ketepatan pengiriman dan kepuasan pelanggan.

---

## Hasil Analisis Berdasarkan Pertanyaan Bisnis

### 1. Kategori produk dengan kontribusi terbesar terhadap pendapatan dan jumlah pesanan
Analisis kategori produk menunjukkan bahwa beberapa kategori memiliki kontribusi yang jauh lebih besar dibanding kategori lain, baik dari sisi pendapatan maupun jumlah order. Hal ini menandakan bahwa tidak semua kategori memiliki peran yang sama dalam mendorong performa bisnis.

Dari perspektif bisnis, kategori dengan kontribusi terbesar dapat dijadikan prioritas dalam:
- promosi dan campaign,
- pengelolaan inventory,
- penentuan produk unggulan,
- dan strategi cross-selling.

Agar analisis tidak hanya terpaku pada nominal revenue, perbandingan juga dilakukan dari sisi jumlah pesanan. Dengan begitu, interpretasi menjadi lebih lengkap: ada kategori yang kuat karena nilai per transaksinya tinggi, dan ada pula kategori yang kuat karena volumenya besar.

### 2. Pengaruh performa pengiriman terhadap kepuasan pelanggan
Analisis menunjukkan bahwa performa pengiriman berkaitan erat dengan review score pelanggan. Pengiriman yang lebih cepat atau tepat waktu cenderung menghasilkan skor review yang lebih tinggi, sedangkan pengiriman yang terlambat cenderung diikuti penurunan kepuasan pelanggan.

Temuan ini penting karena menunjukkan bahwa pengalaman pelanggan tidak hanya dibentuk oleh produk yang dibeli, tetapi juga oleh kualitas pemenuhan pesanan. Dari sisi bisnis, ini berarti peningkatan kualitas logistik dapat memberikan manfaat ganda: memperbaiki pengalaman pelanggan sekaligus menjaga persepsi terhadap layanan.

---

## Analisis Lanjutan

### RFM Analysis
RFM Analysis digunakan untuk mengevaluasi pelanggan berdasarkan tiga dimensi utama:
- **Recency**, yaitu seberapa baru pelanggan melakukan pembelian,
- **Frequency**, yaitu seberapa sering pelanggan melakukan transaksi,
- **Monetary**, yaitu seberapa besar nilai transaksi yang dihasilkan pelanggan.

Berdasarkan kombinasi skor tersebut, pelanggan kemudian dikelompokkan ke dalam beberapa segmen seperti:
- **Best Customers**
- **Loyal Customers**
- **Recent Customers**
- **At Risk**
- **Others**

Pendekatan ini membantu memetakan pelanggan berdasarkan nilai dan perilaku mereka. Dengan segmentasi seperti ini, strategi retensi dan pemasaran dapat dibuat lebih tepat sasaran. Pelanggan bernilai tinggi dapat diprioritaskan untuk program loyalitas, sementara pelanggan berisiko dapat menjadi target kampanye reaktivasi.

### Customer Clustering (Manual Grouping / Binning)
Selain RFM, proyek ini juga menggunakan pendekatan manual grouping atau binning untuk mengelompokkan pelanggan berdasarkan nilai transaksi. Pendekatan ini memang lebih sederhana dibanding clustering berbasis algoritma, tetapi tetap bermanfaat untuk membedakan pelanggan berdasarkan kontribusi nilai secara praktis dan mudah diinterpretasikan.

Pengelompokan ini membantu menjawab pertanyaan sederhana namun penting: pelanggan mana yang benar-benar memberikan kontribusi tinggi terhadap bisnis, dan kelompok mana yang masih memiliki potensi untuk ditingkatkan.

### Geographic Analysis
Analisis geografis dilakukan untuk melihat bagaimana distribusi pelanggan tersebar di berbagai wilayah. Dengan bantuan data geolokasi dan state pelanggan, analisis ini memberikan konteks regional terhadap performa penjualan.

Distribusi ini penting karena dapat dimanfaatkan untuk:
- menentukan prioritas pemasaran regional,
- memahami konsentrasi pasar,
- serta membantu optimasi logistik di wilayah yang berkontribusi paling besar.

Dashboard juga mendukung tampilan geografis berbasis titik koordinat pelanggan maupun fallback centroid per state jika data granular tidak tersedia secara lengkap.

---

## Dashboard Interaktif
Sebagai pelengkap notebook, proyek ini juga menyertakan dashboard interaktif berbasis **Streamlit**. Dashboard dirancang untuk menyajikan hasil analisis dalam bentuk yang lebih mudah dipahami dan dieksplorasi.

Dashboard dibagi ke dalam tiga bagian utama:
- **Business Performance**, yang menampilkan metrik utama, tren revenue, dan kategori produk unggulan,
- **Customer Intelligence**, yang menampilkan hubungan antara keterlambatan pengiriman dan review, segmentasi RFM, serta pengelompokan pelanggan,
- **Geographic Distribution**, yang menampilkan distribusi pendapatan berdasarkan state dan persebaran pelanggan secara geografis.

Dashboard juga dilengkapi dengan filter berdasarkan:
- state pelanggan,
- kategori produk,
- dan bulan transaksi.

Filter ini memungkinkan pengguna melihat perubahan KPI dan visualisasi berdasarkan subset data tertentu, sehingga dashboard menjadi lebih fleksibel untuk eksplorasi.

---

## Struktur Direktori
```text
submission/
├── dashboard/
│   └── dashboard.py
├── data/
│   ├── customers_dataset.csv
│   ├── geolocation_dataset.csv
│   ├── main_data.csv
│   ├── order_items_dataset.csv
│   ├── order_payments_dataset.csv
│   ├── order_reviews_dataset.csv
│   ├── orders_dataset.csv
│   ├── product_category_name_translation.csv
│   ├── products_dataset.csv
│   └── sellers_dataset.csv
├── notebook.ipynb
├── README.md
├── requirements.txt
└── url.txt
```

## Cara Menjalankan Dashboard

### 1. Install Dependencies
Pastikan Python telah terpasang, kemudian install seluruh dependency:

```bash
pip install -r requirements.txt
```
### 2. Jalankan Aplikasi Streamlit

Jalankan dashboard dengan perintah berikut:
```bash
streamlit run dashboard/dashboard.py
```
Dashboard akan terbuka secara otomatis melalui browser.