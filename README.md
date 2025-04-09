# KIZUNA - Quadtree Image Compressor

## 絆 (Kizuna) - Ikatan Digital

   ```bash
        _/    _/  _/  _/_/_/_/  _/    _/  _/    _/    _/_/    
       _/  _/    _/      _/    _/    _/  _/_/  _/  _/    _/   
      _/_/      _/     _/     _/    _/  _/ _/_/_/  _/_/_/_/   
     _/  _/    _/    _/      _/    _/  _/    _/   _/    _/    
    _/    _/  _/   _/_/_/_/  _/_/_/   _/    _/   _/    _/  
   ```

KIZUNA, yang dalam bahasa Jepang berarti "ikatan" atau "hubungan", adalah program kompresi gambar yang mengimplementasikan algoritma Quadtree dengan pendekatan Divide and Conquer. Program ini menghubungkan pixel-pixel dalam gambar secara cerdas, membentuk struktur hierarkis yang mengoptimalkan penyimpanan data visual.

Dengan metode pembagian rekursif (*yottsu no bunkatsu* - pembagian empat arah), KIZUNA menganalisis area gambar berdasarkan keseragaman warna. Area yang homogen disimpan sebagai satu unit, sementara area dengan variasi warna tinggi dibagi lebih lanjut hingga mencapai threshold yang ditentukan pengguna atau ukuran blok minimum.

Hasil akhirnya adalah representasi gambar yang lebih ringkas namun tetap mempertahankan kualitas visual yang baik hingga menciptakan "ikatan" yang efisien antara efektivitas penyimpanan dan kualitas gambar.

## Struktur Program

Program ini dibagi menjadi beberapa bagian utama:
- `Quadtree.hpp` dan `Quadtree.cpp`: Mendefinisikan struktur data dan algoritma quadtree
- `interface.hpp`: Menyediakan antarmuka pengguna
- `main.cpp`: Titik masuk program, menangani interaksi dengan pengguna

## Spesifikasi Program

Program ini mengimplementasikan:
- Metode pengukuran error: Variance, MAD (Mean Absolute Deviation), Max Pixel Difference, Entropy, dan SSIM (bonus)
- Kompresi gambar berbasis threshold
- Mode persentase kompresi otomatis (bonus)
- Visualisasi proses kompresi dalam format GIF (bonus)

## Requirement Program

Program ini membutuhkan:
- C++ compiler (minimal C++17)
- OpenCV library (untuk pemrosesan gambar)
- CMake (untuk build sistem)
- FFmpeg (opsional, untuk pembuatan visualisasi GIF)

### Instalasi Requirement di Windows

1. **OpenCV**:
   - Download OpenCV dari [situs resmi](https://opencv.org/releases/)
   - Ekstrak ke lokasi yang diinginkan, misalnya `C:\opencv`
   - Tambahkan path bin ke PATH environment variable, misalnya `C:\opencv\build\x64\vc15\bin`

2. **CMake**:
   - Download CMake dari [situs resmi](https://cmake.org/download/)
   - Install dengan mengikuti petunjuk instalasi

3. **FFmpeg** (opsional):
   - Download FFmpeg dari [situs resmi](https://ffmpeg.org/download.html)
   - Ekstrak ke lokasi yang diinginkan
   - Tambahkan path bin ke PATH environment variable

### Instalasi Requirement di Linux/Ubuntu

1. **OpenCV**:
   ```bash
   sudo apt update
   sudo apt install libopencv-dev
   ```
2. **CMake**:
    ```bash
   sudo apt install cmake
   ```
3. **FFmpeg** (gif):
    ```bash
   sudo apt install ffmpeg
   ```

## Cara Kompilasi Program

### Menggunakan CMake (Direkomendasikan)

1. Buat direktori build (jika belum ada):
   ```bash
   mkdir build
   ```
2. Masuk ke direktori build dan build project:
    ```bash
   cd build
   cmake ..
   cmake --build .
   ```
3. **FFmpeg** (gif):
    ```bash
   bin/Release/
   ```

## Cara Menjalankan Program

### Menggunakan CMake (Direkomendasikan)

1. Setelah kompilasi, dari direktori build, jalankan program dengan:
   ```bash
   ..\bin\Release\QuadtreeCompression.exe
   ```
2. Atau bisa langsung menjalankan executable dari folder bin:
    ```bash
   cd ..\bin\Release
   QuadtreeCompression.exe
   ```
3. Ikuti petunjuk yang muncul pada layar untuk:
   - Memasukkan path file gambar input
   - Memilih metode pengukuran error
   - Memasukkan nilai threshold
   - Menentukan ukuran blok minimum
   - Mengatur persentase kompresi target (jika diinginkan)
   - Menentukan path output gambar hasil kompresi
   - Memilih apakah ingin membuat visualisasi GIF

## Input dan Parameter

- **Path File Gambar Input**: Path lengkap ke file gambar yang ingin dikompresi (.jpg, .jpeg, .png, dll)
- **Metode Pengukuran Error**:
  1. Variance - Variansi statistik antar kanal warna
  2. MAD (Mean Absolute Deviation) - Rata-rata perbedaan dari warna rata-rata
  3. Max Pixel Difference - Perbedaan warna maksimum dalam blok
  4. Entropy - Pengukuran keacakan warna
  5. SSIM (Structural Similarity Index) - Metrik kesamaan perseptual [BONUS]
- **Threshold**: Nilai ambang batas untuk menentukan apakah blok akan dibagi lagi
- **Ukuran Blok Minimum**: Ukuran terkecil yang diperbolehkan untuk proses pembagian
- **Persentase Kompresi Target** [BONUS]: Nilai untuk mengatur target kompresi yang diinginkan

## Output Program

Program akan menampilkan:
- Waktu eksekusi
- Ukuran gambar sebelum dan sesudah kompresi
- Persentase kompresi yang dicapai
- Kedalaman pohon Quadtree
- Jumlah simpul dalam pohon
- Gambar hasil kompresi (disimpan ke path yang ditentukan)
- Visualisasi GIF (jika dipilih)

## Contoh Penggunaan
   ```bash
   PS C:\Users\DANENDRA\OneDrive\Documents\ITB\SEMESTER 4\IF2211 Strategi Algoritma\Tucil 2> ..\bin\Release\QuadtreeCompression.exe

    _/    _/  _/  _/_/_/_/  _/    _/  _/    _/    _/_/    
   _/  _/    _/      _/    _/    _/  _/_/  _/  _/    _/   
  _/_/      _/     _/     _/    _/  _/ _/_/_/  _/_/_/_/   
 _/  _/    _/    _/      _/    _/  _/    _/   _/    _/    
_/    _/  _/   _/_/_/_/  _/_/_/   _/    _/   _/    _/    

    =======================================================
    ||           QUADTREE IMAGE COMPRESSOR               ||
    =======================================================

    Masukkan path file gambar input: /path/ke/image.jpg
    Pilih metode pengukuran error: 
    Masukkan nilai threshold: 
    Masukkan ukuran blok minimum: 
    Masukkan persentase kompresi target (0.0 untuk menonaktifkan): 
    Masukkan path file gambar output: /path/ke/output.jpg
    Buat visualisasi GIF? [1-Ya, 0-Tidak]: 1
    Masukkan path file output GIF: /path/ke/output.gif
   ```
## Contoh Penggunaan
  - Nama: Danendra Shafi Athallah
  - NIM: 13523136
