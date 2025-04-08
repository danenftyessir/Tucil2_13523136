#define NOMINMAX

#ifdef _WIN32
#include <windows.h>
#endif

#include <iostream>
#include <string>
#include <chrono>
#include <limits>
#include <filesystem>
#include <iomanip>
#include <cmath>
#include <vector>
#include <algorithm>

#include "interface.hpp"
#include "Quadtree.hpp"

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

// ANSI color codes for terminal output
namespace Color {
    const std::string RESET   = "\033[0m";
    const std::string BLACK   = "\033[30m";
    const std::string RED     = "\033[31m";
    const std::string GREEN   = "\033[32m";
    const std::string YELLOW  = "\033[33m";
    const std::string BLUE    = "\033[34m";
    const std::string MAGENTA = "\033[35m";
    const std::string CYAN    = "\033[36m";
    const std::string WHITE   = "\033[37m";
    const std::string BOLD    = "\033[1m";
    const std::string BG_RED  = "\033[41m";
    const std::string BG_GREEN = "\033[42m";
    const std::string BG_BLUE = "\033[44m";
}

void clearInputBuffer() {
    cin.clear();
    cin.ignore(numeric_limits<streamsize>::max(), '\n');
}

string cleanPath(const string& path) {
    string cleanedPath = path;
    
    if (cleanedPath.size() >= 2 && 
        ((cleanedPath.front() == '"' && cleanedPath.back() == '"') ||
         (cleanedPath.front() == '\'' && cleanedPath.back() == '\''))) {
        cleanedPath = cleanedPath.substr(1, cleanedPath.size() - 2);
    }
    
    return cleanedPath;
}

bool validateImageFile(const string& path, string& errorMessage) {
    if (!fs::exists(path)) {
        errorMessage = "File tidak ditemukan pada path: " + path;
        return false;
    }
    
    string extension = fs::path(path).extension().string();
    transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    
    if (extension != ".jpg" && extension != ".jpeg" && extension != ".png" && 
        extension != ".webp" && extension != ".bmp" && extension != ".tiff" && 
        extension != ".tif" && extension != ".ppm" && extension != ".pgm") {
        errorMessage = "Format file tidak didukung. Silakan gunakan gambar JPG, JPEG, PNG, WEBP, BMP, TIFF, atau PPM/PGM.";
        return false;
    }
    
    Mat testImage = imread(path);
    if (testImage.empty()) {
        errorMessage = "Tidak dapat memuat gambar. File mungkin rusak atau bukan gambar yang valid.";
        return false;
    }
    
    if (testImage.channels() != 3) {
        errorMessage = "Gambar harus memiliki 3 kanal warna (RGB). Gambar grayscale atau dengan alpha channel mungkin tidak diproses dengan benar.";
        return false;
    }
    
    return true;
}

bool validateAndCreateDirectory(const string& path, string& errorMessage) {
    try {
        fs::path dirPath = fs::path(path).parent_path();
        if (!dirPath.empty() && !fs::exists(dirPath)) {
            if (!fs::create_directories(dirPath)) {
                errorMessage = "Gagal membuat direktori: " + dirPath.string();
                return false;
            }
        }
        return true;
    } catch (const fs::filesystem_error& e) {
        errorMessage = e.what();
        return false;
    }
}

// Validate threshold based on the selected error method
bool validateThreshold(double threshold, ErrorMethod method, string& errorMessage) {
    if (threshold <= 0.0) {
        errorMessage = "Threshold harus positif";
        return false;
    }
    
    double maxValue = 0.0;
    
    switch (method) {
        case ErrorMethod::VARIANCE:
            maxValue = 10000.0;
            if (threshold < 1.0) {
                errorMessage = "Peringatan: Threshold sangat rendah untuk metode Variance. Kompresi mungkin minimal";
            } else if (threshold > 1000.0) {
                errorMessage = "Peringatan: Threshold sangat tinggi untuk metode Variance. Kualitas gambar mungkin buruk";
            }
            break;
            
        case ErrorMethod::MAD:
            maxValue = 255.0;
            if (threshold < 1.0) {
                errorMessage = "Peringatan: Threshold sangat rendah untuk metode MAD. Kompresi mungkin minimal";
            } else if (threshold > 100.0) {
                errorMessage = "Peringatan: Threshold sangat tinggi untuk metode MAD. Kualitas gambar mungkin buruk";
            }
            break;
            
        case ErrorMethod::MAX_PIXEL_DIFF:
            maxValue = 255.0;
            if (threshold < 1.0) {
                errorMessage = "Peringatan: Threshold sangat rendah untuk metode Max Pixel Difference. Kompresi mungkin minimal";
            } else if (threshold > 200.0) {
                errorMessage = "Peringatan: Threshold sangat tinggi untuk metode Max Pixel Difference. Kualitas gambar mungkin buruk";
            }
            break;
            
        case ErrorMethod::ENTROPY:
            maxValue = 8.0;
            if (threshold < 0.1) {
                errorMessage = "Peringatan: Threshold sangat rendah untuk metode Entropy. Kompresi mungkin minimal";
            } else if (threshold > 5.0) {
                errorMessage = "Peringatan: Threshold sangat tinggi untuk metode Entropy. Kualitas gambar mungkin buruk";
            }
            break;
            
        case ErrorMethod::SSIM:
            maxValue = 1.0;
            if (threshold < 0.01) {
                errorMessage = "Peringatan: Threshold sangat rendah untuk metode SSIM. Kompresi mungkin minimal";
            } else if (threshold > 0.5) {
                errorMessage = "Peringatan: Threshold sangat tinggi untuk metode SSIM. Kualitas gambar mungkin buruk";
            }
            break;
            
        default:
            maxValue = 1000.0;
    }
    
    return true;
}

// Validate minimum block size
bool validateMinBlockSize(int minBlockSize, const Mat& image, string& errorMessage) {
    if (minBlockSize <= 0) {
        errorMessage = "Ukuran blok minimum harus positif";
        return false;
    }
    
    int minDimension = min(image.cols, image.rows);
    if (minBlockSize >= minDimension / 2) {
        errorMessage = "Ukuran blok minimum terlalu besar untuk gambar ini. Maksimum yang direkomendasikan: " + 
                      to_string(minDimension / 4);
        return false;
    }
    
    if (!isPowerOfTwo(minBlockSize)) {
        errorMessage = "Peringatan: Ukuran blok minimum bukan pangkat dari 2. Ini dapat menyebabkan hasil yang tidak terduga";
    }
    
    return true;
}

// Validate target compression percentage
bool validateTargetCompression(double targetCompression, string& errorMessage) {
    if (targetCompression < 0.0 || targetCompression > 100.0) {
        errorMessage = "Target kompresi harus antara 0 dan 100 persen";
        return false;
    }
    
    if (targetCompression > 95.0) {
        errorMessage = "Peringatan: Target kompresi sangat tinggi (>95%). Kualitas gambar mungkin sangat buruk";
    } else if (targetCompression > 0.0 && targetCompression < 10.0) {
        errorMessage = "Peringatan: Target kompresi sangat rendah (<10%). Mungkin sulit dicapai";
    }
    
    return true;
}

void displayRecommendedThresholds(ErrorMethod method) {
    cout << "    " << Color::CYAN << "Rentang threshold yang direkomendasikan untuk " << getErrorMethodName(method) << ":" << Color::RESET << "\n";
    
    double minRecommended = 0.0, maxRecommended = 0.0;
    switch (method) {
        case ErrorMethod::VARIANCE:
            minRecommended = 10.0;
            maxRecommended = 1000.0;
            cout << "    - Minimum yang direkomendasikan: " << Color::GREEN << minRecommended << Color::RESET << " (kompresi minimal)\n";
            cout << "    - Nilai tengah: " << Color::YELLOW << "100.0" << Color::RESET << " (keseimbangan kualitas/kompresi)\n";
            cout << "    - Maksimum yang direkomendasikan: " << Color::RED << maxRecommended << Color::RESET << " (kompresi maksimal)\n";
            cout << "    - Nilai tipikal: " << Color::BOLD << "30-200" << Color::RESET << " untuk sebagian besar gambar\n";
            break;
        case ErrorMethod::MAD:
            minRecommended = 5.0;
            maxRecommended = 50.0;
            cout << "    - Minimum yang direkomendasikan: " << Color::GREEN << minRecommended << Color::RESET << " (kompresi minimal)\n";
            cout << "    - Nilai tengah: " << Color::YELLOW << "20.0" << Color::RESET << " (keseimbangan kualitas/kompresi)\n";
            cout << "    - Maksimum yang direkomendasikan: " << Color::RED << maxRecommended << Color::RESET << " (kompresi maksimal)\n";
            cout << "    - Nilai tipikal: " << Color::BOLD << "10-30" << Color::RESET << " untuk sebagian besar gambar\n";
            break;
        case ErrorMethod::MAX_PIXEL_DIFF:
            minRecommended = 10.0;
            maxRecommended = 100.0;
            cout << "    - Minimum yang direkomendasikan: " << Color::GREEN << minRecommended << Color::RESET << " (kompresi minimal)\n";
            cout << "    - Nilai tengah: " << Color::YELLOW << "40.0" << Color::RESET << " (keseimbangan kualitas/kompresi)\n";
            cout << "    - Maksimum yang direkomendasikan: " << Color::RED << maxRecommended << Color::RESET << " (kompresi maksimal)\n";
            cout << "    - Nilai tipikal: " << Color::BOLD << "20-60" << Color::RESET << " untuk sebagian besar gambar\n";
            break;
        case ErrorMethod::ENTROPY:
            minRecommended = 0.1;
            maxRecommended = 5.0;
            cout << "    - Minimum yang direkomendasikan: " << Color::GREEN << minRecommended << Color::RESET << " (kompresi minimal)\n";
            cout << "    - Nilai tengah: " << Color::YELLOW << "1.0" << Color::RESET << " (keseimbangan kualitas/kompresi)\n";
            cout << "    - Maksimum yang direkomendasikan: " << Color::RED << maxRecommended << Color::RESET << " (kompresi maksimal)\n";
            cout << "    - Nilai tipikal: " << Color::BOLD << "0.5-2.0" << Color::RESET << " untuk sebagian besar gambar\n";
            break;
        case ErrorMethod::SSIM:
            minRecommended = 0.05;
            maxRecommended = 0.5;
            cout << "    - Minimum yang direkomendasikan: " << Color::GREEN << minRecommended << Color::RESET << " (kompresi minimal)\n";
            cout << "    - Nilai tengah: " << Color::YELLOW << "0.2" << Color::RESET << " (keseimbangan kualitas/kompresi)\n";
            cout << "    - Maksimum yang direkomendasikan: " << Color::RED << maxRecommended << Color::RESET << " (kompresi maksimal)\n";
            cout << "    - Nilai tipikal: " << Color::BOLD << "0.1-0.3" << Color::RESET << " untuk sebagian besar gambar\n";
            break;
    }
    cout << "\n";
}

int main() {
    string inputImagePath, outputImagePath, gifOutputPath;
    double threshold, targetCompressionPct;
    int minBlockSize, errorMethodChoice;
    ErrorMethod method;
    bool visualizeGif = false;
    string errorMessage;
    bool saveOutput = true;
    
    // Enable ANSI colors on Windows
    #ifdef _WIN32
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD dwMode = 0;
    GetConsoleMode(hOut, &dwMode);
    dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
    SetConsoleMode(hOut, dwMode);
    #endif
    
    QuadtreeInterface ui(true);
    
    // Show program logo and introduction (still keep this for the nice logo)
    ui.showLogo();
    ui.showIntro();
    
    // Skip showing the main menu and go directly to compression
    // The showMainMenu(ui) call is removed
    
    ui.showSectionHeader("INPUT GAMBAR");
    
    cout << "    " << Color::CYAN << "Format: " << Color::RESET << "Silakan berikan path lengkap ke file gambar.\n";
    cout << "    " << Color::YELLOW << "Contoh: " << Color::RESET << "/home/user/images/sample.jpg atau C:\\Users\\user\\Pictures\\sample.png\n";
    cout << "    " << Color::BLUE << "Format yang didukung: " << Color::RESET << "JPG, JPEG, PNG, WEBP, BMP, TIFF, PPM/PGM\n\n";
    
    bool validInputImage = false;
    while (!validInputImage) {
        cout << "    Masukkan path file gambar input: ";
        getline(cin, inputImagePath);
        
        if (inputImagePath.empty()) {
            ui.showError("Tidak ada input. Silakan masukkan path gambar yang valid.");
            continue;
        }
        
        inputImagePath = cleanPath(inputImagePath);
        
        if (validateImageFile(inputImagePath, errorMessage)) {
            validInputImage = true;
            ui.showSuccess("File gambar berhasil divalidasi.");
        } else {
            ui.showError(errorMessage);
            ui.showInfo("Silakan periksa bahwa file ada dan merupakan gambar valid dengan format yang didukung.");
        }
    }
    
    ui.showLoading("Memuat gambar");
    Mat image = imread(inputImagePath);
    ui.showInfo("Dimensi gambar: " + to_string(image.cols) + "x" + to_string(image.rows) + " piksel");
    ui.showInfo("Kanal warna: " + to_string(image.channels()));
    
    ui.showSectionHeader("METODE PENGUKURAN ERROR");
    
    cout << "    Pilih metode pengukuran error:\n";
    cout << "    " << Color::BLUE << "1. Variance" << Color::RESET << " - Variansi statistik antar kanal warna\n";
    cout << "    " << Color::GREEN << "2. Mean Absolute Deviation (MAD)" << Color::RESET << " - Rata-rata perbedaan dari warna rata-rata\n";
    cout << "    " << Color::YELLOW << "3. Max Pixel Difference" << Color::RESET << " - Perbedaan warna maksimum dalam blok\n";
    cout << "    " << Color::MAGENTA << "4. Entropy" << Color::RESET << " - Pengukuran keacakan warna dari teori informasi\n";
    cout << "    " << Color::CYAN << "5. Structural Similarity Index (SSIM)" << Color::RESET << " - Metrik kesamaan perseptual [BONUS]\n\n";
    
    bool validMethod = false;
    while (!validMethod) {
        cout << "    Masukkan pilihan (1-5): ";
        
        if (!(cin >> errorMethodChoice) || errorMethodChoice < 1 || errorMethodChoice > 5) {
            ui.showError("Pilihan tidak valid. Silakan masukkan angka antara 1 dan 5.");
            clearInputBuffer();
        } else {
            switch (errorMethodChoice) {
                case 1: method = ErrorMethod::VARIANCE; break;
                case 2: method = ErrorMethod::MAD; break;
                case 3: method = ErrorMethod::MAX_PIXEL_DIFF; break;
                case 4: method = ErrorMethod::ENTROPY; break;
                case 5: method = ErrorMethod::SSIM; break;
                default: method = ErrorMethod::VARIANCE;
            }
            validMethod = true;
            ui.showSuccess("Metode terpilih: " + getErrorMethodName(method));
            clearInputBuffer();
        }
    }
    
    ui.showSectionHeader("NILAI THRESHOLD");
    
    cout << "    " << Color::CYAN << "Threshold menentukan seberapa agresif gambar akan dikompresi." << Color::RESET << "\n";
    cout << "    - " << Color::GREEN << "Threshold rendah" << Color::RESET << " = kualitas lebih tinggi, kompresi lebih sedikit\n";
    cout << "    - " << Color::RED << "Threshold tinggi" << Color::RESET << " = kualitas lebih rendah, kompresi lebih banyak\n\n";
    
    displayRecommendedThresholds(method);
    
    double minRecommended = 0.0, maxRecommended = 0.0;
    switch (method) {
        case ErrorMethod::VARIANCE:
            minRecommended = 10.0;
            maxRecommended = 1000.0;
            break;
        case ErrorMethod::MAD:
            minRecommended = 5.0;
            maxRecommended = 50.0;
            break;
        case ErrorMethod::MAX_PIXEL_DIFF:
            minRecommended = 10.0;
            maxRecommended = 100.0;
            break;
        case ErrorMethod::ENTROPY:
            minRecommended = 0.1;
            maxRecommended = 5.0;
            break;
        case ErrorMethod::SSIM:
            minRecommended = 0.05;
            maxRecommended = 0.5;
            break;
    }
    
    bool validThreshold = false;
    while (!validThreshold) {
        cout << "    Masukkan nilai threshold (rentang yang direkomendasikan: " << Color::GREEN << minRecommended << Color::RESET << " sampai " << Color::RED << maxRecommended << Color::RESET << "): ";
        
        if (!(cin >> threshold)) {
            ui.showError("Input tidak valid. Silakan masukkan nilai numerik.");
            clearInputBuffer();
            continue;
        }
        
        string errorMsg;
        if (!validateThreshold(threshold, method, errorMsg)) {
            ui.showError(errorMsg);
            clearInputBuffer();
            continue;
        }
        
        if (!errorMsg.empty()) {
            ui.showWarning(errorMsg);
        }
        
        bool inRecommendedRange = (threshold >= minRecommended && threshold <= maxRecommended);
        
        if (!inRecommendedRange) {
            string warningMsg;
            if (threshold < minRecommended) {
                warningMsg = "Threshold di bawah minimum yang direkomendasikan. Ini mungkin menghasilkan kompresi minimal.";
            } else {
                warningMsg = "Threshold di atas maksimum yang direkomendasikan. Ini mungkin menghasilkan kualitas gambar yang buruk.";
            }
            ui.showWarning(warningMsg);
            
            cout << "    Anda ingin melanjutkan dengan nilai threshold ini? [y/n]: ";
            char confirm;
            cin >> confirm;
            clearInputBuffer();
            
            if (tolower(confirm) != 'y') {
                continue;
            }
        }
        
        validThreshold = true;
        ui.showSuccess("Threshold diatur ke: " + to_string(threshold));
        clearInputBuffer();
    }
    
    ui.showSectionHeader("UKURAN BLOK MINIMUM");
    
    cout << "    " << Color::CYAN << "Ukuran blok minimum menentukan blok terkecil yang akan dibuat oleh Quadtree." << Color::RESET << "\n";
    cout << "    - Nilai lebih kecil (mis., " << Color::GREEN << "2, 4" << Color::RESET << ") mempertahankan detail lebih banyak tapi mengurangi kompresi\n";
    cout << "    - Nilai lebih besar (mis., " << Color::YELLOW << "8, 16, 32" << Color::RESET << ") meningkatkan kompresi tapi detail bisa hilang\n\n";
    
    cout << "    " << Color::BOLD << "Praktik terbaik:" << Color::RESET << "\n";
    cout << "    - Gunakan pangkat dari 2 (" << Color::BOLD << "2, 4, 8, 16, 32" << Color::RESET << ") untuk kinerja optimal\n";
    cout << "    - Untuk gambar dengan detail tinggi, gunakan nilai lebih kecil (" << Color::GREEN << "2-4" << Color::RESET << ")\n";
    cout << "    - Untuk gambar lebih sederhana, nilai lebih besar (" << Color::YELLOW << "8-16" << Color::RESET << ") bisa lebih baik\n";
    cout << "    - Nilai antara " << Color::BOLD << "2 dan 16" << Color::RESET << " biasanya paling berguna\n\n";
    
    int imgMin = std::min(image.cols, image.rows);
    int recommendedMin = 2;
    int recommendedMax = std::min(16, imgMin / 8);
    
    cout << "    Berdasarkan ukuran gambar Anda (" << image.cols << "x" << image.rows << "):\n";
    cout << "    - Minimum yang direkomendasikan: " << Color::GREEN << recommendedMin << Color::RESET << "\n";
    cout << "    - Maksimum yang direkomendasikan: " << Color::YELLOW << recommendedMax << Color::RESET << "\n\n";
    
    bool validBlockSize = false;
    while (!validBlockSize) {
        cout << "    Masukkan ukuran blok minimum: ";
        
        if (!(cin >> minBlockSize)) {
            ui.showError("Input tidak valid. Silakan masukkan nilai numerik.");
            clearInputBuffer();
            continue;
        }
        
        string errorMsg;
        if (!validateMinBlockSize(minBlockSize, image, errorMsg)) {
            ui.showError(errorMsg);
            clearInputBuffer();
            continue;
        }
        
        if (!errorMsg.empty()) {
            ui.showWarning(errorMsg);
        }
        
        if (!isPowerOfTwo(minBlockSize)) {
            cout << "    Pangkat dari 2 terdekat: ";
            
            int lowerPow = (int)pow(2, floor(log2(minBlockSize)));
            int upperPow = (int)pow(2, ceil(log2(minBlockSize)));
            
            cout << Color::GREEN << lowerPow << Color::RESET << " atau " << Color::YELLOW << upperPow << Color::RESET << "\n";
            cout << "    Apakah Anda ingin menggunakan salah satu nilai ini? [y/n]: ";
            
            char adjustSize;
            cin >> adjustSize;
            clearInputBuffer();
            
            if (tolower(adjustSize) == 'y') {
                cout << "    Pilih [1] untuk " << Color::GREEN << lowerPow << Color::RESET << " atau [2] untuk " << Color::YELLOW << upperPow << Color::RESET << ": ";
                int choice;
                cin >> choice;
                clearInputBuffer();
                
                if (choice == 1) {
                    minBlockSize = lowerPow;
                } else if (choice == 2) {
                    minBlockSize = upperPow;
                } else {
                    ui.showWarning("Pilihan tidak valid. Mempertahankan nilai awal: " + to_string(minBlockSize));
                }
            }
        }
        
        validBlockSize = true;
        ui.showSuccess("Ukuran blok minimum diatur ke: " + to_string(minBlockSize));
    }
    
    ui.showSectionHeader("PERSENTASE KOMPRESI TARGET [BONUS]");
    
    cout << "    " << Color::CYAN << "Fitur BONUS ini memungkinkan algoritma menyesuaikan threshold secara otomatis" << Color::RESET << "\n";
    cout << "    untuk mencapai rasio kompresi tertentu, terlepas dari threshold yang Anda atur sebelumnya.\n\n";
    cout << "    " << Color::BOLD << "Panduan:" << Color::RESET << "\n";
    cout << "    - " << Color::BLUE << "0.0" << Color::RESET << " = Nonaktifkan penyesuaian otomatis (gunakan nilai threshold sebelumnya)\n";
    cout << "    - " << Color::GREEN << "1-30%" << Color::RESET << " = Kompresi rendah, kualitas tinggi\n";
    cout << "    - " << Color::CYAN << "30-60%" << Color::RESET << " = Kompresi sedang, kualitas baik\n";
    cout << "    - " << Color::YELLOW << "60-80%" << Color::RESET << " = Kompresi tinggi, kualitas berkurang\n";
    cout << "    - " << Color::RED << "80-95%" << Color::RESET << " = Kompresi sangat tinggi, kualitas turun signifikan\n";
    cout << "    - Nilai di atas " << Color::BG_RED << "95%" << Color::RESET << " mungkin sulit dicapai tanpa penurunan kualitas yang parah\n\n";
    cout << "    " << Color::BOLD << "Catatan:" << Color::RESET << " Algoritma akan mencoba mendekati target Anda sedekat mungkin, tetapi\n";
    cout << "    persentase yang persis mungkin tidak dapat dicapai untuk semua gambar.\n\n";
    
    bool validCompression = false;
    while (!validCompression) {
        cout << "    Masukkan persentase kompresi target (0.0 untuk menonaktifkan, mis., 50.0 untuk 50%): ";
        
        if (!(cin >> targetCompressionPct)) {
            ui.showError("Input tidak valid. Silakan masukkan nilai numerik.");
            clearInputBuffer();
            continue;
        }
        
        string errorMsg;
        if (!validateTargetCompression(targetCompressionPct, errorMsg)) {
            ui.showError(errorMsg);
            clearInputBuffer();
            continue;
        }
        
        if (!errorMsg.empty()) {
            ui.showWarning(errorMsg);
        }
        
        validCompression = true;
        if (targetCompressionPct == 0.0) {
            ui.showInfo("Target kompresi dinonaktifkan. Menggunakan kompresi berbasis threshold saja.");
        } else {
            ui.showSuccess("Target kompresi diatur ke: " + to_string(targetCompressionPct) + "%");
            ui.showInfo("Algoritma akan mencoba menyesuaikan threshold secara otomatis untuk mencapai target ini.");
        }
        clearInputBuffer();
    }
    
    ui.showSectionHeader("GAMBAR OUTPUT");
    
    bool validOutputPath = false;
    while (!validOutputPath) {
        cout << "    Masukkan path file gambar output (kosongkan untuk default): ";
        getline(cin, outputImagePath);
        
        if (outputImagePath.empty()) {
            fs::path exePath = fs::current_path();
            fs::path outputDir = exePath / "hasil";
            
            fs::path inputPath(inputImagePath);
            fs::path outputFileName = inputPath.stem().string() + "_compressed" + inputPath.extension().string();
            outputImagePath = (outputDir / outputFileName).string();
            
            try {
                fs::create_directories(outputDir);
                ui.showInfo("Menggunakan path output default: " + outputImagePath);
                validOutputPath = true;
            } catch (const fs::filesystem_error& e) {
                ui.showError("Gagal membuat direktori output: " + string(e.what()));
                ui.showInfo("Silakan tentukan path output yang valid secara manual.");
                continue;
            }
        } else {
            outputImagePath = cleanPath(outputImagePath);
            
            fs::path outputPath(outputImagePath);
            if (outputPath.extension().empty()) {
                fs::path inputPath(inputImagePath);
                outputImagePath += inputPath.extension().string();
                ui.showInfo("Menambahkan ekstensi file: " + outputImagePath);
            }
            
            if (validateAndCreateDirectory(outputImagePath, errorMessage)) {
                validOutputPath = true;
            } else {
                ui.showError(errorMessage);
            }
        }
    }
    
    ui.showSectionHeader("SIMPAN GAMBAR OUTPUT");

    bool validSaveChoice = false;
    while (!validSaveChoice) {
        cout << "    Simpan gambar terkompresi? [" << Color::GREEN << "1-Ya" << Color::RESET << ", " << Color::RED << "0-Tidak" << Color::RESET << "]: ";
        int saveChoice;
        
        if (!(cin >> saveChoice) || (saveChoice != 0 && saveChoice != 1)) {
            ui.showError("Input tidak valid. Silakan masukkan 0 atau 1.");
            clearInputBuffer();
            continue;
        }
        
        saveOutput = (saveChoice == 1);
        validSaveChoice = true;
        clearInputBuffer();
        
        if (!saveOutput) {
            ui.showInfo("Gambar hasil tidak akan disimpan, hanya ditampilkan.");
        }
    }
    
    ui.showSectionHeader("VISUALISASI GIF [BONUS]");
    
    bool validGifChoice = false;
    while (!validGifChoice) {
        cout << "    Buat visualisasi GIF? [" << Color::GREEN << "1-Ya" << Color::RESET << ", " << Color::RED << "0-Tidak" << Color::RESET << "]: ";
        int gifChoice;
        
        if (!(cin >> gifChoice) || (gifChoice != 0 && gifChoice != 1)) {
            ui.showError("Input tidak valid. Silakan masukkan 0 atau 1.");
            clearInputBuffer();
            continue;
        }
        
        visualizeGif = (gifChoice == 1);
        validGifChoice = true;
        clearInputBuffer();
    }
    if (visualizeGif) {
        bool validGifPath = false;
        while (!validGifPath) {
            cout << "    Masukkan path file output GIF (kosongkan untuk default): ";
            getline(cin, gifOutputPath);
            
            if (gifOutputPath.empty()) {
                fs::path exePath = fs::current_path();
                fs::path outputDir = exePath / "hasil";
                
                try {
                    fs::create_directories(outputDir);
                } catch (const fs::filesystem_error& e) {
                    ui.showError("Gagal membuat direktori output: " + string(e.what()));
                    ui.showInfo("Silakan tentukan path output yang valid secara manual.");
                    continue;
                }
                
                fs::path inputPath(inputImagePath);
                fs::path gifFileName = inputPath.stem().string() + "_process.gif";
                gifOutputPath = (outputDir / gifFileName).string();
                
                ui.showInfo("Menggunakan path GIF default: " + gifOutputPath);
                validGifPath = true;
            } else {
                gifOutputPath = cleanPath(gifOutputPath);
                
                fs::path gifPath(gifOutputPath);
                if (gifPath.extension().empty() || gifPath.extension() != ".gif") {
                    gifOutputPath = fs::path(gifOutputPath).replace_extension(".gif").string();
                    ui.showInfo("Menggunakan ekstensi file .gif: " + gifOutputPath);
                }
                
                if (validateAndCreateDirectory(gifOutputPath, errorMessage)) {
                    validGifPath = true;
                } else {
                    ui.showError(errorMessage);
                }
            }
        }
    }
    
    ui.showSectionHeader("PEMROSESAN");
    
    cout << "\n    " << Color::CYAN << "Ringkasan Parameter Kompresi:" << Color::RESET << "\n";
    cout << "    - Gambar Input: " << Color::YELLOW << inputImagePath << Color::RESET << "\n";
    cout << "    - Metode Error: " << Color::YELLOW << getErrorMethodName(method) << Color::RESET << "\n";
    cout << "    - Threshold: " << Color::YELLOW << threshold << Color::RESET << "\n";
    cout << "    - Ukuran Blok Minimum: " << Color::YELLOW << minBlockSize << Color::RESET << "\n";
    cout << "    - Kompresi Target: " << Color::YELLOW << (targetCompressionPct > 0 ? to_string(targetCompressionPct) + "%" : "Dinonaktifkan") << Color::RESET << "\n";
    cout << "    - Gambar Output: " << Color::YELLOW << outputImagePath << Color::RESET << "\n";
    cout << "    - Buat GIF: " << Color::YELLOW << (visualizeGif ? "Ya" : "Tidak") << Color::RESET << "\n";
    if (visualizeGif) {
        cout << "    - Output GIF: " << Color::YELLOW << gifOutputPath << Color::RESET << "\n";
    }
    
    bool confirmProceed = false;
    while (!confirmProceed) {
        cout << "\n    Lanjutkan dengan kompresi? [" << Color::GREEN << "y" << Color::RESET << "/" << Color::RED << "n" << Color::RESET << "]: ";
        char confirm;
        cin >> confirm;
        clearInputBuffer();
        
        if (tolower(confirm) == 'y') {
            confirmProceed = true;
        } else if (tolower(confirm) == 'n') {
            cout << "    Kompresi dibatalkan. Keluar dari program." << endl;
            return 0;
        } else {
            ui.showError("Input tidak valid. Silakan masukkan 'y' atau 'n'.");
        }
    }
    
    auto start = chrono::high_resolution_clock::now();
    
    ui.showInfo("Memulai proses kompresi...");
    
    try {
        // Show progress without actual delays for better performance
        for (int i = 0; i <= 100; i += 5) {
            ui.showProgressBar(i, 100);
            std::this_thread::sleep_for(std::chrono::milliseconds(5)); // Very short delay
        }
        
        ui.showLoading("Membuat quadtree", 50);
        Quadtree quadtree(image, threshold, minBlockSize, method, targetCompressionPct, visualizeGif);
        
        ui.showLoading("Mengompresi gambar", 50);
        quadtree.compressImage();
        
        ui.showLoading("Merekonstruksi gambar", 50);
        Mat compressedImage = image.clone();
        quadtree.reconstructImage(compressedImage);
        
        auto end = chrono::high_resolution_clock::now();
        double execTime = chrono::duration<double, milli>(end - start).count();
        
        
        if (saveOutput) {
            ui.showLoading("Menyimpan gambar terkompresi", 50);
            
            try {
                fs::path outputDir = fs::path(outputImagePath).parent_path();
                if (!outputDir.empty()) {
                    fs::create_directories(outputDir);
                }
            } catch (const fs::filesystem_error& e) {
                ui.showWarning("Peringatan saat membuat direktori output: " + string(e.what()));
                ui.showInfo("Mencoba menyimpan file output...");
            }
            
            string extension = fs::path(outputImagePath).extension().string();
            transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
            
            vector<int> compressionParams;
            if (extension == ".png") {
                compressionParams.push_back(IMWRITE_PNG_COMPRESSION);
                compressionParams.push_back(9); // Maximum PNG compression
            } else if (extension == ".jpg" || extension == ".jpeg") {
                compressionParams.push_back(IMWRITE_JPEG_QUALITY);
                // Reduce JPEG quality to achieve better file compression
                // Use a lower quality value based on target compression
                int quality = 85; // Default quality
                
                // Adjust quality based on target compression
                if (targetCompressionPct > 0) {
                    // Higher target compression = lower quality
                    if (targetCompressionPct > 80) quality = 60;
                    else if (targetCompressionPct > 60) quality = 70;
                    else if (targetCompressionPct > 40) quality = 75;
                    else if (targetCompressionPct > 20) quality = 80;
                }
                
                compressionParams.push_back(quality);
            } else if (extension == ".webp") {
                compressionParams.push_back(IMWRITE_WEBP_QUALITY);
                compressionParams.push_back(80);
            }
            
            bool saveSuccess = imwrite(outputImagePath, compressedImage, compressionParams);
            
            if (!saveSuccess) {
                ui.showError("Gagal menyimpan gambar terkompresi. Perbandingan masih bisa dilihat.");
            } else {
                ui.showSuccess("Gambar terkompresi berhasil disimpan.");
            }
        } else {
            ui.showInfo("Gambar terkompresi tidak disimpan sesuai permintaan.");
        }
        
        if (visualizeGif) {
            ui.showLoading("Membuat visualisasi GIF", 50);
            
            try {
                fs::path gifDir = fs::path(gifOutputPath).parent_path();
                if (!gifDir.empty()) {
                    fs::create_directories(gifDir);
                }
            } catch (const fs::filesystem_error& e) {
                ui.showWarning("Peringatan saat membuat direktori GIF: " + string(e.what()));
                ui.showInfo("Mencoba menyimpan file GIF...");
            }
            
            bool gifSuccess = quadtree.saveGifAnimation(gifOutputPath);
            if (!gifSuccess) {
                ui.showWarning("Gagal membuat visualisasi GIF. Melanjutkan...");
            } else {
                ui.showSuccess("Visualisasi GIF berhasil disimpan.");
            }
        }
        
        int treeDepth = quadtree.getTreeDepth();
        int nodeCount = quadtree.getNodeCount();
        
        // Hitung persentase kompresi berdasarkan ukuran file (sesuai rumus)
        double compressionPercentage = 0.0;
        
        if (saveOutput) {
            compressionPercentage = quadtree.calculateCompressionPercentage(inputImagePath, outputImagePath);
        } else {
            // Jika gambar tidak disimpan, gunakan node-based metric
            int totalPixels = image.rows * image.cols;
            int leafNodes = quadtree.countLeafNodes(quadtree.getRoot());
            compressionPercentage = (1.0 - (double)leafNodes / totalPixels) * 100.0;
        }
        
        // Untuk informasi ukuran file
        uintmax_t originalSize = 0;
        uintmax_t compressedSize = 0;
        
        try {
            originalSize = fs::file_size(inputImagePath);
        } catch (const fs::filesystem_error& e) {
            ui.showWarning("Tidak dapat mendapatkan ukuran file asli: " + string(e.what()));
        }
        
        try {
            if (fs::exists(outputImagePath)) {
                compressedSize = fs::file_size(outputImagePath);
            }
        } catch (const fs::filesystem_error& e) {
            ui.showWarning("Tidak dapat mendapatkan ukuran file terkompresi: " + string(e.what()));
        }
        
        vector<pair<string, string>> resultData = {
            {"Waktu eksekusi", to_string(execTime) + " ms"},
            {"Ukuran gambar asli", to_string(originalSize) + " bytes"},
            {"Ukuran gambar terkompresi", to_string(compressedSize) + " bytes"},
            {"Persentase kompresi", to_string(compressionPercentage) + "%"},
            {"Threshold akhir", 
                [&quadtree]() {
                    double thresh = quadtree.getThreshold();
                    if (thresh > 1e10) {
                        return string("Auto-adjusted");
                    }
                    stringstream ss;
                    ss << fixed << setprecision(2) << thresh;
                    return ss.str();
                }()
            },
            {"Kedalaman Quadtree", to_string(treeDepth)},
            {"Jumlah node dalam Quadtree", to_string(nodeCount)}
        };
        
        ui.showResultTable("HASIL KOMPRESI", resultData);
        
        cout << "\n    " << Color::BG_GREEN << " Kompresi berhasil diselesaikan! " << Color::RESET << "\n";
        if (saveOutput) {
            cout << "    Gambar terkompresi disimpan ke: " << Color::GREEN << outputImagePath << Color::RESET << "\n";
        } else {
            cout << "    Gambar terkompresi tidak disimpan (mode tampilkan saja).\n";
        }
        
        if (visualizeGif) {
            cout << "    Visualisasi proses disimpan ke: " << Color::GREEN << gifOutputPath << Color::RESET << "\n";
        }
        
    } catch (const exception& e) {
        ui.showError("Kompresi gagal: " + string(e.what()));
        return -1;
    }
    
    cout << "\n    Apakah Anda ingin membuka gambar asli dan terkompresi untuk perbandingan? [" << Color::GREEN << "y" << Color::RESET << "/" << Color::RED << "n" << Color::RESET << "]: ";
    char viewImages;
    cin >> viewImages;
    
    if (tolower(viewImages) == 'y') {
        namedWindow("Gambar Asli", WINDOW_NORMAL);
        imshow("Gambar Asli", image);
        
        Mat compressedImage = imread(outputImagePath);
        if (!compressedImage.empty()) {
            namedWindow("Gambar Terkompresi", WINDOW_NORMAL);
            imshow("Gambar Terkompresi", compressedImage);
            
            ui.showInfo("Tekan sembarang tombol pada jendela gambar untuk menutupnya.");
            waitKey(0);
            destroyAllWindows();
        } else {
            ui.showError("Tidak dapat memuat gambar terkompresi untuk ditampilkan.");
        }
    }
    
    ui.showThankYou();
    
    return 0;
}