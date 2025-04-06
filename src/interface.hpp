#ifndef INTERFACE_HPP
#define INTERFACE_HPP

// Definisikan NOMINMAX untuk mencegah definisi macro min dan max dari Windows
#define NOMINMAX

// Include Windows headers dulu jika menggunakan Windows
#ifdef _WIN32
#include <windows.h>
#include <conio.h>
#endif

#include <iostream>
#include <string>
#include <chrono>
#include <thread>
#include <iomanip>
#include <vector>
#include <algorithm>

// Untuk sistem Unix
#ifndef _WIN32
#include <unistd.h>
#include <termios.h>
#endif

class QuadtreeInterface {
private:
    bool useAnimation;
    
    // Helper untuk membaca karakter dari keyboard tanpa buffering
    int getch() {
        #ifdef _WIN32
            return _getch();
        #else
            struct termios oldt, newt;
            int ch;
            tcgetattr(STDIN_FILENO, &oldt);
            newt = oldt;
            newt.c_lflag &= ~(ICANON | ECHO);
            tcsetattr(STDIN_FILENO, TCSANOW, &newt);
            ch = getchar();
            tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
            return ch;
        #endif
    }
    
    // Helper untuk membersihkan layar
    void clearScreen() {
        #ifdef _WIN32
            system("cls");
        #else
            system("clear");
        #endif
    }
    
    // Tunggu beberapa milidetik
    void delay(int ms) {
        std::this_thread::sleep_for(std::chrono::milliseconds(ms));
    }
    
    // Menampilkan teks dengan efek ketik
    void typeText(const std::string& text, int delayMs = 5) {
        if (!useAnimation) {
            std::cout << text;
            return;
        }
        
        for (char c : text) {
            std::cout << c << std::flush;
            delay(delayMs);
        }
    }
    
public:
    QuadtreeInterface(bool animation = true) : useAnimation(animation) {}
    
    // Tampilkan logo KIZUNA dengan ASCII art
    void showLogo() {
        clearScreen();
        
        std::cout << "\n";
        std::cout << "    _/    _/  _/     _/_/_/_/  _/    _/  _/    _/    _/_/    \n";
        std::cout << "   _/  _/    _/         _/    _/    _/  _/_/  _/  _/    _/   \n";
        std::cout << "  _/_/      _/        _/     _/    _/  _/ _/_/_/  _/_/_/_/   \n"; 
        std::cout << " _/  _/    _/       _/      _/    _/  _/    _/   _/    _/    \n";
        std::cout << "_/    _/  _/_/_/_/ _/_/_/_/  _/_/_/   _/    _/   _/    _/    \n";
        std::cout << "\n";
        std::cout << "    =======================================================\n";
        std::cout << "    ||           QUADTREE IMAGE COMPRESSOR               ||\n";
        std::cout << "    =======================================================\n\n";
        
        if (useAnimation) {
            delay(200);
        }
    }
    
    // Tampilkan deskripsi singkat tentang program dan cara kerja
    void showIntro() {
        typeText("    Selamat datang di KIZUNA Quadtree Image Compressor!\n\n");
        typeText("    Program ini akan membantu Anda mengompres gambar dengan menggunakan\n");
        typeText("    algoritma Divide and Conquer berbasis Quadtree. Metode ini bekerja\n");
        typeText("    dengan membagi gambar menjadi 4 bagian secara rekursif sampai bagian\n");
        typeText("    tersebut memiliki warna yang relatif seragam.\n\n", 5);
        
        typeText("    Tekan ENTER untuk melanjutkan...");
        getch();
        std::cout << std::endl;
    }
    
    // Tampilkan header section
    void showSectionHeader(const std::string& title) {
        std::cout << "\n";
        std::cout << "    +-" << std::string(title.length() + 2, '-') << "-+\n";
        std::cout << "    | " << title << " |\n";
        std::cout << "    +-" << std::string(title.length() + 2, '-') << "-+\n\n";
    }
    
    // Tampilkan pesan error
    void showError(const std::string& message) {
        std::cout << "    [ERROR] " << message << std::endl;
    }
    
    // Tampilkan pesan warning
    void showWarning(const std::string& message) {
        std::cout << "    [PERINGATAN] " << message << std::endl;
    }
    
    // Tampilkan pesan sukses
    void showSuccess(const std::string& message) {
        std::cout << "    [BERHASIL] " << message << std::endl;
    }
    
    // Tampilkan pesan info
    void showInfo(const std::string& message) {
        std::cout << "    [INFO] " << message << std::endl;
    }
    
    // Tampilkan progress bar
    void showProgressBar(int progress, int total, int width = 40) {
        float percent = (float)progress / total;
        int completed = (int)(width * percent);
        
        std::cout << "    [";
        for (int i = 0; i < width; i++) {
            if (i < completed) std::cout << "#";
            else std::cout << " ";
        }
        
        std::cout << "] " << std::fixed << std::setprecision(1) << (percent * 100.0) << "%" << std::endl;
    }
    
    // Tampilkan animasi loading
    void showLoading(const std::string& message, int durationMs = 300) {
        if (!useAnimation) {
            std::cout << "    " << message << "... Selesai!" << std::endl;
            return;
        }
        
        std::cout << "    " << message << "... ";
        std::cout.flush();
        
        const char sequence[] = {'|', '/', '-', '\\'};
        int count = durationMs / 50; // Lebih cepat - 50ms per karakter
        
        for (int i = 0; i < count; i++) {
            std::cout << sequence[i % 4] << "\b" << std::flush;
            delay(50); // Waktu delay lebih pendek
        }
        
        std::cout << "Selesai!" << std::endl;
    }
    
    // Tampilkan tabel hasil kompresi (REVISED)
    void showResultTable(const std::string& title, const std::vector<std::pair<std::string, std::string>>& data) {
        // Determine column widths with reasonable limits
        size_t maxLabelWidth = 0;
        size_t maxValueWidth = 0;
        const size_t VALUE_DISPLAY_LIMIT = 30; // Limit very long values
        
        for (const auto& row : data) {
            maxLabelWidth = std::max(maxLabelWidth, row.first.length());
            // For value column, limit extremely long values
            size_t valueLen = std::min(row.second.length(), VALUE_DISPLAY_LIMIT);
            maxValueWidth = std::max(maxValueWidth, valueLen);
        }
        
        // Add padding
        maxLabelWidth += 2;
        maxValueWidth += 2;
        
        // Total table width
        size_t tableWidth = maxLabelWidth + maxValueWidth + 3;
        
        // Display header
        std::cout << "\n    +" << std::string(tableWidth, '-') << "+\n";
        std::cout << "    |" << std::setw(tableWidth) << std::left << " " + title << "|\n";
        std::cout << "    +" << std::string(maxLabelWidth, '-') << "+" << std::string(maxValueWidth + 2, '-') << "+\n";
        
        // Display data rows
        for (const auto& row : data) {
            std::cout << "    | " << std::setw(maxLabelWidth - 1) << std::left << row.first << "| ";
            
            // Handle long values with truncation and indicator
            std::string displayValue = row.second;
            if (displayValue.length() > VALUE_DISPLAY_LIMIT) {
                displayValue = displayValue.substr(0, VALUE_DISPLAY_LIMIT - 3) + "...";
            }
            
            std::cout << std::setw(maxValueWidth - 1) << std::left << displayValue << "|\n";
        }
        
        // Display footer
        std::cout << "    +" << std::string(maxLabelWidth, '-') << "+" << std::string(maxValueWidth + 2, '-') << "+\n";
    }
    
    // Tampilkan ucapan terima kasih
    void showThankYou() {
        clearScreen();
        std::cout << "\n\n";
        std::cout << "    Terima kasih telah menggunakan KIZUNA Quadtree Image Compressor!\n\n";
        
        typeText("    Sampai jumpa kembali...  ", 20);
        delay(200);
    }
};

#endif // INTERFACE_HPP