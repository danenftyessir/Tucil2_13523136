#ifndef QUADTREE_HPP
#define QUADTREE_HPP

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <filesystem>
#include <vector>
#include <thread>
#include <atomic>
#include <random>

namespace fs = std::filesystem;
using namespace cv;
using namespace std;

enum class ErrorMethod {
    VARIANCE,
    MAD,
    MAX_PIXEL_DIFF,
    ENTROPY,
    SSIM        // Bonus: Structural Similarity Index
};

class QuadtreeNode {
public:
    int x, y, width, height;
    Vec3b avgColor;
    QuadtreeNode* children[4];
    bool isLeaf; 

    QuadtreeNode(int x, int y, int width, int height);
    void calculateAverageColor(const Mat& image);
};

class Quadtree {
private:
    QuadtreeNode* root;
    double threshold;
    int minBlockSize;
    Mat sourceImage;
    ErrorMethod errorMethod;
    double targetCompressionPct; // Bonus
    vector<Mat> gifFrames;       // Bonus
    bool visualizeGif;           // Bonus
    atomic<int> nodeCounter;    
    atomic<bool> timeoutFlag; 
    int maxDepth; 
    bool forceLowCompression;
    double targetNodeRatio; 
    mt19937 rng;            // Random number generator
    bool useHybridCompression;
    double hybridCompressionTarget;
    Rect centerRegion;
    int centerMinBlockSize;
    int centerMaxDepth;
    int outerMinBlockSize;
    int outerMaxDepth;
    
    void quadtreeCompress(Mat& image, QuadtreeNode* node, int depth = 0);
    void reconstructHelper(Mat& image, QuadtreeNode* node);
    int getTreeDepthHelper(QuadtreeNode* node);
    int getNodeCountHelper(QuadtreeNode* node);
    void deleteTree(QuadtreeNode* node);
    
    // Error measurement methods
    double calculateVariance(const Mat& block);
    double calculateMAD(const Mat& block);
    double calculateMaxPixelDiff(const Mat& block);
    double calculateEntropy(const Mat& block);
    double calculateSSIM(const Mat& block, const Mat& avgBlock); // Bonus: SSIM calculation
    double calculateError(const Mat& block, const Mat& avgBlock = Mat());
    string getErrorMethodName(ErrorMethod method);
    
    // Bonus: Dynamic threshold adjustment
    void adjustThresholdForTargetCompression(const Mat& image);
    // Bonus: GIF visualization
    void captureFrameForGif(const Mat& currentImage);
    void drawQuadtreeVisualization(Mat& image, QuadtreeNode* node, int depth);
    Mat getSafeRoi(const Mat& image, int x, int y, int width, int height);
    
public:
    Quadtree(const Mat& image, double threshold, int minBlockSize, 
             ErrorMethod method = ErrorMethod::VARIANCE, 
             double targetCompressionPct = 0.0,
             bool visualizeGif = false);
    ~Quadtree();
    
    void compressImage();
    void reconstructImage(Mat& image);
    int getTreeDepth();
    int getNodeCount();
    int countLeafNodes(QuadtreeNode* node);
    double calculateCompressionPercentage(const string& originalImagePath, const string& compressedImagePath);
    double getThreshold() const { return threshold; }
    QuadtreeNode* getRoot() const { return root; }
    
    // Bonus: Save GIF animation
    bool saveGifAnimation(const string& outputPath);
};

string getErrorMethodName(ErrorMethod method);
bool isPowerOfTwo(int n);

#endif