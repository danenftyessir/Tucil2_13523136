#include "Quadtree.hpp"
#include <cmath>
#include <map>
#include <algorithm>
#include <chrono>
#include <thread>
#include <iomanip>
#include <sstream>
#include <future>
#include <random>
#include <fstream>

bool isPowerOfTwo(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

string getErrorMethodName(ErrorMethod method) {
    switch (method) {
        case ErrorMethod::VARIANCE: return "Variance";
        case ErrorMethod::MAD: return "Mean Absolute Deviation";
        case ErrorMethod::MAX_PIXEL_DIFF: return "Max Pixel Difference";
        case ErrorMethod::ENTROPY: return "Entropy";
        case ErrorMethod::SSIM: return "SSIM";
        default: return "Unknown";
    }
}

QuadtreeNode::QuadtreeNode(int x, int y, int width, int height)
    : x(x), y(y), width(width), height(height), isLeaf(true) {
    for (int i = 0; i < 4; ++i) {
        children[i] = nullptr;
    }
    avgColor = Vec3b(128, 128, 128);
}

void QuadtreeNode::calculateAverageColor(const Mat& image) {
    int endX = std::min(x + width, image.cols);
    int endY = std::min(y + height, image.rows);
    int startX = std::max(0, x);
    int startY = std::max(0, y);
    
    if (startX >= endX || startY >= endY || startX >= image.cols || startY >= image.rows) {
        avgColor = Vec3b(128, 128, 128);
        return;
    }
    
    Vec3d sum(0, 0, 0);
    int count = 0;
    
    for (int py = startY; py < endY; py++) {
        for (int px = startX; px < endX; px++) {
            if (px < 0 || px >= image.cols || py < 0 || py >= image.rows) {
                continue;
            }
            Vec3b pixel = image.at<Vec3b>(py, px);
            sum[0] += pixel[0];
            sum[1] += pixel[1]; 
            sum[2] += pixel[2];
            count++;
        }
    }
    
    if (count > 0) {
        avgColor = Vec3b(
            static_cast<uchar>(sum[0] / count),
            static_cast<uchar>(sum[1] / count),
            static_cast<uchar>(sum[2] / count)
        );
    } else {
        avgColor = Vec3b(200, 200, 200);
    }
}

Quadtree::Quadtree(const Mat& image, double threshold, int minBlockSize, 
                   ErrorMethod method, double targetCompressionPct, bool visualizeGif)
    : threshold(threshold), 
      minBlockSize(minBlockSize), 
      errorMethod(method), 
      targetCompressionPct(targetCompressionPct),
      visualizeGif(visualizeGif),
      nodeCounter(0),
      timeoutFlag(false),
      maxDepth(10),
      forceLowCompression(false),
      useHybridCompression(false),
      targetNodeRatio(1.0),
      centerMinBlockSize(2),
      centerMaxDepth(10),
      outerMinBlockSize(16),
      outerMaxDepth(4) {
          
    sourceImage = image.clone();
    root = new QuadtreeNode(0, 0, image.cols, image.rows);
    
    random_device rd;
    rng = mt19937(rd());
}

Quadtree::~Quadtree() {
    deleteTree(root);
}

void Quadtree::deleteTree(QuadtreeNode* node) {
    if (!node) return;
    for (int i = 0; i < 4; ++i) {
        deleteTree(node->children[i]);
    }
    delete node;
}

Mat Quadtree::getSafeRoi(const Mat& image, int x, int y, int width, int height) {
    int startX = max(0, x);
    int startY = max(0, y);
    int endX = min(x + width, image.cols);
    int endY = min(y + height, image.rows);
    
    if (startX >= endX || startY >= endY) {
        return Mat();
    }
    
    return image(Rect(startX, startY, endX - startX, endY - startY));
}

double Quadtree::calculateVariance(const Mat& block) {
    if (block.empty() || block.rows * block.cols <= 1) return 0.0;
    
    Scalar meanColor = mean(block);
    
    double sumSq[3] = {0, 0, 0};
    int count = 0;
    
    for (int i = 0; i < block.rows; i++) {
        for (int j = 0; j < block.cols; j++) {
            Vec3b pixel = block.at<Vec3b>(i, j);
            for (int c = 0; c < 3; c++) {
                double diff = pixel[c] - meanColor[c];
                sumSq[c] += diff * diff;
            }
            count++;
        }
    }
    
    if (count <= 1) return 0.0;
    
    double variance = (sumSq[0] + sumSq[1] + sumSq[2]) / (3.0 * count);
    return variance;
}

double Quadtree::calculateMAD(const Mat& block) {
    if (block.empty()) return 0.0;
    
    Scalar meanColor = mean(block);
    
    double madSum[3] = {0, 0, 0};
    int count = 0;
    
    for (int i = 0; i < block.rows; i++) {
        for (int j = 0; j < block.cols; j++) {
            Vec3b pixel = block.at<Vec3b>(i, j);
            madSum[0] += std::abs(pixel[0] - meanColor[0]);
            madSum[1] += std::abs(pixel[1] - meanColor[1]);
            madSum[2] += std::abs(pixel[2] - meanColor[2]);
            count++;
        }
    }
    
    if (count == 0) return 0.0;
    
    return (madSum[0] + madSum[1] + madSum[2]) / (3.0 * count);
}

double Quadtree::calculateMaxPixelDiff(const Mat& block) {
    if (block.empty()) return 0.0;
    
    if (block.rows * block.cols <= 4) {
        Vec3b firstPixel = block.at<Vec3b>(0, 0);
        if (block.rows * block.cols == 1) return 0.0;
        
        double maxDiff = 0;
        for (int i = 0; i < block.rows; ++i) {
            for (int j = 0; j < block.cols; ++j) {
                Vec3b pixel = block.at<Vec3b>(i, j);
                double diff = 0;
                for (int c = 0; c < 3; c++) {
                    diff += std::abs(pixel[c] - firstPixel[c]);
                }
                maxDiff = std::max(maxDiff, diff/3.0);
            }
        }
        return maxDiff;
    }
    
    Vec3b minVals(255, 255, 255);
    Vec3b maxVals(0, 0, 0);
    
    for (int i = 0; i < block.rows; i++) {
        for (int j = 0; j < block.cols; j++) {
            Vec3b pixel = block.at<Vec3b>(i, j);
            
            for (int c = 0; c < 3; c++) {
                minVals[c] = std::min(minVals[c], pixel[c]);
                maxVals[c] = std::max(maxVals[c], pixel[c]);
            }
        }
    }
    
    return ((maxVals[0] - minVals[0]) + (maxVals[1] - minVals[1]) + (maxVals[2] - minVals[2])) / 3.0;
}

double Quadtree::calculateEntropy(const Mat& block) {
    if (block.empty() || block.rows * block.cols < 16) {
        return calculateMaxPixelDiff(block) / 255.0;
    }
    
    int hist[768] = {0};
    int sampleCount = 0;
    
    for (int i = 0; i < block.rows; i++) {
        for (int j = 0; j < block.cols; j++) {
            Vec3b pixel = block.at<Vec3b>(i, j);
            hist[pixel[0]]++;
            hist[256 + pixel[1]]++;
            hist[512 + pixel[2]]++;
            sampleCount++;
        }
    }
    
    if (sampleCount == 0) return 0.0;
    
    double entropy = 0.0;
    double count = 0.0;
    
    for (int i = 0; i < 768; i++) {
        if (hist[i] > 0) {
            double p = (double)hist[i] / sampleCount;
            entropy -= p * log2(p);
            count++;
        }
    }
    
    if (count > 0) {
        entropy /= 3.0;
    }
    
    return std::min(entropy, 5.0);
}

double Quadtree::calculateSSIM(const Mat& block, const Mat& avgBlock) {
    if (block.empty() || avgBlock.empty()) return 0.0;
    
    if (block.rows < 4 || block.cols < 4) {
        return calculateVariance(block) / 1000.0;
    }
    
    const double L = 255.0;
    const double k1 = 0.01;
    const double k2 = 0.03;
    const double C1 = pow(k1 * L, 2);
    const double C2 = pow(k2 * L, 2);
    
    const double wR = 0.299;
    const double wG = 0.587;
    const double wB = 0.114;
    
    vector<double> ssimValues(3, 0.0);
    
    for (int c = 0; c < 3; c++) {
        double mu1 = 0, mu2 = 0;
        double sigma1_sq = 0, sigma2_sq = 0, sigma12 = 0;
        int N = 0;
        
        for (int i = 0; i < block.rows; i++) {
            for (int j = 0; j < block.cols; j++) {
                double val1 = block.at<Vec3b>(i, j)[c];
                double val2 = avgBlock.at<Vec3b>(i, j)[c];
                
                mu1 += val1;
                mu2 += val2;
                N++;
            }
        }
        
        if (N == 0) continue;
        
        mu1 /= N;
        mu2 /= N;
        
        for (int i = 0; i < block.rows; i++) {
            for (int j = 0; j < block.cols; j++) {
                double val1 = block.at<Vec3b>(i, j)[c];
                double val2 = avgBlock.at<Vec3b>(i, j)[c];
                
                double diff1 = val1 - mu1;
                double diff2 = val2 - mu2;
                
                sigma1_sq += diff1 * diff1;
                sigma2_sq += diff2 * diff2;
                sigma12 += diff1 * diff2;
            }
        }
        
        sigma1_sq /= (N - 1);
        sigma2_sq /= (N - 1);
        sigma12 /= (N - 1);
        
        double numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2);
        double denominator = (mu1*mu1 + mu2*mu2 + C1) * (sigma1_sq + sigma2_sq + C2);
        
        double ssim = denominator > 0.001 ? numerator / denominator : 0.99;
        ssimValues[c] = std::max(0.0, std::min(1.0, 1.0 - ssim));
    }
    
    double weightedSSIM = wR * ssimValues[2] + wG * ssimValues[1] + wB * ssimValues[0];
    return weightedSSIM * 0.5;
}

double Quadtree::calculateError(const Mat& block, const Mat& avgBlock) {
    if (block.empty() || (block.rows == 1 && block.cols == 1)) return 0.0;
    
    // Special handling for very small blocks (2x2 or 3x3)
    if (block.rows * block.cols <= 9 && minBlockSize == 2) {
        // For small blocks, use a more stable error metric
        switch (errorMethod) {
            case ErrorMethod::VARIANCE:
            case ErrorMethod::MAD:
            case ErrorMethod::ENTROPY:
                return calculateMaxPixelDiff(block) * 0.5; // Use a more stable metric with reduced sensitivity
            default:
                break;
        }
    }
    
    switch (errorMethod) {
        case ErrorMethod::VARIANCE:
            return calculateVariance(block);
        case ErrorMethod::MAD:
            return calculateMAD(block);
        case ErrorMethod::MAX_PIXEL_DIFF:
            return calculateMaxPixelDiff(block);
        case ErrorMethod::ENTROPY:
            return calculateEntropy(block);
        case ErrorMethod::SSIM:
            if (avgBlock.empty()) {
                Scalar meanColor = mean(block);
                Mat uniformBlock(block.size(), block.type(), Scalar(meanColor[0], meanColor[1], meanColor[2]));
                return calculateSSIM(block, uniformBlock);
            } else {
                return calculateSSIM(block, avgBlock);
            }
        default:
            return calculateVariance(block);
    }
}

string Quadtree::getErrorMethodName(ErrorMethod method) {
    return ::getErrorMethodName(method);
}

void Quadtree::adjustThresholdForTargetCompression(const Mat& image) {
    if (targetCompressionPct <= 0.0) {
        cout << "Target compression is disabled. Using standard threshold-based compression." << endl;
        return;
    }
    
    double targetPct = targetCompressionPct;
    
    if (targetPct < 20.0) {
        cout << "Target kompresi " << targetPct << "%. Menggunakan pendekatan presisi tinggi." << endl;
        
        threshold = [this]() {
            switch (errorMethod) {
                case ErrorMethod::VARIANCE: return 5.0;
                case ErrorMethod::MAD: return 2.0;
                case ErrorMethod::MAX_PIXEL_DIFF: return 5.0;
                case ErrorMethod::ENTROPY: return 0.1;
                case ErrorMethod::SSIM: return 0.01;
                default: return 5.0;
            }
        }();
        
        int gridSize = static_cast<int>(sqrt(image.cols * image.rows * (1.0 - targetPct/100.0)));
        int powerOf2 = 1;
        while (powerOf2 * 2 <= gridSize) powerOf2 *= 2;
        minBlockSize = max(2, powerOf2);
        
        return;
    }
    else if (targetPct < 75.0) {
        cout << "Target kompresi " << targetPct << "%. Menggunakan pendekatan fixed-grid." << endl;
        
        double originalThreshold = threshold;
        int originalMinBlockSize = minBlockSize;
        
        int totalPixels = image.rows * image.cols;
        
        double targetNodeRatio = 1.0 - (targetPct / 100.0);
        int targetLeafNodes = std::max(1, static_cast<int>(totalPixels * targetNodeRatio));
        
        cout << "  - Target leaf nodes: " << targetLeafNodes << " dari " << totalPixels << " piksel" << endl;
        
        double avgBlockArea = static_cast<double>(totalPixels) / targetLeafNodes;
        int gridSize = static_cast<int>(std::sqrt(avgBlockArea));
        
        int powerOf2 = 1;
        while (powerOf2 * 2 <= gridSize) powerOf2 *= 2;
        
        minBlockSize = powerOf2;
        maxDepth = static_cast<int>(std::log2(std::max(image.cols, image.rows) / powerOf2)) + 1;
        
        forceLowCompression = true;
        
        int predictedLeafNodes = (image.cols / powerOf2) * (image.rows / powerOf2);
        double predictedCompressionPct = (1.0 - static_cast<double>(predictedLeafNodes) / totalPixels) * 100.0;
        
        cout << "  - Menggunakan mode fixed-grid dengan ukuran blok " << powerOf2 << "x" << powerOf2 << endl;
        cout << "  - Prediksi kompresi: " << predictedCompressionPct << "%" << endl;
        cout << "  - Batas kedalaman: " << maxDepth << endl;
        
        if (std::abs(predictedCompressionPct - targetPct) > 10.0) {
            useHybridCompression = true;
            hybridCompressionTarget = targetPct;
            
            double centerRatio = 0.4;
            if (targetPct < 40.0) centerRatio = 0.6;
            else if (targetPct > 60.0) centerRatio = 0.3;
            
            int centerWidth = static_cast<int>(image.cols * centerRatio);
            int centerHeight = static_cast<int>(image.rows * centerRatio);
            centerRegion = Rect((image.cols - centerWidth) / 2, 
                              (image.rows - centerHeight) / 2,
                              centerWidth, centerHeight);
            
            cout << "  - Region tengah: " << centerRegion.width << "x" << centerRegion.height 
                 << " dengan detail tinggi" << endl;
                 
            centerMinBlockSize = max(2, powerOf2 / 2);
            centerMaxDepth = 10;
            
            outerMinBlockSize = powerOf2 * 2;
            outerMaxDepth = min(4, maxDepth - 1);
        }
        
        return;
    }
    
    cout << "Target kompresi " << targetPct << "%. Menggunakan pendekatan adaptif." << endl;
    
    double low = 0.0001;
    double high = 0.0; 
    
    if (targetPct < 85.0) {
        switch (errorMethod) {
            case ErrorMethod::VARIANCE: high = 50.0; break;
            case ErrorMethod::MAD: high = 15.0; break;
            case ErrorMethod::MAX_PIXEL_DIFF: high = 30.0; break;
            case ErrorMethod::ENTROPY: high = 1.0; break;
            case ErrorMethod::SSIM: high = 0.15; break;
            default: high = 50.0;
        }
    } else if (targetPct < 95.0) {
        switch (errorMethod) {
            case ErrorMethod::VARIANCE: high = 200.0; break;
            case ErrorMethod::MAD: high = 30.0; break;
            case ErrorMethod::MAX_PIXEL_DIFF: high = 75.0; break;
            case ErrorMethod::ENTROPY: high = 2.5; break;
            case ErrorMethod::SSIM: high = 0.3; break;
            default: high = 200.0;
        }
    } else {
        switch (errorMethod) {
            case ErrorMethod::VARIANCE: high = 500.0; break;
            case ErrorMethod::MAD: high = 50.0; break;
            case ErrorMethod::MAX_PIXEL_DIFF: high = 150.0; break;
            case ErrorMethod::ENTROPY: high = 5.0; break;
            case ErrorMethod::SSIM: high = 0.5; break;
            default: high = 500.0;
        }
    }
    
    double bestThreshold = threshold;
    double bestDifference = std::numeric_limits<double>::max();
    int maxIterations = 7;
    double tolerance = 3.0;
    
    Mat testImage;
    double scale = 1.0;
    if (image.rows * image.cols > 1000000) {
        scale = 0.5;
        resize(image, testImage, Size(), scale, scale, INTER_AREA);
    } else {
        testImage = image.clone();
    }
    
    double currentPct = 0.0;
    {
        Quadtree tempTree(testImage, threshold, minBlockSize, errorMethod, 0.0, false);
        tempTree.compressImage();
        
        int totalPixels = testImage.rows * testImage.cols;
        int leafNodes = tempTree.countLeafNodes(tempTree.getRoot());
        currentPct = (1.0 - (double)leafNodes / totalPixels) * 100.0;
        
        double difference = abs(currentPct - targetPct);
        if (difference < bestDifference) {
            bestThreshold = threshold;
            bestDifference = difference;
        }
        
        if (abs(currentPct - targetPct) <= tolerance) {
            cout << "Target compression achieved with initial threshold!" << endl;
            return;
        }
        
        if (currentPct < targetPct) {
            low = threshold;
        } else {
            high = threshold;
        }
    }

    for (int iter = 0; iter < maxIterations; iter++) {
        double weight = 0.5;
        
        if (iter > 0) {
            if (currentPct < targetPct) {
                weight = 0.7;
            } else {
                weight = 0.3;
            }
        }
        
        threshold = low + (high - low) * weight;
        
        if (abs(threshold - bestThreshold) < 0.001 * bestThreshold) {
            break;
        }
        
        cout << "Iteration " << iter+1 << ": Testing threshold = " << threshold << endl;
        
        Quadtree tempTree(testImage, threshold, minBlockSize, errorMethod, 0.0, false);
        tempTree.compressImage();
        
        int totalPixels = testImage.rows * testImage.cols;
        int leafNodes = tempTree.countLeafNodes(tempTree.getRoot());
        currentPct = (1.0 - (double)leafNodes / totalPixels) * 100.0;
        
        cout << "  Current compression: " << currentPct << "%" << endl;
        
        double difference = abs(currentPct - targetPct);
        if (difference < bestDifference) {
            bestThreshold = threshold;
            bestDifference = difference;
        }
        
        if (abs(currentPct - targetPct) <= tolerance) {
            cout << "Target compression achieved with threshold = " << threshold << endl;
            break;
        }
        
        if (currentPct < targetPct) {
            low = threshold;
        } else {
            high = threshold;
        }
        
        if ((high - low) < 0.001 * low) {
            break;
        }
    }
    
    if (bestDifference > tolerance) {
        double extrapolatedThreshold = bestThreshold;
        if (currentPct < targetPct) {
            extrapolatedThreshold = bestThreshold * (targetPct / currentPct);
        } else {
            extrapolatedThreshold = bestThreshold * (currentPct / targetPct);
        }
        
        extrapolatedThreshold = max(low, min(high * 1.2, extrapolatedThreshold));
        
        cout << "Fine-tuning with threshold = " << extrapolatedThreshold << endl;
        
        Quadtree tempTree(testImage, extrapolatedThreshold, minBlockSize, errorMethod, 0.0, false);
        tempTree.compressImage();
        
        int totalPixels = testImage.rows * testImage.cols;
        int leafNodes = tempTree.countLeafNodes(tempTree.getRoot());
        double extrapolatedPct = (1.0 - (double)leafNodes / totalPixels) * 100.0;
        
        double extrapolatedDiff = abs(extrapolatedPct - targetPct);
        if (extrapolatedDiff < bestDifference) {
            bestThreshold = extrapolatedThreshold;
            bestDifference = extrapolatedDiff;
        }
    }
    
    threshold = bestThreshold;
    cout << "Using best threshold = " << threshold << endl;
    cout << "Estimated final compression: within " << bestDifference << "% of target" << endl;
}

void Quadtree::captureFrameForGif(const Mat& currentImage) {
    if (!visualizeGif) return;
    
    static int frameCounter = 0;
    frameCounter++;
    
    bool shouldCapture = false;
    if (gifFrames.size() < 5) {
        shouldCapture = true;
    } else if (frameCounter % 10 == 0 && gifFrames.size() < 30) {
        shouldCapture = true;
    } else if (frameCounter % 30 == 0 && gifFrames.size() < 60) {
        shouldCapture = true;
    }
    
    if (!shouldCapture) return;
    
    int targetWidth = 640;
    int targetHeight = 480;
    
    double scaleX = static_cast<double>(targetWidth) / std::max(1, currentImage.cols);
    double scaleY = static_cast<double>(targetHeight) / std::max(1, currentImage.rows);
    double scale = std::min(scaleX, scaleY);
    
    Mat frame;
    if (scale < 1.0) {
        resize(currentImage, frame, Size(), scale, scale, INTER_AREA);
    } else {
        frame = currentImage.clone();
    }
    
    Mat visImage = frame.clone();
    drawQuadtreeVisualization(visImage, root, 0);
    
    string infoText = "Frame " + to_string(gifFrames.size() + 1);
    putText(visImage, infoText, Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,255), 1);
    
    gifFrames.push_back(visImage);
}

void Quadtree::drawQuadtreeVisualization(Mat& image, QuadtreeNode* node, int depth) {
    if (!node || depth > 10) return;
    
    Rect rect(node->x, node->y, node->width, node->height);
    
    rect = rect & Rect(0, 0, image.cols, image.rows);
    
    if (rect.width <= 0 || rect.height <= 0) return;
    
    if (node->isLeaf) {
        rectangle(image, rect, node->avgColor, FILLED);
        if (node->width >= 8 && node->height >= 8) {
            rectangle(image, rect, Scalar(0, 255, 0), 1);
        }
    } else {
        Scalar color;
        int depthMod = depth % 3;
        switch (depthMod) {
            case 0: color = Scalar(255, 0, 0);   break;
            case 1: color = Scalar(0, 0, 255);   break;
            case 2: color = Scalar(0, 165, 255); break;
            default: color = Scalar(255, 255, 255); break;
        }
        
        rectangle(image, rect, color, 1);
        
        int midX = rect.x + rect.width / 2;
        int midY = rect.y + rect.height / 2;
        
        line(image, Point(midX, rect.y), Point(midX, rect.y + rect.height), color, 1);
        line(image, Point(rect.x, midY), Point(rect.x + rect.width, midY), color, 1);
        
        if (depth < 8) {
            for (int i = 0; i < 4; i++) {
                if (node->children[i]) {
                    drawQuadtreeVisualization(image, node->children[i], depth + 1);
                }
            }
        }
    }
}

void Quadtree::quadtreeCompress(Mat& image, QuadtreeNode* node, int depth) {
    const int MAX_NODES = 150000;
    
    if (timeoutFlag || nodeCounter > MAX_NODES) return;
    if (!node) return;
    
    // Jika posisi node invalid
    if (node->x < 0 || node->y < 0 || node->x >= image.cols || node->y >= image.rows) {
        node->isLeaf = true;
        return;
    }
    
    // Untuk hybrid compression
    if ((useHybridCompression || forceLowCompression) && targetCompressionPct <= 0.0) {
        useHybridCompression = false;
        forceLowCompression = false;
    }
    
    // Kode untuk hybrid compression dan force low compression tetap sama
    if (useHybridCompression && targetCompressionPct > 0.0) {
        Rect nodeRect(node->x, node->y, node->width, node->height);
        bool isInCenterRegion = (nodeRect & centerRegion).area() > 0;
        
        int currentMaxDepth = isInCenterRegion ? centerMaxDepth : outerMaxDepth;
        int currentMinBlockSize = isInCenterRegion ? centerMinBlockSize : outerMinBlockSize;
        
        if (depth > currentMaxDepth || node->width <= currentMinBlockSize || node->height <= currentMinBlockSize) {
            node->calculateAverageColor(image);
            node->isLeaf = true;
            return;
        }
        
        node->isLeaf = false;
        nodeCounter += 4;
        
        int halfWidth = max(1, node->width / 2);
        int halfHeight = max(1, node->height / 2);
        
        if (visualizeGif && (depth <= 2)) {
            Rect rect(node->x, node->y, node->width, node->height);
            rect = rect & Rect(0, 0, image.cols, image.rows);
            if (rect.width > 0 && rect.height > 0) {
                Mat visImage = image.clone();
                rectangle(visImage, rect, Scalar(0, 0, 255), 2);
                captureFrameForGif(visImage);
            }
        }
        
        node->children[0] = new QuadtreeNode(node->x, node->y, halfWidth, halfHeight);
        node->children[1] = new QuadtreeNode(node->x + halfWidth, node->y, node->width - halfWidth, halfHeight);
        node->children[2] = new QuadtreeNode(node->x, node->y + halfHeight, halfWidth, node->height - halfHeight);
        node->children[3] = new QuadtreeNode(node->x + halfWidth, node->y + halfHeight, node->width - halfWidth, node->height - halfHeight);
        
        for (int i = 0; i < 4; i++) {
            quadtreeCompress(image, node->children[i], depth + 1);
            
            if (timeoutFlag) break;
        }
        
        return;
    }
    else if (forceLowCompression && targetCompressionPct > 0.0) {
        if (node->width <= minBlockSize || node->height <= minBlockSize || depth >= maxDepth) {
            node->calculateAverageColor(image);
            node->isLeaf = true;
            return;
        }
        
        node->isLeaf = false;
        nodeCounter += 4;
        
        int halfWidth = max(1, node->width / 2);
        int halfHeight = max(1, node->height / 2);
        
        if (visualizeGif && (depth <= 2)) {
            Rect rect(node->x, node->y, node->width, node->height);
            rect = rect & Rect(0, 0, image.cols, image.rows);
            if (rect.width > 0 && rect.height > 0) {
                Mat visImage = image.clone();
                rectangle(visImage, rect, Scalar(0, 0, 255), 2);
                captureFrameForGif(visImage);
            }
        }
        
        node->children[0] = new QuadtreeNode(node->x, node->y, halfWidth, halfHeight);
        node->children[1] = new QuadtreeNode(node->x + halfWidth, node->y, node->width - halfWidth, halfHeight);
        node->children[2] = new QuadtreeNode(node->x, node->y + halfHeight, halfWidth, node->height - halfHeight);
        node->children[3] = new QuadtreeNode(node->x + halfWidth, node->y + halfHeight, node->width - halfWidth, node->height - halfHeight);
        
        for (int i = 0; i < 4; i++) {
            quadtreeCompress(image, node->children[i], depth + 1);
            
            if (timeoutFlag) break;
        }
        
        return;
    }
    
    // PERBAIKAN UTAMA: Penanganan khusus untuk minBlockSize = 2
    if (minBlockSize == 2) {
        // Jika sudah mencapai batas kedalaman maksimum
        if (depth > maxDepth) {
            node->calculateAverageColor(image);
            node->isLeaf = true;
            return;
        }
        
        // Jika ukuran node terlalu kecil untuk dibagi lagi
        if (node->width < 4 || node->height < 4) {
            node->calculateAverageColor(image);
            node->isLeaf = true;
            return;
        }
        
        // Jika node berada di luar batas gambar
        if (node->x >= image.cols || node->y >= image.rows) {
            node->isLeaf = true;
            return;
        }
        
        // Ambil bagian gambar yang valid
        int endX = std::min(node->x + node->width, image.cols);
        int endY = std::min(node->y + node->height, image.rows);
        int startX = std::max(0, node->x);
        int startY = std::max(0, node->y);
        
        if (startX >= endX || startY >= endY) {
            node->isLeaf = true;
            return;
        }
        
        Rect rect(startX, startY, endX - startX, endY - startY);
        
        // Hitung error dan check subdivisi
        double error;
        try {
            if (rect.width <= 0 || rect.height <= 0 || 
                rect.x + rect.width > image.cols || rect.y + rect.height > image.rows) {
                node->isLeaf = true;
                return;
            }
            
            Mat block = image(rect);
            node->calculateAverageColor(image);
            
            // Perhitungan error untuk blok kecil dengan pendekatan khusus
            if (rect.width * rect.height <= 16) { // Ukuran blok 4x4 atau lebih kecil
                // Gunakan metode MaxPixelDiff untuk blok kecil karena lebih stabil
                error = calculateMaxPixelDiff(block) * 0.5;
            } else if (errorMethod == ErrorMethod::SSIM) {
                Mat avgBlock = Mat(block.size(), block.type(), 
                            Scalar(node->avgColor[0], node->avgColor[1], node->avgColor[2]));
                error = calculateError(block, avgBlock);
            } else {
                error = calculateError(block);
            }
        } catch (const cv::Exception& e) {
            cout << "Warning: " << e.what() << endl;
            node->isLeaf = true;
            return;
        }
        
        // Sesuaikan threshold berdasarkan ukuran blok
        double adjusted_threshold = threshold;
        if (rect.width * rect.height <= 36) { // 6x6 atau lebih kecil
            adjusted_threshold = threshold * 1.5; // Lebih toleran terhadap error untuk blok kecil
        }
        
        if (error < adjusted_threshold) {
            node->isLeaf = true;
            return;
        }
        
        // Cek apakah ukuran anak-anak valid
        int halfWidth = max(2, node->width / 2);
        int halfHeight = max(2, node->height / 2);
        
        // Jika ukuran anak terlalu kecil, jangan bagi lagi
        if (halfWidth < 2 || halfHeight < 2) {
            node->isLeaf = true;
            return;
        }
        
        node->isLeaf = false;
        nodeCounter += 4;
        
        bool shouldCaptureFrame = visualizeGif && (depth <= 2 || depth == 4 || depth == 6);
        if (shouldCaptureFrame) {
            Mat visImage = image.clone();
            rectangle(visImage, rect, Scalar(0, 0, 255), 2);
            captureFrameForGif(visImage);
        }
        
        // Buat node anak dengan ukuran minimum 2x2
        node->children[0] = new QuadtreeNode(node->x, node->y, halfWidth, halfHeight);
        node->children[1] = new QuadtreeNode(node->x + halfWidth, node->y, max(2, node->width - halfWidth), halfHeight);
        node->children[2] = new QuadtreeNode(node->x, node->y + halfHeight, halfWidth, max(2, node->height - halfHeight));
        node->children[3] = new QuadtreeNode(node->x + halfWidth, node->y + halfHeight, max(2, node->width - halfWidth), max(2, node->height - halfHeight));
        
        // Rekursi untuk setiap anak
        for (int i = 0; i < 4; i++) {
            quadtreeCompress(image, node->children[i], depth + 1);
            if (timeoutFlag) break;
        }
    } else {
        // KODE ORIGINAL UNTUK UKURAN > 2
        if (depth > maxDepth || node->width <= minBlockSize || node->height <= minBlockSize) {
            node->calculateAverageColor(image);
            node->isLeaf = true;
            return;
        }
        
        if (node->x >= image.cols || node->y >= image.rows) {
            node->isLeaf = true;
            return;
        }
        
        bool shouldCaptureFrame = visualizeGif && (depth <= 2 || depth == 4 || depth == 6);
        
        int endX = std::min(node->x + node->width, image.cols);
        int endY = std::min(node->y + node->height, image.rows);
        int startX = std::max(0, node->x);
        int startY = std::max(0, node->y);
        
        if (startX >= endX || startY >= endY) {
            node->isLeaf = true;
            return;
        }
        
        Rect rect(startX, startY, endX - startX, endY - startY);
        
        double error;
        try {
            if (rect.width <= 0 || rect.height <= 0 || 
                rect.x + rect.width > image.cols || rect.y + rect.height > image.rows) {
                node->isLeaf = true;
                return;
            }
            
            Mat block = image(rect);
            node->calculateAverageColor(image);
            
            if (errorMethod == ErrorMethod::SSIM) {
                Mat avgBlock = Mat(block.size(), block.type(), 
                            Scalar(node->avgColor[0], node->avgColor[1], node->avgColor[2]));
                error = calculateError(block, avgBlock);
            } else {
                error = calculateError(block);
            }
        } catch (const cv::Exception& e) {
            cout << "Warning: " << e.what() << endl;
            node->isLeaf = true;
            return;
        }
        
        if (error < threshold) {
            node->isLeaf = true;
            
            if (shouldCaptureFrame) {
                Mat visImage = image.clone();
                rectangle(visImage, rect, Scalar(0, 255, 0), 1);
                captureFrameForGif(visImage);
            }
            
            return;
        }
        
        node->isLeaf = false;
        nodeCounter += 4;
        
        int halfWidth = max(1, node->width / 2);
        int halfHeight = max(1, node->height / 2);
        
        if (shouldCaptureFrame) {
            Mat visImage = image.clone();
            rectangle(visImage, rect, Scalar(0, 0, 255), 2);
            captureFrameForGif(visImage);
        }
        
        node->children[0] = new QuadtreeNode(node->x, node->y, halfWidth, halfHeight);
        node->children[1] = new QuadtreeNode(node->x + halfWidth, node->y, node->width - halfWidth, halfHeight);
        node->children[2] = new QuadtreeNode(node->x, node->y + halfHeight, halfWidth, node->height - halfHeight);
        node->children[3] = new QuadtreeNode(node->x + halfWidth, node->y + halfHeight, node->width - halfWidth, node->height - halfHeight);
        
        bool useParallel = (image.cols * image.rows > 500000) && (depth <= 1);
        
        if (useParallel) {
            vector<future<void>> futures;
            
            for (int i = 0; i < 4; i++) {
                futures.push_back(async(launch::async, [this, &image, i, node, depth]() {
                    quadtreeCompress(image, node->children[i], depth + 1);
                }));
            }
            
            for (auto& f : futures) {
                f.wait();
            }
        } else {
            for (int i = 0; i < 4; i++) {
                quadtreeCompress(image, node->children[i], depth + 1);
                
                if (i % 2 == 1 && timeoutFlag) {
                    break;
                }
            }
        }
    }
}

void Quadtree::compressImage() {
    cout << "Compressing image using Quadtree..." << endl;
    
    useHybridCompression = false;
    forceLowCompression = false;
    timeoutFlag = false;
    nodeCounter = 0;
    maxDepth = 10;
    gifFrames.clear();
    
    auto startTime = std::chrono::high_resolution_clock::now();
    const auto timeoutDuration = std::chrono::milliseconds(600);
    
    thread timeoutThread([this, startTime, timeoutDuration]() {
        while (!timeoutFlag) {
            this_thread::sleep_for(chrono::milliseconds(100));
            auto currentTime = chrono::high_resolution_clock::now();
            if (chrono::duration_cast<chrono::milliseconds>(currentTime - startTime) > timeoutDuration) {
                timeoutFlag = true;
                break;
            }
        }
    });
    timeoutThread.detach();
    
    if (targetCompressionPct > 0.0) {
        if (sourceImage.rows * sourceImage.cols > 1000000) {
            Mat scaledImage;
            double scale = 0.5;
            resize(sourceImage, scaledImage, Size(), scale, scale, INTER_AREA);
            
            adjustThresholdForTargetCompression(scaledImage);
        } else {
            adjustThresholdForTargetCompression(sourceImage);
        }
    }
    
    if (visualizeGif) {
        captureFrameForGif(sourceImage);
    }
    
    try {
        cout << "Starting compression with threshold: " << threshold << endl;
        cout << "Method: " << getErrorMethodName(errorMethod) << endl;
        
        if (root) {
            deleteTree(root);
        }
        root = new QuadtreeNode(0, 0, sourceImage.cols, sourceImage.rows);
        
        bool useParallel = sourceImage.rows * sourceImage.cols > 500000;
        
        if (useParallel) {
            int halfWidth = max(1, sourceImage.cols / 2);
            int halfHeight = max(1, sourceImage.rows / 2);
            
            root->isLeaf = false;
            root->children[0] = new QuadtreeNode(0, 0, halfWidth, halfHeight);
            root->children[1] = new QuadtreeNode(halfWidth, 0, sourceImage.cols - halfWidth, halfHeight);
            root->children[2] = new QuadtreeNode(0, halfHeight, halfWidth, sourceImage.rows - halfHeight);
            root->children[3] = new QuadtreeNode(halfWidth, halfHeight, sourceImage.cols - halfWidth, sourceImage.rows - halfHeight);
            
            vector<future<void>> futures;
            for (int i = 0; i < 4; i++) {
                futures.push_back(async(launch::async, [this, i]() {
                    quadtreeCompress(sourceImage, root->children[i], 1);
                }));
            }
            
            for (auto& f : futures) {
                f.wait();
            }
        } else {
            quadtreeCompress(sourceImage, root);
        }
        
        cout << "Quadtree compression completed successfully" << endl;
        
        if (timeoutFlag) {
            cout << "Note: Compression was stopped early due to timeout" << endl;
        }
    } catch (const std::exception& e) {
        cout << "Error during compression: " << e.what() << endl;
    }
    
    if (visualizeGif) {
        Mat finalImage = sourceImage.clone();
        drawQuadtreeVisualization(finalImage, root, 0);
        gifFrames.push_back(finalImage);
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    
    cout << "Compression complete with threshold: " << threshold 
         << " in " << duration.count() << " ms" << endl;
}

void Quadtree::reconstructHelper(Mat& image, QuadtreeNode* node) {
    if (!node) return;
    
    if (node->isLeaf) {
        int endX = std::min(node->x + node->width, image.cols);
        int endY = std::min(node->y + node->height, image.rows);
        int startX = std::max(0, node->x);
        int startY = std::max(0, node->y);
        
        if (startX >= endX || startY >= endY) return;
        
        Rect region(startX, startY, endX - startX, endY - startY);
        
        if (region.x >= 0 && region.y >= 0 && 
            region.x + region.width <= image.cols && 
            region.y + region.height <= image.rows) {
            
            // Pendekatan berbasis rectangle untuk semua ukuran blok
            rectangle(image, region, Scalar(node->avgColor[0], node->avgColor[1], node->avgColor[2]), FILLED);
        }
    } else {
        for (int i = 0; i < 4; ++i) {
            if (node->children[i]) {
                reconstructHelper(image, node->children[i]);
            }
        }
    }
}

void Quadtree::reconstructImage(Mat& image) {
    // Buat gambar kosong
    image = Mat::zeros(sourceImage.size(), sourceImage.type());
    
    if (minBlockSize == 2) {
        // Pendekatan alternatif untuk minBlockSize 2: kita langsung buat blok lebih besar
        // Daripada mengikuti quadtree asli yang terlalu detail
        int blockSize = 16; // Ukuran blok yang lebih besar
        
        for (int y = 0; y < image.rows; y += blockSize) {
            for (int x = 0; x < image.cols; x += blockSize) {
                // Tentukan ukuran blok yang valid (untuk menangani tepi gambar)
                int width = min(blockSize, image.cols - x);
                int height = min(blockSize, image.rows - y);
                
                if (width <= 0 || height <= 0) continue;
                
                // Ambil area dari gambar sumber
                Rect region(x, y, width, height);
                Mat block = sourceImage(region);
                
                // Hitung warna rata-rata untuk blok ini
                Scalar avgColor = mean(block);
                
                // Isi blok dengan warna rata-rata
                rectangle(image, region, avgColor, FILLED);
            }
        }
        
        // Optional: Gunakan median blur untuk menghilangkan noise
        medianBlur(image, image, 3);
    } else {
        // Gunakan pendekatan quadtree normal untuk ukuran blok lebih besar
        reconstructHelper(image, root);
    }
}

int Quadtree::countLeafNodes(QuadtreeNode* node) {
    if (!node) return 0;
    if (node->isLeaf) return 1;
    
    int count = 0;
    for (int i = 0; i < 4; i++) {
        count += countLeafNodes(node->children[i]);
    }
    return count;
}

double Quadtree::calculateCompressionPercentage(const string& originalImagePath, const string& compressedImagePath) {
    try {
        uintmax_t originalSize = fs::file_size(originalImagePath);
        uintmax_t compressedSize = 0;
        
        if (fs::exists(compressedImagePath)) {
            compressedSize = fs::file_size(compressedImagePath);
        }
        
        if (originalSize > 0 && compressedSize > 0) {
            double fileCompressionPct = (1.0 - (double)compressedSize / originalSize) * 100.0;
            
            cout << "Perhitungan kompresi berdasarkan ukuran file:" << endl;
            cout << "  Original: " << originalSize << " bytes" << endl;
            cout << "  Compressed: " << compressedSize << " bytes" << endl;
            cout << "  Persentase kompresi: " << fileCompressionPct << "%" << endl;
            
            return fileCompressionPct;
        } else {
            int totalPixels = sourceImage.rows * sourceImage.cols;
            int leafNodes = countLeafNodes(root);
            double nodeCompressionPct = (1.0 - (double)leafNodes / totalPixels) * 100.0;
            
            cout << "  Tidak dapat mendapatkan ukuran file, menggunakan kompresi berbasis node: " << nodeCompressionPct << "%" << endl;
            return nodeCompressionPct;
        }
    } catch (const exception& e) {
        cout << "Error calculating compression percentage: " << e.what() << endl;
        
        int totalPixels = sourceImage.rows * sourceImage.cols;
        int leafNodes = countLeafNodes(root);
        return (1.0 - (double)leafNodes / totalPixels) * 100.0;
    }
}

bool Quadtree::saveGifAnimation(const string& outputPath) {
    if (!visualizeGif || gifFrames.empty()) {
        cout << "No frames available for animation." << endl;
        return false;
    }
    
    fs::path outputFilePath;
    
    if (fs::path(outputPath).is_absolute()) {
        outputFilePath = fs::path(outputPath);
    } else {
        try {
            outputFilePath = fs::absolute(fs::path(outputPath));
        } catch (const fs::filesystem_error& e) {
            outputFilePath = fs::current_path() / fs::path(outputPath);
        }
    }
    
    if (outputFilePath.extension() != ".gif") {
        outputFilePath = outputFilePath.replace_extension(".gif");
    }
    
    try {
        fs::create_directories(outputFilePath.parent_path());
        cout << "Output directory prepared: " << outputFilePath.parent_path().string() << endl;
    } catch (const fs::filesystem_error& e) {
        cout << "Error creating output directory: " << e.what() << endl;
        return false;
    }
    
    string finalOutputPath;
    
    #ifdef _WIN32
    finalOutputPath = "\"" + outputFilePath.string() + "\"";
    #else
    string unixPath = outputFilePath.string();
    size_t pos = 0;
    while ((pos = unixPath.find(" ", pos)) != string::npos) {
        unixPath.replace(pos, 1, "\\ ");
        pos += 2;
    }
    finalOutputPath = unixPath;
    #endif
    
    try {
        cout << "Creating GIF animation with " << gifFrames.size() << " frames..." << endl;
        cout << "Will save to: " << outputFilePath.string() << endl;
        
        fs::path tempDirPath;
        
        try {
            #ifdef _WIN32
            tempDirPath = fs::temp_directory_path() / ("quadtree_gif_" + to_string(time(nullptr)));
            #else
            tempDirPath = fs::temp_directory_path() / ("quadtree_gif_" + to_string(time(nullptr)));
            #endif
            
            fs::create_directories(tempDirPath);
            cout << "Created temporary directory: " << tempDirPath.string() << endl;
        } catch (const fs::filesystem_error& e) {
            tempDirPath = outputFilePath.parent_path() / "temp_frames";
            fs::create_directories(tempDirPath);
            cout << "Using local temp directory: " << tempDirPath.string() << endl;
        }
        
        vector<fs::path> frameFilePaths;
        for (size_t i = 0; i < gifFrames.size(); ++i) {
            stringstream ss;
            ss << setfill('0') << setw(4) << i;
            string frameIndex = ss.str();
            
            fs::path framePath = tempDirPath / ("frame_" + frameIndex + ".png");
            bool writeSuccess = imwrite(framePath.string(), gifFrames[i]);
            
            if (!writeSuccess) {
                cout << "Error writing frame to: " << framePath.string() << endl;
                continue;
            }
            
            frameFilePaths.push_back(framePath);
        }
        
        if (frameFilePaths.empty()) {
            cout << "Failed to save any frames." << endl;
            return false;
        }
        
        cout << "Saved " << frameFilePaths.size() << " frames to temporary directory." << endl;
        
        string tempDirPathCmd;
        #ifdef _WIN32
        tempDirPathCmd = "\"" + tempDirPath.string() + "\\frame_%04d.png\"";
        #else
        string unixTempPath = tempDirPath.string();
        size_t pos = 0;
        while ((pos = unixTempPath.find(" ", pos)) != string::npos) {
            unixTempPath.replace(pos, 1, "\\ ");
            pos += 2;
        }
        tempDirPathCmd = unixTempPath + "/frame_%04d.png";
        #endif
        
        string ffmpegCmd = "ffmpeg -y -f image2 -framerate 3 -i " + tempDirPathCmd + 
                           " -vf \"split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse\" -loop 0 " + finalOutputPath;
        
        cout << "Executing: " << ffmpegCmd << endl;
        int ffmpegResult = system(ffmpegCmd.c_str());
        
        if (ffmpegResult == 0 && fs::exists(outputFilePath) && fs::file_size(outputFilePath) > 0) {
            cout << "GIF successfully created with ffmpeg!" << endl;
            cout << "Location: " << outputFilePath.string() << endl;
            cout << "Size: " << fs::file_size(outputFilePath) << " bytes" << endl;
            
            for (const auto& path : frameFilePaths) {
                try {
                    fs::remove(path);
                } catch (...) {}
            }
            try {
                fs::remove_all(tempDirPath);
            } catch (...) {}
            
            return true;
        }
        
        cout << "First ffmpeg attempt failed. Trying simpler command..." << endl;
        
        ffmpegCmd = "ffmpeg -y -f image2 -framerate 3 -i " + tempDirPathCmd + " -loop 0 " + finalOutputPath;
        
        cout << "Executing: " << ffmpegCmd << endl;
        ffmpegResult = system(ffmpegCmd.c_str());
        
        if (ffmpegResult == 0 && fs::exists(outputFilePath) && fs::file_size(outputFilePath) > 0) {
            cout << "GIF successfully created with simplified ffmpeg command!" << endl;
            cout << "Location: " << outputFilePath.string() << endl;
            cout << "Size: " << fs::file_size(outputFilePath) << " bytes" << endl;
            
            for (const auto& path : frameFilePaths) {
                try {
                    fs::remove(path);
                } catch (...) {}
            }
            try {
                fs::remove_all(tempDirPath);
            } catch (...) {}
            
            return true;
        }
        
        fs::path framesDir = outputFilePath.parent_path() / (outputFilePath.stem().string() + "_frames");
        
        try {
            fs::create_directories(framesDir);
        } catch (const fs::filesystem_error& e) {
            cout << "Error creating frames directory: " << e.what() << endl;
            return false;
        }
        
        for (size_t i = 0; i < frameFilePaths.size(); ++i) {
            fs::path destPath = framesDir / frameFilePaths[i].filename();
            
            try {
                fs::copy_file(frameFilePaths[i], destPath, fs::copy_options::overwrite_existing);
            } catch (const fs::filesystem_error& e) {
                cout << "Error copying frame: " << e.what() << endl;
            }
        }
        
        fs::path readmePath = framesDir / "README.txt";
        ofstream readme(readmePath);
        
        if (readme.is_open()) {
            readme << "GIF Creation Instructions" << endl;
            readme << "=========================" << endl << endl;
            
            readme << "Proses visualisasi kompresi yang telah dibuat" << frameFilePaths.size() 
                   << " frame-frame yang perlu digabungkan menjadi sebuah GIF" << endl << endl;
            
            readme << "Untuk menyajikan GIF dengan ffmpeg:" << endl;
            readme << "1. Install ffmpeg" << endl;
            readme << "2. Jalankan perintah ini di direktori ini:" << endl;
            readme << "ffmpeg -framerate 3 -i frame_%04d.png -loop 0 ../" 
                   << outputFilePath.filename().string() << endl << endl;
            
            readme.close();
            cout << "Saved instructions to: " << readmePath.string() << endl;
        }
        
        cout << "Individual frames saved to: " << framesDir.string() << endl;        
        try {
            fs::remove_all(tempDirPath);
        } catch (...) {}
        
        return false;
    } catch (const exception& e) {
        cout << "Error in GIF creation process: " << e.what() << endl;
        return false;
    }
}

int Quadtree::getTreeDepthHelper(QuadtreeNode* node) {
    if (!node) return 0;
    if (node->isLeaf) return 1;
    
    int maxDepth = 0;
    for (int i = 0; i < 4; i++) {
        maxDepth = max(maxDepth, getTreeDepthHelper(node->children[i]));
    }
    
    return 1 + maxDepth;
}

int Quadtree::getNodeCountHelper(QuadtreeNode* node) {
    if (!node) return 0;
    
    int count = 1;
    if (!node->isLeaf) {
        for (int i = 0; i < 4; i++) {
            count += getNodeCountHelper(node->children[i]);
        }
    }
    
    return count;
}

int Quadtree::getTreeDepth() {
    return getTreeDepthHelper(root);
}

int Quadtree::getNodeCount() {
    return getNodeCountHelper(root);
}