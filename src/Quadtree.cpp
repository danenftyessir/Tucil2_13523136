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
    avgColor = Vec3b(128, 128, 128); // Initialize with a neutral gray instead of black
}

void QuadtreeNode::calculateAverageColor(const Mat& image) {
    int endX = std::min(x + width, image.cols);
    int endY = std::min(y + height, image.rows);
    int startX = std::max(0, x);
    int startY = std::max(0, y);
    
    // Check if the region is valid
    if (startX >= endX || startY >= endY || startX >= image.cols || startY >= image.rows) {
        // Instead of black, use a distinctive color for debug or inherit from parent
        avgColor = Vec3b(128, 128, 128); // Neutral gray
        return;
    }
    
    Vec3d sum(0, 0, 0);
    int count = 0;
    
    // Use different variable names to avoid confusion with member variables
    for (int py = startY; py < endY; py++) {
        for (int px = startX; px < endX; px++) {
            if (px < 0 || px >= image.cols || py < 0 || py >= image.rows) {
                continue; // Skip invalid coordinates
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
        // If somehow we still have no valid pixels, use a fallback color
        avgColor = Vec3b(200, 200, 200); // Light gray as fallback
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
    } else if (frameCounter % 40 == 0 && gifFrames.size() < 15) {
        shouldCapture = true;
    } else if (frameCounter % 80 == 0 && gifFrames.size() < 30) {
        shouldCapture = true;
    }
    
    if (!shouldCapture) return;
    
    int targetWidth = 640;
    int targetHeight = 480;
    
    double scaleX = static_cast<double>(targetWidth) / max(1, currentImage.cols);
    double scaleY = static_cast<double>(targetHeight) / max(1, currentImage.rows);
    double scale = min(scaleX, scaleY);
    
    Mat frame;
    if (scale < 1.0) {
        resize(currentImage, frame, Size(), scale, scale, INTER_AREA);
    } else {
        frame = currentImage.clone();
    }
    
    string infoText = "Frame " + to_string(gifFrames.size() + 1);
    putText(frame, infoText, Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,255), 1);
    
    gifFrames.push_back(frame);
}

void Quadtree::drawQuadtreeVisualization(Mat& image, QuadtreeNode* node, int depth) {
    if (!node || depth > 10) return;
    
    Rect rect(node->x, node->y, node->width, node->height);
    
    // Make sure the rectangle is within image bounds
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
        
        // Draw the lines that divide the rectangle
        line(image, Point(midX, rect.y), Point(midX, rect.y + rect.height), color, 1);
        line(image, Point(rect.x, midY), Point(rect.x + rect.width, midY), color, 1);
        
        // Recursively draw children, but limit depth for performance
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
    const int MAX_NODES = 150000; // Increased max nodes for better quality
    
    if (timeoutFlag || nodeCounter > MAX_NODES) return;
    if (!node) return;
    
    // Make sure node coordinates are valid
    if (node->x < 0 || node->y < 0 || node->x >= image.cols || node->y >= image.rows) {
        node->isLeaf = true;
        return;
    }
    
    if ((useHybridCompression || forceLowCompression) && targetCompressionPct <= 0.0) {
        useHybridCompression = false;
        forceLowCompression = false;
    }
    
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
        
        if (visualizeGif && (depth <= 1 || depth == 2)) {
            Rect rect(node->x, node->y, node->width, node->height);
            // Ensure rect is within image bounds
            rect = rect & Rect(0, 0, image.cols, image.rows);
            if (rect.width > 0 && rect.height > 0) {
                Mat visImage = image.clone();
                rectangle(visImage, rect, Scalar(0, 0, 255), 2);
                captureFrameForGif(visImage);
            }
        }
        
        // Create children with validated dimensions
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
        
        if (visualizeGif && (depth <= 1 || depth == 2)) {
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
    
    // Normal mode processing
    if (depth > maxDepth || node->width <= minBlockSize || node->height <= minBlockSize) {
        node->calculateAverageColor(image);
        node->isLeaf = true;
        return;
    }
    
    // Validate node position
    if (node->x >= image.cols || node->y >= image.rows) {
        node->isLeaf = true;
        return;
    }
    
    bool shouldCaptureFrame = visualizeGif && (depth <= 1 || depth == 3 || depth == 5);
    
    // Calculate valid region coordinates
    int endX = std::min(node->x + node->width, image.cols);
    int endY = std::min(node->y + node->height, image.rows);
    int startX = std::max(0, node->x);
    int startY = std::max(0, node->y);
    
    // Skip if region is invalid
    if (startX >= endX || startY >= endY) {
        node->isLeaf = true;
        return;
    }
    
    Rect rect(startX, startY, endX - startX, endY - startY);
    
    double error;
    try {
        // Final validation check
        if (rect.width <= 0 || rect.height <= 0 || 
            rect.x + rect.width > image.cols || rect.y + rect.height > image.rows) {
            node->isLeaf = true;
            return;
        }
        
        // Get the block and calculate error
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
    
    // If error is below threshold, this is a leaf node
    if (error < threshold) {
        node->isLeaf = true;
        
        if (shouldCaptureFrame) {
            captureFrameForGif(image);
        }
        
        return;
    }
    
    // Otherwise, subdivide into children
    node->isLeaf = false;
    nodeCounter += 4;
    
    int halfWidth = max(1, node->width / 2);
    int halfHeight = max(1, node->height / 2);
    
    if (shouldCaptureFrame) {
        Mat visImage = image.clone();
        rectangle(visImage, rect, Scalar(0, 0, 255), 2);
        captureFrameForGif(visImage);
    }
    
    // Create children nodes
    node->children[0] = new QuadtreeNode(node->x, node->y, halfWidth, halfHeight);
    node->children[1] = new QuadtreeNode(node->x + halfWidth, node->y, node->width - halfWidth, halfHeight);
    node->children[2] = new QuadtreeNode(node->x, node->y + halfHeight, halfWidth, node->height - halfHeight);
    node->children[3] = new QuadtreeNode(node->x + halfWidth, node->y + halfHeight, node->width - halfWidth, node->height - halfHeight);
    
    // Process children in parallel for large images at top level
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
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    
    cout << "Compression complete with threshold: " << threshold 
         << " in " << duration.count() << " ms" << endl;
}

void Quadtree::reconstructHelper(Mat& image, QuadtreeNode* node) {
    if (!node) return;
    
    if (node->isLeaf) {
        // Calculate valid coordinates within image bounds
        int endX = std::min(node->x + node->width, image.cols);
        int endY = std::min(node->y + node->height, image.rows);
        int startX = std::max(0, node->x);
        int startY = std::max(0, node->y);
        
        // Skip invalid regions
        if (startX >= endX || startY >= endY) return;
        
        // Define the rectangle to fill
        Rect region(startX, startY, endX - startX, endY - startY);
        
        // Final validation check
        if (region.x >= 0 && region.y >= 0 && 
            region.x + region.width <= image.cols && 
            region.y + region.height <= image.rows) {
            // Fill the rectangle with the average color
            rectangle(image, region, Scalar(node->avgColor[0], node->avgColor[1], node->avgColor[2]), FILLED);
        }
    } else {
        // Process all valid children recursively
        for (int i = 0; i < 4; ++i) {
            if (node->children[i]) {
                reconstructHelper(image, node->children[i]);
            }
        }
    }
}

void Quadtree::reconstructImage(Mat& image) {
    // Start with a blank canvas of the same size as the original
    image = Mat::zeros(sourceImage.size(), sourceImage.type());
    
    // Fill the image by recursively processing the quadtree
    reconstructHelper(image, root);
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
    
    try {
        cout << "Creating animation with " << gifFrames.size() << " frames..." << endl;
        
        VideoWriter writer;
        int codec = VideoWriter::fourcc('M', 'J', 'P', 'G');
        double fps = 2.0;
        
        Size frameSize = gifFrames[0].size();
        writer.open(outputPath, codec, fps, frameSize);
        
        if (!writer.isOpened()) {
            cout << "Could not create animation file." << endl;
            return false;
        }
        
        int total = gifFrames.size();
        for (int i = 0; i < total; i++) {
            writer.write(gifFrames[i]);
            
            if (i % max(1, total/5) == 0 || i == total-1) {
                cout << "Writing frame " << i+1 << "/" << total << endl;
            }
        }
        
        writer.release();
        cout << "Animation saved to: " << outputPath << endl;
        return true;
    } catch (const exception& e) {
        cout << "Error creating animation: " << e.what() << endl;
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