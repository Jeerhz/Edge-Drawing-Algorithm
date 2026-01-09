#pragma once

#include <opencv2/opencv.hpp>
#include <cmath>
#include "Chain.h"

#define ANCHOR_PIXEL 254
#define EDGE_PIXEL 255

// Follow recommandations and set the number of false alarms to 1 per image
#define EPSILON 1.0
#define MAX_GRAD_VALUE 128 * 256

// Sobel is the default value for ED
// Prewitt operator is the default value for EDPF (changing to Sobel needs to modify gradient threshold due to quantization differences)
enum GradientOperator
{
    PREWITT_OPERATOR = 101,
    SOBEL_OPERATOR = 102,
    LSD_OPERATOR = 103
};

class ED
{

public:
    ED(cv::Mat _srcImage, GradientOperator _gradOperator, int _gradThresh = 20, int _anchorThresh = 0, int _minPathLen = 10, double _sigma = 1.0, bool _sumFlag = true);
    ED(const ED &cpyObj);
    ED();

    cv::Mat getEdgeImage();
    cv::Mat getAnchorImage();
    cv::Mat getSmoothImage();
    cv::Mat getGradImage();

    std::vector<std::vector<cv::Point>> getSegmentPoints();

protected:
    int image_width;      // width of source image
    int image_height;     // height of source image
    uchar *srcImgPointer; // pointer to source image data
    std::vector<std::vector<cv::Point>> segmentPoints;
    double sigma;            // Gaussian sigma
    cv::Mat smoothImage;     // smoothed image after applying Gaussian filter
    uchar *edgeImgPointer;   // pointer to edge image data
    uchar *smoothImgPointer; // pointer to smoothed image data (gaussian applied)
    short *gradImgPointer;   // pointer to gradient image data
    int minPathLen;          // minimum length of an anchor chain
    cv::Mat srcImage;        // source image

    // for EDColor
    uchar *smoothB_ptr;
    uchar *smoothG_ptr;
    uchar *smoothR_ptr;

private:
    void ComputeGradient();
    void ComputeGradientMapByDiZenzo();
    void ComputeAnchorPoints();
    void JoinAnchorPointsUsingSortedAnchors();
    void exploreChain(StackNode &current_node, Chain *current_chain, int &total_pixels_in_anchor_chain);
    void extractSegmentsFromChain(Chain *chain, std::vector<std::vector<cv::Point>> &anchorSegments);
    void extractSecondChildChains(Chain *anchor_chain_root, std::vector<cv::Point> &anchorSegment);
    void extractFirstChildChains(Chain *anchor_chain_root, std::vector<cv::Point> &anchorSegment);
    void extractOtherChains(Chain *anchor_chain_root, std::vector<std::vector<cv::Point>> &anchorSegments);
    int *sortAnchorsByGradValue();

    void cleanUpSurroundingAnchorPixels(StackNode &current_node);
    StackNode getNextChainPixel(StackNode &current_node);
    bool validateNode(StackNode &node);
    bool areNeighbors(int offset1, int offset2);
    void cleanUpPenultimateSegmentPixel(Chain *chain, std::vector<cv::Point> &anchorSegment, bool is_first_child);
    void revertChainEdgePixel(Chain *&chain);

    int anchorNb;
    std::vector<cv::Point> anchorPoints;

    cv::Mat edgeImage;
    cv::Mat gradImage;
    // used for color images processing
    cv::Mat smooth_R;
    cv::Mat smooth_G;
    cv::Mat smooth_B;

    GradOrientation *gradOrientationImgPointer; // pointer to direction image data. Used only in constructor for computing edges and segments
    ProcessStack process_stack;                 // stack for processing edge pixels during anchor joining

    int gradThresh;                // gradient threshold
    int anchorThresh;              // anchor point threshold
    GradientOperator gradOperator; // mask used in gradient calculation
    bool sumFlag;                  // flag for using sum of terms to compute gradient magnitude
};