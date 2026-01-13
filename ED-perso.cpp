#include "ED-perso.h"
#include "Chain.h"
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace cv;
using namespace std;

ED::ED(cv::Mat _srcImage, GradientOperator _gradOperator, int _gradThresh, int _anchorThresh, double _sigma, bool _sumFlag)
{
    srcImage = _srcImage;
    // detect if input is grayscale or BGR and prepare per-channel buffers for later use (Di Zenzo)
    image_height = srcImage.rows;
    image_width = srcImage.cols;
    gradOperator = _gradOperator;
    gradThresh = _gradThresh;
    anchorThresh = _anchorThresh;
    sigma = _sigma;
    sumFlag = _sumFlag;
    process_stack = ProcessStack();
    segmentPoints = vector<vector<Point>>();
    edgeImage = Mat(image_height, image_width, CV_8UC1, Scalar(0));
    srcImgPointer = srcImage.data;
    gradImage = Mat(image_height, image_width, CV_16SC1);
    gradImgPointer = (short *)gradImage.data;
    edgeImgPointer = edgeImage.data;
    gradOrientationImgPointer = new GradOrientation[image_width * image_height];

    bool isColorImage = (srcImage.channels() == 3);
    std::cout << "Input image is " << (isColorImage ? "color" : "grayscale") << std::endl;

    if (isColorImage)
    {
        if (srcImage.type() != CV_8UC3)
            srcImage.convertTo(srcImage, CV_8UC3);

        std::vector<cv::Mat> ch(3);
        cv::split(srcImage, ch);

        smooth_B = Mat(image_height, image_width, CV_8UC1);
        smooth_G = Mat(image_height, image_width, CV_8UC1);
        smooth_R = Mat(image_height, image_width, CV_8UC1);

        if (sigma == 1.0)
        {
            GaussianBlur(ch[0], smooth_B, Size(5, 5), sigma);
            GaussianBlur(ch[1], smooth_G, Size(5, 5), sigma);
            GaussianBlur(ch[2], smooth_R, Size(5, 5), sigma);
        }
        else
        {
            GaussianBlur(ch[0], smooth_B, Size(), sigma);
            GaussianBlur(ch[1], smooth_G, Size(), sigma);
            GaussianBlur(ch[2], smooth_R, Size(), sigma);
        }

        smoothR_ptr = smooth_R.data;
        smoothG_ptr = smooth_G.data;
        smoothB_ptr = smooth_B.data;

        ComputeGradientMapByDiZenzo();
    }

    else
    {
        smoothImage = Mat(image_height, image_width, CV_8UC1);

        if (sigma == 1.0)
            GaussianBlur(srcImage, smoothImage, Size(5, 5), sigma);
        else
            GaussianBlur(srcImage, smoothImage, Size(), sigma);

        smoothImgPointer = smoothImage.data;
        std::cout << "Computing gradient map..." << std::endl;
        ComputeGradient();
    }

    ComputeAnchorPoints();
    JoinAnchorPointsUsingSortedAnchors();

    delete[] gradOrientationImgPointer;
}

// needed for EDLines constructor
ED::ED(const ED &cpyObj)
{
    image_height = cpyObj.image_height;
    image_width = cpyObj.image_width;

    srcImage = cpyObj.srcImage.clone();

    gradThresh = cpyObj.gradThresh;
    anchorThresh = cpyObj.anchorThresh;
    sigma = cpyObj.sigma;
    sumFlag = cpyObj.sumFlag;

    edgeImage = cpyObj.edgeImage.clone();
    smoothImage = cpyObj.smoothImage.clone();
    gradImage = cpyObj.gradImage.clone();

    srcImgPointer = srcImage.data;

    smoothImgPointer = smoothImage.data;
    gradImgPointer = (short *)gradImage.data;
    edgeImgPointer = edgeImage.data;

    segmentPoints = cpyObj.segmentPoints;
}

ED::ED()
{
}

Mat ED::getEdgeImage()
{
    return edgeImage;
}

Mat ED::getAnchorImage()
{
    Mat anchorImage = Mat(edgeImage.size(), edgeImage.type(), Scalar(0));
    for (const Point &p : anchorPoints)
        anchorImage.at<uchar>(p) = 255;
    return anchorImage;
}

Mat ED::getSmoothImage()
{
    return smoothImage;
}

Mat ED::getGradImage()
{
    Mat result8UC1;
    convertScaleAbs(gradImage, result8UC1);
    return result8UC1;
}
// Compute gradient magnitude and orientation using Sobel or Prewitt operator
void ED::ComputeGradient()
{
    // Initialize gradient image for row = 0, row = height-1, column=0, column=width-1
    for (int j = 0; j < image_width; j++)
    {
        gradImgPointer[j] = gradImgPointer[(image_height - 1) * image_width + j] = gradThresh - 1;
    }
    for (int i = 1; i < image_height - 1; i++)
    {
        gradImgPointer[i * image_width] = gradImgPointer[(i + 1) * image_width - 1] = gradThresh - 1;
    }

    for (int i = 1; i < image_height - 1; i++)
    {
        for (int j = 1; j < image_width - 1; j++)
        {

            int com1 = smoothImgPointer[(i + 1) * image_width + j + 1] - smoothImgPointer[(i - 1) * image_width + j - 1];
            int com2 = smoothImgPointer[(i - 1) * image_width + j + 1] - smoothImgPointer[(i + 1) * image_width + j - 1];

            int gx;
            int gy;

            switch (gradOperator)
            {
            case PREWITT_OPERATOR:
                gx = abs(com1 + com2 + (smoothImgPointer[i * image_width + j + 1] - smoothImgPointer[i * image_width + j - 1]));
                gy = abs(com1 - com2 + (smoothImgPointer[(i + 1) * image_width + j] - smoothImgPointer[(i - 1) * image_width + j]));
                break;
            case SOBEL_OPERATOR:
                gx = abs(com1 + com2 + 2 * (smoothImgPointer[i * image_width + j + 1] - smoothImgPointer[i * image_width + j - 1]));
                gy = abs(com1 - com2 + 2 * (smoothImgPointer[(i + 1) * image_width + j] - smoothImgPointer[(i - 1) * image_width + j]));
                break;
            case LSD_OPERATOR:
                // com1 and com2 differs from previous operators, because LSD has 2x2 kernel
                int com1 = smoothImgPointer[(i + 1) * image_width + j + 1] - smoothImgPointer[i * image_width + j];
                int com2 = smoothImgPointer[i * image_width + j + 1] - smoothImgPointer[(i + 1) * image_width + j];

                gx = abs(com1 + com2);
                gy = abs(com1 - com2);
            }

            int sum;

            if (sumFlag)
                sum = gx + gy;
            else
                sum = (int)sqrt((double)gx * gx + gy * gy);

            int index = i * image_width + j;
            gradImgPointer[index] = sum;

            if (sum >= gradThresh)
            {
                if (gx >= gy)
                    gradOrientationImgPointer[index] = EDGE_VERTICAL;
                else
                    gradOrientationImgPointer[index] = EDGE_HORIZONTAL;
            }
        }
    }
}

// Function that we we want to maximize to compute the gradient in a multi-level image using the DiZenzo method
double F(double t, int gxx, int gyy, int gxy)
{
    return gxx * cos(t) * cos(t) + 2.0 * gxy * sin(t) * cos(t) + gyy * sin(t) * sin(t);
};

// This is part of EDColor, in this variant we use BGR channels and not Lab
void ED::ComputeGradientMapByDiZenzo()
{
    // Initialize gradient buffer
    memset(gradImgPointer, 0, sizeof(short) * image_width * image_height);

    int max_val = 0;

    for (int i = 1; i < image_height - 1; ++i)
    {
        for (int j = 1; j < image_width - 1; ++j)
        {
            int idx = i * image_width + j;

            // Prewitt-like differences for R channel
            int com1 = (int)smoothR_ptr[(i + 1) * image_width + j + 1] - (int)smoothR_ptr[(i - 1) * image_width + j - 1];
            int com2 = (int)smoothR_ptr[(i - 1) * image_width + j + 1] - (int)smoothR_ptr[(i + 1) * image_width + j - 1];
            int gxR = com1 + com2 + ((int)smoothR_ptr[i * image_width + j + 1] - (int)smoothR_ptr[i * image_width + j - 1]);
            int gyR = com1 - com2 + ((int)smoothR_ptr[(i + 1) * image_width + j] - (int)smoothR_ptr[(i - 1) * image_width + j]);

            // Prewitt-like differences for G channel
            com1 = (int)smoothG_ptr[(i + 1) * image_width + j + 1] - (int)smoothG_ptr[(i - 1) *
                                                                                          image_width +
                                                                                      j - 1];
            com2 = (int)smoothG_ptr[(i - 1) * image_width + j + 1] - (int)smoothG_ptr[(i + 1) * image_width + j - 1];
            int gxG = com1 + com2 + ((int)smoothG_ptr[i * image_width + j + 1] - (int)smoothG_ptr[i * image_width + j - 1]);
            int gyG = com1 - com2 + ((int)smoothG_ptr[(i + 1) * image_width + j] - (int)smoothG_ptr[(i - 1) * image_width + j]);

            // Prewitt-like differences for B channel
            com1 = (int)smoothB_ptr[(i + 1) * image_width + j + 1] - (int)smoothB_ptr[(i - 1) * image_width + j - 1];
            com2 = (int)smoothB_ptr[(i - 1) * image_width + j + 1] - (int)smoothB_ptr[(i + 1) * image_width + j - 1];
            int gxB = com1 + com2 + ((int)smoothB_ptr[i * image_width + j + 1] - (int)smoothB_ptr[i * image_width + j - 1]);
            int gyB = com1 - com2 + ((int)smoothB_ptr[(i + 1) * image_width + j] - (int)smoothB_ptr[(i - 1) * image_width + j]);

            // Di Zenzo tensor components
            int gxx = gxR * gxR + gxG * gxG + gxB * gxB; // g11
            int gyy = gyR * gyR + gyG * gyG + gyB * gyB; // g22
            int gxy = gxR * gyR + gxG * gyG + gxB * gyB; // g12

            // atan2(Y,X) is the arctan function of Y / X and return values in the interval [−π/2, π/2].
            // We add M_PI / 2 to shift the range to [0, π]. As suggested in DiZenzo article
            double theta0 = 0.5 * atan2(2.0 * (double)gxy,
                                        (double)(gxx - gyy)) +
                            M_PI / 2.0;

            double theta1;
            // We have two candidate angles
            if (theta0 < M_PI / 2.0)
                theta1 = theta0 + M_PI / 2.0;
            else
                theta1 = theta0 - M_PI / 2.0;

            // Evaluate F at both candidate angles (once) and keep the maximum
            double F_theta0 = F(theta0, gxx, gyy, gxy);
            double F_theta1 = F(theta1, gxx, gyy, gxy);

            double val = (F_theta1 > F_theta0) ? F_theta1 : F_theta0;
            // 'Edge strength'computed as the square root of the maximum value
            int grad = (int)sqrt(std::max(0.0, val));

            // store gradient magnitude and update global max
            gradImgPointer[idx] = grad;

            // Update maximum gradient value needed for scaling
            if (grad > max_val)
                max_val = grad;

            // set orientation based on the chosen angle's components (reuse which F was larger)
            if (grad >= gradThresh)
            {
                double chosenTheta = (F_theta1 > F_theta0) ? theta1 : theta0;
                double cos_theta = cos(chosenTheta), sin_theta = sin(chosenTheta);
                gradOrientationImgPointer[idx] = (std::abs(cos_theta) >= std::abs(sin_theta)) ? EDGE_VERTICAL : EDGE_HORIZONTAL;
            }
        }
    }

    // Scale to 0-255
    double scale = (max_val > 0) ? (255.0 / max_val) : 1.0;
    for (int k = 0; k < image_width * image_height; ++k)
        gradImgPointer[k] = (short)(gradImgPointer[k] * scale);
}

void ED::ComputeAnchorPoints()
{
    for (int i = 2; i < image_height - 2; i++)
    {
        int start = 2;
        int inc = 1;

        for (int j = start; j < image_width - 2; j++)
        {
            if (gradImgPointer[i * image_width + j] < gradThresh)
                continue;

            if (gradOrientationImgPointer[i * image_width + j] == EDGE_VERTICAL)
            {
                int diff1 = gradImgPointer[i * image_width + j] - gradImgPointer[i * image_width + j - 1];
                int diff2 = gradImgPointer[i * image_width + j] - gradImgPointer[i * image_width + j + 1];
                if (diff1 >= anchorThresh && diff2 >= anchorThresh)
                {
                    edgeImgPointer[i * image_width + j] = ANCHOR_PIXEL;
                    anchorPoints.push_back(Point(j, i));
                }
            }
            else
            {
                int diff1 = gradImgPointer[i * image_width + j] - gradImgPointer[(i - 1) * image_width + j];
                int diff2 = gradImgPointer[i * image_width + j] - gradImgPointer[(i + 1) * image_width + j];
                if (diff1 >= anchorThresh && diff2 >= anchorThresh)
                {
                    edgeImgPointer[i * image_width + j] = ANCHOR_PIXEL;
                    anchorPoints.push_back(Point(j, i));
                }
            }
        }
    }
    anchorNb = anchorPoints.size();
}

// Helper to delete a chain tree given a root pointer and nulify it.
// https://stackoverflow.com/questions/60380985/c-delete-all-nodes-from-binary-tree
void RemoveAll(Chain *&chain)
{
    if (!chain)
        return;

    RemoveAll(chain->first_childChain);
    RemoveAll(chain->second_childChain);

    delete chain;
    chain = nullptr;
}

int *ED::sortAnchorsByGradValue()
{
    int SIZE = 128 * 256;
    int *C = new int[SIZE];
    memset(C, 0, sizeof(int) * SIZE);

    // Count the number of grad values
    for (int i = 1; i < image_height - 1; i++)
    {
        for (int j = 1; j < image_width - 1; j++)
        {
            if (edgeImgPointer[i * image_width + j] != ANCHOR_PIXEL)
                continue;

            int grad = gradImgPointer[i * image_width + j];
            C[grad]++;
        }
    }

    // Compute indices
    for (int i = 1; i < SIZE; i++)
        C[i] += C[i - 1];

    int noAnchors = C[SIZE - 1];
    int *A = new int[noAnchors];
    memset(A, 0, sizeof(int) * noAnchors);

    for (int i = 1; i < image_height - 1; i++)
    {
        for (int j = 1; j < image_width - 1; j++)
        {
            if (edgeImgPointer[i * image_width + j] != ANCHOR_PIXEL)
                continue;

            int grad = gradImgPointer[i * image_width + j];
            int index = --C[grad];
            A[index] = i * image_width + j; // anchor's offset
        }
    }

    delete[] C;

    return A;
}

void setChildToChain(Chain *parent, Chain *child)
{

    if (parent->first_childChain == nullptr)
    {
        parent->first_childChain = child;
        return;
    }

    parent->second_childChain = child;
    return;
}

void ED::revertChainEdgePixel(Chain *&chain)
{

    if (!chain)
        return;

    for (int pixel_index = 0; pixel_index < chain->pixels.size(); pixel_index++)
    {
        int pixel_offset = chain->pixels[pixel_index];
        edgeImgPointer[pixel_offset] = 0;
    }

    revertChainEdgePixel(chain->first_childChain);
    revertChainEdgePixel(chain->second_childChain);
}

bool ED::areNeighbors(int offset1, int offset2)
{
    int row_distance = abs(offset1 / image_width - offset2 / image_width);
    int col_distance = abs(offset1 % image_width - offset2 % image_width);
    return (row_distance <= 1 && col_distance <= 1);
}

// We take the last or first pixel of the current processed chain and clean its neighbors in the segment
void ED::cleanUpPenultimateSegmentPixel(Chain *chain, std::vector<cv::Point> &anchorSegment, bool is_first_child)
{
    if (!chain || chain->pixels.empty())
        return;

    int chain_pixel_offset = is_first_child ? chain->pixels.front() : chain->pixels.back();

    // Start with the second last pixel in the segment
    while (anchorSegment.size() > 1)
    {
        int segment_penultimate_index = anchorSegment.size() - 2;
        Point penultimate_segment_pixel = anchorSegment[segment_penultimate_index];
        if (areNeighbors(chain_pixel_offset, penultimate_segment_pixel.y * image_width + penultimate_segment_pixel.x))
            anchorSegment.pop_back();
        else
            break;
    }
}

// Backward extraction, we start from the end of the latest sub chain and move towards the anchor root
void ED::extractSecondChildChains(Chain *anchor_chain_root, std::vector<Point> &anchorSegment)
{
    if (!anchor_chain_root || !anchor_chain_root->second_childChain)
        return;

    std::pair<int, std::vector<Chain *>> resp = anchor_chain_root->second_childChain->getAllChains(true);
    std::vector<Chain *> all_second_child_chains_in_longest_path = resp.second;

    // iterate through all sub chains in the longest path, clean and add pixels to the anchor segment
    for (size_t chain_index = all_second_child_chains_in_longest_path.size() - 1; chain_index > 0; --chain_index)
    {
        Chain *chain = all_second_child_chains_in_longest_path[chain_index];
        if (!chain || chain->is_extracted)
            continue;

        cleanUpPenultimateSegmentPixel(chain, anchorSegment, false);

        // add the chain pixels to the anchor segment
        for (int pixel_index = (int)chain->pixels.size() - 1; pixel_index >= 0; --pixel_index)
        {
            int pixel_offset = chain->pixels[pixel_index];
            anchorSegment.push_back(Point(pixel_offset % image_width, pixel_offset / image_width));
        }
        chain->is_extracted = true;
    }
}

// Forward extraction, we start from the anchor root, and go deeper
void ED::extractFirstChildChains(Chain *anchor_chain_root, std::vector<Point> &anchorSegment)
{
    if (!anchor_chain_root || !anchor_chain_root->first_childChain)
        return;

    std::pair<int, std::vector<Chain *>> resp = anchor_chain_root->first_childChain->getAllChains(true);
    std::vector<Chain *> all_first_child_chains_in_longest_path = resp.second;

    for (size_t chain_index = 0; chain_index < all_first_child_chains_in_longest_path.size(); ++chain_index)
    {
        Chain *chain = all_first_child_chains_in_longest_path[chain_index];
        if (!chain || chain->is_extracted)
            continue;

        cleanUpPenultimateSegmentPixel(chain, anchorSegment, true);

        for (size_t pixel_index = 0; pixel_index < chain->pixels.size(); ++pixel_index)
        {
            int pixel_offset = chain->pixels[pixel_index];
            anchorSegment.push_back(Point(pixel_offset % image_width, pixel_offset / image_width));
        }
        chain->is_extracted = true;
    }
}

// Extract the remaining significant chains that are not part of the main anchor segment
void ED::extractOtherChains(Chain *anchor_chain_root, std::vector<std::vector<Point>> &anchorSegments)
{
    if (!anchor_chain_root)
        return;

    std::pair<int, std::vector<Chain *>> resp_all = anchor_chain_root->getAllChains(false);
    // This is all chains in the anchor root, traversed depth-first adding the first child first.
    std::vector<Chain *> all_anchor_root_chains = resp_all.second;

    // Start the iteration from the anchor root and go deeper
    for (size_t k = 0; k < all_anchor_root_chains.size(); ++k)
    {
        Chain *other_chain = all_anchor_root_chains[k];
        if (!other_chain)
            continue;

        std::vector<Point> otherAnchorSegment;
        other_chain->pruneToLongestChain();

        std::pair<int, std::vector<Chain *>> other_resp = other_chain->getAllChains(true);
        int other_chain_total_length = other_resp.first;
        std::vector<Chain *> other_chain_chainChilds_in_longest_path = other_resp.second;

        // Check the significance of the chain in another way
        // if (other_chain_total_length < minPathLen)
        // continue;

        for (size_t chain_index = 0; chain_index < other_chain_chainChilds_in_longest_path.size(); ++chain_index)
        {
            Chain *other_chain_childChain = other_chain_chainChilds_in_longest_path[chain_index];
            if (!other_chain_childChain || other_chain_childChain->is_extracted)
                continue;

            cleanUpPenultimateSegmentPixel(other_chain_childChain, otherAnchorSegment, true);

            for (size_t pixel_index = 0; pixel_index < other_chain_childChain->pixels.size(); ++pixel_index)
            {
                int pixel_offset = other_chain_childChain->pixels[pixel_index];
                otherAnchorSegment.push_back(Point(pixel_offset % image_width, pixel_offset / image_width));
            }
            other_chain_childChain->is_extracted = true;
        }

        if (!otherAnchorSegment.empty())
            anchorSegments.push_back(otherAnchorSegment);
    }
}

void ED::extractSegmentsFromChain(Chain *anchor_chain_root, std::vector<std::vector<Point>> &anchorSegments)
{
    if (!anchor_chain_root)
        return;

    std::vector<Point> anchorSegment;

    // second child (backward)
    extractSecondChildChains(anchor_chain_root, anchorSegment);

    // first child (forward)
    extractFirstChildChains(anchor_chain_root, anchorSegment);

    // Clean possible boucle at the beginning of the segment
    if (anchorSegment.size() > 1 && areNeighbors(anchorSegment[1].y * image_width + anchorSegment[1].x,
                                                 anchorSegment.back().y * image_width + anchorSegment.back().x))
        anchorSegment.erase(anchorSegment.begin());

    // Add the main anchor segment to the anchor segments (only if non-empty)
    if (!anchorSegment.empty())
        anchorSegments.push_back(anchorSegment);

    // other long segments attached to anchor root
    extractOtherChains(anchor_chain_root, anchorSegments);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////// SUPPLEMENTARY FUNCTION FOR VERSION WITHOUT MINLENPATH ////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////

void ED::computeGradientCDF()
{
    // Cumulative distribution (CDF) of gradient magnitudes:
    // gradient_cdf[i] = proportion of pixels with gradient <= i
    gradient_cdf = new double[MAX_GRAD_VALUE];
    int *gradient_cumulative_histogram = new int[MAX_GRAD_VALUE];

    // initialize histogram to zero
    memset(gradient_cumulative_histogram, 0, sizeof(int) * MAX_GRAD_VALUE);

    for (int i = 0; i < image_width * image_height; i++)
        gradient_cumulative_histogram[gradImgPointer[i]]++;

    // Compute cumulative histogram
    for (int i = 1; i <= MAX_GRAD_VALUE; i++)
        gradient_cumulative_histogram[i] += gradient_cumulative_histogram[i - 1];

    // Compute gradient CDF array
    for (int i = 0; i <= MAX_GRAD_VALUE; i++)
        gradient_cdf[i] = ((double)gradient_cumulative_histogram[i] / (double)(image_height * image_width));

    delete[] gradient_cumulative_histogram;
}

void ED::computeNumberSegmentPieces()
{
    number_segment_pieces = 0;
    for (int i = 0; i < segmentPoints.size(); i++)
    {
        int len = (int)segmentPoints[i].size();
        number_segment_pieces += (len * (len - 1)) / 2;
    }
}

double ED::NFA(double prob, int len)
{
    double nfa = number_segment_pieces;
    for (int i = 0; i < (int)(len / 2) && nfa > EPSILON; i++)
        nfa *= prob;

    return nfa;
}

//
void ED::addPixelsToSegment(std::vector<Point> &segment, Chain *pruned_chain)
{
    if (!pruned_chain)
        return;

    std::vector<int> chain_pixels_offset = pruned_chain->pixels;

    for (size_t pixel_index = 0; pixel_index < chain_pixels_offset.size(); ++pixel_index)
    {
        int pixel_offset = chain_pixels_offset[pixel_index];
        segment.push_back(Point(pixel_offset % image_width, pixel_offset / image_width));
    }
}

//////////////////// END SUPPLEMENTARY FUNCTION FOR VERSION WITHOUT MINLENPATH ///////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////

void ED::JoinAnchorPointsUsingSortedAnchors()
{
    int *SortedAnchors = sortAnchorsByGradValue();
    for (int anchor_index = anchorNb - 1; anchor_index >= 0; anchor_index--)
    {
        int anchorPixelOffset = SortedAnchors[anchor_index];

        // Skip if already processed
        if (edgeImgPointer[anchorPixelOffset] != ANCHOR_PIXEL)
            continue;

        int total_pixels_in_anchor_chain = 0; // Count total pixels in the anchor chain and its children

        Chain *anchor_chain_root = new Chain();
        // We initialize two distinct nodes to start anchor chain exploration in order to avoid duplicate pixels from the start.
        // This is not done in the original implementation where we set the anchor point two times
        if (gradOrientationImgPointer[anchorPixelOffset] == EDGE_VERTICAL)
        {
            process_stack.push(StackNode(anchorPixelOffset, DOWN, anchor_chain_root));
            process_stack.push(StackNode(anchorPixelOffset - image_width, UP, anchor_chain_root));
        }
        else
        {
            process_stack.push(StackNode(anchorPixelOffset, RIGHT, anchor_chain_root));
            process_stack.push(StackNode(anchorPixelOffset - 1, LEFT, anchor_chain_root));
        }

        while (!process_stack.empty())
        {
            StackNode currentNode = process_stack.top();
            process_stack.pop();

            // processed stack pixel are in two chains in opposite directions, we track duplicates
            if (edgeImgPointer[currentNode.offset] != EDGE_PIXEL)
                total_pixels_in_anchor_chain--;

            Chain *new_process_stack_chain = new Chain(currentNode.node_direction, currentNode.parent_chain);
            setChildToChain(new_process_stack_chain->parent_chain, new_process_stack_chain);
            // Explore from the stack node to add more pixels to the new created chain
            exploreChain(currentNode, new_process_stack_chain, total_pixels_in_anchor_chain);
        }

        // Check significance of the chain another way
        // if (total_pixels_in_anchor_chain < minPathLen)
        //     revertChainEdgePixel(anchor_chain_root);

        anchor_chain_root->first_childChain->pruneToLongestChain();
        anchor_chain_root->second_childChain->pruneToLongestChain();
        // Create a segment corresponding to this anchor chain
        std::vector<std::vector<Point>> anchorSegments;
        extractSegmentsFromChain(anchor_chain_root, anchorSegments);
        segmentPoints.insert(segmentPoints.end(), anchorSegments.begin(), anchorSegments.end());

        RemoveAll(anchor_chain_root);
    }
    delete[] SortedAnchors;
}

// Clean pixel perpendicular to edge direction
void ED::cleanUpSurroundingAnchorPixels(StackNode &current_node)
{
    int offset = current_node.offset;
    int offset_diff = (current_node.node_direction == LEFT || current_node.node_direction == RIGHT) ? image_width : 1;

    // Left/up neighbor
    if (edgeImgPointer[offset - offset_diff] == ANCHOR_PIXEL)
        edgeImgPointer[offset - offset_diff] = 0;
    // Right/down neighbor
    if (edgeImgPointer[offset + offset_diff] == ANCHOR_PIXEL)
        edgeImgPointer[offset + offset_diff] = 0;
}

// Get next pixel in the chain based on current node direction and gradient values
StackNode ED::getNextChainPixel(StackNode &current_node)
{
    const int offset = current_node.offset;
    const Direction dir = current_node.node_direction;

    //
    const int exploration_offset_diff = (dir == LEFT) ? -1 : (dir == RIGHT) ? 1
                                                         : (dir == UP)      ? -image_width
                                                                            : image_width;
    // Perpendicular component: for vertical movement it's a column shift, for horizontal it's a row shift
    const int perpendicular_offset_diff = (dir == LEFT || dir == RIGHT) ? image_width : 1;
    const int perpendicular_steps[3] = {0, 1, -1};

    int best_grad = -1;
    int best_offset = -1;

    for (int k = 0; k < 3; ++k)
    {
        const int perpendicular_step = perpendicular_steps[k];

        int neighbor_offset = offset + exploration_offset_diff + perpendicular_step * perpendicular_offset_diff;

        bool is_neighbor_anchor = (edgeImgPointer[neighbor_offset] == ANCHOR_PIXEL), is_neighbor_edge = (edgeImgPointer[neighbor_offset] == EDGE_PIXEL);
        if (is_neighbor_anchor || is_neighbor_edge)
            return StackNode(neighbor_offset, dir, current_node.parent_chain);

        const int grad = gradImgPointer[neighbor_offset];
        if (grad > best_grad)
        {
            best_grad = grad;
            best_offset = neighbor_offset;
        }
    }

    return StackNode(best_offset, dir, current_node.parent_chain);
}

void ED::exploreChain(StackNode &current_node, Chain *current_chain, int &total_pixels_in_anchor_chain)
{

    GradOrientation chain_orientation = current_chain->direction == LEFT || current_chain->direction == RIGHT ? EDGE_HORIZONTAL : EDGE_VERTICAL;
    // Explore until we find change direction or we hit an edge pixel or the gradient is below threshold
    while (gradOrientationImgPointer[current_node.offset] == chain_orientation)
    {
        current_chain->pixels.push_back(current_node.offset);
        total_pixels_in_anchor_chain++;
        edgeImgPointer[current_node.offset] = EDGE_PIXEL;
        cleanUpSurroundingAnchorPixels(current_node);

        current_node = getNextChainPixel(current_node);

        if (edgeImgPointer[current_node.offset] == EDGE_PIXEL || gradImgPointer[current_node.offset] < gradThresh)
            return;
    }

    // We have a valid pixel which gradient orientation does not match the exploration direction
    // This does not match original implementation where this node is the starting of the perpendicular sub-chains
    // We decide to add this last pixel to the current chain and add two other distinct pixels for the new chains in order to avoid duplicate pixels
    current_chain->pixels.push_back(current_node.offset);
    total_pixels_in_anchor_chain++;
    edgeImgPointer[current_node.offset] = EDGE_PIXEL;
    cleanUpSurroundingAnchorPixels(current_node);

    // We add new nodes to the process stack in perpendicular directions to the edge with reference to this chain as a parent
    // This is different from the original implementation where the above node is the starting of the perpendicular sub-chains
    if (chain_orientation == EDGE_HORIZONTAL)
    {
        // Add UP and DOWN for horizontal chains if the pixels are valid
        // The border pixels were set to a low gradient threshold, so we do not need to check for out of bounds access
        if (edgeImgPointer[current_node.offset + image_width] == EDGE_PIXEL || gradImgPointer[current_node.offset + image_width] < gradThresh)
            process_stack.push(StackNode(current_node.offset, DOWN, current_chain));
        if (edgeImgPointer[current_node.offset - image_width] == EDGE_PIXEL || gradImgPointer[current_node.offset - image_width] < gradThresh)
            process_stack.push(StackNode(current_node.offset, UP, current_chain));
    }
    else
    {
        // Add LEFT and RIGHT for vertical chains
        if (edgeImgPointer[current_node.offset + 1] == EDGE_PIXEL || gradImgPointer[current_node.offset + 1] < gradThresh)
            process_stack.push(StackNode(current_node.offset, RIGHT, current_chain));
        if (edgeImgPointer[current_node.offset - 1] == EDGE_PIXEL || gradImgPointer[current_node.offset - 1] < gradThresh)
            process_stack.push(StackNode(current_node.offset, LEFT, current_chain));
    }
}

std::vector<std::vector<cv::Point>> ED::getSegmentPoints()
{
    return segmentPoints;
}