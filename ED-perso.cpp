#include "ED-perso.h"
#include "Chain.h"
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace cv;
using namespace std;

ED::ED(cv::Mat _srcImage, GradientOperator _gradOperator, int _gradThresh, int _anchorThresh, int _minPathLen, double _sigma, bool _sumFlag)
{
    srcImage = _srcImage;
    // detect if input is grayscale or BGR and prepare per-channel buffers for later use (Di Zenzo)
    image_height = srcImage.rows;
    image_width = srcImage.cols;
    gradOperator = _gradOperator;
    gradThresh = _gradThresh;
    anchorThresh = _anchorThresh;
    minPathLen = _minPathLen;
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
    minPathLen = cpyObj.minPathLen;
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

// We take the last (or first for the first child chain) pixel of the current processed chain and clean its neighbors in the segment
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

// Explore the binary graph by depth first from the children of the anchor root.
// Prune the considered node to its longest path. Add the childnode that is not from longest path to the processed stack
// Call cleanUpPenultimateSegmentPixel with a forward pass, add the chains pixels to the segment before processing to the next stack node
// To reconstruct the main segment, when we arrive to the segment from the second main child, we revert the segment and add it to the first segment.
void ED::extractChainsRecur(Chain *anchor_chain_root, std::vector<std::vector<Point>> &anchorSegments)
{
    // We assume that main children chains have already been extracted
    // Store the pointer to the main children in order to reconstruct the 'main segment'
    Chain *main_first_child = anchor_chain_root->first_childChain;
    Chain *main_second_child = anchor_chain_root->second_childChain;
    bool is_first_pass = true;

    // Explore the whole chain depth first
    std::stack<Chain *> chain_stack;
    chain_stack.push(anchor_chain_root);

    while (!chain_stack.empty())
    {
        Chain *current_chain = chain_stack.top();
        chain_stack.pop();

        if (current_chain == nullptr)
            continue;

        // Compute the longest path from the current chain
        int nb_pixels_longest_path = current_chain->pruneToLongestChain();

        // add the child that is not part of the longest path in the stack
        if (current_chain->is_first_childChain_longest_path)
            chain_stack.push(current_chain->second_childChain);
        else
            chain_stack.push(current_chain->first_childChain);

        // Skip the extraction of the main children chains (already extracted) or if the longest path is too short
        if (nb_pixels_longest_path < minPathLen)
            continue;

        // Extraction of the longest path of the current chain into a new segment
        std::pair<int, std::vector<Chain *>> all_chains_resp = current_chain->getAllChains(true);
        std::vector<Chain *> chainChilds_in_longest_path = all_chains_resp.second;

        // Initialize a segment that will hold the points of the current chain longest path
        std::vector<Point> currentSegment;
        // Extract the pixels and add them to the current segment
        for (size_t chain_index = 0; chain_index < chainChilds_in_longest_path.size(); ++chain_index)
        {
            Chain *child_chain = chainChilds_in_longest_path[chain_index];
            if (!child_chain)
                break;

            cleanUpPenultimateSegmentPixel(child_chain, currentSegment, true);

            for (size_t pixel_index = 0; pixel_index < child_chain->pixels.size(); ++pixel_index)
                currentSegment.push_back(Point(child_chain->pixels[pixel_index] % image_width, child_chain->pixels[pixel_index] / image_width));
        }

        if (currentSegment.empty())
        {
            is_first_pass = false;
            continue;
        }

        // If the current segment is not empty, add it to the anchor segments or to the main segment for the main childs
        // Clean possible boucle at the beginning of the segment
        if (currentSegment.size() > 1 && areNeighbors(currentSegment[1].y * image_width + currentSegment[1].x,
                                                      currentSegment.back().y * image_width + currentSegment.back().x))
            currentSegment.erase(currentSegment.begin());

        if ((current_chain == main_first_child || current_chain == main_second_child) && !is_first_pass)
        {
            // Revert the order of the main child segment and add it to the first segment
            std::reverse(currentSegment.begin(), currentSegment.end());
            anchorSegments[0].insert(anchorSegments[0].end(), currentSegment.begin(), currentSegment.end());
        }
        else
            anchorSegments.push_back(currentSegment);

        is_first_pass = false;
    }
}

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

        if (total_pixels_in_anchor_chain < minPathLen)
            revertChainEdgePixel(anchor_chain_root);

        else
        {
            anchor_chain_root->first_childChain->pruneToLongestChain();
            anchor_chain_root->second_childChain->pruneToLongestChain();
            // Create a segment corresponding to this anchor chain
            std::vector<std::vector<Point>> anchorSegments;
            extractChainsRecur(anchor_chain_root, anchorSegments);
            segmentPoints.insert(segmentPoints.end(), anchorSegments.begin(), anchorSegments.end());
        }

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