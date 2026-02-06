
#include "EDPF.h"

// EDPF definition
// Value of gradient threshold should be computed based on the impact of pixel quantization on gradient magnitude calculation
// Then it is dependent of the operator mask used (Sobel, Prewitt, etc.)
// The gratdient threshold is declared to be 0 in the article, the default value in official implementation is 8.45
EDPF::EDPF(cv::Mat _srcImage)
    : ED(_srcImage, LSD_OPERATOR, 8, 0)
{
    computeNumberSegmentPieces();
    computeGradientCDF();
    validateEdgeSegments();
}

void EDPF::computeNumberSegmentPieces()
{
    number_segment_pieces = 0;
    for (int i = 0; i < segmentPoints.size(); i++)
    {
        int len = (int)segmentPoints[i].size();
        number_segment_pieces += (len * (len - 1)) / 2;
    }
}

void EDPF::computeGradientCDF()
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

double EDPF::NFA(double prob, int len)
{
    double nfa = number_segment_pieces;
    for (int i = 0; i < (int)(len / 2) && nfa > EPSILON; i++)
        nfa *= prob;

    return nfa;
}

// void EDPF::testSegmentPiece(int segment_idx, int start_idx, int end_idx)
// {
//     int chainLen = end_idx - start_idx + 1;
//     if (chainLen < minPathLen)
//         return;

//     // First find the min. gradient along the segment
//     int minGrad = 1 << 30; // 1 << 30 computes 1 * 2^30 = 1073741824
//     int minGradIndex = start_idx;
//     for (int k = start_idx; k <= end_idx; k++)
//     {
//         int point_row = segmentPoints[segment_idx][k].y;
//         int point_col = segmentPoints[segment_idx][k].x;
//         if (gradImgPointer[point_row * image_width + point_col] < minGrad)
//         {
//             minGrad = gradImgPointer[point_row * image_width + point_col];
//             minGradIndex = k;
//         }
//     } // end-for

//     // Compute nfa
//     double prob = gradient_cdf[minGrad];
//     double nfa = NFA(1 - prob, (int)chainLen);

//     if (nfa <= EPSILON)
//     {
//         for (int k = start_idx; k <= end_idx; k++)
//         {
//             int point_row = segmentPoints[segment_idx][k].y;
//             int point_col = segmentPoints[segment_idx][k].x;
//             edgeImgPointer[point_row * image_width + point_col] = 255;
//         }

//         return;
//     }

//     // We divide at the point where the gradient is the minimum starting from both ends
//     int minGradIndexFromEnd = minGradIndex - 1;
//     while (minGradIndexFromEnd > start_idx)
//     {
//         int point_row = segmentPoints[segment_idx][minGradIndexFromEnd].y;
//         int point_col = segmentPoints[segment_idx][minGradIndexFromEnd].x;

//         if (gradImgPointer[point_row * image_width + point_col] <= minGrad)
//             minGradIndexFromEnd--;
//         else
//             break;
//     } // end-while

//     int minGradIndexFromStart = minGradIndex + 1;
//     while (minGradIndexFromStart < end_idx)
//     {
//         int point_row = segmentPoints[segment_idx][minGradIndexFromStart].y;
//         int point_col = segmentPoints[segment_idx][minGradIndexFromStart].x;

//         if (gradImgPointer[point_row * image_width + point_col] <= minGrad)
//             minGradIndexFromStart++;
//         else
//             break;
//     }

//     testSegmentPiece(segment_idx, start_idx, minGradIndexFromEnd);
//     testSegmentPiece(segment_idx, minGradIndexFromStart, end_idx);
// }

void EDPF::testSegmentPiece(int segment_idx, int start_idx, int end_idx)
{
    int chainLen = end_idx - start_idx + 1;
    if (chainLen < minPathLen)
        return;

    // Try subsegments of decreasing size starting from chainLen - 1
    for (int subSegLen = chainLen - 1; subSegLen >= minPathLen; subSegLen--)
    {
        // Test all possible subsegments of length subSegLen
        for (int subStart = start_idx; subStart + subSegLen - 1 <= end_idx; subStart++)
        {
            int subEnd = subStart + subSegLen - 1;
            
            // Find the min gradient along this subsegment
            int minGrad = 1 << 30;
            int minGradIndex = subStart;
            for (int k = subStart; k <= subEnd; k++)
            {
                int point_row = segmentPoints[segment_idx][k].y;
                int point_col = segmentPoints[segment_idx][k].x;
                if (gradImgPointer[point_row * image_width + point_col] < minGrad)
                {
                    minGrad = gradImgPointer[point_row * image_width + point_col];
                    minGradIndex = k;
                }
            }

            // Compute nfa
            double prob = gradient_cdf[minGrad];
            double nfa = NFA(1 - prob, subSegLen);

            if (nfa <= EPSILON)
            {
                // Mark this subsegment as an edge
                for (int k = subStart; k <= subEnd; k++)
                {
                    int point_row = segmentPoints[segment_idx][k].y;
                    int point_col = segmentPoints[segment_idx][k].x;
                    edgeImgPointer[point_row * image_width + point_col] = 255;
                }
                
                // Recursively test the remaining pieces
                if (subStart > start_idx)
                    testSegmentPiece(segment_idx, start_idx, subStart - 1);
                if (subEnd < end_idx)
                    testSegmentPiece(segment_idx, subEnd + 1, end_idx);
                
                return; // Found a valid subsegment, exit
            }
        }
    }
    
    // No valid subsegment found for any size >= minPathLen
    // The segment is discarded (no edges marked)
}

void EDPF::validateEdgeSegments()
{
    memset(edgeImgPointer, 0, image_width * image_height); // clear edge image as we will re-draw the valid edges
    // Validate segments
    for (int segment_idx = 0; segment_idx < segmentPoints.size(); segment_idx++)
        testSegmentPiece(segment_idx, 0, (int)segmentPoints[segment_idx].size() - 1);

    extractNewSegments();

    // clean space
    delete[] gradient_cdf;
}

//----------------------------------------------------------------------------------------------
// After the validation of the edge segments, extracts the valid ones
// In other words, updates the valid segments' pixel arrays and their lengths
//
void EDPF::extractNewSegments()
{
    std::vector<std::vector<cv::Point>> validSegments;
    int noSegments = 0;

    for (int i = 0; i < segmentPoints.size(); i++)
    {
        int start = 0;
        while (start < segmentPoints[i].size())
        {
            while (start < segmentPoints[i].size())
            {
                int r = segmentPoints[i][start].y;
                int c = segmentPoints[i][start].x;

                if (edgeImgPointer[r * image_width + c])
                    break;
                start++;
            }

            int end = start + 1;
            while (end < segmentPoints[i].size())
            {
                int r = segmentPoints[i][end].y;
                int c = segmentPoints[i][end].x;

                if (edgeImgPointer[r * image_width + c] == 0)
                    break;
                end++;
            }

            int len = end - start;
            if (len >= 10)
            {
                std::vector<cv::Point> subVec;
                subVec.assign(
                    segmentPoints[i].begin() + start,
                    segmentPoints[i].begin() + end);

                validSegments.push_back(std::move(subVec));
                noSegments++;
            }

            start = end + 1;
            if (start >= segmentPoints[i].size())
                break;
        }
    } // end-for

    // Copy to ed
    segmentPoints = validSegments;
}