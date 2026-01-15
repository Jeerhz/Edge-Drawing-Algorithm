#include "EDLines.h"
#include "EDPF.h"
#include <iostream>
#include "opencv2/ximgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;
using namespace cv::ximgproc;

int main(int argc, char **argv)
{
    const char *filename;
    if (argc > 1)
        filename = argv[1];
    else
        filename = "./images/maison.jpg";

    Mat testImg, colorImg;
    colorImg = imread(filename);

    if (colorImg.empty())
    {
        cerr << "Error: Could not load image " << filename << endl;
        return -1;
    }

    cvtColor(colorImg, testImg, COLOR_BGR2GRAY);

    Ptr<EdgeDrawing> ed = createEdgeDrawing();
    cout << "\n#################################################";
    cout << "\n##### NEW ED IMPLEMENTATION #########";
    cout << "\n#################################################\n";

    ed->params.EdgeDetectionOperator = EdgeDrawing::SOBEL;
    ed->params.GradientThresholdValue = 36;
    ed->params.AnchorThresholdValue = 8;
    ed->params.Sigma = 1.0;

    TickMeter tm;

    // -------------------------------
    // 1. NEW ED
    // -------------------------------
    tm.reset();
    tm.start();
    ED testED = ED(testImg, LSD_OPERATOR, 6, 0, 10, 1.0, true);
    tm.stop();
    double newEdTime = tm.getTimeMilli();
    cout << "New ED Implementation          : " << newEdTime << " ms" << endl;

    Mat newEdgeImg = testED.getEdgeImage();

    std::vector<std::vector<cv::Point>> segmentPoints = testED.getSegmentPoints();

    // Create a color image and draw each segment in a different deterministic color
    Mat segmentsImg = Mat::zeros(newEdgeImg.size(), CV_8UC3);

    for (size_t i = 0; i < segmentPoints.size(); ++i)
    {
        const auto &seg = segmentPoints[i];
        if (seg.empty())
            continue;
        // deterministic but varied color per segment (B,G,R)
        cv::Scalar color((i * 53) % 256, (i * 97) % 256, (i * 193) % 256);

        // Color every point in the same segment with the same color
        for (const cv::Point &p : seg)
        {
            if (p.x >= 0 && p.x < segmentsImg.cols && p.y >= 0 && p.y < segmentsImg.rows)
                segmentsImg.at<cv::Vec3b>(p.y, p.x) = cv::Vec3b((uchar)color[0], (uchar)color[1], (uchar)color[2]);
        }
    }

    // Save colored segments image
    imwrite("SegmentsEdgeImage.png", segmentsImg);

    // Save anchor and edge images
    imwrite("EdgeImage_New.png", newEdgeImg);

    cout << "\nSaved edge images:" << endl;
    cout << "  - EdgeImage_New.png" << endl;

    // -------------------------------
    // 2. NEW EDLINES
    // -------------------------------
    tm.reset();
    tm.start();
    EDLines newEDLines(testED);
    tm.stop();
    double newEdLinesTime = tm.getTimeMilli();
    cout << "New EDLines Implementation    : " << newEdLinesTime << " ms" << endl;

    // Get line images
    Mat newLinesImg = newEDLines.getLineImage();

    // Save images
    imwrite("LineImage_New.png", newLinesImg);

    cout << "\nSaved line images:" << endl;
    cout << "  - LineImage_New.png" << endl;

    // -------------------------------
    // 3. NEW EDPF
    // -------------------------------
    tm.reset();
    tm.start();
    EDPF edpf = EDPF(testImg);
    tm.stop();
    double edpfTime = tm.getTimeMilli();
    cout << "EDPF Implementation           : " << edpfTime << " ms" << endl;

    // Get EDPF edge image and segments
    Mat edpfEdgeImg = edpf.getEdgeImage();
    std::vector<std::vector<cv::Point>> edpfSegmentPoints = edpf.getSegmentPoints();

    // Save EDPF edge image
    imwrite("EdgeImage_EDPF.png", edpfEdgeImg);
    cout << "  - EdgeImage_EDPF.png" << endl;

    // Create a color image and draw each EDPF segment in a different deterministic color
    Mat edpfSegmentsImg = Mat::zeros(edpfEdgeImg.size(), CV_8UC3);

    for (size_t i = 0; i < edpfSegmentPoints.size(); ++i)
    {
        const auto &seg = edpfSegmentPoints[i];
        if (seg.empty())
            continue;
        // deterministic but varied color per segment (B,G,R)
        cv::Scalar color((i * 53) % 256, (i * 97) % 256, (i * 193) % 256);

        // Color every point in the same segment with the same color
        for (const cv::Point &p : seg)
        {
            if (p.x >= 0 && p.x < edpfSegmentsImg.cols && p.y >= 0 && p.y < edpfSegmentsImg.rows)
                edpfSegmentsImg.at<cv::Vec3b>(p.y, p.x) = cv::Vec3b((uchar)color[0], (uchar)color[1], (uchar)color[2]);
        }
    }

    // Save colored segments image for EDPF
    imwrite("SegmentsEdgeImage_EDPF.png", edpfSegmentsImg);
    cout << "  - SegmentsEdgeImage_EDPF.png" << endl;

    cout << "\n#################################################" << endl;
    cout << "All comparisons completed successfully.\n";
    cout << "#################################################\n";

    return 0;
}
