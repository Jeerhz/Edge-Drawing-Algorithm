
#include "ED-perso.h"

// Declaration of EDPF class inheriting from ED
class EDPF : public ED
{
public:
    EDPF(cv::Mat _srcImage);

private:
    int number_segment_pieces; //  number of segment pieces used in nfa calculation
    double *gradient_cdf;      // CDF of gradient magnitudes also used in nfa calculation

    // Method to compute the nfa of a segment and validate edge segments based on Helmholtz principle
    void validateEdgeSegments();
    void computeNumberSegmentPieces();
    void computeGradientCDF();
    void extractValidatedEdgeSegments();
    void testSegmentPiece(int segment_idx, int start_idx, int end_idx);
    void testSegmentPieceSimple(int segment_idx); // Simple variant of the segment testing
    void extractNewSegments();
    double NFA(double prob, int len);
    double NFASimple(double prob, int len); // Simple variant of the segment testing
};
