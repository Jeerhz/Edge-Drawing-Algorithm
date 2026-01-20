#ifndef ED_H
#define ED_H

#include "Chain.h"
#include "image.h"

class ED
{
public:
    ED(const Image<float>& G, const Image<float>& Theta,
       float gradThresh=6, float anchorGap=2, int minPathLen=10);

    std::vector<std::vector<Point>> edges;

    typedef bool Orientation;
    typedef char State;
protected:
    Image<float> G;
    Image<Orientation> O;
    Image<State> S;
    float minGrad;
    int minLen;

private:
    void computeAnchors(float anchorThresh);
    Point* sortedAnchors(int& n) const;
    void joinAnchors();
    void exploreChain(StackNode node, Chain* chain, std::stack<StackNode>& S);
    bool nextPixelChain(StackNode& node);
    void buildChainTree(Chain* root, Point p);
    void extractEdgesFromTree(Chain* root);
    void buildRootEdge(Chain* root);
};

#endif
