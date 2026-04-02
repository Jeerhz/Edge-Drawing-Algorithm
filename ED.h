// SPDX-License-Identifier: MPL-2.0
/**
 * @file ED.h
 * @brief edge drawing
 * @author Adle Ben Salem
 *         Pascal Monasse <pascal.monasse@enpc.fr>
 * @date 2025-2026
 */

#ifndef ED_H
#define ED_H

#include "Chain.h"
#include "image.h"

class ED {
public:
    ED(const Image<float>& G, const Image<float>& Theta,
       float gradMin=6, float anchorGap=2, int minPathLen=10);
    void validateNFA(float lEpsNFA=0);

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
    std::vector<int> cumulHistoGradAnchors() const;
    std::vector<Point> sortedAnchors() const;
    void joinAnchors();
    void exploreChain(StackNode node, Chain* chain, std::stack<StackNode>& S);
    bool nextPixelChain(StackNode& node);
    void buildChainTree(Chain* root, Point p);
    void extractEdgesFromTree(Chain* root);
    void buildRootEdge(Chain* root);
    void validateEdge(const std::vector<Point>& e,
                      const std::vector<float>& lProba,
                      float lTests, float lEpsNFA,
                      std::vector<std::vector<Point>>& valid) const;
};

#endif
