// SPDX-License-Identifier: MPL-2.0
/**
 * @file ED.cpp
 * @brief edge drawing
 * @author Adle Ben Salem
 *         Pascal Monasse <pascal.monasse@enpc.fr>
 * @date 2025-2026
 */

#include "ED.h"
#include "Chain.h"
#include <stack>
#include <algorithm>
#include <numeric>
#include <cmath>

/// \file
// The above command is meant to include comment on #define in Doxygen.
/// Uncomment to validate portions of lines. Otherwise, any valid portion
/// validates the whole line.
//#define VALID_SUBLINE

const ED::Orientation HORIZONTAL=true;
const ED::Orientation VERTICAL=false;
const ED::State ANCHOR=1;
const ED::State EDGE=2;

inline ED::Orientation orient(Direction d) {
    return (d==LEFT || d== RIGHT)? HORIZONTAL: VERTICAL;
}
inline Direction dir(ED::Orientation o, int i) {
    if(o==HORIZONTAL) return i==0? LEFT: RIGHT;
    return i==0? UP: DOWN;
}
inline Point neighbor(Point p, Direction d) {
    Point q(p);
    switch(d) {
    case LEFT: --q.x; break;
    case RIGHT: ++q.x; break;
    case UP: --q.y; break;
    case DOWN: ++q.y; break;
    case UNDEFINED: default: break;
    }
    return q;
}

/// Erase chain-tree in state image \a S.
void erase_chain(const Chain* c, Image<ED::State>& S) {
    if(! c) return;
    std::vector<Point>::const_iterator i;
    for(i=c->pts.begin(); i!=c->pts.end(); ++i)
        S(*i) = 0;
    erase_chain(c->child[0], S);
    erase_chain(c->child[1], S);
}

/// Constructor. Does all the computation, output in field \c edges.
ED::ED(const Image<float>& grad, const Image<float>& Theta,
       float gradMin, float anchorThresh, int minPathLen)
: G(grad), O(G.w,G.h), S(G.w,G.h), minGrad(gradMin), minLen(minPathLen) {
    for(int y=0; y<G.h; y++)
        G(0,y) = G(G.w-1,y) = 0;
    for(int x=0; x<G.w; x++)
        G(x,0) = G(x,G.h-1) = 0;
    for(int y=0; y<O.h; y++)
        for(int x=0; x<O.w; x++) {
            float o = std::abs(Theta(x,y));
            O(x,y) = (o<M_PI/4 || o>3*M_PI/4)? VERTICAL: HORIZONTAL;
        }
    computeAnchors(anchorThresh);
    joinAnchors();
}

/// Compute anchor pixels: local max of gradient (with minimal gap and value).
/// Pixels satisfying the condition get the label in state image \c S.
void ED::computeAnchors(float anchorThresh) {
    S.fill(0);
    for(Point p={1,1}; p.y+1<S.h; p.y++)
        for(p.x=1; p.x+1<S.w; p.x++) {
            float g = G(p);
            if(g < minGrad)
                continue;
            Point q1 = neighbor(p, dir(!O(p),0));
            Point q2 = neighbor(p, dir(!O(p),1));
            if(g >= std::max(G(q1),G(q2))+anchorThresh && S(q1)==0 && S(q2)==0)
                S(p) = ANCHOR;
        }
}

/// Build histogram of G values for anchor points, return number of beans.
std::vector<int> ED::cumulHistoGradAnchors() const {
    int n = (int)std::round(*std::max_element(G.begin(), G.end()))+1;
    std::vector<int> H(n, 0);
    for(Point p={1,1}; p.y+1<S.h; p.y++)
        for(p.x=1; p.x+1<S.w; p.x++)
            if(S(p) == ANCHOR)
                ++H[(int)std::round(G(p))];
    std::partial_sum(H.begin(), H.end(), H.begin());
    return H;
}

/// Return anchors ordered by increasing gradient.
std::vector<Point> ED::sortedAnchors() const {
    std::vector<int> H = cumulHistoGradAnchors();
    std::vector<Point> anchors;
    if(H.empty())
        return anchors;

    // Sort
    const int n = H.back();
    anchors.resize(n);
    for(Point p={1,1}; p.y+1<S.h; p.y++)
        for(p.x=1; p.x+1<S.w; p.x++)
            if(S(p) == ANCHOR) {
                int i = --H[(int)std::round(G(p))];
                anchors[i] = p;
            }
    return anchors;
}

/// Extract edges from anchors.
void ED::joinAnchors() {
    std::vector<Point> anchors = sortedAnchors();
    std::vector<Point>::const_reverse_iterator it, end=anchors.rend();
    for(it=anchors.rbegin(); it!=end; ++it) {
        const Point& p = *it;
        if(S(p)!=ANCHOR) continue;
        Chain* root = new Chain;
        buildChainTree(root, p);
        int l0 = root->child[0]->length();
        int l1 = root->child[1]->length();
        root->len = l0+l1+1;
        if(root->len>=minLen)
            extractEdgesFromTree(root);
        else {
            erase_chain(root->child[0], S);
            erase_chain(root->child[1], S);
            S(p) = 0;
        }
        delete root;
    }
}

/// Get next pixel in chain based on node direction and gradient values.
bool ED::nextPixelChain(StackNode& node) {
    Point q[3];
    q[0] = neighbor(node.pos,node.dir);
    for(int i=1; i<3; i++) {
        q[i] = neighbor(q[0], dir(!orient(node.dir),i-1));
    }

    float bestGrad = -1;
    for (int i = 0; i < 3; i++) {
        if(S(q[i]) != 0) {
            node.pos = q[i];
            return S(node.pos)!=EDGE;
        }
        float g = G(q[i]);
        if (g > bestGrad) {
            bestGrad = g;
            node.pos = q[i];
        }
    }
    return S(node.pos)!=EDGE && bestGrad>=minGrad;
}

/// Explore edge until finding a changed direction, hitting an edge pixel, or
/// too low gradient. In the first case, two anchors are appended to \a stack.
void ED::exploreChain(StackNode node, Chain* chain,
                      std::stack<StackNode>& stack) {
    Orientation ori = orient(chain->dir);
    while (O(node.pos) == ori) {
        for(int i=0; i<2; i++) { // Remove adjacent anchors
            Point p = neighbor(node.pos, dir(!ori,i));
            if(S(p) == ANCHOR)
                S(p)=0;
        }
        if(! nextPixelChain(node))
            return;
        chain->pts.push_back(node.pos);
        S(node.pos) = EDGE;
    }

    // Add new nodes in perpendicular direction
    stack.emplace(node.pos, dir(!ori,0), chain);
    stack.emplace(node.pos, dir(!ori,1), chain);
}

/// Build chain tree issued from anchor point \a p.
void ED::buildChainTree(Chain* root, Point p) {
    root->pts.push_back(p);
    S(p) = EDGE;
    std::stack<StackNode> stack;
    stack.emplace(p, dir(O(p),0), root);
    stack.emplace(p, dir(O(p),1), root);
    while(! stack.empty()) {
        StackNode node = stack.top();
        stack.pop();
        Chain* c = new Chain(node.dir, node.parent);
        exploreChain(node, c, stack);
    }
}

/// Build edge segment from the two children of \a root.
void ED::buildRootEdge(Chain* root) {
    edges.emplace_back();
    std::vector<Point>& v = edges.back();
    Chain* child = root->child[0];
    v.insert(v.end(), child->pts.rbegin(), child->pts.rend());
    v.push_back(root->pts.back());
    child = root->child[1];
    v.insert(v.end(),child->pts.begin(), child->pts.end());
}

/// From the chain tree at \a root, extract edge segments.
/// Find the longest paths from nodes, prune them, yielding orphan trees,
/// which are themselves handled in the same manner.
void ED::extractEdgesFromTree(Chain* root) {
    std::stack<Chain*> orphans;
    for(int i=0; i<2; i++)
        root->child[i]->pruneLongestPath(orphans);
    buildRootEdge(root);
    while(!orphans.empty()) {
        Chain* c = orphans.top();
        orphans.pop();
        if(c->len>=minLen) {
            c->pruneLongestPath(orphans);
            edges.push_back(c->pts);
        } else
            erase_chain(c, S);
    }
}

/// Functor for sorting based on gradient along edge.
struct CompareGradEdge {
    const Image<float>& G;
    const std::vector<Point>& E;
    CompareGradEdge(const Image<float>& g, const std::vector<Point>& e)
    : G(g), E(e) {}
    bool operator()(int i, int j) const {
        float vi=G(E[i]), vj=G(E[j]);
        return (vi<vj);
    }
};

/// A contrario validation. \a lEpsNFA is the log10 of detection threshold.
/// Its normal value is 0, or negative for more requiring detection.
void ED::validateNFA(float lEpsNFA) {
    for(Point p={1,1}; p.y+1<S.h; p.y++)
        for(p.x=1; p.x+1<S.w; p.x++)
            S(p) = G(p)<minGrad? 0: ANCHOR;
    std::vector<int> H = cumulHistoGradAnchors();
    if(H.empty()) {
        edges.clear();
        return;
    }

    std::vector<float> lProba(H.size(), 0);
    const int n = H.back();
    const float v = std::log10(n);
    for(size_t i=1; i<H.size(); i++)
        lProba[i] = log10(n-H[i-1])-v;

    int nTests = 0;
    std::vector<std::vector<Point>>::const_iterator it, end=edges.end();
    for(it=edges.begin(); it!=end; ++it)
        nTests += it->size()*(it->size()+1)/2;
    const float lTests = log10(nTests);

    std::vector<std::vector<Point>> valid;
    for(it=edges.begin(); it!=end; ++it)
        validateEdge(*it, lProba, lTests, lEpsNFA, valid);
    std::swap(edges, valid);
}

/// Find_root of Union/Find algorithm.
int root(std::vector<int>& zpar, int i) {
    if(zpar[i]==i)
        return i;
    return (zpar[i] = root(zpar, zpar[i]));
}

/// Max-tree of edge intervals
struct Interval {
    Interval* parent;
    std::vector<Interval*> child;
    int min, max;
    float v;
    Interval(int i, float v0): parent(0), min(i), max(i), v(v0) {}
    ~Interval() {
        std::vector<Interval*>::iterator it, end=child.end();
        for(it=child.begin(); it!=end; ++it)
            delete *it;
    }
    void addChild(Interval* c) {
        c->parent = this;
        child.push_back(c);
    }
    void add(int i) {
        if(i<min)
            min = i;
        if(i>max)
            max = i;
    }
    Interval* findMinValue() {
        Interval* min = this;
        std::vector<Interval*>::iterator it, end=child.end();
        for(it=child.begin(); it!=end; ++it) {
            Interval* m = (*it)->findMinValue();
            if(m->v < min->v)
                min = m;
        }
        return min;
    }
    void fillBounds() {
        std::vector<Interval*>::iterator it, end=child.end();
        for(it=child.begin(); it!=end; ++it) {
            (*it)->fillBounds();
            if(min > (*it)->min)
                min = (*it)->min;
            if(max < (*it)->max)
                max = (*it)->max;
        }
    }
};

/// Step 3 of algorithm in ED::validateEdge.
void extract_valid_segments(const std::vector<Point>& e,
                            Interval* r, float lEpsNFA,
                            std::vector<std::vector<Point>>& valid) {
    Interval* m = r->findMinValue();
    if(m->v > lEpsNFA)
        return;
#ifdef VALID_SUBLINE
    std::vector<Point> v(e.begin()+m->min, e.begin()+m->max+1);
    valid.push_back(v);
    for(; m->parent; m = m->parent) {
        std::vector<Interval*>::iterator it, end=m->parent->child.end();
        for(it=m->parent->child.begin(); it!=end; ++it)
            if(*it!=m) {
                (*it)->parent = 0;
                extract_valid_segments(e, *it, lEpsNFA, valid);
            }
    }
#else
    valid.push_back(e);
#endif
}

/// Append to \a valid the maximally contrasted segments of \a e.
/// \a lProba gives the log10 probability of contrast at least index.
/// \a lTests is log10 of the number of tests and \a lEpsNFA is log10 of the
/// upper bound threshold for meaningfulness.
/// Algo:
/// 1. Compute the max-tree of gradients on \a e (Berger algorithm).
/// 2. Find most meaningful segment of tree.
/// 3. If meaningful, validate and go back to 2 for all disjoint segments.
void ED::validateEdge(const std::vector<Point>& e,
                      const std::vector<float>& lProba,
                      float lTests, float lEpsNFA,
                      std::vector<std::vector<Point>>& valid) const {
    const size_t n=e.size();
#ifndef VALID_SUBLINE // shortcut: if whole line is valid, no need for max-tree
    float min=G(e[0]);
    for(size_t i=1; i<n; i++)
        if(min > G(e[i]))
            min = G(e[i]);
    if(lTests+n*0.5f*lProba[(int)std::round(min)] <= lEpsNFA) {
        valid.push_back(e);
        return;
    }
#endif
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), CompareGradEdge(G,e));
    std::vector<int> par(n,-1);
    std::vector<int> zpar(n,-1);
    // Build tree
    for(int i=(int)n-1; i>=0; i--) {
        int j=idx[i];
        par[j] = zpar[j] = j;
        if(j>0 && zpar[j-1]>=0) {
            int k = root(zpar,j-1);
            par[k] = zpar[k] = j;
        }
        if(j+1<(int)n && zpar[j+1]>=0) {
            int k = root(zpar,j+1);
            par[k] = zpar[k] = j;
        }
    }
    // Canonize
    for(size_t i=1; i<n; i++) {
        int j=idx[i], k=par[j];
        if(std::round(G(e[par[k]])) == std::round(G(e[k])))
            par[j] = par[k];
    }
    size_t root = idx[0];

    std::vector<Interval*> tree(n, 0);
    for(size_t i=0; i<n; i++) { // Build tree nodes
        float v = std::round(G(e[i]));
        if(i==root || std::round(G(e[par[i]]))!=v)
            tree[i] = new Interval(i,v);
    }
    for(size_t i=0; i<n; i++)
        if(i!=root) { // Build tree edges and fill info
            if(tree[i])
                tree[par[i]]->addChild(tree[i]);
            else
                tree[par[i]]->add(i);
        }
    tree[root]->fillBounds();
    for(size_t i=0; i<n; i++) // Compute log NFA
        if(tree[i])
            tree[i]->v = lTests +
                         (tree[i]->max-tree[i]->min+1) * 0.5f *
                         lProba[(int)std::round(tree[i]->v)];
    extract_valid_segments(e, tree[root], lEpsNFA, valid);
    delete tree[root];
}
