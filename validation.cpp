// SPDX-License-Identifier: MPL-2.0
/**
 * @file validation.cpp
 * @brief NFA validation of edge drawing
 * @author Pascal Monasse <pascal.monasse@enpc.fr>
 * @date 2026
 */

#include "ED.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cassert>

/// Test if path is closed, that is, first and last points are 8-neighbors.
static bool closed(const std::vector<Point>& e) {
    short dx = (short)e.front().x - (short)e.back().x,
          dy = (short)e.front().y - (short)e.back().y;
    return dx*dy*dx*dy==1;
}

/// Max-tree of edge intervals.
/// Stores bounds \c min and \c max, a value \a v (initially the min gradient
/// on the interval, later log10 of NFA). If the edge is circular, the interval
/// may loop and then max < min.
struct Interval {
    Interval* parent;
    std::vector<Interval*> child;
    int min, max;
    float v; ///< First min gradient, then log10(NFA)
    bool loop; ///< For closed ege

    Interval(int i, float v0): parent(0), min(i), max(i), v(v0), loop(false) {}
    ~Interval();
    void addChild(Interval* c);
    void add(int i, int ext);
    void fillBounds(int ext);
    Interval* findMinValue();
};

/// Destructor. Recursive deallocation of all subtree.
Interval::~Interval() {
    std::vector<Interval*>::iterator it, end=child.end();
    for(it=child.begin(); it!=end; ++it)
        delete *it;
}

/// Add \a c as child.
void Interval::addChild(Interval* c) {
    c->parent = this;
    child.push_back(c);
}

/// Add index \a i, adjusting \c min and \c max of interval. If the interval is
/// looping, it needs \a ext, a point outside the interval.
void Interval::add(int i, int ext) {
    if(i<min && (!loop || ext<i))
        min = i;
    if(i>max && (!loop || i<ext))
        max = i;
}

/// Integrate sub-intervals to update fields \c min and \c max.
void Interval::fillBounds(int ext) {
    std::vector<Interval*>::iterator it, end=child.end();
    for(it=child.begin(); it!=end; ++it) {
        (*it)->fillBounds(ext);
        add((*it)->min, ext);
        add((*it)->max, ext);
    }
}

/// Find interval with minimal value \c v in all subtree.
Interval* Interval::findMinValue() {
    Interval* min = this;
    std::vector<Interval*>::iterator it, end=child.end();
    for(it=child.begin(); it!=end; ++it) {
        Interval* m = (*it)->findMinValue();
        if(m->v < min->v)
            min = m;
    }
    return min;
}

/// Functor for sorting based on gradient along edge.
struct CompareGradEdge {
    const Image<int>& G;
    const std::vector<Point>& E;
    CompareGradEdge(const Image<int>& g, const std::vector<Point>& e)
    : G(g), E(e) {}
    bool operator()(int i, int j) const { return (G(E[i])<G(E[j])); }
};

/// A contrario validation. \a lEpsNFA is the log10 of detection threshold.
/// Its normal value is 0, or negative for more requiring detection.
/// @param lEpsNFA log10 of max NFA for validation. 
/// @param bSubLines Validate only portions of lines.
void ED::validateNFA(float lEpsNFA, bool bSubLines) {
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
        validateEdge(*it, lProba, lTests, lEpsNFA, bSubLines, valid);
    std::swap(edges, valid);
}

/// Find_root of Union/Find algorithm.
int root(std::vector<int>& zpar, int i) {
    if(zpar[i]==i)
        return i;
    return (zpar[i] = root(zpar, zpar[i]));
}

/// Lowest common ancestor on max-tree.
/// Rely on increasing value while going down-tree.
static Interval* lca(Interval* i1, Interval* i2) {
    while(i1 != i2) {
        assert(i1 && i2);
        if(i2->v <= i1->v)
            i1 = i1->parent;
        else
            i2 = i2->parent;
    }
    return i1;
}

/// Step 3 of algorithm in ED::validateEdge.
void extract_valid_segments(const std::vector<Point>& e,
                            Interval* r, float lEpsNFA, bool bSubLines,
                            std::vector<std::vector<Point>>& valid) {
    Interval* m = r->findMinValue();
    if(m->v > lEpsNFA)
        return;
    if(! bSubLines) {
        valid.push_back(e);
        return;
    }
    std::vector<Point> v;
    if(m->loop) {
        v.insert(v.end(), e.begin()+m->min, e.end());
        v.insert(v.end(), e.begin(), e.begin()+m->max+1);
    } else
        v.insert(v.end(), e.begin()+m->min, e.begin()+m->max+1);
    valid.push_back(v);
    for(; m->parent; m = m->parent) {
        std::vector<Interval*>::iterator it, end=m->parent->child.end();
        for(it=m->parent->child.begin(); it!=end; ++it)
            if(*it!=m) {
                (*it)->parent = 0;
                extract_valid_segments(e, *it, lEpsNFA, bSubLines, valid);
            }
    }
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
                      float lTests, float lEpsNFA, bool bSubLines,
                      std::vector<std::vector<Point>>& valid) const {
    const size_t n=e.size();
    const bool circular = closed(e);
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    if(! bSubLines) { // shortcut: if whole line is valid, no need for max-tree
        int min=*std::min_element(idx.begin(), idx.end(), CompareGradEdge(G,e));
        if(lTests+n*0.5f*lProba[G(e[min])] <= lEpsNFA) {
            valid.push_back(e);
            return;
        }
    }
    std::sort(idx.begin(), idx.end(), CompareGradEdge(G,e));
    std::vector<int> par(n,-1);
    std::vector<int> zpar(n,-1);
    // Build tree
    for(int i=(int)n-1; i>=0; i--) {
        int j=idx[i];
        par[j] = zpar[j] = j;
        int l = j-1;
        if(circular && l<0)
            l = (int)n-1;
        if(l>=0 && zpar[l]>=0) {
            int k = root(zpar,l);
            par[k] = zpar[k] = j;
        }
        l = j+1;
        if(circular && l>=(int)n)
            l = 0;
        if(l<(int)n && zpar[l]>=0) {
            int k = root(zpar,l);
            par[k] = zpar[k] = j;
        }
    }
    // Canonize
    for(size_t i=1; i<n; i++) {
        int j=idx[i], k=par[j];
        if(G(e[par[k]]) == G(e[k]))
            par[j] = par[k];
    }
    const size_t root = idx[0];

    std::vector<Interval*> tree(n, 0);
    for(size_t i=0; i<n; i++) { // Build tree nodes
        int v = G(e[i]);
        if(i==root || G(e[par[i]])!=v)
            tree[i] = new Interval(i,v);
    }
    for(size_t i=0; i<n; i++)  // Build tree edges
        if(i!=root && tree[i])
            tree[par[i]]->addChild(tree[i]);
    const int ext=tree[root]->min; // Index of point outside any loop
    if(circular) { // Tag circular intervals
        Interval* i1 = tree[0]? tree[0]: tree[par[0]];
        Interval* i2 = tree[n-1]? tree[n-1]: tree[par[n-1]];
        for(i1 = lca(i1,i2); i1->parent; i1=i1->parent) {
            i1->loop = true;
            if(i1->min < ext)
                i1->min=(int)n-1;
            else
                i1->max=0;
        }
    }
    for(size_t i=0; i<n; i++) // Fill bounds (without sub-intervals)
        if(! tree[i])
            tree[par[i]]->add(i, ext);
    tree[root]->fillBounds(ext); // Integrate sub-intervals in computing bounds
    tree[root]->min=0; tree[root]->max=(int)n-1; // Fix root bounds
    for(size_t i=0; i<n; i++) // Compute log NFA
        if(tree[i]) {
            int len = (tree[i]->loop? (int)n-tree[i]->max+tree[i]->min+1:
                       tree[i]->max-tree[i]->min+1);
            tree[i]->v = lTests+len*0.5f*lProba[tree[i]->v];
        }
    extract_valid_segments(e, tree[root], lEpsNFA, bSubLines, valid);
    delete tree[root];
}
