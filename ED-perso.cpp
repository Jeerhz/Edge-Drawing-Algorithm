#include "ED-perso.h"
#include "Chain.h"
#include <stack>
#include <algorithm>
#include <numeric>
#include <cmath>

const ED::Orientation HORIZONTAL=true;
const ED::Orientation VERTICAL=false;

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

const ED::State ANCHOR=1;
const ED::State EDGE=2;

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
       float gradMin, float anchorThresh, int minPathLen, float epsNFA)
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
    validateNFA(epsNFA);
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
int* ED::cumulHistoGradAnchors(int& nbins) const {
    nbins = (int)std::round(*std::max_element(G.begin(), G.end()))+1;
    int* H = new int[nbins];
    std::fill_n(H, nbins, 0);
    for(Point p={1,1}; p.y+1<S.h; p.y++)
        for(p.x=1; p.x+1<S.w; p.x++)
            if(S(p) == ANCHOR)
                ++H[(int)std::round(G(p))];
    std::partial_sum(H, H+nbins, H);
    return H;
}

Point* ED::sortedAnchors(int& n) const {
    int nbins;
    int* H = cumulHistoGradAnchors(nbins);
    int min = std::min(1,(int)std::floor(minGrad));
    if(min>=nbins) {
        delete [] H;
        n = 0;
        return 0;
    }

    // Sort
    n = H[nbins-1]-H[min-1]; min = H[min-1];
    Point* anchors = new Point[n];
    for(Point p={1,1}; p.y+1<S.h; p.y++)
        for(p.x=1; p.x+1<S.w; p.x++)
            if(S(p) == ANCHOR) {
                int i = --H[(int)std::round(G(p))];
                anchors[i-min] = p;
            }
    delete [] H;
    return anchors;
}

void ED::joinAnchors() {
    int n;
    Point* anchors = sortedAnchors(n);
    while(--n >= 0) {
        Point p=anchors[n];
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
    delete [] anchors;
}

// Get next pixel in the chain based on current node direction and gradient values
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

void ED::exploreChain(StackNode node, Chain* chain,
                      std::stack<StackNode>& stack) {
    Orientation ori = orient(chain->dir);
    // Explore until we find change direction or we hit an edge pixel or the gradient is below threshold
    while (O(node.pos) == ori) {
        // Remove adjacent anchors
        for(int i=0; i<2; i++) {
            Point p = neighbor(node.pos, dir(!ori,i));
            if(S(p) == ANCHOR)
                S(p)=0;
        }
        if(! nextPixelChain(node))
            return;
        chain->pts.push_back(node.pos);
        S(node.pos) = EDGE;
    }

    // We add new nodes to the process stack in perpendicular directions to the edge with reference to this chain as a parent
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

/// Fonctor for sorting based on gradient along edge.
struct CompareGradEdge {
    const Image<float>& G;
    const std::vector<Point>& E;
    CompareGradEdge(const Image<float>& g, const std::vector<Point>& e)
    : G(g), E(e) {}
    bool operator()(int i, int j) const {
        float vi=G(E[i]), vj=G(E[j]);
        if(vi != vj)
            return (vi<vj);
        return i>j; // Equal value pts handled lower to upper index
    }
};

/// A contrario validation.
void ED::validateNFA(float epsNFA) {
    if(epsNFA<=0)
        return;
    for(Point p={1,1}; p.y+1<S.h; p.y++)
        for(p.x=1; p.x+1<S.w; p.x++)
            S(p) = G(p)<minGrad? 0: ANCHOR;
    int nbins;
    int* H = cumulHistoGradAnchors(nbins);
    int min = std::min(1,(int)std::floor(minGrad));
    if(min>=nbins) {
        edges.clear();
        delete [] H;
        return;
    }

    int n = H[nbins-1]-H[min-1];
    float* lProba = new float[nbins];
    float v = std::log10(H[nbins-1]);
    for(int i=1; i<nbins; i++)
        lProba[i] = log10(H[nbins-1]-H[i-1])-v;
    delete [] H;

    int nTests = 0;
    std::vector<std::vector<Point>>::const_iterator it=edges.begin(), end;
    for(end=edges.end(); it!=end; ++it)
        nTests += it->size()*(it->size()+1)/2;
    const float lTests = log10(nTests);
    const float lEpsNFA = log10(epsNFA);
    
    std::vector<std::vector<Point>> valid;
    for(it=edges.begin(); it!=end; ++it)
        validateEdge(*it, lProba, nbins, lTests, lEpsNFA, valid);
    std::swap(edges, valid);
    delete [] lProba;
}

/// Find_root of Union/Find algorithm.
int root(int* zpar, int i) {
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

void extract_valid_segments(const std::vector<Point>& e,
                            Interval* r, float lEpsNFA,
                            std::vector<std::vector<Point>>& valid) {
    Interval* m = r->findMinValue();
    if(m->v > lEpsNFA)
        return;
    std::vector<Point> v(e.begin()+r->min, e.begin()+r->max+1);
    valid.push_back(v);
    for(; m->parent; m = m->parent) {
        std::vector<Interval*>::iterator it, end=m->parent->child.end();
        for(it=m->parent->child.begin(); it!=end; ++it)
            if(*it!=m) {
                (*it)->parent = 0;
                extract_valid_segments(e, *it, lEpsNFA, valid);
            }
    }
}

void ED::validateEdge(const std::vector<Point>& e, float* lProba, int nbins,
                      float lTests, float lEpsNFA,
                      std::vector<std::vector<Point>>& valid) const {
    size_t n=e.size();
    int* idx = new int[n];
    std::iota(idx, idx+n, 0);
    std::sort(idx, idx+n, CompareGradEdge(G,e));
    int* par = new int[n];
    int* zpar = new int[n];
    std::fill(par, par+n, -1);
    std::fill(zpar, zpar+n, -1);
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
        if(G(e[par[k]]) == G(e[k]))
            par[j] = par[k];
    }
    size_t root = idx[0];
    delete [] idx;
    delete [] zpar;

    std::vector<Interval*> tree(n, 0);
    for(size_t i=0; i<n; i++) { // Build tree nodes
        float v = G(e[i]);
        if(i==root || G(e[par[i]])!=v)
            tree[i] = new Interval(i,v);
    }
    for(size_t i=0; i<n; i++)
        if(i!=root) { // Build hierarchy and fill info
            if(tree[i])
                tree[par[i]]->addChild(tree[i]);
            else
                tree[par[i]]->add(i);
        }
    tree[root]->fillBounds();
    delete [] par;
    for(size_t i=0; i<n; i++) // Compute log NFA
        if(tree[i])
            tree[i]->v = lTests +
                         (tree[i]->max-tree[i]->min+1) *
                         lProba[(int)std::round(tree[i]->v)];
    extract_valid_segments(e, tree[root], lEpsNFA, valid);
    delete tree[root];
}
