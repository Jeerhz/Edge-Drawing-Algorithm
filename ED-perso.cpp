#include "ED-perso.h"
#include "Chain.h"
#include <stack>
#include <algorithm>
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

void erase_chain(const Chain* c, Image<ED::State>& S)
{
    if(! c) return;
    std::vector<Point>::const_iterator i;
    for(i=c->pts.begin(); i!=c->pts.end(); ++i)
        S(*i) = 0;
    erase_chain(c->child[0], S);
    erase_chain(c->child[1], S);
}

ED::ED(const Image<float>& grad, const Image<float>& Theta,
       float gradThresh, float anchorThresh, int minPathLen)
: G(grad), O(G.w,G.h), S(G.w,G.h), minGrad(gradThresh), minLen(minPathLen)
{
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


void ED::computeAnchors(float anchorThresh)
{
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

Point* ED::sortedAnchors(int& n) const
{
    // Build histogram of G values
    int min=(int)std::floor(minGrad);
    int nbins = (int)std::round(*std::max_element(G.begin(), G.end()))-min+1;
    int* H = new int[nbins];
    std::fill_n(H, nbins, 0);
    for(Point p={1,1}; p.y+1<S.h; p.y++)
        for(p.x=1; p.x+1<S.w; p.x++)
            if(S(p) == ANCHOR)
                ++H[(int)std::round(G(p))-min];
    // Cumulate histogram
    for(int i=1; i<nbins; i++)
        H[i] += H[i-1];
    n = H[nbins-1];

    // Sort
    Point* anchors = new Point[n];
    for(Point p={1,1}; p.y+1<S.h; p.y++)
        for(p.x=1; p.x+1<S.w; p.x++)
            if(S(p) == ANCHOR) {
                int i = --H[(int)std::round(G(p))-min];
                anchors[i] = p;
            }
    delete [] H;
    return anchors;
}

void ED::joinAnchors()
{
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
bool ED::nextPixelChain(StackNode& node)
{
    Point q[3];
    q[0] = neighbor(node.pos,node.dir);
    for(int i=1; i<3; i++) {
        q[i] = neighbor(q[0], dir(!orient(node.dir),i-1));
    }

    float bestGrad = -1;
    for (int i = 0; i < 3; i++)
    {
        if(S(q[i]) != 0) {
            node.pos = q[i];
            return S(node.pos)!=EDGE;
        }
        float g = G(q[i]);
        if (g > bestGrad)
        {
            bestGrad = g;
            node.pos = q[i];
        }
    }
    return S(node.pos)!=EDGE && bestGrad>=minGrad;
}

void ED::exploreChain(StackNode node, Chain* chain,
                      std::stack<StackNode>& stack)
{
    Orientation ori = orient(chain->dir);
    // Explore until we find change direction or we hit an edge pixel or the gradient is below threshold
    while (O(node.pos) == ori)
    {
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
