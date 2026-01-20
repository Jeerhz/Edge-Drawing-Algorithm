#include "Chain.h"

Chain::Chain()
: len(0), path(-1), dir(LEFT)
{
    child[0] = child[1] = nullptr;
}

Chain::Chain(Direction direction, Chain* p)
: len(0), path(-1), dir(direction)
{
    child[0] = child[1] = nullptr;
    if(p) {
        int i = p->child[0]? 1: 0;
        p->child[i] = this;
    }
}

Chain::~Chain()
{
    delete child[0];
    delete child[1];
}

int Chain::length()
{
    int l[2];
    l[0] = child[0] ? child[0]->length() : 0;
    l[1] = child[1] ? child[1]->length() : 0;
    path = (l[0]>=l[1])? 0: 1;
    len = pts.size() + l[path];
    return len;
}

void Chain::pruneLongestPath(std::stack<Chain*>& orphans)
{
    if(! child[path])
        return;
    if(child[1-path])
        orphans.push(child[1-path]);
    child[path]->pruneLongestPath(orphans);
    pts.insert(pts.end(), child[path]->pts.begin(), child[path]->pts.end());
}
