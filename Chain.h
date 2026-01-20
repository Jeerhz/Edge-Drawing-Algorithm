#ifndef CHAIN_H
#define CHAIN_H

#include "image.h"
#include <vector>
#include <stack>

enum Direction
{
    LEFT = 1,
    RIGHT = 2,
    UP = 3,
    DOWN = 4,
    UNDEFINED = -1
};

struct Chain
{
    std::vector<Point> pts;
    Chain* child[2];
    int len;
    int path;
    const Direction dir;

    Chain();
    Chain(Direction direction, Chain* parent);
    ~Chain();

    int length();
    void pruneLongestPath(std::stack<Chain*>& orphans);
};

class StackNode
{
public:
    Point pos;
    Direction dir;         // Direction of exploration
    Chain* parent;

    StackNode(Point p, Direction d, Chain* par): pos(p), dir(d), parent(par) {}
};

#endif
