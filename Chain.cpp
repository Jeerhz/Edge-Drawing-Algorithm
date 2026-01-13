#include "Chain.h"
#include <iostream>
#include <algorithm>
#include "ED-perso.h"

using namespace cv;

Chain::Chain()
    : pixels(), parent_chain(nullptr), first_childChain(nullptr), second_childChain(nullptr), direction(UNDEFINED)
{
}

Chain::Chain(Direction _direction, Chain *_parent_chain)
    : pixels(), parent_chain(_parent_chain), first_childChain(nullptr), second_childChain(nullptr), direction(_direction)
{
}

Chain::~Chain()
{
}

int Chain::pruneToLongestChain()
{
    int first_childChain_totalLen = first_childChain ? first_childChain->pruneToLongestChain() : 0;
    int second_childChain_totalLen = second_childChain ? second_childChain->pruneToLongestChain() : 0;

    if (first_childChain_totalLen >= second_childChain_totalLen)
    {

        is_first_childChain_longest_path = true;
        return pixels.size() + first_childChain_totalLen;
    }
    else
    {
        is_first_childChain_longest_path = false;
        return pixels.size() + second_childChain_totalLen;
    }
}

// DFS traversal to collect all chains
std::pair<int, std::vector<Chain *>> Chain::getAllChains(bool only_longest_path)
{
    std::vector<Chain *> all_chains;
    int total_length = 0;
    appendAllChains(all_chains, total_length, only_longest_path);
    return {total_length, all_chains};
}

void Chain::appendAllChains(std::vector<Chain *> &allChains, int &total_length, bool only_longest_path)
{
    if (only_longest_path && this->is_extracted)
        return;

    allChains.push_back(this);
    total_length += pixels.size();
    if (first_childChain && (!only_longest_path || is_first_childChain_longest_path))
        first_childChain->appendAllChains(allChains, total_length, only_longest_path);

    if (second_childChain && (!only_longest_path || !is_first_childChain_longest_path))
        second_childChain->appendAllChains(allChains, total_length, only_longest_path);
}

int Chain::getTotalLength(bool only_longest_path)
{
    int total_length = pixels.size();

    if (first_childChain && (!only_longest_path || is_first_childChain_longest_path))
        total_length += first_childChain->getTotalLength(only_longest_path);

    if (second_childChain && (!only_longest_path || !is_first_childChain_longest_path))
        total_length += second_childChain->getTotalLength(only_longest_path);

    return total_length;
}

// StackNode implementation
StackNode::StackNode(int _offset, Direction direction, Chain *_parent_chain)
{
    offset = _offset;
    node_direction = direction;
    parent_chain = _parent_chain;
}

void ProcessStack::clear()
{
    std::stack<StackNode>().swap(*this);
}
