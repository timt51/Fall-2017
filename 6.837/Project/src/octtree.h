#ifndef OCTTREE_H
#define OCTTREE_H

#include <vector>
#include <vecmath.h>

const unsigned NUMBER_OF_CHILDREN = 8;
const unsigned MAX_COORDS = 100;

struct Node
{
    Vector3f minCoords;
    Vector3f maxCoords;
    Node* children[NUMBER_OF_CHILDREN];
    float mass;
    float width;
    bool hasChildren;

    Node(Vector3f minCoords, Vector3f maxCoords) {
        for (unsigned i = 0; i < NUMBER_OF_CHILDREN; ++i) {
            children[i] = nullptr;
        }
    }

    ~Node() {
        for (unsigned i = 0; i < NUMBER_OF_CHILDREN; ++i) {
            delete children[i];
        }
    }

    void insertParticle(Vector3f position, float mass);
    Vector3f forceOnParticle(Vector3f position, float mass);
};

class OctTree
{
public:
    OctTree(Vector3f minCoords, Vector3f maxCoords) {
        root = new Node(minCoords, maxCoords);
    };
    ~OctTree() {
        delete root;
    };
    void insertParticle(Vector3f position, float mass);
    Vector3f forceOnParticle(Vector3f position, float mass);
private:
    Node* root;
};

#endif
