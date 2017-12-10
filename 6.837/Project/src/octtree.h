#ifndef OCTTREE_H
#define OCTTREE_H

#include <vector>
#include <vecmath.h>

const unsigned NUMBER_OF_CHILDREN = 8;
const unsigned MAX_COORDS = 100;

struct BoundingBox
{
    const Vector3f minCoords;
    const Vector3f maxCoords;
    float width;

    BoundingBox(const Vector3f& minCoords, const Vector3f& maxCoords) : minCoords(minCoords), maxCoords(maxCoords) {
        const Vector3f diff = maxCoords - minCoords;
        const float x = std::abs(diff.x());
        const float y = std::abs(diff.y());
        const float z = std::abs(diff.z());
        width = std::max(x, std::max(y, z));
    }
};

struct Particle
{
    Vector3f position;
    Vector3f velocity;
    float mass;
};

struct Node
{
    const BoundingBox bounds;
    Node* children[NUMBER_OF_CHILDREN];
    Particle particle;
    bool hasParticle;
    bool hasChildren;
    float mass;
    Vector3f centerOfMass;

    Node(const BoundingBox& bounds) : bounds(bounds) {
        for (unsigned i = 0; i < NUMBER_OF_CHILDREN; ++i) {
            children[i] = nullptr;
        }
    }

    ~Node() {
        for (unsigned i = 0; i < NUMBER_OF_CHILDREN; ++i) {
            delete children[i];
        }
    }

    void insertParticle(Particle& particle);
    Vector3f particleAcceleration(const Particle& particle);
    bool contains(const Particle& particle);
    void createChildren();
    void insertParticleIntoChildren(Particle& particle);
    void establishRepInvariant();
};

class OctTree
{
public:
    OctTree(const BoundingBox& bounds) {
        root = new Node(bounds);
    };

    ~OctTree() {
        delete root;
    };

    void insertParticles(const std::vector<Particle>& particles);
    Vector3f particleAcceleration(const Particle& particle);
private:
    Node* root;
};

#endif
