#include "Vector3f.h"
#include "octtree.h"

#include <vector>

const float MAX_MAC = 0.5;

void Node::insertParticle(Particle& particle) {
    if (hasParticle) {
        // determine child this->particle belongs to
        // insert this->particle into child
        if (!hasChildren) {
            // create children
        }
        // determine child particle belongs to
        // insert particle into child
    } else {
        particle = particle;
        hasParticle = true;
    }
}

Vector3f Node::particleAcceleration(const Particle& particle) {
    Vector3f acc;

    if (!hasChildren) {
        if (hasParticle) {
        }
    } else {
        float distanceToNode = (particle.position - centerOfMass).abs();
        float mac = bounds.width / distanceToNode;
        if (mac < MAX_MAC) {

        } else {
            for (auto child : children) {
                acc += child->particleAcceleration(particle);
            }
        }
    }

    return acc;
}

bool Node::contains(const Particle& particle){

}

void Node::establishRepInvariant() {
    mass = 0;
    centerOfMass = Vector3f::ZERO;

    if (hasChildren) {
        for (auto child : children) {
            child->establishRepInvariant();
            mass += child->mass;
        }

        for (auto child : children) {
            centerOfMass += (child->mass / mass) * child->centerOfMass;
        }
    } else {
        if (hasParticle) {
            mass = particle.mass;
            centerOfMass = particle.position;
        }
    }
}

void OctTree::insertParticles(const std::vector<Particle>& particles) {
    for (auto particle : particles) {
        root->insertParticle(particle);
    }
    root->establishRepInvariant();
}

Vector3f OctTree::particleAcceleration(const Particle& particle){
    return root->particleAcceleration(particle);
}