#include "Vector3f.h"
#include "octtree.h"

#include <vector>

const float MAX_MAC = 0.5;
const float EPSILON = 0.01f;

void Node::insertParticle(Particle& particle) {
    if (hasParticle) {
        if (!hasChildren) {
            createChildren();
        }
        insertParticleIntoChildren(this->particle);
        insertParticleIntoChildren(particle);
    } else {
        particle = particle;
        hasParticle = true;
    }
}

Vector3f Node::particleAcceleration(const Particle& particle) {
    const Vector3f displacement = centerOfMass - particle.position;
    const float distance = displacement.abs();
    if (!hasChildren) {
        if (hasParticle) {
            return this->particle.mass * displacement / (std::pow(distance, 3.0) + EPSILON);
        } else {
            return Vector3f::ZERO;
        }
    }

    const float mac = bounds.width / distance;
    if (mac < MAX_MAC) {
        return mass * displacement / (std::pow(distance, 3.0) + EPSILON);
    } else {
        Vector3f acc;
        for (auto child : children) {
            acc += child->particleAcceleration(particle);
        }
        return acc;
    }
}

bool Node::contains(const Particle& particle){
    const bool withinXBounds = particle.position.x() >= bounds.minCoords.x()
                            && particle.position.x() <= bounds.maxCoords.x();
    const bool withinYBounds = particle.position.y() >= bounds.minCoords.y()
                            && particle.position.y() <= bounds.maxCoords.y();
    const bool withinZBounds = particle.position.z() >= bounds.minCoords.z()
                            && particle.position.z() <= bounds.maxCoords.z();
    return withinXBounds && withinYBounds && withinZBounds;
}

void Node::createChildren() {
    const Vector3f newCoords = (bounds.maxCoords - bounds.minCoords) / 2;
    const float xIntervals[2][2] = { { bounds.minCoords.x(), newCoords.x() },
                                   { newCoords.x(), bounds.maxCoords.x() }};
    const float yIntervals[2][2] = { { bounds.minCoords.y(), newCoords.y() },
                                   { newCoords.y(), bounds.maxCoords.y() }};
    const float zIntervals[2][2] = { { bounds.minCoords.z(), newCoords.z() },
                                   { newCoords.z(), bounds.maxCoords.z() }};
    const int index = 0;
    for (auto xInterval : xIntervals) {
        for (auto yInterval : yIntervals) {
            for (auto zInterval : zIntervals) {
                const BoundingBox bounds(Vector3f(xInterval[0], yInterval[0], zInterval[0]),
                                         Vector3f(xInterval[1], yInterval[1], zInterval[1]));
                children[0] = new Node(bounds);
            }
        }
    }
}

void Node::insertParticleIntoChildren(Particle& particle) {
    // determine child particle belongs to (lazy way)
    // insert particle into child
    for (auto child : children) {
        if (contains(particle)) {
            child->insertParticle(particle);
            break;
        }
    }
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