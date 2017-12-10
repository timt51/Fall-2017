#include "galaxysystem.h"

#include <cassert>
#include <iostream>
#include "camera.h"
#include "vertexrecorder.h"
#include "octtree.h"

// TODO adjust to number of particles.
const unsigned NUM_PARTICLES = 250;
const Vector3f ACC_DUE_TO_GRAVITY = Vector3f(0, -9.81f, 0);
const float MASS = 1.0f;
const float DRAG_COEF = 0.3f;
const float STIFFNESS = 32.0f;
const float REST_LENGTH = 0.15;
const float EPSILON = 0.01f;
const float MIN_COORD = -100;
const float MAX_COORD = 100;
const BoundingBox BOUNDS = BoundingBox(Vector3f(MIN_COORD), Vector3f(MAX_COORD));

GalaxySystem::GalaxySystem()
{
    // To add a bit of randomness, use e.g.
    // float f = rand_uniform(-0.5f, 0.5f);
    // in your initial conditions.
    for (unsigned idx = 0; idx < NUM_PARTICLES/2; ++idx) {
        m_vVecState.push_back(Vector3f(rand_uniform(-1.0f, 1.0f), rand_uniform(-4.0f, -2.0f), rand_uniform(-1.0f, 1.0f)));
        m_vVecState.push_back(Vector3f(2, 0, 0));
    }

    for (unsigned idx = 0; idx < NUM_PARTICLES/2; ++idx) {
        m_vVecState.push_back(Vector3f(rand_uniform(-1.0f, 1.0f), rand_uniform(2.0f, 4.0f), rand_uniform(-1.0f, 1.0f)));
        m_vVecState.push_back(Vector3f(-2, 0, 0));
    }

    octTree = nullptr;
    createOctTree();
}

void GalaxySystem::createOctTree() {
    delete octTree;
    octTree = new OctTree(BOUNDS);
    // Create octtree
    particles.clear();
    for (unsigned idx = 0; idx < NUM_PARTICLES/2; idx += 2) {
        particles.push_back(Particle(m_vVecState[idx], MASS));
    }
    std::cout << "inserting particles" << std::endl;
    octTree->insertParticles(particles);
    std::cout << "finished inserting particles" << std::endl;
}

std::vector<Vector3f> GalaxySystem::evalF(const std::vector<Vector3f>& state)
{
    std::vector<Vector3f> f(state.size());

    for (unsigned idx = 0; idx < 2*NUM_PARTICLES; idx += 2) {
        const auto currentPosition = state[idx];
        const auto currentVelocity = state[idx+1];

        const auto nextPositionDerivative = currentVelocity;
        const Particle particle(currentPosition, MASS);
        const auto nextVelocityDerivative = octTree->particleAcceleration(particle);
        f[idx] = nextPositionDerivative;
        f[idx+1] = nextVelocityDerivative;
    }
    return f;
}

void GalaxySystem::setState(const std::vector<Vector3f>  & newState) {
    m_vVecState = newState;
    createOctTree();
}

// render the system (ie draw the particles)
void GalaxySystem::draw(GLProgram& gl)
{
    Vector3f PENDULUM_COLOR(0.73f, 0.0f, 0.83f);
    gl.updateMaterial(PENDULUM_COLOR, PENDULUM_COLOR);

    // TODO 4.2, 4.3

    // example code. Replace with your own drawing  code
    for (unsigned idx = 0; idx < NUM_PARTICLES; idx += 2) {
        gl.updateModelMatrix(Matrix4f::translation(m_vVecState[idx]));
        drawSphere(0.03f, 10, 10);
    }

    PENDULUM_COLOR = Vector3f(1.0f, 1.0f, 1.0f);
    gl.updateMaterial(PENDULUM_COLOR, PENDULUM_COLOR);

    // TODO 4.2, 4.3

    // example code. Replace with your own drawing  code
    for (unsigned idx = NUM_PARTICLES; idx < 2*NUM_PARTICLES; idx += 2) {
        gl.updateModelMatrix(Matrix4f::translation(m_vVecState[idx]));
        drawSphere(0.03f, 10, 10);
    }
}
