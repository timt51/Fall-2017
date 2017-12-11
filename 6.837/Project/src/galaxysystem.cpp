#include "galaxysystem.h"

#include <cassert>
#include <iostream>
#include <chrono>
#include "camera.h"
#include "vertexrecorder.h"
#include "octtree.h"

// TODO adjust to number of particles.
const unsigned NUM_PARTICLES = 1000;
const Vector3f ACC_DUE_TO_GRAVITY = Vector3f(0, -9.81f, 0);
float MASS = 1.0f;
const float DRAG_COEF = 0.3f;
const float STIFFNESS = 32.0f;
const float REST_LENGTH = 0.15;
const float EPSILON = 0.01f;
const float MIN_COORD = -5;
const float MAX_COORD = 5;
const BoundingBox BOUNDS = BoundingBox(Vector3f(MIN_COORD), Vector3f(MAX_COORD));
const Vector3f COLOR_SPECTRUM[] = { Vector3f(4, 49, 104) / 255.0, 
                                    Vector3f(7, 63, 120) / 255.0,
                                    Vector3f(7, 77, 129) / 255.0,
                                    Vector3f(12, 102, 154) / 255.0,
                                    Vector3f(21, 117, 168) / 255.0,
                                    Vector3f(24, 124, 173) / 255.0,
                                    Vector3f(51, 160, 191) / 255.0,
                                    Vector3f(118, 200, 219) / 255.0,
                                    Vector3f(253, 232, 162) / 255.0,
                                    Vector3f(246, 222, 148) / 255.0,
                                    Vector3f(242, 178, 114) / 255.0,
                                    Vector3f(241, 161, 94) / 255.0,
                                    Vector3f(236, 149, 79) / 255.0,
                                    Vector3f(227, 138, 75) / 255.0,
                                    Vector3f(224, 124, 64) / 255.0,
                                    Vector3f(215, 109, 51) / 255.0 };
const unsigned NUM_COLORS = 16;

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
    for (unsigned idx = 0; idx < 2*NUM_PARTICLES; idx += 2) {
        particles.push_back(Particle(m_vVecState[idx], MASS));
    }
    octTree->insertParticles(particles);
}

std::vector<Vector3f> GalaxySystem::evalF(const std::vector<Vector3f>& state)
{
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<Vector3f> f(state.size());

    for (unsigned idx = 0; idx < 2*NUM_PARTICLES; idx += 2) {
        auto currentPosition = state[idx];
        const auto currentVelocity = state[idx+1];

        const auto nextPositionDerivative = currentVelocity;
        const Particle particle(currentPosition, MASS);
        const auto nextVelocityDerivative = octTree->particleAcceleration(particle);
        
        f[idx] = nextPositionDerivative;
        f[idx+1] = nextVelocityDerivative;
    }
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "Evalf time : " << elapsed.count() << " s\n";

    lastEvalF = f;
    return f;
}

void GalaxySystem::setState(const std::vector<Vector3f>  & newState) {
    m_vVecState = newState;
    createOctTree();
}

// render the system (ie draw the particles)
void GalaxySystem::draw(GLProgram& gl)
{
    float minForce = std::numeric_limits<float>::max();
    float maxForce = 0;
    for (unsigned idx = 0; idx < 2*NUM_PARTICLES; idx += 2) {
        float force = lastEvalF[idx+1].abs();
        minForce = std::min(force, minForce);
        maxForce = std::max(force, maxForce);
    }

    for (unsigned idx = 0; idx < NUM_PARTICLES; idx += 2) {
        // Vector3f color(0.73f, 0.0f, 0.83f);
        // gl.updateMaterial(Vector3f::ZERO, color, Vector3f::ZERO, 1.0F, 0.8F);
        // gl.updateModelMatrix(Matrix4f::translation(m_vVecState[idx]));
        // drawSphere(0.02f, 10, 10);
        float force = lastEvalF[idx+1].abs();
        unsigned normed = std::round((NUM_COLORS-1) * (force - minForce) / (maxForce - minForce));
        Vector3f color = COLOR_SPECTRUM[normed];
        gl.updateMaterial(Vector3f::ZERO, color, Vector3f::ZERO, 1.0F, 0.8F);
        gl.updateModelMatrix(Matrix4f::translation(m_vVecState[idx]));
        drawSphere(0.02f, 10, 10);
    }

    for (unsigned idx = NUM_PARTICLES; idx < 2*NUM_PARTICLES; idx += 2) {
        float force = lastEvalF[idx+1].abs();
        unsigned normed = std::round((NUM_COLORS-1) * (force - minForce) / (maxForce - minForce));
        Vector3f color = COLOR_SPECTRUM[normed];
        gl.updateMaterial(Vector3f::ZERO, color, Vector3f::ZERO, 1.0F, 0.8F);
        gl.updateModelMatrix(Matrix4f::translation(m_vVecState[idx]));
        drawSphere(0.02f, 10, 10);
    }
}
