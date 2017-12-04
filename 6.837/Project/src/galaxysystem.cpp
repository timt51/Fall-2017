#include "galaxysystem.h"

#include <cassert>
#include "camera.h"
#include "vertexrecorder.h"

// TODO adjust to number of particles.
const unsigned NUM_PARTICLES = 250;
const Vector3f ACC_DUE_TO_GRAVITY = Vector3f(0, -9.81f, 0);
const float MASS = 1.0f;
const float DRAG_COEF = 0.3f;
const float STIFFNESS = 32.0f;
const float REST_LENGTH = 0.15;
const float EPSILON = 0.01f;

GalaxySystem::GalaxySystem()
{
    // TODO 4.2 Add particles for simple pendulum
    // TODO 4.3 Extend to multiple particles

    // To add a bit of randomness, use e.g.
    // float f = rand_uniform(-0.5f, 0.5f);
    // in your initial conditions.
    for (unsigned idx = 0; idx < NUM_PARTICLES/2; ++idx) {
        m_vVecState.push_back(Vector3f(rand_uniform(-1.0f, 1.0f), rand_uniform(-4.0f, -2.0f), rand_uniform(-1.0f, 1.0f)));
        m_vVecState.push_back(Vector3f(2, 0, 0));
    }

    m_vVecState.push_back(Vector3f(1,1,1));
    m_vVecState.push_back(Vector3f::ZERO);
    for (unsigned idx = 0; idx < NUM_PARTICLES/2; ++idx) {
        m_vVecState.push_back(Vector3f(rand_uniform(-1.0f, 1.0f), rand_uniform(2.0f, 4.0f), rand_uniform(-1.0f, 1.0f)));
        m_vVecState.push_back(Vector3f(-2, 0, 0));
    }
}

std::vector<Vector3f> GalaxySystem::evalF(std::vector<Vector3f> state)
{
    std::vector<Vector3f> f(state.size());

    for (unsigned idx = 0; idx < 2*NUM_PARTICLES; idx += 2) {
        const auto currentPosition = state[idx];
        const auto currentVelocity = state[idx+1];

        const auto nextPositionDerivative = currentVelocity;
        auto nextVelocityDerivative = Vector3f::ZERO;
        for (unsigned otherIdx = 0; otherIdx < 2*NUM_PARTICLES; otherIdx += 2) {
            if (otherIdx == idx) {
                continue;
            }
            const auto positionOfOther = state[otherIdx];
            const auto displacement = positionOfOther - currentPosition;
            nextVelocityDerivative += displacement / (std::pow(displacement.abs(), 3.0) + EPSILON);
        }
        // // calculate vel change due to gravity
        // nextVelocityDerivative += MASS*ACC_DUE_TO_GRAVITY;
        // // calculate vel change due to drag
        // nextVelocityDerivative += -DRAG_COEF*currentVelocity;
        // // calculate vel change due to springs
        // const auto prevParticlePosition = state[idx-2];
        // nextVelocityDerivative += -STIFFNESS*((currentPosition-prevParticlePosition).abs()-REST_LENGTH)*(currentPosition-prevParticlePosition).normalized();
        // if (idx != 2*(NUM_PARTICLES - 1)) {
        //     const auto nextParticlePosition = state[idx+2];
        //     nextVelocityDerivative += -STIFFNESS*((currentPosition-nextParticlePosition).abs()-REST_LENGTH)*(currentPosition-nextParticlePosition).normalized();
        // }

        f[idx] = nextPositionDerivative;
        f[idx+1] = nextVelocityDerivative;
    }
    return f;
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
