#include "pendulumsystem.h"

#include <cassert>
#include "camera.h"
#include "vertexrecorder.h"

// TODO adjust to number of particles.
const unsigned NUM_PARTICLES = 4;
const Vector3f ACC_DUE_TO_GRAVITY = Vector3f(0, -9.81f, 0);
const float MASS = 1.0f;
const float DRAG_COEF = 0.3f;
const float STIFFNESS = 32.0f;
const float REST_LENGTH = 0.15;

PendulumSystem::PendulumSystem()
{

    // TODO 4.2 Add particles for simple pendulum
    // TODO 4.3 Extend to multiple particles

    // To add a bit of randomness, use e.g.
    // float f = rand_uniform(-0.5f, 0.5f);
    // in your initial conditions.
    m_vVecState.push_back(Vector3f(-0.5, 1, 0));
    m_vVecState.push_back(Vector3f::ZERO);
    for (unsigned idx = 1; idx < NUM_PARTICLES; ++idx) {
        m_vVecState.push_back(Vector3f(-0.5, rand_uniform(-0.5f, 0.5f), 0));
        m_vVecState.push_back(Vector3f::ZERO);
    }
}


std::vector<Vector3f> PendulumSystem::evalF(std::vector<Vector3f> state)
{
    std::vector<Vector3f> f(state.size());
    // TODO 4.1: implement evalF
    //  - gravity
    //  - viscous drag
    //  - springs
    f[0] = state[1];
    f[1] = Vector3f::ZERO;

    for (unsigned idx = 2; idx < 2*NUM_PARTICLES; idx += 2) {
        const auto currentPosition = state[idx];
        const auto currentVelocity = state[idx+1];

        const auto nextPositionDerivative = currentVelocity;
        auto nextVelocityDerivative = Vector3f::ZERO;
        // calculate vel change due to gravity
        nextVelocityDerivative += MASS*ACC_DUE_TO_GRAVITY;
        // calculate vel change due to drag
        nextVelocityDerivative += -DRAG_COEF*currentVelocity;
        // calculate vel change due to springs
        const auto prevParticlePosition = state[idx-2];
        nextVelocityDerivative += -STIFFNESS*((currentPosition-prevParticlePosition).abs()-REST_LENGTH)*(currentPosition-prevParticlePosition).normalized();
        if (idx != 2*(NUM_PARTICLES - 1)) {
            const auto nextParticlePosition = state[idx+2];
            nextVelocityDerivative += -STIFFNESS*((currentPosition-nextParticlePosition).abs()-REST_LENGTH)*(currentPosition-nextParticlePosition).normalized();
        }

        f[idx] = nextPositionDerivative;
        f[idx+1] = nextVelocityDerivative;
    }
    return f;
}

// render the system (ie draw the particles)
void PendulumSystem::draw(GLProgram& gl)
{
    const Vector3f PENDULUM_COLOR(0.73f, 0.0f, 0.83f);
    gl.updateMaterial(PENDULUM_COLOR);

    // TODO 4.2, 4.3

    // example code. Replace with your own drawing  code
    for (unsigned idx = 0; idx < 2*NUM_PARTICLES; idx += 2) {
        gl.updateModelMatrix(Matrix4f::translation(m_vVecState[idx]));
        drawSphere(0.075f, 10, 10);
    }
}
