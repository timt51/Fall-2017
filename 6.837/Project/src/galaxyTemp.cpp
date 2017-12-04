#include "galaxysystem.h"

#include <cassert>
#include "camera.h"
#include "vertexrecorder.h"

// TODO adjust to number of particles.
const unsigned NUM_PARTICLES = std::pow(10.0, 11.0);
const double G = 6.674 * std::pow(10.0, -11.0);                 // m^3 kg^-1 s^-2
const double MASS = 1.99 * std::pow(10.0, 30.0);                // kg
const double RADIUS = 6.957 * std::pow(10.0, 8.0);              // m
const double LIGHT_YEAR = 9.461 * std::pow(10.0, 15.0);         // m
const double VOLUME = 3.3 * std::pow(10.0, 61.0);               // m^3
const double SIDE_LENGTH = 3.21 * std::pow(10.0, 20.0);         // m

GalaxySystem::GalaxySystem()
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


std::vector<Vector3f> GalaxySystem::evalF(std::vector<Vector3f> state)
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
void GalaxySystem::draw(GLProgram& gl)
{
    const Vector3f PENDULUM_COLOR(0.73f, 0.0f, 0.83f);
    gl.updateMaterial(PENDULUM_COLOR, PENDULUM_COLOR);

    // TODO 4.2, 4.3

    // example code. Replace with your own drawing  code
    for (unsigned idx = 0; idx < 2*NUM_PARTICLES; idx += 2) {
        gl.updateModelMatrix(Matrix4f::translation(m_vVecState[idx]));
        drawSphere(RADIUS, 10, 10);
    }
}
