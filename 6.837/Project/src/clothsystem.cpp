#include "clothsystem.h"
#include "camera.h"
#include "vertexrecorder.h"
#include "math.h"

 // your system should at least contain 8x8 particles.
const unsigned W = 8;
const unsigned H = 8;
const unsigned NUM_PARTICLES = W*H;
const Vector3f ACC_DUE_TO_GRAVITY = Vector3f(0, -9.81f, 0);
const float MASS = 0.05f;
const float DRAG_COEF = 0.5f;
const float STIFFNESS = 32.0f;
const float CLOTH_LENGTH = 2.0f;
const float REST_LENGTH = (CLOTH_LENGTH/W) * 1.1;

using namespace std;

ClothSystem::ClothSystem()
{
    // TODO 5. Initialize m_vVecState with cloth particles. 
    // You can again use rand_uniform(lo, hi) to make things a bit more interesting
    wind = Vector3f::ZERO;
    smooth = false;
    const float xMin = 0.4;
    const float yMax = 1.0;
    for (unsigned row = 0; row < W; ++row) {
        for (unsigned col = 0; col < H; ++col) {
            m_vVecState.push_back(Vector3f(xMin+col*(CLOTH_LENGTH/(H-1)), yMax-row*(CLOTH_LENGTH/(W-1)), 0));
            m_vVecState.push_back(Vector3f::ZERO);
        }
    }
}


std::vector<Vector3f> ClothSystem::evalF(const std::vector<Vector3f>& state)
{
    std::vector<Vector3f> f(state.size());
    // TODO 5. implement evalF
    // - gravity
    // - viscous drag
    // - structural springs
    // - shear springs
    // - flexion springs
    for (unsigned idx = 0; idx < 2*NUM_PARTICLES; idx += 2) {
        if (idx == 0 || idx == 2*(W-1)) {
            f[idx] = state[idx+1];
            f[idx+1] = Vector3f::ZERO;
            continue;
        }

        const auto currentPosition = state[idx];
        const auto currentVelocity = state[idx+1];

        const auto nextPositionDerivative = currentVelocity;
        auto nextVelocityDerivative = Vector3f::ZERO;
        // calculate vel change due to gravity
        nextVelocityDerivative += MASS*ACC_DUE_TO_GRAVITY;
        // calculate vel change due to drag
        nextVelocityDerivative += -DRAG_COEF*currentVelocity;
        // calculate vel change due to wind
        nextVelocityDerivative += wind;            
        // calculate vel change due to springs
        const int currentRow = floor(idx/2/W);
        const int currentCol = (idx / 2) % H;
        const auto possiblyConnectedParticles = { Vector2f(currentRow,currentCol-1), Vector2f(currentRow-1,currentCol),
                                                  Vector2f(currentRow,currentCol+1), Vector2f(currentRow+1,currentCol),
                                                  Vector2f(currentRow-1,currentCol-1), Vector2f(currentRow-1,currentCol+1),
                                                  Vector2f(currentRow+1,currentCol-1), Vector2f(currentRow+1,currentCol+1),
                                                  Vector2f(currentRow,currentCol-2), Vector2f(currentRow-2,currentCol),
                                                  Vector2f(currentRow,currentCol+2), Vector2f(currentRow+2,currentCol) };
        for (const auto possiblyConnectedParticle : possiblyConnectedParticles) {
            const auto row = possiblyConnectedParticle.x();
            const auto col = possiblyConnectedParticle.y();
            if (row >= 0 && row < W && col >= 0 && col < H) {
                const auto particleNumber = row * W + col;
                const auto particlePosition = state[2*particleNumber];
                if (abs(currentRow-row) == 2 || abs(currentCol-col) == 2) {
                    nextVelocityDerivative += 
                    -STIFFNESS*((currentPosition-particlePosition).abs()-2*REST_LENGTH)*(currentPosition-particlePosition).normalized();
                } else if (abs(currentRow-row) == 1 && abs(currentCol-col) == 1) {
                    nextVelocityDerivative += 
                    -STIFFNESS*((currentPosition-particlePosition).abs()-1.414*REST_LENGTH)*(currentPosition-particlePosition).normalized();
                } else {
                    nextVelocityDerivative += 
                    -STIFFNESS*((currentPosition-particlePosition).abs()-REST_LENGTH)*(currentPosition-particlePosition).normalized();    
                }
            }
        }

        f[idx] = nextPositionDerivative;
        f[idx+1] = nextVelocityDerivative;
    }
    return f;
}


void ClothSystem::draw(GLProgram& gl)
{
    //TODO 5: render the system 
    //         - ie draw the particles as little spheres
    //         - or draw the springs as little lines or cylinders
    //         - or draw wireframe mesh

    const Vector3f CLOTH_COLOR(0.9f, 0.9f, 0.9f);
    gl.updateMaterial(CLOTH_COLOR);
    
    if (!smooth) {
        // EXAMPLE for how to render cloth particles.
        //  - you should replace this code.
        for (unsigned idx = 0; idx < 2*NUM_PARTICLES; idx += 2) {
            gl.updateModelMatrix(Matrix4f::translation(m_vVecState[idx]));
            drawSphere(0.04f, 8, 8);
        }

        // EXAMPLE: This shows you how to render lines to debug the spring system.
        //
        //          You should replace this code.
        //
        //          Since lines don't have a clearly defined normal, we can't use
        //          a regular lighting model.
        //          GLprogram has a "color only" mode, where illumination
        //          is disabled, and you specify color directly as vertex attribute.
        //          Note: enableLighting/disableLighting invalidates uniforms,
        //          so you'll have to update the transformation/material parameters
        //          after a mode change.
        gl.disableLighting();
        gl.updateModelMatrix(Matrix4f::identity()); // update uniforms after mode change
        VertexRecorder rec;
        for (unsigned idx = 0; idx < 2*NUM_PARTICLES; idx += 2) {
            const auto currentPosition = m_vVecState[idx];
            const int currentRow = floor(idx/2/W);
            const int currentCol = (idx / 2) % H;
            const auto possiblyConnectedParticles = { Vector2f(currentRow,currentCol-1), Vector2f(currentRow-1,currentCol),
                                                    Vector2f(currentRow,currentCol+1), Vector2f(currentRow+1,currentCol) };
            for (const auto possiblyConnectedParticle : possiblyConnectedParticles) {
                const auto row = possiblyConnectedParticle.x();
                const auto col = possiblyConnectedParticle.y();
                if (row >= 0 && row < W && col >= 0 && col < H) {
                    const auto particleNumber = row * W + col;
                    const auto particlePosition = m_vVecState[2*particleNumber];
                    rec.record(currentPosition, CLOTH_COLOR);
                    rec.record(particlePosition, CLOTH_COLOR);
                }
            }
        }
        glLineWidth(3.0f);
        rec.draw(GL_LINES);

        gl.enableLighting(); // reset to default lighting model
    } else {
        gl.updateModelMatrix(Matrix4f::identity()); // update uniforms after mode change
        VertexRecorder rec;
        for (unsigned idx = 0; idx < 2*NUM_PARTICLES; idx += 2) {
            const auto currentPosition = m_vVecState[idx];
            const int currentRow = floor(idx/2/W);
            const int currentCol = (idx / 2) % H;
            const auto possibleTriangles = { Vector4f(currentRow, currentCol+1, currentRow+1, currentCol),
                                             Vector4f(currentRow-1, currentCol+1, currentRow, currentCol+1) };
            for (const auto possibleTriangle : possibleTriangles) {
                const auto row1 = possibleTriangle.x();
                const auto col1 = possibleTriangle.y();
                if (!(row1 >= 0 && row1 < W && col1 >= 0 && col1 < H)) {
                    continue;
                }
                const auto row2 = possibleTriangle.z();
                const auto col2 = possibleTriangle.w();
                if (!(row2 >= 0 && row2 < W && col2 >= 0 && col2 < H)) {
                    continue;
                }
    
                const auto particleNumber1 = row1 * W + col1;
                const auto p1 = m_vVecState[2*particleNumber1];
                const auto particleNumber2 = row2 * W + col2;
                const auto p2 = m_vVecState[2*particleNumber2];
                const auto normal = Vector3f::cross(p1-currentPosition, p2-currentPosition);
    
                rec.record(currentPosition, -normal);
                rec.record(p1, -normal);
                rec.record(p2, -normal);
            }
        }
        rec.draw();    
    }
}

