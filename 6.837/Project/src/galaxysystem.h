#ifndef GALAXYSYSTEM_H
#define GALAXYSYSTEM_H

#include <vector>

#include "particlesystem.h"

class GalaxySystem : public ParticleSystem
{
public:
    GalaxySystem();

    std::vector<Vector3f> evalF(std::vector<Vector3f> state) override;
    void draw(GLProgram&);

    // inherits 
    // std::vector<Vector3f> m_vVecState;
};

#endif
