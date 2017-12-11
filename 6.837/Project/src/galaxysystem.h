#ifndef GALAXYSYSTEM_H
#define GALAXYSYSTEM_H

#include <vector>

#include "particlesystem.h"
#include "octtree.h"

class GalaxySystem : public ParticleSystem
{
public:
    GalaxySystem();

    std::vector<Vector3f> evalF(const std::vector<Vector3f>& state) override;
    void setState(const std::vector<Vector3f>  & newState) override;
    void draw(GLProgram&);

    // inherits 
    // std::vector<Vector3f> m_vVecState;
private:
    OctTree* octTree;
    std::vector<Particle> particles;
    std::vector<Vector3f> lastEvalF;
    void createOctTree();
};

#endif
