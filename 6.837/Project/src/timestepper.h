#ifndef INTEGRATOR_H
#define INTEGRATOR_H

#include "vecmath.h"
#include <vector>
#include "particlesystem.h"

using namespace std;

class TimeStepper
{
public:
    virtual ~TimeStepper() {}
	virtual void takeStep(ParticleSystem* particleSystem, float stepSize) = 0;
protected:
	vector<Vector3f> sum(const vector<Vector3f>& left, const vector<Vector3f>& right, float leftCoef, float rightCoef);
};

//IMPLEMENT YOUR TIMESTEPPERS

class ForwardEuler : public TimeStepper
{
	void takeStep(ParticleSystem* particleSystem, float stepSize) override;
};

class Trapezoidal : public TimeStepper
{
	void takeStep(ParticleSystem* particleSystem, float stepSize) override;
};

class RK4 : public TimeStepper
{
	void takeStep(ParticleSystem* particleSystem, float stepSize) override;
};

/////////////////////////
#endif
