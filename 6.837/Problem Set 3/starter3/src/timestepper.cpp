#include "timestepper.h"

#include <iostream>
#include <cstdio>
#include <vector>

using namespace std;

vector<Vector3f> TimeStepper::sum(const vector<Vector3f>& left, const vector<Vector3f>& right, float leftCoef, float rightCoef) {
    std::vector<Vector3f> sum;
    for (unsigned idx = 0; idx < left.size(); ++idx) {
        sum.push_back(leftCoef * left.at(idx) + rightCoef * right.at(idx));
    }
    return sum;
}

void ForwardEuler::takeStep(ParticleSystem* particleSystem, float stepSize)
{
   //TODO: See handout 3.1 
    vector<Vector3f> currentState = particleSystem->getState();
    vector<Vector3f> currentDerivative = particleSystem->evalF(currentState);

    vector<Vector3f> nextState = sum(currentState, currentDerivative, 1.0f, stepSize);
    particleSystem->setState(nextState);
}

void Trapezoidal::takeStep(ParticleSystem* particleSystem, float stepSize)
{
   //TODO: See handout 3.1 
    vector<Vector3f> currentState = particleSystem->getState();
    vector<Vector3f> currentDerivative = particleSystem->evalF(currentState);
    vector<Vector3f> nextStepState = sum(currentState, currentDerivative, 1.0f, stepSize);
    vector<Vector3f> nextStepDerivative = particleSystem->evalF(nextStepState);

    vector<Vector3f> nextStep = sum(currentDerivative, nextStepDerivative, stepSize/2, stepSize/2);
    vector<Vector3f> nextState = sum(currentState, nextStep, 1.0, 1.0);
    particleSystem->setState(nextState);
}


void RK4::takeStep(ParticleSystem* particleSystem, float stepSize)
{
    vector<Vector3f> x0 = particleSystem->getState();
    vector<Vector3f> k1 = particleSystem->evalF(x0);
    vector<Vector3f> x1 = sum(x0, k1, 1.0f, stepSize/2);
    vector<Vector3f> k2 = particleSystem->evalF(x1);
    vector<Vector3f> x2 = sum(x0, k2, 1.0f, stepSize/2);
    vector<Vector3f> k3 = particleSystem->evalF(x2);
    vector<Vector3f> x3 = sum(x0, k3, 1.0f, stepSize);
    vector<Vector3f> k4 = particleSystem->evalF(x3);

    vector<Vector3f> nextState = sum(x0, k1, 1.0, stepSize/6);
    nextState = sum(nextState, k2, 1.0, stepSize/3);
    nextState = sum(nextState, k3, 1.0, stepSize/3);
    nextState = sum(nextState, k4, 1.0, stepSize/6);    
    particleSystem->setState(nextState);
}

