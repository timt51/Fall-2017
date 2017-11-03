#include "math.h"
#include "Object3D.h"

bool Sphere::intersect(const Ray &r, float tmin, Hit &h) const
{
    // BEGIN STARTER

    // We provide sphere intersection code for you.
    // You should model other intersection implementations after this one.

    // Locate intersection point ( 2 pts )
    const Vector3f &rayOrigin = r.getOrigin(); //Ray origin in the world coordinate
    const Vector3f &dir = r.getDirection();

    Vector3f origin = rayOrigin - _center;      //Ray origin in the sphere coordinate

    float a = dir.absSquared();
    float b = 2 * Vector3f::dot(dir, origin);
    float c = origin.absSquared() - _radius * _radius;

    // no intersection
    if (b * b - 4 * a * c < 0) {
        return false;
    }

    float d = sqrt(b * b - 4 * a * c);

    float tplus = (-b + d) / (2.0f*a);
    float tminus = (-b - d) / (2.0f*a);

    // the two intersections are at the camera back
    if ((tplus < tmin) && (tminus < tmin)) {
        return false;
    }

    float t = 10000;
    // the two intersections are at the camera front
    if (tminus > tmin) {
        t = tminus;
    }

    // one intersection at the front. one at the back 
    if ((tplus > tmin) && (tminus < tmin)) {
        t = tplus;
    }

    if (t < h.getT()) {
        Vector3f normal = r.pointAtParameter(t) - _center;
        normal = normal.normalized();
        h.set(t, this->material, normal);
        return true;
    }
    // END STARTER
    return false;
}

// Add object to group
void Group::addObject(Object3D *obj) {
    m_members.push_back(obj);
}

// Return number of objects in group
int Group::getGroupSize() const {
    return (int)m_members.size();
}

bool Group::intersect(const Ray &r, float tmin, Hit &h) const
{
    // BEGIN STARTER
    // we implemented this for you
    bool hit = false;
    for (Object3D* o : m_members) {
        if (o->intersect(r, tmin, h)) {
            hit = true;
        }
    }
    return hit;
    // END STARTER
}


Plane::Plane(const Vector3f &normal, float d, Material *m) : Object3D(m) {
    _normal = normal;
    _d = d;
    _material = m;
}
bool Plane::intersect(const Ray &r, float tmin, Hit &h) const
{
    const Vector3f &rayOrigin = r.getOrigin(); //Ray origin in the world coordinate
    const Vector3f &dir = r.getDirection();
    
    // TODO: or... t = 0 ...
    const float denom = Vector3f::dot(_normal, dir);
    if (abs(denom) < 1E-6) {
        return false;
    } else {
        const float t = -(-_d + Vector3f::dot(_normal, rayOrigin)) / denom;
        if (t > tmin && t < h.getT()) {
            h.set(t, this->material, _normal.normalized());
            return true;
        }
        return false;
    }
}
bool Triangle::intersect(const Ray &r, float tmin, Hit &h) const 
{
    const Vector3f &rayOrigin = r.getOrigin(); //Ray origin in the world coordinate
    const Vector3f &dir = r.getDirection();

    // a -> 0, b -> 1, c -> 2
    const Vector3f aMinusB = _v[0] - _v[1];
    const Vector3f aMinusC = _v[0] - _v[2];
    const bool setColumns = true;
    const Matrix3f matrix = Matrix3f(aMinusB, aMinusC, dir, setColumns);

    if (abs(matrix.determinant()) < 1E-6) {
        return false;
    } else {
        const Vector3f aMinusOrigin = _v[0] - rayOrigin;
        const Vector3f x = matrix.inverse() * aMinusOrigin;
        const float t = x.z();
        const float beta = x.x();
        const float gamma = x.y();
        const float alpha = 1 - beta - gamma;
        if (t > tmin && t < h.getT() && alpha >= 0 && beta >= 0 && gamma >= 0) {
            const Vector3f normal = alpha * _normals[0] + beta * _normals[1] + gamma * _normals[2];
            h.set(t, this->material, normal.normalized());
            return true;
        }
        return false;
    }
}


Transform::Transform(const Matrix4f &m,
    Object3D *obj) : _object(obj) {
    _object = obj;
    _mInverse = m.inverse();
    _mNormal = _mInverse.transposed();
}
bool Transform::intersect(const Ray &r, float tmin, Hit &h) const
{
    const Vector4f &rayOrigin = Vector4f(r.getOrigin(), 1); //Ray origin in the world coordinate
    const Vector4f &dir = Vector4f(r.getDirection(), 0);

    const Vector3f rayOriginObjectSpace = (_mInverse * rayOrigin).xyz();
    const Vector3f dirObjectSpace = (_mInverse * dir).xyz();
    const Ray rayInObjectSpace(rayOriginObjectSpace, dirObjectSpace);

    bool hit = false;
    if (_object->intersect(rayInObjectSpace, tmin, h)) {
        hit = true;
        // fix h
        h.normal = (_mNormal * Vector4f(h.normal, 0)).xyz().normalized();
    }

    return hit;
}