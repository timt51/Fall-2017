#include <math.h>
#include "Material.h"

Vector3f Material::shade(const Ray &ray,
    const Hit &hit,
    const Vector3f &dirToLight,
    const Vector3f &lightIntensity)
{
    const float diffuseClamp = std::max(0.0f, Vector3f::dot(dirToLight, hit.normal));
    const Vector3f iDiffuse = diffuseClamp * lightIntensity * _diffuseColor;

    // TODO: check against lecture notes
    const Vector3f rayDir = ray.getDirection();
    const Vector3f perfectReflectionDir = rayDir - 2 * Vector3f::dot(rayDir, hit.normal) * hit.normal;
    const float specularClamp = std::max(0.0f, Vector3f::dot(dirToLight, perfectReflectionDir));
    const Vector3f iSpecular = pow(specularClamp, _shininess) * lightIntensity * _specularColor;

    return iDiffuse + iSpecular;
}
