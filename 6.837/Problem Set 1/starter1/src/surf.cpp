#include <math.h>
#include "surf.h"
#include "vertexrecorder.h"
using namespace std;

namespace
{
    // We're only implenting swept surfaces where the profile curve is
    // flat on the xy-plane.  This is a check function.
    static bool checkFlat(const Curve &profile)
    {
        for (unsigned i=0; i<profile.size(); i++)
            if (profile[i].V[2] != 0.0 ||
                profile[i].T[2] != 0.0 ||
                profile[i].N[2] != 0.0)
                return false;
    
        return true;
    }
}

// DEBUG HELPER
Surface quad() { 
	Surface ret;
	ret.VV.push_back(Vector3f(-1, -1, 0));
	ret.VV.push_back(Vector3f(+1, -1, 0));
	ret.VV.push_back(Vector3f(+1, +1, 0));
	ret.VV.push_back(Vector3f(-1, +1, 0));

	ret.VN.push_back(Vector3f(0, 0, 1));
	ret.VN.push_back(Vector3f(0, 0, 1));
	ret.VN.push_back(Vector3f(0, 0, 1));
	ret.VN.push_back(Vector3f(0, 0, 1));

	ret.VF.push_back(Tup3u(0, 1, 2));
	ret.VF.push_back(Tup3u(0, 2, 3));
	return ret;
}

Matrix3f rotationMatrixYAxis(double radians) {
    return Matrix3f(cos(radians), 0 , sin(radians),
                    0, 1, 0,
                    -sin(radians), 0, cos(radians));
}

vector<Vector3f> rotateCurve(const vector<Vector3f>& curve, const Matrix3f& rotationMatrix) {
    vector<Vector3f> rotatedCurve;
    for (unsigned pointIndex=0; pointIndex<curve.size(); pointIndex++) {
        const Vector3f rotatedPoint = rotationMatrix*curve[pointIndex];
        rotatedCurve.push_back(rotatedPoint);
    }
    return rotatedCurve;
}

vector<Vector3f> rotateCurveNormals(const vector<Vector3f>& curveNormals, const Matrix3f& normalRotationMatrix) {
    vector<Vector3f> rotatedCurveNormals;
    for (unsigned pointIndex=0; pointIndex<curveNormals.size(); pointIndex++) {
        const Vector3f rotatedPoint = normalRotationMatrix*curveNormals[pointIndex];
        rotatedCurveNormals.push_back(rotatedPoint);
    }
    return rotatedCurveNormals;
}

// TODO: there may be a problem with repeating vertices and normals in surface.VV and surface.VN?
Surface generateTriangleMesh(const vector<vector<Vector3f>>& profiles, const vector<vector<Vector3f>>& normalProfiles) {
    Surface surface = Surface();

    for (unsigned profileIndex=0; profileIndex < profiles.size(); profileIndex++) {
        surface.VV.insert(surface.VV.end(), profiles[profileIndex].begin(), profiles[profileIndex].end());
        surface.VN.insert(surface.VN.end(), normalProfiles[profileIndex].begin(), normalProfiles[profileIndex].end());
    }

    unsigned profileLength = profiles.front().size();
    for (unsigned profileIndex=0; profileIndex < profiles.size()-1; profileIndex++) {
        for (unsigned index=1; index < profileLength; index++) {
            surface.VF.push_back(Tup3u(profileIndex*profileLength + index,
                                       (profileIndex+1)*profileLength + index-1,
                                       profileIndex*profileLength + index-1));
            surface.VF.push_back(Tup3u(profileIndex*profileLength + index,
                                       (profileIndex+1)*profileLength + index,
                                       (profileIndex+1)*profileLength + index-1));
        }
    }
    return surface;
}

Surface makeSurfRev(const Curve &profile, unsigned steps)
{    
    if (!checkFlat(profile))
    {
        cerr << "surfRev profile curve must be flat on xy plane." << endl;
        exit(0);
    }

    const double radians = 2 * M_PI / (steps + 1);
    const Matrix3f rotationMatrix = rotationMatrixYAxis(radians);
    const Matrix3f normalRotationMatrix = rotationMatrix.transposed().inverse();
    vector<Vector3f> profilePoints;
    vector<Vector3f> profileNormals;
    for (unsigned pointIndex=0; pointIndex<profile.size(); pointIndex++) {
        profilePoints.push_back(profile[pointIndex].V);
        profileNormals.push_back(-profile[pointIndex].N);
    }
    
    vector<vector<Vector3f>> rotatedProfiles;
    rotatedProfiles.push_back(profilePoints);
    vector<vector<Vector3f>> rotatedProfileNormals;
    rotatedProfileNormals.push_back(profileNormals);
    for (unsigned curveIndex=1; curveIndex<=steps; curveIndex++) {
        const vector<Vector3f> rotatedProfile = rotateCurve(rotatedProfiles.back(), rotationMatrix);
        rotatedProfiles.push_back(rotatedProfile);
        const vector<Vector3f> rotatedProfileNormal = rotateCurve(rotatedProfileNormals.back(), normalRotationMatrix);
        rotatedProfileNormals.push_back(rotatedProfileNormal);
    }

    rotatedProfiles.push_back(rotatedProfiles.front());
    rotatedProfileNormals.push_back(rotatedProfileNormals.front());
    Surface surface = generateTriangleMesh(rotatedProfiles, rotatedProfileNormals);

    cerr << "\t>>> makeSurfRev called." << endl;
 
    return surface;
}

Surface makeGenCyl(const Curve &profile, const Curve &sweep )
{
    if (!checkFlat(profile))
    {
        cerr << "genCyl profile curve must be flat on xy plane." << endl;
        exit(0);
    }

    vector<vector<Vector3f>> profiles;
    vector<vector<Vector3f>> normals;
    for (unsigned sweepIndex=0; sweepIndex<sweep.size(); sweepIndex++) {
        const Matrix4f M(Vector4f(sweep[sweepIndex].N,0),
                        Vector4f(sweep[sweepIndex].B,0),
                        Vector4f(sweep[sweepIndex].T,0),
                        Vector4f(sweep[sweepIndex].V,1), true);
        const Matrix3f M_inv_transpose = M.getSubmatrix3x3(0,0).inverse().transposed();
        
        vector<Vector3f> thisProfile;
        vector<Vector3f> thisNormals;
        for (unsigned profileIndex=0; profileIndex<profile.size(); profileIndex++) {
            thisProfile.push_back((M*Vector4f(profile[profileIndex].V,1)).xyz());
            thisNormals.push_back(M_inv_transpose*-profile[profileIndex].N);
        }
        
        profiles.push_back(thisProfile);
        normals.push_back(thisNormals);
    }
    Surface surface = generateTriangleMesh(profiles, normals);

    cerr << "\t>>> makeGenCyl called." <<endl;

    return surface;
}

void recordSurface(const Surface &surface, VertexRecorder* recorder) {
	const Vector3f WIRECOLOR(0.4f, 0.4f, 0.4f);
    for (int i=0; i<(int)surface.VF.size(); i++)
    {
		recorder->record(surface.VV[surface.VF[i][0]], surface.VN[surface.VF[i][0]], WIRECOLOR);
		recorder->record(surface.VV[surface.VF[i][1]], surface.VN[surface.VF[i][1]], WIRECOLOR);
		recorder->record(surface.VV[surface.VF[i][2]], surface.VN[surface.VF[i][2]], WIRECOLOR);
    }
}

void recordNormals(const Surface &surface, VertexRecorder* recorder, float len)
{
	const Vector3f NORMALCOLOR(0, 1, 1);
    for (int i=0; i<(int)surface.VV.size(); i++)
    {
		recorder->record_poscolor(surface.VV[i], NORMALCOLOR);
		recorder->record_poscolor(surface.VV[i] + surface.VN[i] * len, NORMALCOLOR);
    }
}

void outputObjFile(ostream &out, const Surface &surface)
{
    
    for (int i=0; i<(int)surface.VV.size(); i++)
        out << "v  "
            << surface.VV[i][0] << " "
            << surface.VV[i][1] << " "
            << surface.VV[i][2] << endl;

    for (int i=0; i<(int)surface.VN.size(); i++)
        out << "vn "
            << surface.VN[i][0] << " "
            << surface.VN[i][1] << " "
            << surface.VN[i][2] << endl;

    out << "vt  0 0 0" << endl;
    
    for (int i=0; i<(int)surface.VF.size(); i++)
    {
        out << "f  ";
        for (unsigned j=0; j<3; j++)
        {
            unsigned a = surface.VF[i][j]+1;
            out << a << "/" << "1" << "/" << a << " ";
        }
        out << endl;
    }
}
