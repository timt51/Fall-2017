#include "curve.h"
#include "vertexrecorder.h"
// #include <stdlib.h>
using namespace std;

const float c_pi = 3.14159265358979323846f;

namespace
{
// Approximately equal to.  We don't want to use == because of
// precision issues with floating point.
inline bool approx(const Vector3f& lhs, const Vector3f& rhs)
{
	const float eps = 1e-8f;
	return (lhs - rhs).absSquared() < eps;
}
}

// Returns the CurvePoint representation of a point in 3D given
// the previous CurvePoint on the Curve
CurvePoint pointToCurvePoint(const Vector3f& point, const Vector3f& tangent, const Vector3f& prevBinormal) {
	// Vertex - point
	// Tanget - ???
	// Normal - binormal from prev cross tangent
	// Binormal - tangent cross normal
	const Vector3f normal = Vector3f::cross(prevBinormal, tangent).normalized();
	return CurvePoint { point, tangent.normalized(), normal, prevBinormal.normalized() };
}

Curve interpolateBezier(const vector<Vector3f>& points, unsigned steps) {
	return Curve();
}

// Want a subroutine interpolateBezier(vector<Vector3f>& points, unsigned steps)
// it returns points that are not the first or last point 
// (b/c the caller should already know these points)
Curve evalBezier(const vector< Vector3f >& P, unsigned steps)
{
	// Check
	if (P.size() < 4 || P.size() % 3 != 1)
	{
		cerr << "evalBezier must be called with 3n+1 control points." << endl;
		exit(0);
	}

	Curve curve;
	// Calculate initial curve point
	// Create initial binormal B = anything that is not parallel to initial Tangent
	// Vertex - first point in P
	// Tangent - (P[1] - P[0]).normalized()
	// Normal - initial binormal cross tangent
	// Binormal - tangent cross normal
	Vector3f firstTangent = (P.at(1)-P.at(0)).normalized();
	Vector3f firstBinormal;
	while(firstBinormal == NULL) {
		Vector3f binormal(rand(), rand(), rand());
		binormal.normalize();
		if (Vector3f::dot(firstTangent, binormal) > 1E-6) {
			firstBinormal = Vector3f(binormal);
		}
	}
	curve.push_back(pointToCurvePoint(P.front(), firstTangent, firstBinormal));
	for (unsigned segmentIndex=0; segmentIndex<P.size(); segmentIndex+=3) {
		// compute points on this segment using interpBezier
		// add those points
		const vector<Vector3f> controlPoints(P.begin()+segmentIndex, P.begin()+segmentIndex+4);
		const Curve segmentPoints = interpolateBezier(controlPoints, steps);
		curve.insert(curve.end(), segmentPoints.begin(), segmentPoints.end());
		// add the final point of this segment
		const Vector3f tangent = (P.at(segmentIndex+3)-P.at(segmentIndex+2)).normalized();
		const Vector3f prevBinormal = curve.back().B;
		const CurvePoint lastSegmentPoint = pointToCurvePoint(P.at(segmentIndex+3), tangent, prevBinormal);
		curve.push_back(lastSegmentPoint);
	}
	
	// TODO:
	// You should implement this function so that it returns a Curve
	// (e.g., a vector< CurvePoint >).  The variable "steps" tells you
	// the number of points to generate on each piece of the spline.
	// At least, that's how the sample solution is implemented and how
	// the SWP files are written.  But you are free to interpret this
	// variable however you want, so long as you can control the
	// "resolution" of the discretized spline curve with it.

	// Make sure that this function computes all the appropriate
	// Vector3fs for each CurvePoint: V,T,N,B.
	// [NBT] should be unit and orthogonal.

	// Also note that you may assume that all Bezier curves that you
	// receive have G1 continuity.  Otherwise, the TNB will not be
	// be defined at points where this does not hold.

	cerr << "\t>>> evalBezier has been called with the following input:" << endl;

	cerr << "\t>>> Control points (type vector< Vector3f >): " << endl;
	for (int i = 0; i < (int)P.size(); ++i)
	{
		cerr << "\t>>> " << P[i] << endl;
	}

	cerr << "\t>>> Steps (type steps): " << steps << endl;
	cerr << "\t>>> Returning empty curve." << endl;

	// Right now this will just return this empty curve.
	return Curve();
}

Curve evalBspline(const vector< Vector3f >& P, unsigned steps)
{
	// Check
	if (P.size() < 4)
	{
		cerr << "evalBspline must be called with 4 or more control points." << endl;
		exit(0);
	}

	// TODO:
	// It is suggested that you implement this function by changing
	// basis from B-spline to Bezier.  That way, you can just call
	// your evalBezier function.

	cerr << "\t>>> evalBSpline has been called with the following input:" << endl;

	cerr << "\t>>> Control points (type vector< Vector3f >): " << endl;
	for (int i = 0; i < (int)P.size(); ++i)
	{
		cerr << "\t>>> " << P[i] << endl;
	}

	cerr << "\t>>> Steps (type steps): " << steps << endl;
	cerr << "\t>>> Returning empty curve." << endl;

	// Return an empty curve right now.
	return Curve();
}

Curve evalCircle(float radius, unsigned steps)
{
	// This is a sample function on how to properly initialize a Curve
	// (which is a vector< CurvePoint >).

	// Preallocate a curve with steps+1 CurvePoints
	Curve R(steps + 1);

	// Fill it in counterclockwise
	for (unsigned i = 0; i <= steps; ++i)
	{
		// step from 0 to 2pi
		float t = 2.0f * c_pi * float(i) / steps;

		// Initialize position
		// We're pivoting counterclockwise around the y-axis
		R[i].V = radius * Vector3f(cos(t), sin(t), 0);

		// Tangent vector is first derivative
		R[i].T = Vector3f(-sin(t), cos(t), 0);

		// Normal vector is second derivative
		R[i].N = Vector3f(-cos(t), -sin(t), 0);

		// Finally, binormal is facing up.
		R[i].B = Vector3f(0, 0, 1);
	}

	return R;
}

void recordCurve(const Curve& curve, VertexRecorder* recorder)
{
	const Vector3f WHITE(1, 1, 1);
	for (int i = 0; i < (int)curve.size() - 1; ++i)
	{
		recorder->record_poscolor(curve[i].V, WHITE);
		recorder->record_poscolor(curve[i + 1].V, WHITE);
	}
}
void recordCurveFrames(const Curve& curve, VertexRecorder* recorder, float framesize)
{
	Matrix4f T;
	const Vector3f RED(1, 0, 0);
	const Vector3f GREEN(0, 1, 0);
	const Vector3f BLUE(0, 0, 1);
	
	const Vector4f ORGN(0, 0, 0, 1);
	const Vector4f AXISX(framesize, 0, 0, 1);
	const Vector4f AXISY(0, framesize, 0, 1);
	const Vector4f AXISZ(0, 0, framesize, 1);

	for (int i = 0; i < (int)curve.size(); ++i)
	{
		T.setCol(0, Vector4f(curve[i].N, 0));
		T.setCol(1, Vector4f(curve[i].B, 0));
		T.setCol(2, Vector4f(curve[i].T, 0));
		T.setCol(3, Vector4f(curve[i].V, 1));
 
		// Transform orthogonal frames into model space
		Vector4f MORGN  = T * ORGN;
		Vector4f MAXISX = T * AXISX;
		Vector4f MAXISY = T * AXISY;
		Vector4f MAXISZ = T * AXISZ;

		// Record in model space
		recorder->record_poscolor(MORGN.xyz(), RED);
		recorder->record_poscolor(MAXISX.xyz(), RED);

		recorder->record_poscolor(MORGN.xyz(), GREEN);
		recorder->record_poscolor(MAXISY.xyz(), GREEN);

		recorder->record_poscolor(MORGN.xyz(), BLUE);
		recorder->record_poscolor(MAXISZ.xyz(), BLUE);
	}
}

