#include "curve.h"
#include "vertexrecorder.h"
using namespace std;

const float c_pi = 3.14159265358979323846f;
const Matrix4f bernstein(1, -3, 3, -1,
						 0, 3, -6, 3,
						 0, 0, 3, -3,
						 0, 0, 0, 1);
const Matrix4f bernstein_deriv(-3, 6, -3, 0,
						 	   3, -12, 9, 0,
						 	   0, 6, -9, 0,
							   0, 0, 3, 0);
const Matrix4f bernstein_inv = bernstein.inverse();
const Matrix4f b_spline(1/6., -3/6., 3/6., -1/6.,
						4/6., 0, -6/6., 3/6.,
						1/6., 3/6., 3/6., -3/6.,
						0, 0, 0, 1/6.);
const Matrix4f b_to_bezier = b_spline*bernstein_inv;

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

Curve interpolateBezier(const vector<Vector3f>& points, const CurvePoint& first, unsigned steps) {
	const Matrix4f geometry(Vector4f(points[0],0),
							Vector4f(points[1],0),
							Vector4f(points[2],0),
							Vector4f(points[3],0), true);
	const Matrix4f toPoint = geometry * bernstein;
	const Matrix4f toVelocity = geometry * bernstein_deriv;

	// TODO: what happens when steps is 1 or something weird?
	Curve curve;
	curve.push_back(first);
	for (unsigned step_index=1; step_index<=steps; step_index++) {
		const double t = step_index/(steps+1.0);
		const Vector4f t_vec(1,t,t*t,t*t*t);
		const Vector3f point = (toPoint*t_vec).xyz();
		const Vector3f tangent = (toVelocity*t_vec).xyz().normalized();
		const Vector3f prevBinormal = curve.back().B;
		const CurvePoint segmentPoint = pointToCurvePoint(point, tangent, prevBinormal);
		curve.push_back(segmentPoint);
	}
	curve.erase(curve.begin());
	return curve;
}

CurvePoint generateFirstCurvePoint(const Vector3f& firstControlPoint, const Vector3f& secondControlPoint) {
	// Calculate initial curve point
	// Create initial binormal B = anything that is not parallel to initial Tangent
	// Vertex - first point in P
	// Tangent - (P[1] - P[0]).normalized()
	// Normal - initial binormal cross tangent
	// Binormal - tangent cross normal
	Vector3f firstTangent = (secondControlPoint-firstControlPoint).normalized();
	Vector3f firstBinormal = Vector3f(0,0,1);
	while(approx(firstBinormal, firstTangent) || approx(firstBinormal, -firstTangent)) {
		firstBinormal = Vector3f(rand(), rand(), rand()).normalized();
	}
	const CurvePoint firstCurvePoint = pointToCurvePoint(firstControlPoint, firstTangent, firstBinormal);
	return firstCurvePoint;
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
	curve.push_back(generateFirstCurvePoint(P[0], P[1]));
	for (unsigned segmentIndex=0; segmentIndex<P.size()-1; segmentIndex+=3) {
		// compute points on this segment using interpBezier
		// add those points
		const vector<Vector3f> controlPoints(P.begin()+segmentIndex, P.begin()+segmentIndex+4);
		const Curve segmentPoints = interpolateBezier(controlPoints, curve.back(), steps);
		curve.insert(curve.end(), segmentPoints.begin(), segmentPoints.end());
		// add the final point of this segment
		const Vector3f tangent = (P.at(segmentIndex+3)-P.at(segmentIndex+2)).normalized();
		const Vector3f prevBinormal = curve.back().B;
		const CurvePoint lastSegmentPoint = pointToCurvePoint(P.at(segmentIndex+3), tangent, prevBinormal);
		curve.push_back(lastSegmentPoint);
	}
	
	cerr << "\t>>> evalBezier has been called." << endl;
	cerr << "\t>>> Steps (type steps): " << steps << endl;

	return curve;
}

Curve evalBspline(const vector< Vector3f >& P, unsigned steps)
{
	// Check
	if (P.size() < 4)
	{
		cerr << "evalBspline must be called with 4 or more control points." << endl;
		exit(0);
	}

	vector<Vector3f> bezierPoints;
	bezierPoints.push_back(Vector3f(0,0,0));
	for (unsigned segmentIndex=0; segmentIndex<P.size()-3; segmentIndex++) {
		Matrix4f bControlPoints(Vector4f(P[segmentIndex],0),
							    Vector4f(P[segmentIndex+1],0),
							    Vector4f(P[segmentIndex+2],0),
							    Vector4f(P[segmentIndex+3],0));
		Matrix4f bernsteinControlPoints = bControlPoints*b_to_bezier;
		bezierPoints.erase(bezierPoints.end());
		bezierPoints.push_back(bernsteinControlPoints.getCol(0).xyz());
		bezierPoints.push_back(bernsteinControlPoints.getCol(1).xyz());
		bezierPoints.push_back(bernsteinControlPoints.getCol(2).xyz());
		bezierPoints.push_back(bernsteinControlPoints.getCol(3).xyz());
	}

	cerr << "\t>>> evalBSpline has been called." << endl;
	cerr << "\t>>> Steps (type steps): " << steps << endl;

	return evalBezier(bezierPoints, steps);
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

