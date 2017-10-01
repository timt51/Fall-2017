#include <fstream>

#include "skeletalmodel.h"
#include <cassert>

#include "starter2_util.h"
#include "vertexrecorder.h"

using namespace std;

const unsigned TRANSLATE_COL_IDX = 3;

SkeletalModel::SkeletalModel() {
    program = compileProgram(c_vertexshader, c_fragmentshader_light);
    if (!program) {
        printf("Cannot compile program\n");
        assert(false);
    }
}

SkeletalModel::~SkeletalModel() {
    // destructor will release memory when SkeletalModel is deleted
    while (m_joints.size()) {
        delete m_joints.back();
        m_joints.pop_back();
    }

    glDeleteProgram(program);
}

void SkeletalModel::load(const char *skeletonFile, const char *meshFile, const char *attachmentsFile)
{
    loadSkeleton(skeletonFile);

    m_mesh.load(meshFile);
    m_mesh.loadAttachments(attachmentsFile, (int)m_joints.size());

    computeBindWorldToJointTransforms();
    updateCurrentJointToWorldTransforms();
}

void SkeletalModel::draw(const Camera& camera, bool skeletonVisible)
{
    // draw() gets called whenever a redraw is required
    // (after an update() occurs, when the camera moves, the window is resized, etc)

    m_matrixStack.clear();

    glUseProgram(program);
    updateShadingUniforms();
    if (skeletonVisible)
    {
        drawJoints(camera);
        drawSkeleton(camera);
    }
    else
    {
        // Tell the mesh to draw itself.
        // Since we transform mesh vertices on the CPU,
        // There is no need to set a Model matrix as uniform
        camera.SetUniforms(program, Matrix4f::identity());
        m_mesh.draw();
    }
    glUseProgram(0);
}

void SkeletalModel::updateShadingUniforms() {
    // UPDATE MATERIAL UNIFORMS
    GLfloat diffColor[] = { 0.4f, 0.4f, 0.4f, 1 };
    GLfloat specColor[] = { 0.9f, 0.9f, 0.9f, 1 };
    GLfloat shininess[] = { 50.0f };
    int loc = glGetUniformLocation(program, "diffColor");
    glUniform4fv(loc, 1, diffColor);
    loc = glGetUniformLocation(program, "specColor");
    glUniform4fv(loc, 1, specColor);
    loc = glGetUniformLocation(program, "shininess");
    glUniform1f(loc, shininess[0]);

    // UPDATE LIGHT UNIFORMS
    GLfloat lightPos[] = { 3.0f, 3.0f, 5.0f, 1.0f };
    loc = glGetUniformLocation(program, "lightPos");
    glUniform4fv(loc, 1, lightPos);

    GLfloat lightDiff[] = { 120.0f, 120.0f, 120.0f, 1.0f };
    loc = glGetUniformLocation(program, "lightDiff");
    glUniform4fv(loc, 1, lightDiff);
}

// https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
// TODO : document this in README
Matrix3f roationMatrix(const Vector3f start, const Vector3f end) {
    if (end == Vector3f::ZERO) {
        return Matrix3f::identity();
    }
    const Vector3f normalizedEnd = end.normalized();    
    const Vector3f v = Vector3f::cross(start, normalizedEnd);
    const float c = Vector3f::dot(start, normalizedEnd);
    const Matrix3f vX = Matrix3f(0, -v.z(), v.y(), 
                                  v.z(), 0, -v.x(),
                                  -v.y(), v.x(), 0);    const Matrix3f res = Matrix3f::identity()+vX+vX*vX*(1/(1+c));
    return res;
}

// Assumes first row of file describes the root node
// TODO: document this fact in the readme
void SkeletalModel::loadSkeleton(const char* filename)
{
    // Load the skeleton from file here.
    ifstream skelFile;
    skelFile.open(filename);
    if (!skelFile) {
        cout << "Unable to open " << filename << endl;
    }

    float x;
    float y;
    float z;
    signed int parentJointIndex;
    while (skelFile >> x >> y >> z >> parentJointIndex) {
        // Create new joint.
        Joint *joint = new Joint;
        // Determine rotation matrix
        joint->transform = Matrix4f::identity();
        // if (parentJointIndex != -1) {
        //     joint->transform.setSubmatrix3x3(0,0,roationMatrix(Vector3f::FORWARD, Vector3f(x,y,z)));        
        //     joint->transform.setCol(TRANSLATE_COL_IDX, Vector4f(Vector3f(x,y,z).abs(),0,0,1));
        // } else {
        //     joint->transform.setCol(TRANSLATE_COL_IDX, Vector4f(x,y,z,1));
        // }
        //joint->transform.setCol(TRANSLATE_COL_IDX, Vector4f(.5,0,0,1));
        joint->transform.setCol(TRANSLATE_COL_IDX, Vector4f(x,y,z,1));
        joint->children = vector<Joint*>();
        if (parentJointIndex != -1) {
            m_joints[parentJointIndex]->children.push_back(joint);
        }

        // Add the joint pointer
        m_joints.push_back(joint);
    }

    // Set the root joint
    m_rootJoint = m_joints[0];

    skelFile.close();
}

void drawJoint(const Joint *joint, MatrixStack m_matrixStack, const Camera& camera, const GLuint program) {
    m_matrixStack.push(joint->transform);
    camera.SetUniforms(program, m_matrixStack.top());
    drawSphere(0.025f, 12, 12);
    for (const Joint* childJoint : joint->children) {
        drawJoint(childJoint, m_matrixStack, camera, program);
    }
    m_matrixStack.pop();
}

void SkeletalModel::drawJoints(const Camera& camera)
{
    // Draw a sphere at each joint. You will need to add a recursive
    // helper function to traverse the joint hierarchy.
    //
    // We recommend using drawSphere( 0.025f, 12, 12 )
    // to draw a sphere of reasonable size.
    //
    // You should use your MatrixStack class. A function
    // should push it's changes onto the stack, and
    // use stack.pop() to revert the stack to the original
    // state.

    // m_matrixStack should be cleared here
    // TODO : maybe assert this
    drawJoint(m_rootJoint, m_matrixStack, camera, program);
}

void drawBone(const Joint *joint, MatrixStack m_matrixStack, const Camera& camera, const GLuint program) {
    // you can use the stack with push/pop like this
    // m_matrixStack.push(Matrix4f::translation(+0.6f, +0.5f, -0.5f))
    // camera.SetUniforms(program, m_matrixStack.top());
    // drawCylinder(6, 0.02f, 0.2f);
    // callChildFunction();
    // m_matrixStack.pop();
    //return;

    m_matrixStack.push(joint->transform);
    for (const Joint* childJoint : joint->children) {
        Matrix4f m = Matrix4f::identity();
        m.setSubmatrix3x3(0,0,roationMatrix(Vector3f::UP, childJoint->transform.getCol(3).xyz()));
        camera.SetUniforms(program, m_matrixStack.top()*m);
        drawCylinder(6, 0.02f, childJoint->transform.getCol(3).xyz().abs());
        camera.SetUniforms(program, m_matrixStack.top());
        drawBone(childJoint, m_matrixStack, camera, program);
    }
    m_matrixStack.pop();
}

void SkeletalModel::drawSkeleton(const Camera& camera)
{
    // Draw cylinders between the joints. You will need to add a recursive 
    // helper function to traverse the joint hierarchy.
    //
    // We recommend using drawCylinder(6, 0.02f, <height>);
    // to draw a cylinder of reasonable diameter.

    drawBone(m_rootJoint, m_matrixStack, camera, program);
}

void SkeletalModel::setJointTransform(int jointIndex, float rX, float rY, float rZ)
{
    // Set the rotation part of the joint's transformation matrix based on the passed in Euler angles.
    Matrix3f rotX(1,0,0,
                  0,cos(rX),-sin(rX),
                  0,sin(rX),cos(rX));
    Matrix3f rotY(cos(rY), 0, sin(rY),
                  0,1,0,
                  -sin(rY),0,cos(rY));
    Matrix3f rotZ(cos(rZ),-sin(rZ),0,
                  sin(rZ),cos(rZ),0,
                  0,0,1);
    Matrix3f m = rotX * rotY * rotZ;
    m_joints[jointIndex]->transform.setSubmatrix3x3(0,0,m);
}

void bindWorldToJointTransform(Joint *joint, MatrixStack m_matrixStack) {
    m_matrixStack.push(joint->transform);
    Matrix4f m = Matrix4f::identity();
    m.setCol(3, -m_matrixStack.top().getCol(3));
    m(3,3) = 1;
    joint->bindWorldToJointTransform = m;
    for (Joint* childJoint : joint->children) {
        bindWorldToJointTransform(childJoint, m_matrixStack);
    }
    m_matrixStack.pop();
}

void SkeletalModel::computeBindWorldToJointTransforms()
{
    // 2.3.1. Implement this method to compute a per-joint transform from
    // world-space to joint space in the BIND POSE.
    //
    // Note that this needs to be computed only once since there is only
    // a single bind pose.
    //
    // This method should update each joint's bindWorldToJointTransform.
    // You will need to add a recursive helper function to traverse the joint hierarchy.
    bindWorldToJointTransform(m_rootJoint, m_matrixStack);
}

void currentJointToWorldTransform(Joint *joint, MatrixStack m_matrixStack) {
    m_matrixStack.push(joint->transform);
    joint->currentJointToWorldTransform = m_matrixStack.top();
    for (Joint* childJoint : joint->children) {
        currentJointToWorldTransform(childJoint, m_matrixStack);
    }
    m_matrixStack.pop();
}

void SkeletalModel::updateCurrentJointToWorldTransforms()
{
    // 2.3.2. Implement this method to compute a per-joint transform from
    // joint space to world space in the CURRENT POSE.
    //
    // The current pose is defined by the rotations you've applied to the
    // joints and hence needs to be *updated* every time the joint angles change.
    //
    // This method should update each joint's currentJointToWorldTransform.
    // You will need to add a recursive helper function to traverse the joint hierarchy.
    currentJointToWorldTransform(m_rootJoint, m_matrixStack);
}

void SkeletalModel::updateMesh()
{
    // 2.3.2. This is the core of SSD.
    // Implement this method to update the vertices of the mesh
    // given the current state of the skeleton.
    // You will need both the bind pose world --> joint transforms.
    // and the current joint --> world transforms.
    for (unsigned vertexIndex = 0; vertexIndex < m_mesh.bindVertices.size(); ++vertexIndex) {
        const Vector4f v(m_mesh.bindVertices[vertexIndex],1);
        Vector3f newCurrentVector = Vector3f::ZERO;
        const vector<float> weights = m_mesh.attachments[vertexIndex];
        for (unsigned weightIndex = 0; weightIndex < weights.size(); ++weightIndex) {
            const float w = weights[weightIndex];
            if (w < 1E-6) { continue; }
            const Matrix4f b = m_joints[weightIndex]->bindWorldToJointTransform;
            const Matrix4f t = m_joints[weightIndex]->currentJointToWorldTransform;
             newCurrentVector += w*((t*b*v).xyz());
        }

        m_mesh.currentVertices[vertexIndex] = newCurrentVector;
    }
}
