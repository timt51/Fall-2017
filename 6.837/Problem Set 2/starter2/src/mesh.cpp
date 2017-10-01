#include "mesh.h"

#include "vertexrecorder.h"

using namespace std;

void Mesh::load( const char* filename )
{
	// 4.1. load() should populate bindVertices, currentVertices, and faces
    ifstream meshFile;
    meshFile.open(filename);
    if (!meshFile) {
		cout << "Unable to open " << filename << endl;
	}

	string line;
	while (getline(meshFile, line)) {	
		stringstream ss(line);
		string line_type;
		ss >> line_type;
		if(line_type == "v") {
			Vector3f vertex;
			ss >> vertex[0] >> vertex[1] >> vertex[2];
			bindVertices.push_back(vertex);
		} else if (line_type == "f") {
			for (int i=0; i<3; i++) {
				unsigned v1;
				unsigned v2;
				unsigned v3;
				ss >> v1 >> v2 >> v3;
				faces.push_back(Tuple3u(v1-1,v2-1,v3-1));
			}
		} else {
			continue;
		}
    }

	// make a copy of the bind vertices as the current vertices
	currentVertices = bindVertices;
}

void Mesh::draw()
{
	// 4.2 Since these meshes don't have normals
	// be sure to generate a normal per triangle.
	// Notice that since we have per-triangle normals
	// rather than the analytical normals from
	// assignment 1, the appearance is "faceted".
	VertexRecorder rec;
    for (unsigned int i=0; i<faces.size(); i++) {
		Tuple3u indicies = faces[i];
		Vector3f pos1 = currentVertices[indicies[0]];
		Vector3f pos2 = currentVertices[indicies[1]];
		Vector3f pos3 = currentVertices[indicies[2]];
        Vector3f normal = Vector3f::cross(pos2-pos1,pos3-pos1);
		rec.record(pos1, normal);
		rec.record(pos2, normal);
		rec.record(pos3, normal);
    }
	rec.draw();
}

void Mesh::loadAttachments( const char* filename, int numJoints )
{
	// 4.3. Implement this method to load the per-vertex attachment weights
	// this method should update m_mesh.attachments
    ifstream attachFile;
    attachFile.open(filename);
    if (!attachFile) {
		cout << "Unable to open " << filename << endl;
	}

	string line;
	while (getline(attachFile, line)) {	
		stringstream ss(line);
		vector<float> vec;
		vec.push_back(0);
		float f;
		for (int i=0; i<(numJoints-1); ++i){
			ss >> f;
			vec.push_back(f);
		}
		attachments.push_back(vec);
	}
}
