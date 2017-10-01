#include "matrixstack.h"

MatrixStack::MatrixStack()
{
	// Initialize the matrix stack with the identity matrix.
}

void MatrixStack::clear()
{
	// Revert to just containing the identity matrix.
	m_matrices.clear();
}

Matrix4f MatrixStack::top()
{
	// Return the top of the stack
	Matrix4f top = Matrix4f::identity();
	for (const Matrix4f& matrix : m_matrices) {
		top = top * matrix;
	}
    return top;
}

void MatrixStack::push( const Matrix4f& m )
{
	// Push m onto the stack.
	// The new top should be "old * m", so that conceptually the new matrix
	// is applied first in right-to-left evaluation.
	m_matrices.push_back(m);
}

void MatrixStack::pop()
{
	// Remove the top element from the stack
	m_matrices.pop_back();
}
