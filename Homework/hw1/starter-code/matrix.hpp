#ifndef _MATRIX_HPP
#define _MATRIX_HPP
#include <iostream>
#include <ostream>
#include <stdexcept>
#include <cstring>

// same idea as typedef
using uint = unsigned int;

template <typename T>

class Matrix
{
public:
    // //constructor
    // virtual Matrix(uint rows, uint cols)=0;
    // virtual Matrix(uint n)=0;
    // //destructor
    // virtual ~Matrix()=0;
    // //arithmetic operators
    // virtual Matrix<T> operator +(const Matrix<T>& m)=0;

    // virtual Matrix<T> operator -(const Matrix<T>& m)=0;

    // virtual Matrix<T> operator *(const Matrix<T>& m)=0;

    // virtual Matrix<T> operator /(const Matrix<T>& m)=0;
    //access operator
    virtual T& operator ()(uint i, uint j)=0;
    //l0 norm
    virtual unsigned int l0norm()=0;
};

template <typename T>
class MatrixSymmetric: public Matrix<T>
{
public:
    MatrixSymmetric(uint n)
    {
      this->n = n;
      this->size = n * (n+1)/2;
      this->elements = new T[this->size];
    }

    MatrixSymmetric(const MatrixSymmetric<T>& m){
        this->n = m.n;
        this->size = this->n*(this->n+1)/2;
        this->elements = new T[this->size];
        std::memcpy (this->elements, m.elements,this->size*sizeof(T));
    }

    ~MatrixSymmetric()
    {
      if (this->elements)
        {
            delete[] this->elements;
            this->elements = nullptr;
        }
    }
    
    MatrixSymmetric<T> operator +(const MatrixSymmetric<T>& m)
    {
      if (this->n != m.n)
          throw std::invalid_argument("Invalid matrix dimension.");
      MatrixSymmetric<T> temp(this->n);
      for (uint i=0; i < this->size; i++)
      {
        temp.elements[i] = this->elements[i] + m.elements[i];
      }   
      return temp;
    }

    MatrixSymmetric<T> operator -(const MatrixSymmetric<T>& m)
    {
      if (this->n != m.n)
          throw std::invalid_argument("Invalid matrix dimension.");
      MatrixSymmetric<T> temp(this->n);
      for (uint i=0; i < this->size; i++)
      {
        temp.elements[i] = this->elements[i] - m.elements[i];
      }  
      return temp; 
    }

    MatrixSymmetric<T> operator *(const MatrixSymmetric<T>& m)
    {
      if (this->n != m.n)
          throw std::invalid_argument("Invalid matrix dimension.");
      MatrixSymmetric<T> temp(this->n);
      for (uint i=0; i < this->size; i++)
      {
        temp.elements[i] = this->elements[i] * m.elements[i];
      }   
      return temp;
    }

    MatrixSymmetric<T> operator /(const MatrixSymmetric<T>& m)
    {
      if (this->n != m.n)
          throw std::invalid_argument("Invalid matrix dimension.");
      MatrixSymmetric<T> temp(this->n);
      for (uint i=0; i < this->size; i++)
      {
        temp.elements[i] = this->elements[i] / m.elements[i];
      }   
      return temp;
    }

    const T& operator ()(uint row, uint col) const
    {
      if (col>=row)
        return this->elements[this->n*row+col-row*(row+1)/2];
      else
        return this->elements[this->n*col+row-col*(col+1)/2];
    }

    T& operator ()(uint row, uint col)
    {
      if (col>=row)
        return this->elements[this->n*row+col-row*(row+1)/2];
      else
        return this->elements[this->n*col+row-col*(col+1)/2];
    }

    uint l0norm()
    {
      uint num=0;
      for(uint i=0; i < this->size; i++)
      {
        if(this->elements[i]!=0){
          num++;
        }
      }
      return num;
    }

    template <typename U>
    friend std::ostream& operator <<(std::ostream& stream, MatrixSymmetric<U> m)
    {
        m.print(stream);
        return stream;
    }

    uint size_of_matrix()
    {
      return this->n;
    }
private:
    uint n;
    uint size;
    T *elements;

    void print(std::ostream& stream)
    {
        for (uint i = 0; i < this->n; i++)
        {
            for (uint j = 0; j < this->n; j++)
                stream << (*this)(i, j) << " ";
            stream << std::endl;
        }   
    }
};

#endif /* MATRIX_HPP */