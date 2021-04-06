#include "matrix.hpp"


int main()
{
    // TODO: Write your tests here //
    MatrixSymmetric<double> a(3);
    MatrixSymmetric<double> b(3);
    MatrixSymmetric<double> X(4);
    //3(3+1)/2 = 6
    for (int i = 0; i < 3; i++){
        for (int j = i; j < 3; j++){
            a(i,j) = i;
            b(i,j) = i+j+1;
        }
    }
    for (int i = 0; i < 4; i++){
        for (int j = i; j < 4; j++){
            X(i,j) = i;
        }
    }
    //show matrix a and b
    std::cout<<"Symmetric matrix A: "<<std::endl;
    std::cout<<a<<std::endl;
    std::cout<<"Size of symmetric matrix A:"<<a.size_of_matrix()<<std::endl;
    std::cout<<"Symmetric matrix B: "<<std::endl;
    std::cout<<b<<std::endl;
    std::cout<<"Size of symmetric matrix B:"<<b.size_of_matrix()<<std::endl;
    std::cout<<"Symmetric matrix X: "<<std::endl;
    std::cout<<X<<std::endl;
    std::cout<<"Size of symmetric matrix X:"<<X.size_of_matrix()<<std::endl;
    MatrixSymmetric<double> c = a + b;
    std::cout << "C=A+B: " << std::endl<<c<<std::endl;

    MatrixSymmetric<double> d = a - b;
    std::cout << "D=A-B: " << std::endl<<d<<std::endl;

    MatrixSymmetric<double> e = a * b;
    std::cout << "E=A*B: " << std::endl<<e<<std::endl;

    MatrixSymmetric<double> f = a / b;
    std::cout << "f=A/B: " << std::endl<<f<<std::endl;

    //test l0norm
    std::cout <<"l0 norm of A: "<<a.l0norm()<<std::endl;
    std::cout <<"l0 norm of B: "<<b.l0norm()<<std::endl;

    //change element
    a(0,0)=10;
    std::cout<<"Symmetric matrix A after change: "<<std::endl;
    std::cout<<a<<std::endl;    


    return 0;
}