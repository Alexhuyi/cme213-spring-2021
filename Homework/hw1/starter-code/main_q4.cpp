#include <iostream>
#include <algorithm>
#include <vector>
#include <list>
#include <numeric>
#include <stdexcept>
#include <cassert>

/**********  Q4a: DAXPY **********/
template <typename T>
std::vector<T> daxpy(T a, const std::vector<T>& x, const std::vector<T>& y)
{
    // TODO
    size_t i=0;
    std::vector<T> z(x.size());
    std::for_each(z.begin(), z.end(),[&](T& z_element){z_element=a*x[i]+y[i];i++;});
    return z;
}


/**********  Q4b: All students passed **********/
constexpr double HOMEWORK_WEIGHT = 0.20;
constexpr double MIDTERM_WEIGHT = 0.35;
constexpr double FINAL_EXAM_WEIGHT = 0.45;

struct Student
{
    double homework;
    double midterm;
    double final_exam;

    Student(double hw, double mt, double fe) : 
           homework(hw), midterm(mt), final_exam(fe) { }
};

bool all_students_passed(const std::vector<Student>& students, double pass_threshold) 
{
    // TODO 
    return std::all_of(students.begin(), students.end(),[&](const Student & std){ return pass_threshold <= std.homework*HOMEWORK_WEIGHT+std.midterm*MIDTERM_WEIGHT+std.final_exam*FINAL_EXAM_WEIGHT;});
}


/**********  Q4c: Odd first, even last **********/
void sort_odd_even(std::vector<int>& data)
{
    // TODO
    std::sort(data.begin(), data.end(),[](int i, int j){
        if (i%2==0 && j%2!=0){
            return false;
        }
        else if (i%2!=0 && j%2==0){
            return true;
        }
        else
            return i<j;
    });
    return;
}

/**********  Q4d: Sparse matrix list sorting **********/
template <typename T>
struct SparseMatrixCoordinate
{
    int row;
    int col;
    T data;
    
    SparseMatrixCoordinate(int r, int c, T d) :
        row(r), col(c), data(d) { }
};

template <typename T>
void sparse_matrix_sort(std::list<SparseMatrixCoordinate<T> >& list) 
{
    // TODO
    list.sort([](SparseMatrixCoordinate<T> entry_1, SparseMatrixCoordinate<T> entry_2){
        if (entry_1.row < entry_2.row)
            return true;
        else if (entry_1.row == entry_2.row && entry_1.col < entry_2.col)
            return true;
        else
            return false;
    });
    return;
}

int main() 
{    
    // Q4a test
    const int Q4_A = 2;
    const std::vector<int> q4a_x = {-2, -1, 0, 1, 2};
    const std::vector<int> q4_y = {-2, -1, 0, 1, 2};

    // TODO: Verify your Q4a implementation
    std::vector<int> z = daxpy(Q4_A,q4a_x,q4_y);
    bool sign_q4a = true;
    for (size_t i = 0; i < q4a_x.size();i++){
        if (z[i]!=Q4_A*q4a_x[i]+q4_y[i]){
            sign_q4a=false;
            break;
        }
    }
    assert(sign_q4a);
    std::cout <<"Q4a passes the test."<< std::endl;
    // Q4b test
    std::vector<Student> all_pass_students = {
            Student(1., 1., 1.),
            Student(0.6, 0.6, 0.6),
            Student(0.8, 0.65, 0.7)};

    std::vector<Student> not_all_pass_students = {
            Student(1., 1., 1.),
            Student(0, 0, 0)};

    // TODO: Verify your Q4b implementation
    bool case_1 = all_students_passed(all_pass_students,0.6);
    assert(case_1);
    std::cout <<"Q4b: all students pass the test."<< std::endl;
    
    bool case_2 = all_students_passed(not_all_pass_students,0.6);
    assert(!case_2);
    std::cout <<"Q4b: not all students pass the test."<< std::endl;
    // Q4c test
    std::vector<int> odd_even_sorted = {-5, -3, -1, 1, 3, -4, -2, 0, 2, 4};
    sort_odd_even(odd_even_sorted);
    std::cout <<"Q4c sort test:"<< std::endl;
    for(auto iter= odd_even_sorted.begin(); iter<=odd_even_sorted.end();iter++){
        std::cout <<*iter<<", ";
    }
    std::cout<<std::endl;
    auto cmp1=[](int i, int j){
        if (i%2==0 && j%2!=0){
            return false;
        }
        else if (i%2!=0 && j%2==0){
            return true;
        }
        else
            return i<j;
    };
    size_t sign_q4c = true;
    for (size_t i=0; i<odd_even_sorted.size()-1; i++) {
        if (!cmp1(odd_even_sorted[i],odd_even_sorted[i+1])) {
            sign_q4c=false;
            break;
        }
    }
    assert(sign_q4c);
    std::cout<<"Q4c test passes"<<std::endl;
    // TODO: Verify your Q4c implementation
    
    // Q4d test
    std::list<SparseMatrixCoordinate<int>> sparse = {
            SparseMatrixCoordinate<int>(2, 5, 1),
            SparseMatrixCoordinate<int>(2, 2, 2),
            SparseMatrixCoordinate<int>(3, 4, 3)};

    // TODO: Verify your Q4d implementation
    auto cmp_2=[](SparseMatrixCoordinate<int> entry_1, SparseMatrixCoordinate<int> entry_2)
    {
        if (entry_1.row < entry_2.row)
            return true;
        else if (entry_1.row == entry_2.row && entry_1.col < entry_2.col)
            return true;
        else
            return false;
    };

    sparse_matrix_sort(sparse);
    bool sign_q4d=true;
    for(auto iter = sparse.begin();std::next(iter,1) != sparse.end();iter++)
    {
        if (!cmp_2(*iter,*(std::next(iter,1)))){
            sign_q4d=false;
            break;
        }
    }
    assert(sign_q4d);
    std::cout <<"Q4d passes the test."<< std::endl;
    return 0;
}
