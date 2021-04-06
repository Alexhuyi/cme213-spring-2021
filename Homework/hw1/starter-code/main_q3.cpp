#include <iostream>
#include <random>
#include <set>

// TODO: add your function here //
int range(std::set<double> & data,double low,double high)
{
    std::set<double>::iterator iter1 = data.lower_bound(low);
    std::set<double>::iterator iter2 = data.upper_bound(high);
    std::set<double>::iterator iter;
    int num = 0;
    for (iter = iter1; iter != iter2; ++iter)
    {
        num++;
    }
    return num;
}

int main()
{
    // Test with N(0,1) data.
    std::cout << "Generating N(0,1) data" << std::endl;
    std::set<double> data;
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, 1.0);
    for (unsigned int i = 0; i < 1000; ++i)
        data.insert(distribution(generator));

    // TODO: print out number of points in [2, 10] //
    std::cout<<"The number of points in the range [lb, ub] = [2, 10] is "<<range(data,2,10)<<std::endl;
    return 0;
}
