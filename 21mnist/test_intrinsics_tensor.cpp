#include "include/tensor.h"
#include <iostream>

int main() {
    tensor<float,1,1,1,28> tens1; // normal vectorization
    tensor<float,1,1,1,28> tens2;
    tens1.init_const(1,1);
    tens2.init_const(1,2);
    
    std::cout << _mm512_reduce_add_ps(_mm512_mul_ps(tens1.V(0,0,0,0),tens2.V(0,0,0,0))) << std::endl; // 32
    
    tensor<float,1,1,1,12> tens3; // vectorisation on an array with less than 16 elements
    tensor<float,1,1,1,12> tens4;
    tens3.init_const(1,1);
    tens4.init_const(1,2);
    
    std::cout << _mm512_reduce_add_ps(_mm512_mul_ps(tens3.V(0,0,0,0),tens4.V(0,0,0,0))) << std::endl; // nan
    
    return 0;
}