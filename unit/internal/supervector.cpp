/*
 * Copyright (c) 2015-2017, Intel Corporation
 * Copyright (c) 2020-2021, VectorCamp PC
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of Intel Corporation nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include<iostream>
#include<immintrin.h>
#include<cstring>
#include"gtest/gtest.h"
#include"src/util/simd/types.hpp"


/*Gust a debugin message to know if the supervector.cpp executed*/
int mes(){
    std::cout<<"\x1B[35m Info-Print: SuperVector init tests initilizing...\x1B[0m"<<std::endl;
    return 0;
}
static int a = mes();


struct SuperVector_zeroes {
    operator m128_t() {return SuperVector<16>::Zeroes();}
};

struct SuperVector_ones {
    operator m128_t() {return SuperVector<16>::Ones();}
};

/*
void SuperVector_loadu(m128_t *a, const void *ptr) { *a = SuperVector<16>::loadu(ptr); }
void SuperVector_loadu(m256_t *a, const void *ptr) { *a = SuperVector<32>::loadu(ptr); }
void SuperVector_loadu(m512_t *a, const void *ptr) { *a = SuperVector<64>::loadu(ptr); }

void SuperVector_load(m128_t *a, const void *ptr) { *a = SuperVector<16>::load(ptr); }
void SuperVector_load(m256_t *a, const void *ptr) { *a = SuperVector<32>::load(ptr); }
void SuperVector_load(m512_t *a, const void *ptr) { *a = SuperVector<64>::load(ptr); }
*/


template<typename T>
class SuperVectorUtilsTest : public testing::Test {
    // empty
};

//Declare test types
typedef ::testing::Types<m128_t> SuperVectorTypes;

TYPED_TEST_CASE(SuperVectorUtilsTest, SuperVectorTypes);


/*Begin init tests*/
TYPED_TEST(SuperVectorUtilsTest, zero){
    const TypeParam zeroes = SuperVector_zeroes();
    // Should have no bits on.
    char cmp[sizeof(zeroes)];
    std::memset(cmp, 0, sizeof(zeroes));
    ASSERT_EQ(0, std::memcmp(cmp, &zeroes, sizeof(zeroes)));
}

TYPED_TEST(SuperVectorUtilsTest, ones) {
    const TypeParam ones = SuperVector_ones();
    // Should have all bits on.
    char cmp[sizeof(ones)];
    memset(cmp, 0xff, sizeof(ones));
    ASSERT_EQ(0, memcmp(cmp, &ones, sizeof(ones)));
}


// Unaligned load
/*
TYPED_TEST(SuperVectorUtilsTest, loadu) {
    const TypeParam ones = SuperVector_ones();

    const size_t mem_len = sizeof(ones) * 2;
    unique_ptr<char[]> mem_array = ue2::make_unique<char[]>(mem_len);
    char *mem = mem_array.get();

    for (size_t offset = 1; offset < sizeof(ones); offset++) {
        std::memset(mem, 0, mem_len);
        std::memset(mem + offset, 0xff, sizeof(ones));
        TypeParam a;
        SuperVector_loadu(&a, mem + offset);
        ASSERT_EQ(0, simd_diff(a, ones));
    }
}
*/