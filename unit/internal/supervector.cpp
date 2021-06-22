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
#include<cstring>
#include<time.h>
#include"gtest/gtest.h"
#include"ue2common.h"
#include"util/arch.h"
#include"util/simd_utils.h"
#include"util/simd/types.hpp"


typedef union uni128i{__m128i f; int8_t vec[16];}u128i;

TEST(SuperVectorUtilsTest, Zero128) {
    m128_t zeroes = SuperVector<16>::Zeroes();
    u128i z;
    z.f = _mm_set1_epi8(0);
    for(int i=0; i<16; i++){ASSERT_EQ(zeroes.u.s8[i],z.vec[i]);}     
}

TEST(SuperVectorUtilsTest, Ones128) {
    m128_t ones = SuperVector<16>::Ones();
    u128i z;
    z.f = _mm_set1_epi8(0xff);
    for(int i=0; i<16; i++){ASSERT_EQ(ones.u.s8[i],z.vec[i]);}
}


TEST(SuperVectorUtilsTest, Loadu128) {
    int vec[4];
    srand(time(NULL));
    for (int i=0; i<4; i++) {vec[i]=rand() %1000 +1;}
    m128_t SP = SuperVector<16>::loadu((__m128i*)vec);
    u128i test_vector;
    test_vector.f = _mm_lddqu_si128((__m128i*)vec);
    for(int i=0; i<16; i++){ ASSERT_EQ(SP.u.s8[i],test_vector.vec[i]);}
}

TEST(SuperVectorUtilsTest, Load128) {
    int vec[4] __attribute__((aligned(16)));
    srand(time(NULL));
    for (int i=0; i<4; i++) {vec[i]=rand() %1000 +1;}
    m128_t SP = SuperVector<16>::load((__m128i*)vec);
    u128i test_vector;
    test_vector.f = _mm_load_si128((__m128i*)vec);
    for(int i=0; i<16; i++){ ASSERT_EQ(SP.u.s8[i],test_vector.vec[i]);}
}

TEST(SuperVectorUtilsTest,Equal128){
    int vec[8];
    srand(time(NULL));
    for (int i=0; i<8; i++) {vec[i]=rand() %1000 +1;}
    m128_t SP1 = SuperVector<16>::loadu((__m128i*)vec);
    m128_t SP2 = SuperVector<16>::loadu((__m128i*)vec+4);
    u128i test_vector1;
    u128i test_vector2;
    test_vector1.f = _mm_loadu_si128((__m128i*)vec);
    test_vector2.f = _mm_loadu_si128((__m128i*)vec+4);
    m128_t SPResult = SP1.eq(SP2);
    u128i test_result;
    test_result.f = _mm_cmpeq_epi8(test_vector1.f,test_vector2.f);
    for (int i=0; i<16; i++){ASSERT_EQ(SPResult.u.s8[i],test_result.vec[i]);}
}

TEST(SuperVectorUtilsTest,And128){
    m128_t SPResult = SuperVector<16>::Zeroes() & SuperVector<16>::Ones();
    __m128i test_vector1 = _mm_set1_epi8(0);
    __m128i test_vector2 = _mm_set1_epi8(0xff);
    u128i test_result;
    test_result.f = _mm_and_si128(test_vector1,test_vector2);
    for (int i=0; i<16; i++){ASSERT_EQ(SPResult.u.s8[i],test_result.vec[i]);}
}

TEST(SuperVectorUtilsTest,Movemask128){
    int vec[4];
    srand(time(NULL));
    for (int i=0; i<4; i++) {vec[i]=rand() %1000 +1;}
    m128_t SP = SuperVector<16>::loadu((__m128i*)vec);
    __m128i test_vector = _mm_loadu_si128((__m128i*)vec);
    int SP_Mask = SP.movemask();
    int test_result = _mm_movemask_epi8(test_vector);
    ASSERT_EQ(SP_Mask,test_result);
}

TEST(SuperVectorUtilsTest,Eqmask128){
    int vec[8];
    srand(time(NULL));
    for (int i=0; i<8; i++) {vec[i]=rand() %1000 +1;}
    m128_t SP = SuperVector<16>::loadu((__m128i*)vec);
    m128_t SP1 = SuperVector<16>::loadu((__m128i*)vec+4);
    __m128i test_vector1 = _mm_loadu_si128((__m128i*)vec);
    __m128i test_vector2 = _mm_loadu_si128((__m128i*)vec+4);
    __m128i test_result = _mm_cmpeq_epi8(test_vector1,test_vector2);
    int SP_Mask = SP.eqmask(SP1);
    int test_res = _mm_movemask_epi8(test_result);
    ASSERT_EQ(SP_Mask,test_res);
}

/*Define SHIFT128 macro*/
#define TEST_SHIFT128(l)                                                                     \
SP_after_shift = SP<<(l);                                                                    \
test_vector_after_shift.f = _mm_slli_si128(test_vector.f,l);                                 \
for(int i=0; i<16; i++) {ASSERT_EQ(SP_after_shift.u.s8[i],test_vector_after_shift.vec[i]);}  \

TEST(SuperVectorUtilsTest,Shift128){
    int vec[4];
    srand(time(NULL));
    for (int i=0; i<4; i++) {vec[i]=rand() %1000 +1;}
    m128_t SP = SuperVector<16>::loadu((__m128i*)vec);
    u128i test_vector;
    test_vector.f = _mm_loadu_si128((__m128i*)vec);
    u128i test_vector_after_shift;
    m128_t SP_after_shift = SP<<(0);
    TEST_SHIFT128(1)
    TEST_SHIFT128(2)
    TEST_SHIFT128(3)
    TEST_SHIFT128(4)
    TEST_SHIFT128(5)
    TEST_SHIFT128(6)
    TEST_SHIFT128(7)
    TEST_SHIFT128(8)
    TEST_SHIFT128(9)
    TEST_SHIFT128(10)
    TEST_SHIFT128(11)
    TEST_SHIFT128(12)
    TEST_SHIFT128(13)
    TEST_SHIFT128(14)
    TEST_SHIFT128(15)
    TEST_SHIFT128(16)
}

#define ALIGNR128(l)                                                    \
al_test.f = _mm_alignr_epi8(test_vector1,test_vector2,l);               \
SP_test = SP.alignr(SP1,l);                                             \
for (int i=0; i<16; i++) {ASSERT_EQ(SP_test.u.s8[i],al_test.vec[i]);}   \

TEST(SuperVectorUtilsTest,Alignr128){
    int vec[8];
    srand(time(NULL));
    for (int i=0; i<8; i++) {vec[i]=rand() %1000 +1;}
    m128_t SP = SuperVector<16>::loadu((__m128i*)vec);
    m128_t SP1 = SuperVector<16>::loadu((__m128i*)vec+4);
    __m128i test_vector1 = _mm_loadu_si128((__m128i*)vec);
    __m128i test_vector2 = _mm_loadu_si128((__m128i*)vec+4);
    u128i al_test;
    m128_t SP_test = SP.alignr(SP1,0);
    ALIGNR128(1);
    ALIGNR128(2);
    ALIGNR128(3);
    ALIGNR128(4);
    ALIGNR128(5);
    ALIGNR128(6);
    ALIGNR128(7);
    ALIGNR128(8);
    ALIGNR128(9);
    ALIGNR128(10);
    ALIGNR128(11);
    ALIGNR128(12);
    ALIGNR128(13);
    ALIGNR128(14);
    ALIGNR128(15);
    ALIGNR128(16);
}

