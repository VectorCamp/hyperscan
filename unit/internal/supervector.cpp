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
#include<stdlib.h>     
#include"gtest/gtest.h"
#include"ue2common.h"
#include"util/arch.h"
#include"util/simd_utils.h"
#include"util/simd/types.hpp"


TEST(SuperVectorUtilsTest, Zero128c) {
    m128_t zeroes = SuperVector<16>::Zeroes();
    char buf[16]{0};
    for(int i=0; i<16; i++){ASSERT_EQ(zeroes.u.s8[i],buf[i]);}
}


TEST(SuperVectorUtilsTest, Ones128c) {
    m128_t ones = SuperVector<16>::Ones();
    char buf[16];
    for (int i=0; i<16; i++){buf[i]=0xff;}
    for(int i=0; i<16; i++){ASSERT_EQ(ones.u.s8[i],buf[i]);}
}


TEST(SuperVectorUtilsTest, Loadu128c) {
    char vec[32];
    for(int i=0; i<32;i++){vec[i]=i;}
    for(int i=0; i<=16;i++){
        m128_t SP = SuperVector<16>::loadu(vec+i);
        for(int j=0; j<16; j++){
            ASSERT_EQ(SP.u.s8[j],vec[j+i]);
        }
    }
}

TEST(SuperVectorUtilsTest, Load128c) {
    char vec[128] __attribute__((aligned(16)));
    for(int i=0; i<128;i++){vec[i]=i;}
    for(int i=0;i<=16;i+=16){
        m128_t SP = SuperVector<16>::loadu(vec+i);
        for(int j=0; j<16; j++){
            ASSERT_EQ(SP.u.s8[j],vec[j+i]);
        }
    }    
}

TEST(SuperVectorUtilsTest,Equal128c){
    char vec[32];
     for (int i=0; i<32; i++) {vec[i]=i;};
    m128_t SP1 = SuperVector<16>::loadu(vec);
    m128_t SP2 = SuperVector<16>::loadu(vec+16);
    char buf[16]={0};
    /*check for equality byte by byte*/
    for (int s=0; s<16; s++){
        if(vec[s]==vec[s+16]){
            buf[s]=1;
        }
    }
    m128_t SPResult = SP1.eq(SP2);
    for (int i=0; i<16; i++){ASSERT_EQ(SPResult.u.s8[i],buf[i]);}
}

TEST(SuperVectorUtilsTest,And128c){
    m128_t SPResult = SuperVector<16>::Zeroes() & SuperVector<16>::Ones();
    for (int i=0; i<16; i++){ASSERT_EQ(SPResult.u.s8[i],0);}
}

TEST(SuperVectorUtilsTest,OPAnd128c){
    m128_t SP1 = SuperVector<16>::Zeroes(); 
    m128_t SP2 = SuperVector<16>::Ones();
    SP2 = SP2.opand(SP1);
    for (int i=0; i<16; i++){ASSERT_EQ(SP2.u.s8[i],0);}
}


TEST(SuperVectorUtilsTest,OR128c){
    m128_t SPResult = SuperVector<16>::Zeroes() | SuperVector<16>::Ones();
    for (int i=0; i<16; i++){ASSERT_EQ(SPResult.u.s8[i],-1);}
}

TEST(SuperVectorUtilsTest,OPANDNOT128c){
    m128_t SP1 = SuperVector<16>::Zeroes(); 
    m128_t SP2 = SuperVector<16>::Ones();
    SP2 = SP2.opandnot(SP1);
    for (int i=0; i<16; i++){ASSERT_EQ(SP2.u.s8[i],0);}
}

TEST(SuperVectorUtilsTest,Movemask128c){
    uint8_t vec[16] = {0,0xff,0xff,3,4,5,6,7,8,9,0xff,11,12,13,14,0xff};
    /*according to the array above the movemask outcome must be the following:
      10000100000000110 or 0x8406*/
    m128_t SP = SuperVector<16>::loadu(vec);
    int SP_Mask = SP.movemask();
    ASSERT_EQ(SP_Mask,0x8406);
}

TEST(SuperVectorUtilsTest,Eqmask128c){
    uint8_t vec[16] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    uint8_t vec2[16] = {16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31};
    uint8_t vec3[16] = {16,17,3,4,5,6,7,8,1,2,11,12,13,14,15,16};
    m128_t SP = SuperVector<16>::loadu(vec);
    m128_t SP1 = SuperVector<16>::loadu(vec);
    int SP_Mask = SP.eqmask(SP1);
    /*if masks are equal the outcome is 1111111111111111 or 0xffff*/
    ASSERT_EQ(SP_Mask,0xffff);
    SP = SuperVector<16>::loadu(vec);
    SP1 = SuperVector<16>::loadu(vec2);
    SP_Mask = SP.eqmask(SP1);
    ASSERT_EQ(SP_Mask,0);
    SP = SuperVector<16>::loadu(vec2);
    SP1 = SuperVector<16>::loadu(vec3);
    SP_Mask = SP.eqmask(SP1);
    ASSERT_EQ(SP_Mask,3);
}


TEST(SuperVectorUtilsTest,pshufbc){
    srand (time(NULL));
    uint8_t vec[16];
    for (int i=0; i<16; i++){vec[i]=rand() % 100 + 1;};
    uint8_t vec2[16];
    for (int i=0; i<16; i++){vec2[i]=i;};
    m128_t SP1 = SuperVector<16>::loadu(vec);
    m128_t SP2 = SuperVector<16>::loadu(vec2);
    m128_t SResult = SP1.pshufb(SP2);
    for (int i=0; i<16; i++){
            ASSERT_EQ(vec[vec2[i]],SResult.u.u8[i]);
        }
}

TEST(SuperVectorUtilsTest,LShift64_128c){
    u_int64_t vec[2] = {128, 512}; 
    m128_t SP = SuperVector<16>::loadu(vec);
    for(int s = 0; s<16; s++){
        m128_t SP_after_shift = SP.lshift64(s);
        for (int i=0; i<2; i++){ASSERT_EQ(SP_after_shift.u.u64[i],vec[i]<<s);}
    }   
}

TEST(SuperVectorUtilsTest,RShift64_128c){
    u_int64_t vec[2] = {128, 512}; 
    m128_t SP = SuperVector<16>::loadu(vec);
    for(int s = 0; s<16; s++){
        m128_t SP_after_shift = SP.rshift64(s);
        for (int i=0; i<2; i++){ASSERT_EQ(SP_after_shift.u.u64[i],vec[i]>>s);}
    }   
}


/*Define LSHIFT128 macro*/
#define TEST_LSHIFT128(l)   {   SP_after_Lshift = SP<<(l);                                              \
                                buf[l-1]=0;                                                             \
                                for(int i=0; i<16; i++){ASSERT_EQ(SP_after_Lshift.u.s8[i],buf[i]);}     \
                            }           

TEST(SuperVectorUtilsTest,LShift128c){
    char vec[16];
    for (int i=0; i<16; i++) {vec[i]=0xff;}
    m128_t SP = SuperVector<16>::loadu(vec);
    char buf[16];
    for (int i=0; i<16; i++){buf[i]=0xff;}
    m128_t SP_after_Lshift = SP<<(0);
    TEST_LSHIFT128(1)
    TEST_LSHIFT128(2)
    TEST_LSHIFT128(3)
    TEST_LSHIFT128(4)
    TEST_LSHIFT128(5)
    TEST_LSHIFT128(6)
    TEST_LSHIFT128(7)
    TEST_LSHIFT128(8)
    TEST_LSHIFT128(9)
    TEST_LSHIFT128(10)
    TEST_LSHIFT128(11)
    TEST_LSHIFT128(12)
    TEST_LSHIFT128(13)
    TEST_LSHIFT128(14)
    TEST_LSHIFT128(15)
    TEST_LSHIFT128(16)
}


/*Define RSHIFT128 macro*/
#define TEST_RSHIFT128(l)   {   SP_after_Rshift = SP>>(l);                                           \
                                buf[16-l] = 0;                                                       \
                                for(int i=0; i<16; i++) {ASSERT_EQ(SP_after_Rshift.u.u8[i],buf[i]);} \
                            }   

TEST(SuperVectorUtilsTest,RShift128c){
    char vec[16];
    for (int i=0; i<16; i++) {vec[i]=0xff;}
    m128_t SP = SuperVector<16>::loadu(vec);
    uint8_t buf[16];
    for (int i=0; i<16; i++){buf[i]=0xff;}
    m128_t SP_after_Rshift = SP>>(0);
    TEST_RSHIFT128(1)
    TEST_RSHIFT128(2)
    TEST_RSHIFT128(3)
    TEST_RSHIFT128(4)
    TEST_RSHIFT128(5)
    TEST_RSHIFT128(6)
    TEST_RSHIFT128(7)
    TEST_RSHIFT128(8)
    TEST_RSHIFT128(9)
    TEST_RSHIFT128(10)
    TEST_RSHIFT128(11)
    TEST_RSHIFT128(12)
    TEST_RSHIFT128(13)
    TEST_RSHIFT128(14)
    TEST_RSHIFT128(15)
    TEST_RSHIFT128(16)
}


/*Define ALIGNR128 macro*/
#define TEST_ALIGNR128(l)   {  SP_test = SP1.alignr(SP,l);                                       \
                                for (int i=0; i<16; i++){ASSERT_EQ(SP_test.u.u8[i],vec[i+l]);}   \
                            }

TEST(SuperVectorUtilsTest,Alignr128c){
    uint8_t vec[32];
    for (int i=0; i<32; i++) {vec[i]=i;}
    m128_t SP = SuperVector<16>::loadu(vec);
    m128_t SP1 = SuperVector<16>::loadu(vec+16);
    m128_t SP_test = SP1.alignr(SP,0);
    TEST_ALIGNR128(1)
    TEST_ALIGNR128(2)
    TEST_ALIGNR128(3)
    TEST_ALIGNR128(4)
    TEST_ALIGNR128(5)
    TEST_ALIGNR128(6)
    TEST_ALIGNR128(7)
    TEST_ALIGNR128(8)
    TEST_ALIGNR128(9)
    TEST_ALIGNR128(10)
    TEST_ALIGNR128(11)
    TEST_ALIGNR128(12)
    TEST_ALIGNR128(13)
    TEST_ALIGNR128(14)
    TEST_ALIGNR128(15)
    TEST_ALIGNR128(16)
    
}


#if defined(HAVE_AVX2)
typedef union  uni256i{__m256i f; int8_t vec[32];}u256i;

TEST(SuperVectorUtilsTest, Ones256) {
    m256_t zeroes = SuperVector<32>::Ones();
    u256i z;
    z.f = _mm256_set1_epi8(0xff);
    for(int i=0; i<32; i++){ASSERT_EQ(zeroes.u.s8[i],z.vec[i]);}
}

TEST(SuperVectorUtilsTest, Zero256) {
    m256_t ones = SuperVector<32>::Zeroes();
    u256i z;
    z.f = _mm256_set1_epi8(0);
    for(int i=0; i<32; i++){ASSERT_EQ(ones.u.s8[i],z.vec[i]);}
}

TEST(SuperVectorUtilsTest, Load256) {
    int vec[8] __attribute__((aligned(16)));
    srand(time(NULL));
    for (int i=0; i<8; i++) {vec[i]=rand() %1000 +1;}
    m256_t SP = SuperVector<32>::load((__m256i*)vec);
    u256i test_vector;
    test_vector.f = _mm256_load_si256((__m256i*)vec);
    for(int i=0; i<32; i++){ ASSERT_EQ(SP.u.s8[i],test_vector.vec[i]);}
}

TEST(SuperVectorUtilsTest, Loadu256) {
    int vec[8];
    srand(time(NULL));
    for (int i=0; i<8; i++) {vec[i]=rand() %1000 +1;}
    m256_t SP = SuperVector<32>::loadu((__m256i*)vec);
    u256i test_vector;
    test_vector.f = _mm256_lddqu_si256((__m256i*)vec);
    for(int i=0; i<32; i++){ ASSERT_EQ(SP.u.s8[i],test_vector.vec[i]);}
}

TEST(SuperVectorUtilsTest,Equal256){
    int vec[16];
    srand(time(NULL));
    for (int i=0; i<16; i++) {vec[i]=rand() %1000 +1;}
    m256_t SP1 = SuperVector<32>::loadu((__m256i*)vec);
    m256_t SP2 = SuperVector<32>::loadu((__m256i*)vec+8);
    u256i test_vector1;
    u256i test_vector2;
    test_vector1.f = _mm256_loadu_si256((__m256i*)vec);
    test_vector2.f = _mm256_loadu_si256((__m256i*)vec+8);
    m256_t SPResult = SP1.eq(SP2);
    u256i test_result;
    test_result.f = _mm256_cmpeq_epi8(test_vector1.f,test_vector2.f);
    for (int i=0; i<32; i++){ASSERT_EQ(SPResult.u.s8[i],test_result.vec[i]);}
}

TEST(SuperVectorUtilsTest,And256){
    m256_t SPResult = SuperVector<32>::Zeroes() & SuperVector<32>::Ones();
    __m256i test_vector1 = _mm256_set1_epi8(0);
    __m256i test_vector2 = _mm256_set1_epi8(0xff);
    u256i test_result;
    test_result.f = _mm256_and_si256(test_vector1,test_vector2);
    for (int i=0; i<32; i++){ASSERT_EQ(SPResult.u.s8[i],test_result.vec[i]);}
}

TEST(SuperVectorUtilsTest,Movemask256){
    int vec[8];
    srand(time(NULL));
    for (int i=0; i<8; i++) {vec[i]=rand() %1000 +1;}
    m256_t SP = SuperVector<32>::loadu((__m256i*)vec);
    __m256i test_vector = _mm256_loadu_si256((__m256i*)vec);
    int SP_Mask = SP.movemask();
    int test_result = _mm256_movemask_epi8(test_vector);
    ASSERT_EQ(SP_Mask,test_result);
}

TEST(SuperVectorUtilsTest,Eqmask256){
    int vec[16];
    srand(time(NULL));
    for (int i=0; i<16; i++) {vec[i]=rand() %1000 +1;}
    m256_t SP = SuperVector<32>::loadu((__m256i*)vec);
    m256_t SP1 = SuperVector<32>::loadu((__m256i*)vec+8);
    __m256i test_vector1 = _mm256_loadu_si256((__m256i*)vec);
    __m256i test_vector2 = _mm256_loadu_si256((__m256i*)vec+8);
    __m256i test_result = _mm256_cmpeq_epi8(test_vector1,test_vector2);
    int SP_Mask = SP.eqmask(SP1);
    int test_res = _mm256_movemask_epi8(test_result);
    ASSERT_EQ(SP_Mask,test_res);
}

/*Define SHIFT256 macro*/
#define TEST_SHIFT256(l)                                                                     \
SP_after_shift = SP<<(l);                                                                    \
test_vector_after_shift.f = _mm256_slli_si256(test_vector.f,l);                              \
for(int i=0; i<32; i++) {ASSERT_EQ(SP_after_shift.u.s8[i],test_vector_after_shift.vec[i]);}  \

TEST(SuperVectorUtilsTest,Shift256){
    int vec[8];
    srand(time(NULL));
    for (int i=0; i<8; i++) {vec[i]=rand() %1000 +1;}
    m256_t SP = SuperVector<32>::loadu((__m128i*)vec);
    u256i test_vector;
    test_vector.f = _mm256_loadu_si256((__m256i*)vec);
    u256i test_vector_after_shift;
    m256_t SP_after_shift = SP<<(0);
    TEST_SHIFT256(1)
    TEST_SHIFT256(2)
    TEST_SHIFT256(3)
    TEST_SHIFT256(4)
    TEST_SHIFT256(5)
    TEST_SHIFT256(6)
    TEST_SHIFT256(7)
    TEST_SHIFT256(8)
    TEST_SHIFT256(9)
    TEST_SHIFT256(10)
    TEST_SHIFT256(11)
    TEST_SHIFT256(12)
    TEST_SHIFT256(13)
    TEST_SHIFT256(14)
    TEST_SHIFT256(15)
    TEST_SHIFT256(16)
}

#define ALIGNR256(l)                                                    \
al_test.f = _mm256_alignr_epi8(test_vector1,test_vector2,l);            \
SP_test = SP.alignr(SP1,l);                                             \
for (int i=0; i<32; i++) {ASSERT_EQ(SP_test.u.s8[i],al_test.vec[i]);}   \

TEST(SuperVectorUtilsTest,Alignr256){
    int vec[16];
    srand(time(NULL));
    for (int i=0; i<16; i++) {vec[i]=rand() %1000 +1;}
    m256_t SP = SuperVector<32>::loadu((__m256i*)vec);
    m256_t SP1 = SuperVector<32>::loadu((__m256i*)vec+8);
    __m256i test_vector1 = _mm256_loadu_si256((__m256i*)vec);
    __m256i test_vector2 = _mm256_loadu_si256((__m256i*)vec+8);
    u256i al_test;
    m256_t SP_test = SP.alignr(SP1,0);
    ALIGNR256(1);
    ALIGNR256(2);
    ALIGNR256(3);
    ALIGNR256(4);
    ALIGNR256(5);
    ALIGNR256(6);
    ALIGNR256(7);
    ALIGNR256(8);
    ALIGNR256(9);
    ALIGNR256(10);
    ALIGNR256(11);
    ALIGNR256(12);
    ALIGNR256(13);
    ALIGNR256(14);
    ALIGNR256(15);
    ALIGNR256(16);
}
#endif

#if defined(HAVE_AVX512)
typedef union  uni512i{__m512i f; int8_t vec[64];}u512i;

TEST(SuperVectorUtilsTest, Ones512) {
    m512_t zeroes = SuperVector<64>::Ones();
    u512i z;
    z.f = _mm512_set1_epi8(0xff);
    for(int i=0; i<64; i++){ASSERT_EQ(zeroes.u.s8[i],z.vec[i]);}
}

TEST(SuperVectorUtilsTest, Zero512) {
    m512_t ones = SuperVector<64>::Zeroes();
    u512i z;
    z.f = _mm512_set1_epi8(0);
    for(int i=0; i<64; i++){ ASSERT_EQ(ones.u.s8[i],z.vec[i]);}
}

TEST(SuperVectorUtilsTest, Load512) {
    int vec[16] __attribute__((aligned(64)));
    srand(time(NULL));
    for (int i=0; i<16; i++) {vec[i]=rand() %1000 +1;}
    m512_t SP = SuperVector<64>::load((__m512i*)vec);
    u512i test_vector;
    test_vector.f = _mm512_load_si512((__m512i*)vec);
    for(int i=0; i<64; i++){ ASSERT_EQ(SP.u.s8[i],test_vector.vec[i]);}
}

TEST(SuperVectorUtilsTest, Loadu512) {
    int vec[16];
    srand(time(NULL));
    for (int i=0; i<16; i++) {vec[i]=rand() %1000 +1;}
    m512_t SP = SuperVector<64>::loadu((__m512i*)vec);
    u512i test_vector;
    test_vector.f = _mm512_loadu_si512((__m512i*)vec);
    for(int i=0; i<64; i++){ ASSERT_EQ(SP.u.s8[i],test_vector.vec[i]);}
}

/* This method is under construction
TEST(SuperVectorUtilsTest,Equal512){}
*/

TEST(SuperVectorUtilsTest,And512){
    m512_t SPResult = SuperVector<64>::Zeroes() & SuperVector<64>::Ones();
    __m512i test_vector1 = _mm512_set1_epi8(0);
    __m512i test_vector2 = _mm512_set1_epi8(0xff);
    u512i test_result;
    test_result.f = _mm512_and_si512(test_vector1,test_vector2);
    for (int i=0; i<64; i++){ASSERT_EQ(SPResult.u.s8[i],test_result.vec[i]);}
}

/* This methos is under construction
TEST(SuperVectorUtilsTest,Movemask256){}
*/

TEST(SuperVectorUtilsTest,Eqmask512){
    int vec[16];
    srand(time(NULL));
    for (int i=0; i<16; i++) {vec[i]=rand() %1000 +1;}
    m256_t SP = SuperVector<32>::loadu((__m256i*)vec);
    m256_t SP1 = SuperVector<32>::loadu((__m256i*)vec+8);
    __m256i test_vector1 = _mm256_loadu_si256((__m256i*)vec);
    __m256i test_vector2 = _mm256_loadu_si256((__m256i*)vec+8);
    __m256i test_result = _mm256_cmpeq_epi8(test_vector1,test_vector2);
    int SP_Mask = SP.eqmask(SP1);
    int test_res = _mm256_movemask_epi8(test_result);
    ASSERT_EQ(SP_Mask,test_res);
}
/*
This methos is under construction
TEST(SuperVectorUtilsTest,Shift256){}
*/

#define ALIGNR512(l)                                                    \
al_test.f = _mm512_alignr_epi8(test_vector1,test_vector2,l);            \
SP_test = SP.alignr(SP1,l);                                             \
for (int i=0; i<64; i++) {ASSERT_EQ(SP_test.u.s8[i],al_test.vec[i]);}   \

TEST(SuperVectorUtilsTest,Alignr512){
    int vec[32];
    srand(time(NULL));
    for (int i=0; i<32; i++) {vec[i]=rand() %1000 +1;}
    m512_t SP = SuperVector<64>::loadu((__m512i*)vec);
    m512_t SP1 = SuperVector<64>::loadu((__m512i*)vec+16);
    __m512i test_vector1 = _mm512_loadu_si512((__m512i*)vec);
    __m512i test_vector2 = _mm512_loadu_si512((__m512i*)vec+16);
    u512i al_test;
    m512_t SP_test = SP.alignr(SP1,0);
    ALIGNR512(1);
    ALIGNR512(2);
    ALIGNR512(3);
    ALIGNR512(4);
    ALIGNR512(5);
    ALIGNR512(6);
    ALIGNR512(7);
    ALIGNR512(8);
    ALIGNR512(9);
    ALIGNR512(10);
    ALIGNR512(11);
    ALIGNR512(12);
    ALIGNR512(13);
    ALIGNR512(14);
    ALIGNR512(15);
    ALIGNR512(16);
}
<<<<<<< HEAD
=======

>>>>>>> upstream/feature/templates-refactor
#endif
