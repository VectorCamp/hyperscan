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


TEST(SuperVectorUtilsTest,pshufbc){
    srand (time(NULL));
    uint8_t vec[16];
    for (int i=0; i<16; i++){vec[i]=rand() % 100 + 1;;};
    uint8_t vec2[16];
    for (int i=0; i<16; i++){vec2[i]=i;};
    m128_t SP1 = SuperVector<16>::loadu(vec);
    m128_t SP2 = SuperVector<16>::loadu(vec2);
    m128_t SResult = SP1.pshufb(SP2);
    for (int i=0; i<16; i++){ASSERT_EQ(vec[vec2[i]],SResult.u.u8[i]);}
}


/*Define ALIGNR128 macro*/
#define TEST_ALIGNR128(l)       {  SP_test = SP1.alignr(SP,l);                                             \
                                   for (int i=0; i<16; i++){ASSERT_EQ(SP_test.u.u8[i],vec[i+l]);}          \
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
