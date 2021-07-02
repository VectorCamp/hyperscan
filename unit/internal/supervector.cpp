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
    auto zeroes = SuperVector<16>::Zeroes();
    u8 buf[16]{0};
    for(int i=0; i<16; i++) {
        ASSERT_EQ(zeroes.u.u8[i],buf[i]);
    }
}

TEST(SuperVectorUtilsTest, Ones128c) {
    auto ones = SuperVector<16>::Ones();
    u8 buf[16];
    for (int i=0; i<16; i++) { buf[i]=0xff; }
    for(int i=0; i<16; i++) {
        ASSERT_EQ(ones.u.u8[i],buf[i]);
    }
}

TEST(SuperVectorUtilsTest, Loadu128c) {
    u8 vec[32];
    for(int i=0; i<32;i++) { vec[i]=i; }
    for(int i=0; i<=16;i++) {
        auto SP = SuperVector<16>::loadu(vec+i);
        for(int j=0; j<16; j++) {
            ASSERT_EQ(SP.u.u8[j],vec[j+i]);
        }
    }
}

TEST(SuperVectorUtilsTest, Load128c) {
    u8 ALIGN_ATTR(16) vec[32];
    for(int i=0; i<32;i++) { vec[i]=i; }
    for(int i=0;i<=16;i+=16) {
        auto SP = SuperVector<16>::loadu(vec+i);
        for(int j=0; j<16; j++){
            ASSERT_EQ(SP.u.u8[j],vec[j+i]);
        }
    }    
}

TEST(SuperVectorUtilsTest,Equal128c){
    u8 vec[32];
     for (int i=0; i<32; i++) {vec[i]=i;};
    auto SP1 = SuperVector<16>::loadu(vec);
    auto SP2 = SuperVector<16>::loadu(vec+16);
    u8 buf[16]={0};
    /*check for equality byte by byte*/
    for (int s=0; s<16; s++){
        if(vec[s]==vec[s+16]){
            buf[s]=1;
        }
    }
    auto SPResult = SP1.eq(SP2);
    for (int i=0; i<16; i++) {
        ASSERT_EQ(SPResult.u.s8[i],buf[i]);
    }
}

TEST(SuperVectorUtilsTest,And128c){
    auto SPResult = SuperVector<16>::Zeroes() & SuperVector<16>::Ones();
    for (int i=0; i<16; i++) {
        ASSERT_EQ(SPResult.u.u8[i],0);
    }
}

TEST(SuperVectorUtilsTest,OPAnd128c){
    auto SP1 = SuperVector<16>::Zeroes(); 
    auto SP2 = SuperVector<16>::Ones();
    SP2 = SP2.opand(SP1);
    for (int i=0; i<16; i++) {
        ASSERT_EQ(SP2.u.u8[i],0);
    }
}

TEST(SuperVectorUtilsTest,OR128c){
    auto SPResult = SuperVector<16>::Zeroes() | SuperVector<16>::Ones();
    for (int i=0; i<16; i++) {
        ASSERT_EQ(SPResult.u.u8[i],0xff);
    }
}

TEST(SuperVectorUtilsTest,XOR128c){
    srand (time(NULL));
    u8 vec[16];
    for (int i=0; i<16; i++) {
        vec[i] = rand() % 100 + 1;
    }
    u8 vec2[16];
    for (int i=0; i<16; i++) {
        vec2[i] = rand() % 100 + 1;
    }
    auto SP1 = SuperVector<16>::loadu(vec);
    auto SP2 = SuperVector<16>::loadu(vec2);
    auto SPResult = SP1 ^ SP2;
    for (int i=0; i<16; i++) {
        ASSERT_EQ(SPResult.u.u8[i],vec[i] ^ vec2[i]);
    }
}


TEST(SuperVectorUtilsTest,OPXOR128c){
    srand (time(NULL));
    u8 vec[16];
    for (int i=0; i<16; i++) {
        vec[i] = rand() % 100 + 1;
    }
    u8 vec2[16];
    for (int i=0; i<16; i++) {
        vec2[i] = rand() % 100 + 1;
    }
    auto SP1 = SuperVector<16>::loadu(vec);
    auto SP2 = SuperVector<16>::loadu(vec2);
    auto SPResult = SP1.opxor(SP2);
    for (int i=0; i<16; i++) {
        ASSERT_EQ(SPResult.u.u8[i],vec[i] ^ vec2[i]);
    }
}

TEST(SuperVectorUtilsTest,OPANDNOT128c){
    auto SP1 = SuperVector<16>::Zeroes(); 
    auto SP2 = SuperVector<16>::Ones();
    SP2 = SP2.opandnot(SP1);
    for (int i=0; i<16; i++) {
        ASSERT_EQ(SP2.u.s8[i],0);
    }
}

TEST(SuperVectorUtilsTest,Movemask128c){
    u8 vec[16] = { 0, 0xff, 0xff, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 0, 0, 0, 0xff };
    /*according to the array above the movemask outcome must be the following:
      10000100000000110 or 0x8406*/
    auto SP = SuperVector<16>::loadu(vec);
    int mask = SP.movemask();
    ASSERT_EQ(mask, 0x8c06);
}

TEST(SuperVectorUtilsTest,Eqmask128c){
    u8 vec[16]  = {  0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15 };
    u8 vec2[16] = { 16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31 };
    u8 vec3[16] = { 16,17, 3, 4, 5, 6, 7, 8, 1, 2,11,12,13,14,15,16 };
    auto SP = SuperVector<16>::loadu(vec);
    auto SP1 = SuperVector<16>::loadu(vec2);
    auto SP2 = SuperVector<16>::loadu(vec3);
    int mask = SP.eqmask(SP);
    /*if vectors are equal the mask is 1111111111111111 or 0xffff*/
    ASSERT_EQ(mask,0xffff);
    mask = SP.eqmask(SP2);
    ASSERT_EQ(mask,0);
    mask = SP1.eqmask(SP2);
    ASSERT_EQ(mask,3);
}

/*Define LSHIFT128 macro*/
#define TEST_LSHIFT128(buf, vec, v, l) {                                                  \
                                           auto v_shifted = v << (l);                     \
                                           for (int i=15; i>= l; --i) {                   \
                                               buf[i] = vec[i-l];                         \
                                           }                                              \
                                           for (int i=0; i<l; i++) {                      \
                                               buf[i] = 0;                                \
                                           }                                              \
                                           for(int i=0; i<16; i++) {                      \
                                               ASSERT_EQ(v_shifted.u.u8[i], buf[i]);      \
                                           }                                              \
                                       }

TEST(SuperVectorUtilsTest,LShift128c){
    u8 vec[16] = { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 };
    auto SP = SuperVector<16>::loadu(vec);
    u8 buf[16];
    TEST_LSHIFT128(buf, vec, SP, 0);
    TEST_LSHIFT128(buf, vec, SP, 1);
    TEST_LSHIFT128(buf, vec, SP, 2);
    TEST_LSHIFT128(buf, vec, SP, 3);
    TEST_LSHIFT128(buf, vec, SP, 4);
    TEST_LSHIFT128(buf, vec, SP, 5);
    TEST_LSHIFT128(buf, vec, SP, 6);
    TEST_LSHIFT128(buf, vec, SP, 7);
    TEST_LSHIFT128(buf, vec, SP, 8);
    TEST_LSHIFT128(buf, vec, SP, 9);
    TEST_LSHIFT128(buf, vec, SP, 10);
    TEST_LSHIFT128(buf, vec, SP, 11);
    TEST_LSHIFT128(buf, vec, SP, 12);
    TEST_LSHIFT128(buf, vec, SP, 13);
    TEST_LSHIFT128(buf, vec, SP, 14);
    TEST_LSHIFT128(buf, vec, SP, 15);
    TEST_LSHIFT128(buf, vec, SP, 16);
}

TEST(SuperVectorUtilsTest,LShift64_128c){
    u64a vec[2] = {128, 512};
    auto SP = SuperVector<16>::loadu(vec);
    for(int s = 0; s<16; s++) {
        auto SP_after_shift = SP.lshift64(s);
        for (int i=0; i<2; i++) {
            ASSERT_EQ(SP_after_shift.u.u64[i], vec[i] << s);
        }
    }   
}

TEST(SuperVectorUtilsTest,RShift64_128c){
    u64a vec[2] = {128, 512};
    auto SP = SuperVector<16>::loadu(vec);
    for(int s = 0; s<16; s++) {
        auto SP_after_shift = SP.rshift64(s);
        for (int i=0; i<2; i++) {
            ASSERT_EQ(SP_after_shift.u.u64[i], vec[i] >> s);
        }
    }   
}

/*Define RSHIFT128 macro*/
#define TEST_RSHIFT128(buf, vec, v, l) {                                                  \
                                           auto v_shifted = v >> (l);                     \
                                           for (int i=0; i<16-l; i++) {                   \
                                               buf[i] = vec[i+l];                         \
                                           }                                              \
                                           for (int i=16-l; i<16; i++) {                  \
                                               buf[i] = 0;                                \
                                           }                                              \
                                           for(int i=0; i<16; i++) {                      \
                                               ASSERT_EQ(v_shifted.u.u8[i], buf[i]);      \
                                           }                                              \
                                       }

TEST(SuperVectorUtilsTest,RShift128c){
    u8 vec[16] = { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 };
    auto SP = SuperVector<16>::loadu(vec);
    u8 buf[16];
    TEST_RSHIFT128(buf, vec, SP, 0);
    TEST_RSHIFT128(buf, vec, SP, 1);
    TEST_RSHIFT128(buf, vec, SP, 2);
    TEST_RSHIFT128(buf, vec, SP, 3);
    TEST_RSHIFT128(buf, vec, SP, 4);
    TEST_RSHIFT128(buf, vec, SP, 5);
    TEST_RSHIFT128(buf, vec, SP, 6);
    TEST_RSHIFT128(buf, vec, SP, 7);
    TEST_RSHIFT128(buf, vec, SP, 8);
    TEST_RSHIFT128(buf, vec, SP, 9);
    TEST_RSHIFT128(buf, vec, SP, 10);
    TEST_RSHIFT128(buf, vec, SP, 11);
    TEST_RSHIFT128(buf, vec, SP, 12);
    TEST_RSHIFT128(buf, vec, SP, 13);
    TEST_RSHIFT128(buf, vec, SP, 14);
    TEST_RSHIFT128(buf, vec, SP, 15);
    TEST_RSHIFT128(buf, vec, SP, 16);
}

TEST(SuperVectorUtilsTest,pshufbc) {
    srand (time(NULL));
    u8 vec[16];
    for (int i=0; i<16; i++) {
        vec[i] = rand() % 100 + 1;
    }
    u8 vec2[16];
    for (int i=0; i<16; i++) {
        vec2[i]=i;
    }
    auto SP1 = SuperVector<16>::loadu(vec);
    auto SP2 = SuperVector<16>::loadu(vec2);
    auto SResult = SP1.pshufb(SP2);
    for (int i=0; i<16; i++) {
        ASSERT_EQ(vec[vec2[i]],SResult.u.u8[i]);
    }
}

/*Define ALIGNR128 macro*/
#define TEST_ALIGNR128(v1, v2, buf, l) {                                                     \
                                           auto v_aligned = v2.alignr(v1,l);                 \
                                           printv_u8("v1", v1);                              \
                                           printv_u8("v2", v2);                              \
                                           printv_u8("v_aligned", v_aligned);                \
                                           for (size_t i=0; i<16; i++) {                     \
                                               ASSERT_EQ(v_aligned.u.u8[i], vec[16 -l + i]); \
                                           }                                                 \
                                       }

TEST(SuperVectorUtilsTest,Alignr128c){
    u8 vec[32];
    for (int i=0; i<32; i++) {
        vec[i]=i;
    }
    auto SP1 = SuperVector<16>::loadu(vec);
    auto SP2 = SuperVector<16>::loadu(vec+16);
    TEST_ALIGNR128(SP1, SP2, vec, 0);
    TEST_ALIGNR128(SP1, SP2, vec, 1);
    TEST_ALIGNR128(SP1, SP2, vec, 2);
    TEST_ALIGNR128(SP1, SP2, vec, 3);
    TEST_ALIGNR128(SP1, SP2, vec, 4);
    TEST_ALIGNR128(SP1, SP2, vec, 5);
    TEST_ALIGNR128(SP1, SP2, vec, 6);
    TEST_ALIGNR128(SP1, SP2, vec, 7);
    TEST_ALIGNR128(SP1, SP2, vec, 8);
    TEST_ALIGNR128(SP1, SP2, vec, 9);
    TEST_ALIGNR128(SP1, SP2, vec, 10);
    TEST_ALIGNR128(SP1, SP2, vec, 11);
    TEST_ALIGNR128(SP1, SP2, vec, 12);
    TEST_ALIGNR128(SP1, SP2, vec, 13);
    TEST_ALIGNR128(SP1, SP2, vec, 14);
    TEST_ALIGNR128(SP1, SP2, vec, 15);
    TEST_ALIGNR128(SP1, SP2, vec, 16);
}
