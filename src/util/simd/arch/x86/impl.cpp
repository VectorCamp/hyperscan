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

#ifndef SIMD_IMPL_HPP
#define SIMD_IMPL_HPP

#include <cstdint>
#include <cstdio>

#include "ue2common.h"
#include "util/arch.h"
#include "util/unaligned.h"
#include "util/simd/types.hpp"

#if !defined(m128) && defined(HAVE_SSE2)
typedef __m128i m128;
#endif

#if !defined(m256) && defined(HAVE_AVX2)
typedef __m256i m256;
#endif

#if !defined(m512) && defined(HAVE_AVX512)
typedef __m512i m512;
#endif

#ifdef DEBUG
static inline void print_m128_16x8(const char *label, m128 vector) {
    uint8_t ALIGN_ATTR(16) data[16];
    _mm_store_si128 ((m128 *)data, vector);
    DEBUG_PRINTF("%s: ", label);
    for(int i=0; i < 16; i++)
        printf("%02x ", data[i]);
    printf("\n");
}
#endif

// 128-bit SSE implementation

template<>
really_inline SuperVector<16>::SuperVector(SuperVector const &o)
{
	u.v128[0] = o.u.v128[0];
}

template<>
really_inline SuperVector<16>::SuperVector(typename base_type::type const v)
{
	u.v128[0] = v;
};

template<>
template<>
really_inline SuperVector<16>::SuperVector<int8_t>(int8_t const o)
{
	u.v128[0] = _mm_set1_epi8(o);
}

template<>
template<>
really_inline SuperVector<16>::SuperVector<uint8_t>(uint8_t const o)
{
	u.v128[0] = _mm_set1_epi8(static_cast<int8_t>(o));
}

template<>
template<>
really_inline SuperVector<16>::SuperVector<int16_t>(int16_t const o)
{
	u.v128[0] = _mm_set1_epi16(o);
}

template<>
template<>
really_inline SuperVector<16>::SuperVector<uint16_t>(uint16_t const o)
{
	u.v128[0] = _mm_set1_epi16(static_cast<int16_t>(o));
}

template<>
template<>
really_inline SuperVector<16>::SuperVector<int32_t>(int32_t const o)
{
	u.v128[0] = _mm_set1_epi32(o);
}

template<>
template<>
really_inline SuperVector<16>::SuperVector<uint32_t>(uint32_t const o)
{
	u.v128[0] = _mm_set1_epi32(static_cast<int32_t>(o));
}

template<>
template<>
really_inline SuperVector<16>::SuperVector<int64_t>(int64_t const o)
{
	u.v128[0] = _mm_set1_epi64x(o);
}

template<>
template<>
really_inline SuperVector<16>::SuperVector<uint64_t>(uint64_t const o)
{
	u.v128[0] = _mm_set1_epi64x(static_cast<int64_t>(o));
}

// Constants
template<>
really_inline SuperVector<16> SuperVector<16>::Ones(void)
{
    return {_mm_set1_epi8(0xFF)};
}

template<>
really_inline SuperVector<16> SuperVector<16>::Zeroes(void)
{
    return {_mm_set1_epi8(0)};
}

// Methods

template <>
really_inline void SuperVector<16>::operator=(SuperVector<16> const &o)
{
    u.v128[0] = o.u.v128[0];
}

template <>
really_inline SuperVector<16> SuperVector<16>::operator&(SuperVector<16> const b) const
{
    return {_mm_and_si128(u.v128[0], b.u.v128[0])};
}

template <>
really_inline SuperVector<16> SuperVector<16>::mand(SuperVector<16> const b) const
{
    return *this & b;
}

template <>
really_inline SuperVector<16> SuperVector<16>::mandnot(SuperVector<16> const b) const
{
    return {_mm_andnot_si128(u.v128[0], b.u.v128[0])};
}

template <>
really_inline SuperVector<16> SuperVector<16>::eq(SuperVector<16> const b) const
{
	return {_mm_cmpeq_epi8(u.v128[0], b.u.v128[0])};
}

template <>
really_inline typename SuperVector<16>::movemask_type SuperVector<16>::movemask(void)const
{
	return _mm_movemask_epi8(u.v128[0]);
}

template <>
really_inline typename SuperVector<16>::movemask_type SuperVector<16>::eqmask(SuperVector<16> const b) const
{
	return eq(b).movemask();
}

#ifdef HS_OPTIMIZE
template <>
really_inline SuperVector<16> SuperVector<16>::operator<<(uint8_t const N) const
{
	return {_mm_slli_si128(u.v128[0], N)};
}
#else
template <>
really_inline SuperVector<16> SuperVector<16>::operator<<(uint8_t const N) const
{
	switch(N) {
	case 0: return {_mm_slli_si128(u.v128[0], 0)}; break;
	case 1: return {_mm_slli_si128(u.v128[0], 1)}; break;
	case 2: return {_mm_slli_si128(u.v128[0], 2)}; break;
	case 3: return {_mm_slli_si128(u.v128[0], 3)}; break;
	case 4: return {_mm_slli_si128(u.v128[0], 4)}; break;
	case 5: return {_mm_slli_si128(u.v128[0], 5)}; break;
	case 6: return {_mm_slli_si128(u.v128[0], 6)}; break;
	case 7: return {_mm_slli_si128(u.v128[0], 7)}; break;
	case 8: return {_mm_slli_si128(u.v128[0], 8)}; break;
	case 9: return {_mm_slli_si128(u.v128[0], 9)}; break;
	case 10: return {_mm_slli_si128(u.v128[0], 10)}; break;
	case 11: return {_mm_slli_si128(u.v128[0], 11)}; break;
	case 12: return {_mm_slli_si128(u.v128[0], 12)}; break;
	case 13: return {_mm_slli_si128(u.v128[0], 13)}; break;
	case 14: return {_mm_slli_si128(u.v128[0], 14)}; break;
	case 15: return {_mm_slli_si128(u.v128[0], 15)}; break;
	default: break;
	}
	return *this;
}
#endif

#ifdef HS_OPTIMIZE
template <>
really_inline SuperVector<16> SuperVector<16>::operator>>(uint8_t const N) const
{
	return {_mm_srli_si128(u.v128[0], N)};
}
#else
template <>
really_inline SuperVector<16> SuperVector<16>::operator>>(uint8_t const N) const
{
	switch(N) {
	case 0: return {_mm_srli_si128(u.v128[0], 0)}; break;
	case 1: return {_mm_srli_si128(u.v128[0], 1)}; break;
	case 2: return {_mm_srli_si128(u.v128[0], 2)}; break;
	case 3: return {_mm_srli_si128(u.v128[0], 3)}; break;
	case 4: return {_mm_srli_si128(u.v128[0], 4)}; break;
	case 5: return {_mm_srli_si128(u.v128[0], 5)}; break;
	case 6: return {_mm_srli_si128(u.v128[0], 6)}; break;
	case 7: return {_mm_srli_si128(u.v128[0], 7)}; break;
	case 8: return {_mm_srli_si128(u.v128[0], 8)}; break;
	case 9: return {_mm_srli_si128(u.v128[0], 9)}; break;
	case 10: return {_mm_srli_si128(u.v128[0], 10)}; break;
	case 11: return {_mm_srli_si128(u.v128[0], 11)}; break;
	case 12: return {_mm_srli_si128(u.v128[0], 12)}; break;
	case 13: return {_mm_srli_si128(u.v128[0], 13)}; break;
	case 14: return {_mm_srli_si128(u.v128[0], 14)}; break;
	case 15: return {_mm_srli_si128(u.v128[0], 15)}; break;
	default: break;
	}
	return *this;
}
#endif

template <>
really_inline SuperVector<16> SuperVector<16>::loadu(void const *ptr)
{
    return _mm_loadu_si128((const m128 *)ptr);
}

template <>
really_inline SuperVector<16> SuperVector<16>::load(void const *ptr)
{
    assert(ISALIGNED_N(ptr, alignof(SuperVector::size)));
    ptr = assume_aligned(ptr, SuperVector::size);
    return _mm_load_si128((const m128 *)ptr);
}

template <>
really_inline SuperVector<16> SuperVector<16>::loadu_maskz(void const *ptr, uint8_t const len)
{
	uint8_t alignment = (uintptr_t)(ptr) & 15;
	printf("alignment = %d\n", alignment);
	SuperVector<16> maskb = Ones() << alignment;
	SuperVector<16> maske = Ones() >> (16 -len - alignment);
	print_m128_16x8("maskb", maskb.u.v128[0]);
	print_m128_16x8("maske", maske.u.v128[0]);
	SuperVector<16> v = _mm_loadu_si128((const m128 *)ptr);
	print_m128_16x8("v", v.u.v128[0]);
    return {maskb.u.v128[0] & maske.u.v128[0] & v.u.v128[0]};
}

#ifdef HS_OPTIMIZE
template<>
really_inline SuperVector<16> SuperVector<16>::alignr(SuperVector<16> l, int8_t offset)
{
    return {_mm_alignr_epi8(u.v128[0], l.u.v128[0], offset)};
}
#else
template<>
really_inline SuperVector<16> SuperVector<16>::alignr(SuperVector<16> l, int8_t offset)
{
	switch(offset) {
	case 0: return {_mm_alignr_epi8(u.v128[0], l.u.v128[0], 0)};; break;
	case 1: return {_mm_alignr_epi8(u.v128[0], l.u.v128[0], 1)}; break;
	case 2: return {_mm_alignr_epi8(u.v128[0], l.u.v128[0], 2)}; break;
	case 3: return {_mm_alignr_epi8(u.v128[0], l.u.v128[0], 3)}; break;
	case 4: return {_mm_alignr_epi8(u.v128[0], l.u.v128[0], 4)}; break;
	case 5: return {_mm_alignr_epi8(u.v128[0], l.u.v128[0], 5)}; break;
	case 6: return {_mm_alignr_epi8(u.v128[0], l.u.v128[0], 6)}; break;
	case 7: return {_mm_alignr_epi8(u.v128[0], l.u.v128[0], 7)}; break;
	case 8: return {_mm_alignr_epi8(u.v128[0], l.u.v128[0], 8)}; break;
	case 9: return {_mm_alignr_epi8(u.v128[0], l.u.v128[0], 9)}; break;
	case 10: return {_mm_alignr_epi8(u.v128[0], l.u.v128[0], 10)}; break;
	case 11: return {_mm_alignr_epi8(u.v128[0], l.u.v128[0], 11)}; break;
	case 12: return {_mm_alignr_epi8(u.v128[0], l.u.v128[0], 12)}; break;
	case 13: return {_mm_alignr_epi8(u.v128[0], l.u.v128[0], 13)}; break;
	case 14: return {_mm_alignr_epi8(u.v128[0], l.u.v128[0], 14)}; break;
	case 15: return {_mm_alignr_epi8(u.v128[0], l.u.v128[0], 15)}; break;
	default: break;
	}
	return *this;
}
#endif

template<>
really_inline SuperVector<16> SuperVector<16>::pshufb(SuperVector<16> b)
{
	return {_mm_shuffle_epi8(u.v128[0], b.u.v128[0])};
}

#ifdef HS_HS_OPTIMIZE
template<>
really_inline SuperVector<16> SuperVector<16>::lshift64(uint8_t const l)
{
	return {_mm_slli_epi64(u.v128[0], l)};
}
#else
template<>
really_inline SuperVector<16> SuperVector<16>::lshift64(uint8_t const l)
{
	switch(l) {
	case 0: return {_mm_slli_epi64(u.v128[0], 0)}; break;
	case 1: return {_mm_slli_epi64(u.v128[0], 1)}; break;
	case 2: return {_mm_slli_epi64(u.v128[0], 2)}; break;
	case 3: return {_mm_slli_epi64(u.v128[0], 3)}; break;
	case 4: return {_mm_slli_epi64(u.v128[0], 4)}; break;
	case 5: return {_mm_slli_epi64(u.v128[0], 5)}; break;
	case 6: return {_mm_slli_epi64(u.v128[0], 6)}; break;
	case 7: return {_mm_slli_epi64(u.v128[0], 7)}; break;
	case 8: return {_mm_slli_epi64(u.v128[0], 8)}; break;
	case 9: return {_mm_slli_epi64(u.v128[0], 9)}; break;
	case 10: return {_mm_slli_epi64(u.v128[0], 10)}; break;
	case 11: return {_mm_slli_epi64(u.v128[0], 11)}; break;
	case 12: return {_mm_slli_epi64(u.v128[0], 12)}; break;
	case 13: return {_mm_slli_epi64(u.v128[0], 13)}; break;
	case 14: return {_mm_slli_epi64(u.v128[0], 14)}; break;
	case 15: return {_mm_slli_epi64(u.v128[0], 15)}; break;
	default: break;
	}
	return *this;
}
#endif

#ifdef HS_HS_OPTIMIZE
template<>
really_inline SuperVector<16> SuperVector<16>::rshift64(uint8_t const l)
{
	return {_mm_srli_epi64(u.v128[0], l)};
}
#else
template<>
really_inline SuperVector<16> SuperVector<16>::rshift64(uint8_t const l)
{
	switch(l) {
	case 0: return {_mm_srli_epi64(u.v128[0], 0)}; break;
	case 1: return {_mm_srli_epi64(u.v128[0], 1)}; break;
	case 2: return {_mm_srli_epi64(u.v128[0], 2)}; break;
	case 3: return {_mm_srli_epi64(u.v128[0], 3)}; break;
	case 4: return {_mm_srli_epi64(u.v128[0], 4)}; break;
	case 5: return {_mm_srli_epi64(u.v128[0], 5)}; break;
	case 6: return {_mm_srli_epi64(u.v128[0], 6)}; break;
	case 7: return {_mm_srli_epi64(u.v128[0], 7)}; break;
	case 8: return {_mm_srli_epi64(u.v128[0], 8)}; break;
	case 9: return {_mm_srli_epi64(u.v128[0], 9)}; break;
	case 10: return {_mm_srli_epi64(u.v128[0], 10)}; break;
	case 11: return {_mm_srli_epi64(u.v128[0], 11)}; break;
	case 12: return {_mm_srli_epi64(u.v128[0], 12)}; break;
	case 13: return {_mm_srli_epi64(u.v128[0], 13)}; break;
	case 14: return {_mm_srli_epi64(u.v128[0], 14)}; break;
	case 15: return {_mm_srli_epi64(u.v128[0], 15)}; break;
	default: break;
	}
	return *this;
}
#endif

// 256-bit AVX2 implementation
#if defined(HAVE_AVX2)
template<>
really_inline SuperVector<32>::SuperVector(SuperVector const &o)
{
	u.v256[0] = o.u.v256[0];
}

template<>
really_inline SuperVector<32>::SuperVector(typename base_type::type const v)
{
	u.v256[0] = v;
};

template<>
template<>
really_inline SuperVector<32>::SuperVector<int8_t>(int8_t const o)
{
	u.v256[0] = _mm256_set1_epi8(o);
}

template<>
template<>
really_inline SuperVector<32>::SuperVector<uint8_t>(uint8_t const o)
{
	u.v256[0] = _mm256_set1_epi8(static_cast<int8_t>(o));
}

template<>
template<>
really_inline SuperVector<32>::SuperVector<int16_t>(int16_t const o)
{
	u.v256[0] = _mm256_set1_epi16(o);
}

template<>
template<>
really_inline SuperVector<32>::SuperVector<uint16_t>(uint16_t const o)
{
	u.v256[0] = _mm256_set1_epi16(static_cast<int16_t>(o));
}

template<>
template<>
really_inline SuperVector<32>::SuperVector<int32_t>(int32_t const o)
{
	u.v256[0] = _mm256_set1_epi32(o);
}

template<>
template<>
really_inline SuperVector<32>::SuperVector<uint32_t>(uint32_t const o)
{
	u.v256[0] = _mm256_set1_epi32(static_cast<int32_t>(o));
}

template<>
template<>
really_inline SuperVector<32>::SuperVector<int64_t>(int64_t const o)
{
	u.v256[0] = _mm256_set1_epi64x(o);
}

template<>
template<>
really_inline SuperVector<32>::SuperVector<uint64_t>(uint64_t const o)
{
    u.v256[0] = _mm256_set1_epi64x(static_cast<int64_t>(o));
}

template <>
really_inline void SuperVector<32>::operator=(SuperVector<32> const &o)
{
    u.v256[0] = o.u.v256[0];
}

template <>
really_inline SuperVector<32> SuperVector<32>::operator&(SuperVector<32> const b) const
{
    return {_mm256_and_si256(u.v256[0], b.u.v256[0])};
}

template <>
really_inline SuperVector<32> SuperVector<32>::eq(SuperVector<32> const b) const
{
    return {_mm256_cmpeq_epi8(u.v256[0], b.u.v256[0])};
}

template <>
really_inline typename SuperVector<32>::movemask_type SuperVector<32>::movemask(void)const
{
    return _mm256_movemask_epi8(u.v256[0]);
}

template <>
really_inline typename SuperVector<32>::movemask_type SuperVector<32>::eqmask(SuperVector<32> const b) const
{
    return eq(b).movemask();
}

#ifndef DEBUG
template <>
really_inline SuperVector<32> SuperVector<32>::operator<<(uint8_t const N) const
{
    return {_mm256_slli_si256(u.v256[0], N)};
}
#else
template <>
really_inline SuperVector<32> SuperVector<32>::operator<<(uint8_t const N) const
{
	switch(N) {
	case 0: return {_mm256_slli_si256(u.v256[0], 0)}; break;
	case 1: return {_mm256_slli_si256(u.v256[0], 1)}; break;
	case 2: return {_mm256_slli_si256(u.v256[0], 2)}; break;
	case 3: return {_mm256_slli_si256(u.v256[0], 3)}; break;
	case 4: return {_mm256_slli_si256(u.v256[0], 4)}; break;
	case 5: return {_mm256_slli_si256(u.v256[0], 5)}; break;
	case 6: return {_mm256_slli_si256(u.v256[0], 6)}; break;
	case 7: return {_mm256_slli_si256(u.v256[0], 7)}; break;
	case 8: return {_mm256_slli_si256(u.v256[0], 8)}; break;
	case 9: return {_mm256_slli_si256(u.v256[0], 9)}; break;
	case 10: return {_mm256_slli_si256(u.v256[0], 10)}; break;
	case 11: return {_mm256_slli_si256(u.v256[0], 11)}; break;
	case 12: return {_mm256_slli_si256(u.v256[0], 12)}; break;
	case 13: return {_mm256_slli_si256(u.v256[0], 13)}; break;
	case 14: return {_mm256_slli_si256(u.v256[0], 14)}; break;
	case 15: return {_mm256_slli_si256(u.v256[0], 15)}; break;
	default: break;
	}
	return *this;
}
#endif

template <>
really_inline SuperVector<32> SuperVector<32>::loadu(void const *ptr)
{
    return {_mm256_loadu_si256((const m256 *)ptr)};
}

template <>
really_inline SuperVector<32> SuperVector<32>::load(void const *ptr)
{
    assert(ISALIGNED_N(ptr, alignof(SuperVector::size)));
    ptr = assume_aligned(ptr, SuperVector::size);
    return {_mm256_load_si256((const m256 *)ptr)};
}

template <>
really_inline SuperVector<32> SuperVector<32>::loadu_mask(void const *ptr, size_t const len)
{

    return {_mm256_loadu_si256((const m256 *)ptr)};
}

#ifndef DEBUG
template<>
really_inline SuperVector<32> SuperVector<32>::alignr(SuperVector<32> l, int8_t offset)
{
    return {_mm256_alignr_epi8(u.v256[0], l.u.v256[0], offset)};
}
#else
template<>
really_inline SuperVector<32> SuperVector<32>::alignr(SuperVector<32> l, int8_t offset)
{
	switch(offset) {
	case 0: return {_mm256_alignr_epi8(u.v256[0], l.u.v256[0], 0)};; break;
	case 1: return {_mm256_alignr_epi8(u.v256[0], l.u.v256[0], 1)}; break;
	case 2: return {_mm256_alignr_epi8(u.v256[0], l.u.v256[0], 2)}; break;
	case 3: return {_mm256_alignr_epi8(u.v256[0], l.u.v256[0], 3)}; break;
	case 4: return {_mm256_alignr_epi8(u.v256[0], l.u.v256[0], 4)}; break;
	case 5: return {_mm256_alignr_epi8(u.v256[0], l.u.v256[0], 5)}; break;
	case 6: return {_mm256_alignr_epi8(u.v256[0], l.u.v256[0], 6)}; break;
	case 7: return {_mm256_alignr_epi8(u.v256[0], l.u.v256[0], 7)}; break;
	case 8: return {_mm256_alignr_epi8(u.v256[0], l.u.v256[0], 8)}; break;
	case 9: return {_mm256_alignr_epi8(u.v256[0], l.u.v256[0], 9)}; break;
	case 10: return {_mm256_alignr_epi8(u.v256[0], l.u.v256[0], 10)}; break;
	case 11: return {_mm256_alignr_epi8(u.v256[0], l.u.v256[0], 11)}; break;
	case 12: return {_mm256_alignr_epi8(u.v256[0], l.u.v256[0], 12)}; break;
	case 13: return {_mm256_alignr_epi8(u.v256[0], l.u.v256[0], 13)}; break;
	case 14: return {_mm256_alignr_epi8(u.v256[0], l.u.v256[0], 14)}; break;
	case 15: return {_mm256_alignr_epi8(u.v256[0], l.u.v256[0], 15)}; break;
	default: break;
	}
	return *this;
}
#endif
/*
template<>
really_inline SuperVector<32> SuperVector<32>::alignr(SuperVector<32> l, int8_t offset)
{
	printf("offset = %d\n", offset);
	//u.v256[0] = _mm256_set_epi8(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32);
	//l.u.v256[0] = _mm256_set_epi8(101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132);
	print_m256_32x8("this", u.v256[0]);
	print_m256_32x8("l", l.u.v256[0]);
	__m128i v1 = _mm256_extracti128_si256(u.v256[0], 0);
	print1_m128_16x8("v1", v1);
        __m128i v2 = _mm256_extracti128_si256(u.v256[0], 1);
	print1_m128_16x8("v2", v2);
        __m128i l1 = _mm256_extracti128_si256(l.u.v256[0], 0);
	print1_m128_16x8("l1", l1);
        __m128i y1 = _mm_alignr_epi8(v2, l1, 16 - offset);
	print1_m128_16x8("y1", y1);
        __m128i y2 = _mm_alignr_epi8(v2, v1, 16 - offset);
	print1_m128_16x8("y2", y2);
	print_m256_32x8("this", _mm256_set_m128i(y1, y2));
	return {_mm256_set_m128i(y1, y2)};
}*/

// Constants
template<>
really_inline SuperVector<32> SuperVector<32>::Ones(void)
{
    return {_mm256_set1_epi8(0xFF)};
}

template<>
really_inline SuperVector<32> SuperVector<32>::Zeroes(void)
{
    return {_mm256_set1_epi8(0)};
}

#endif // HAVE_AVX2

// 512-bit AVX512 implementation
#if defined(HAVE_AVX512)
template<>
really_inline SuperVector<64>::SuperVector(SuperVector const &o)
{
    u.v512[0] = o.u.v512[0];
}

template<>
really_inline SuperVector<64>::SuperVector(typename base_type::type const v)
{
    u.v512[0] = v;
};

template<>
template<>
really_inline SuperVector<64>::SuperVector<int8_t>(int8_t const o)
{
    u.v512[0] = _mm512_set1_epi8(o);
}

template<>
template<>
really_inline SuperVector<64>::SuperVector<uint8_t>(uint8_t const o)
{
    u.v512[0] = _mm512_set1_epi8(static_cast<int8_t>(o));
}

template<>
template<>
really_inline SuperVector<64>::SuperVector<int16_t>(int16_t const o)
{
    u.v512[0] = _mm512_set1_epi16(o);
}

template<>
template<>
really_inline SuperVector<64>::SuperVector<uint16_t>(uint16_t const o)
{
    u.v512[0] = _mm512_set1_epi16(static_cast<int16_t>(o));
}

template<>
template<>
really_inline SuperVector<64>::SuperVector<int32_t>(int32_t const o)
{
    u.v512[0] = _mm512_set1_epi32(o);
}

template<>
template<>
really_inline SuperVector<64>::SuperVector<uint32_t>(uint32_t const o)
{
    u.v512[0] = _mm512_set1_epi32(static_cast<int32_t>(o));
}

template<>
template<>
really_inline SuperVector<64>::SuperVector<int64_t>(int64_t const o)
{
    u.v512[0] = _mm512_set1_epi64(o);
}

template<>
template<>
really_inline SuperVector<64>::SuperVector<uint64_t>(uint64_t const o)
{
    u.v512[0] = _mm512_set1_epi64(static_cast<int64_t>(o));
}

template <>
really_inline void SuperVector<64>::operator=(SuperVector<64> const &o)
{
    u.v512[0] = o.u.v512[0];
}

template <>
really_inline SuperVector<64> SuperVector<64>::operator&(SuperVector<64> const b) const
{
    return {_mm512_and_si512(u.v512[0], b.u.v512[0])};
}

template <>
really_inline typename SuperVector<64>::movemask_type SuperVector<64>::eqmask(SuperVector<64> const b) const
{
    return _mm512_cmpeq_epi8_mask(u.v512[0], b.u.v512[0]);
}

// template <>
// really_inline SuperVector<64> SuperVector<64>::operator<<(uint8_t const N) const
// {
// 	return {_mm512_slli_si512(u.v512[0], N)};
// }

template <>
really_inline SuperVector<64> SuperVector<64>::loadu(void const *ptr)
{
    return {_mm512_loadu_si512((const m512 *)ptr)};
}

template <>
really_inline SuperVector<64> SuperVector<64>::load(void const *ptr)
{
    assert(ISALIGNED_N(ptr, alignof(SuperVector::size)));
    ptr = assume_aligned(ptr, SuperVector::size);
    return {_mm512_load_si512((const m512 *)ptr)};
}

#ifndef DEBUG
template<>
really_inline SuperVector<64> SuperVector<64>::alignr(SuperVector<64> l, int8_t offset)
{
    return {_mm512_alignr_epi8(u.v512[0], l.u.v512[0], offset)};
}
#else
template<>
really_inline SuperVector<64> SuperVector<64>::alignr(SuperVector<64> l, int8_t offset)
{
	switch(offset) {
	case 0: return {_mm512_alignr_epi8(u.v512[0], l.u.v512[0], 0)};; break;
	case 1: return {_mm512_alignr_epi8(u.v512[0], l.u.v512[0], 1)}; break;
	case 2: return {_mm512_alignr_epi8(u.v512[0], l.u.v512[0], 2)}; break;
	case 3: return {_mm512_alignr_epi8(u.v512[0], l.u.v512[0], 3)}; break;
	case 4: return {_mm512_alignr_epi8(u.v512[0], l.u.v512[0], 4)}; break;
	case 5: return {_mm512_alignr_epi8(u.v512[0], l.u.v512[0], 5)}; break;
	case 6: return {_mm512_alignr_epi8(u.v512[0], l.u.v512[0], 6)}; break;
	case 7: return {_mm512_alignr_epi8(u.v512[0], l.u.v512[0], 7)}; break;
	case 8: return {_mm512_alignr_epi8(u.v512[0], l.u.v512[0], 8)}; break;
	case 9: return {_mm512_alignr_epi8(u.v512[0], l.u.v512[0], 9)}; break;
	case 10: return {_mm512_alignr_epi8(u.v512[0], l.u.v512[0], 10)}; break;
	case 11: return {_mm512_alignr_epi8(u.v512[0], l.u.v512[0], 11)}; break;
	case 12: return {_mm512_alignr_epi8(u.v512[0], l.u.v512[0], 12)}; break;
	case 13: return {_mm512_alignr_epi8(u.v512[0], l.u.v512[0], 13)}; break;
	case 14: return {_mm512_alignr_epi8(u.v512[0], l.u.v512[0], 14)}; break;
	case 15: return {_mm512_alignr_epi8(u.v512[0], l.u.v512[0], 15)}; break;
	default: break;
	}
	return *this;
}
#endif

// Constants
template<>
really_inline SuperVector<64> SuperVector<64>::Ones(void)
{
    return {_mm512_set1_epi8(0xFF)};
}

template<>
really_inline SuperVector<64> SuperVector<64>::Zeroes(void)
{
    return {_mm512_set1_epi8(0)};
}

#endif // HAVE_AVX512


#endif // SIMD_IMPL_HPP
