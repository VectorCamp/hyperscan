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
#include "util/supervector/supervector.hpp"

// 128-bit SSE implementation

template<>
really_inline SuperVector<16>::SuperVector(SuperVector const &other)
{
	u.v128[0] = other.u.v128[0];
}

template<>
really_inline SuperVector<16>::SuperVector(typename base_type::type const v)
{
	u.v128[0] = v;
};

template<>
template<>
really_inline SuperVector<16>::SuperVector<int8_t>(int8_t const other)
{
	u.v128[0] = _mm_set1_epi8(other);
}

template<>
template<>
really_inline SuperVector<16>::SuperVector<uint8_t>(uint8_t const other)
{
	u.v128[0] = _mm_set1_epi8(static_cast<int8_t>(other));
}

template<>
template<>
really_inline SuperVector<16>::SuperVector<int16_t>(int16_t const other)
{
	u.v128[0] = _mm_set1_epi16(other);
}

template<>
template<>
really_inline SuperVector<16>::SuperVector<uint16_t>(uint16_t const other)
{
	u.v128[0] = _mm_set1_epi16(static_cast<int16_t>(other));
}

template<>
template<>
really_inline SuperVector<16>::SuperVector<int32_t>(int32_t const other)
{
	u.v128[0] = _mm_set1_epi32(other);
}

template<>
template<>
really_inline SuperVector<16>::SuperVector<uint32_t>(uint32_t const other)
{
	u.v128[0] = _mm_set1_epi32(static_cast<int32_t>(other));
}

template<>
template<>
really_inline SuperVector<16>::SuperVector<int64_t>(int64_t const other)
{
	u.v128[0] = _mm_set1_epi64x(other);
}

template<>
template<>
really_inline SuperVector<16>::SuperVector<uint64_t>(uint64_t const other)
{
	u.v128[0] = _mm_set1_epi64x(static_cast<int64_t>(other));
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
really_inline void SuperVector<16>::operator=(SuperVector<16> const &other)
{
    u.v128[0] = other.u.v128[0];
}

template <>
really_inline SuperVector<16> SuperVector<16>::operator&(SuperVector<16> const &b) const
{
    return {_mm_and_si128(u.v128[0], b.u.v128[0])};
}

template <>
really_inline SuperVector<16> SuperVector<16>::operator|(SuperVector<16> const &b) const
{
    return {_mm_or_si128(u.v128[0], b.u.v128[0])};
}

template <>
really_inline SuperVector<16> SuperVector<16>::operator^(SuperVector<16> const &b) const
{
    return {_mm_xor_si128(u.v128[0], b.u.v128[0])};
}

template <>
really_inline SuperVector<16> SuperVector<16>::opandnot(SuperVector<16> const &b) const
{
    return {_mm_andnot_si128(u.v128[0], b.u.v128[0])};
}

template <>
really_inline SuperVector<16> SuperVector<16>::eq(SuperVector<16> const &b) const
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
really_inline SuperVector<16> SuperVector<16>::operator>>(uint8_t const N) const
{
    return {_mm_srli_si128(u.v128[0], N)};
}
#else
template <>
really_inline SuperVector<16> SuperVector<16>::operator>>(uint8_t const N) const
{
    switch(N) {
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
    case 16: return Zeroes(); break;
    default: break;
    }
    return *this;
}
#endif

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
    case 16: return Zeroes(); break;
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
	SuperVector<16> mask = Ones() >> (16 -len);
	mask.print8("mask");
	SuperVector<16> v = _mm_loadu_si128((const m128 *)ptr);
	v.print8("v");
	return mask & v;
}

#ifdef HS_OPTIMIZE
template<>
really_inline SuperVector<16> SuperVector<16>::alignr(SuperVector<16> &other, int8_t offset)
{
    return {_mm_alignr_epi8(u.v128[0], other.u.v128[0], offset)};
}
#else
template<>
really_inline SuperVector<16> SuperVector<16>::alignr(SuperVector<16> &other, int8_t offset)
{
	switch(offset) {
	case 0: return other; break;
	case 1: return {_mm_alignr_epi8(u.v128[0], other.u.v128[0], 1)}; break;
	case 2: return {_mm_alignr_epi8(u.v128[0], other.u.v128[0], 2)}; break;
	case 3: return {_mm_alignr_epi8(u.v128[0], other.u.v128[0], 3)}; break;
	case 4: return {_mm_alignr_epi8(u.v128[0], other.u.v128[0], 4)}; break;
	case 5: return {_mm_alignr_epi8(u.v128[0], other.u.v128[0], 5)}; break;
	case 6: return {_mm_alignr_epi8(u.v128[0], other.u.v128[0], 6)}; break;
	case 7: return {_mm_alignr_epi8(u.v128[0], other.u.v128[0], 7)}; break;
	case 8: return {_mm_alignr_epi8(u.v128[0], other.u.v128[0], 8)}; break;
	case 9: return {_mm_alignr_epi8(u.v128[0], other.u.v128[0], 9)}; break;
	case 10: return {_mm_alignr_epi8(u.v128[0], other.u.v128[0], 10)}; break;
	case 11: return {_mm_alignr_epi8(u.v128[0], other.u.v128[0], 11)}; break;
	case 12: return {_mm_alignr_epi8(u.v128[0], other.u.v128[0], 12)}; break;
	case 13: return {_mm_alignr_epi8(u.v128[0], other.u.v128[0], 13)}; break;
	case 14: return {_mm_alignr_epi8(u.v128[0], other.u.v128[0], 14)}; break;
	case 15: return {_mm_alignr_epi8(u.v128[0], other.u.v128[0], 15)}; break;
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

#ifdef HS_OPTIMIZE
template<>
really_inline SuperVector<16> SuperVector<16>::lshift64(uint8_t const N)
{
	return {_mm_slli_epi64(u.v128[0], N)};
}
#else
template<>
really_inline SuperVector<16> SuperVector<16>::lshift64(uint8_t const N)
{
	switch(N) {
	case 0: return *this; break;
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

#ifdef HS_OPTIMIZE
template<>
really_inline SuperVector<16> SuperVector<16>::rshift64(uint8_t const N)
{
	return {_mm_srli_epi64(u.v128[0], N)};
}
#else
template<>
really_inline SuperVector<16> SuperVector<16>::rshift64(uint8_t const N)
{
	switch(N) {
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
really_inline SuperVector<32>::SuperVector(SuperVector const &other)
{
	u.v256[0] = other.u.v256[0];
}

template<>
really_inline SuperVector<32>::SuperVector(typename base_type::type const v)
{
	u.v256[0] = v;
};

template<>
template<>
really_inline SuperVector<32>::SuperVector(m128 const v)
{
	u.v256[0] = _mm256_broadcastsi128_si256(v);
};

template<>
template<>
really_inline SuperVector<32>::SuperVector<int8_t>(int8_t const other)
{
	u.v256[0] = _mm256_set1_epi8(other);
}

template<>
template<>
really_inline SuperVector<32>::SuperVector<uint8_t>(uint8_t const other)
{
	u.v256[0] = _mm256_set1_epi8(static_cast<int8_t>(other));
}

template<>
template<>
really_inline SuperVector<32>::SuperVector<int16_t>(int16_t const other)
{
	u.v256[0] = _mm256_set1_epi16(other);
}

template<>
template<>
really_inline SuperVector<32>::SuperVector<uint16_t>(uint16_t const other)
{
	u.v256[0] = _mm256_set1_epi16(static_cast<int16_t>(other));
}

template<>
template<>
really_inline SuperVector<32>::SuperVector<int32_t>(int32_t const other)
{
	u.v256[0] = _mm256_set1_epi32(other);
}

template<>
template<>
really_inline SuperVector<32>::SuperVector<uint32_t>(uint32_t const other)
{
	u.v256[0] = _mm256_set1_epi32(static_cast<int32_t>(other));
}

template<>
template<>
really_inline SuperVector<32>::SuperVector<int64_t>(int64_t const other)
{
	u.v256[0] = _mm256_set1_epi64x(other);
}

template<>
template<>
really_inline SuperVector<32>::SuperVector<uint64_t>(uint64_t const other)
{
    u.v256[0] = _mm256_set1_epi64x(static_cast<int64_t>(other));
}

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

template <>
really_inline void SuperVector<32>::operator=(SuperVector<32> const &other)
{
    u.v256[0] = other.u.v256[0];
}

template <>
really_inline SuperVector<32> SuperVector<32>::operator&(SuperVector<32> const &b) const
{
    return {_mm256_and_si256(u.v256[0], b.u.v256[0])};
}

template <>
really_inline SuperVector<32> SuperVector<32>::operator|(SuperVector<32> const &b) const
{
    return {_mm256_or_si256(u.v256[0], b.u.v256[0])};
}

template <>
really_inline SuperVector<32> SuperVector<32>::operator^(SuperVector<32> const &b) const
{
    return {_mm256_xor_si256(u.v256[0], b.u.v256[0])};
}

template <>
really_inline SuperVector<32> SuperVector<32>::opandnot(SuperVector<32> const &b) const
{
    return {_mm256_andnot_si256(u.v256[0], b.u.v256[0])};
}

template <>
really_inline SuperVector<32> SuperVector<32>::eq(SuperVector<32> const &b) const
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

#ifdef HS_OPTIMIZE
template <>
really_inline SuperVector<32> SuperVector<32>::operator>>(uint8_t const N) const
{
	// As found here: https://stackoverflow.com/questions/25248766/emulating-shifts-on-32-bytes-with-avx
	if (N < 16) {
		return {_mm256_alignr_epi8(_mm256_permute2x128_si256(A, A, _MM_SHUFFLE(2, 0, 0, 1)), A, N)};
	} else if (N == 16) {
		return {_mm256_permute2x128_si256(A, A, _MM_SHUFFLE(2, 0, 0, 1))};
	} else {
        return {_mm256_srli_si256(_mm256_permute2x128_si256(A, A, _MM_SHUFFLE(2, 0, 0, 1)), N - 16)};
    }
}
#else
template <>
really_inline SuperVector<32> SuperVector<32>::operator>>(uint8_t const N) const
{
	switch(N) {
	case 1: return {_mm256_alignr_epi8(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(2, 0, 0, 1)), u.v256[0], 1)}; break;
	case 2: return {_mm256_alignr_epi8(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(2, 0, 0, 1)), u.v256[0], 2)}; break;
	case 3: return {_mm256_alignr_epi8(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(2, 0, 0, 1)), u.v256[0], 3)}; break;
	case 4: return {_mm256_alignr_epi8(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(2, 0, 0, 1)), u.v256[0], 4)}; break;
	case 5: return {_mm256_alignr_epi8(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(2, 0, 0, 1)), u.v256[0], 5)}; break;
	case 6: return {_mm256_alignr_epi8(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(2, 0, 0, 1)), u.v256[0], 6)}; break;
	case 7: return {_mm256_alignr_epi8(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(2, 0, 0, 1)), u.v256[0], 7)}; break;
	case 8: return {_mm256_alignr_epi8(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(2, 0, 0, 1)), u.v256[0], 8)}; break;
	case 9: return {_mm256_alignr_epi8(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(2, 0, 0, 1)), u.v256[0], 9)}; break;
	case 10: return {_mm256_alignr_epi8(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(2, 0, 0, 1)), u.v256[0], 10)}; break;
	case 11: return {_mm256_alignr_epi8(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(2, 0, 0, 1)), u.v256[0], 11)}; break;
	case 12: return {_mm256_alignr_epi8(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(2, 0, 0, 1)), u.v256[0], 12)}; break;
	case 13: return {_mm256_alignr_epi8(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(2, 0, 0, 1)), u.v256[0], 13)}; break;
	case 14: return {_mm256_alignr_epi8(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(2, 0, 0, 1)), u.v256[0], 14)}; break;
	case 15: return {_mm256_alignr_epi8(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(2, 0, 0, 1)), u.v256[0], 15)}; break;
	case 16: return {_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(2, 0, 0, 1))}; break;
	case 17: return {_mm256_srli_si256(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(2, 0, 0, 1)), 1)}; break;
	case 18: return {_mm256_srli_si256(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(2, 0, 0, 1)), 2)}; break;
	case 19: return {_mm256_srli_si256(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(2, 0, 0, 1)), 3)}; break;
	case 20: return {_mm256_srli_si256(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(2, 0, 0, 1)), 4)}; break;
	case 21: return {_mm256_srli_si256(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(2, 0, 0, 1)), 5)}; break;
	case 22: return {_mm256_srli_si256(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(2, 0, 0, 1)), 6)}; break;
	case 23: return {_mm256_srli_si256(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(2, 0, 0, 1)), 7)}; break;
	case 24: return {_mm256_srli_si256(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(2, 0, 0, 1)), 8)}; break;
	case 25: return {_mm256_srli_si256(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(2, 0, 0, 1)), 9)}; break;
	case 26: return {_mm256_srli_si256(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(2, 0, 0, 1)), 10)}; break;
	case 27: return {_mm256_srli_si256(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(2, 0, 0, 1)), 11)}; break;
	case 28: return {_mm256_srli_si256(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(2, 0, 0, 1)), 12)}; break;
	case 29: return {_mm256_srli_si256(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(2, 0, 0, 1)), 13)}; break;
	case 30: return {_mm256_srli_si256(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(2, 0, 0, 1)), 14)}; break;
	case 31: return {_mm256_srli_si256(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(2, 0, 0, 1)), 15)}; break;
	case 32: return Zeroes(); break;
	default: break;
	}
	return *this;
}
#endif

#ifdef HS_OPTIMIZE
template <>
really_inline SuperVector<32> SuperVector<32>::operator<<(uint8_t const N) const
{
	// As found here: https://stackoverflow.com/questions/25248766/emulating-shifts-on-32-bytes-with-avx
    if (N < 16) {
		return {_mm256_alignr_epi8(A, _mm256_permute2x128_si256(A, A, _MM_SHUFFLE(0, 0, 2, 0)), 16 - N)};
	} else if (N == 16) {
		return {_mm256_permute2x128_si256(A, A, _MM_SHUFFLE(0, 0, 2, 0))};
	} else {
        return {_mm256_slli_si256(_mm256_permute2x128_si256(A, A, _MM_SHUFFLE(0, 0, 2, 0)), N - 16)};
    }
}
#else
template <>
really_inline SuperVector<32> SuperVector<32>::operator<<(uint8_t const N) const
{
	switch(N) {
	case 1: return {_mm256_alignr_epi8(u.v256[0], _mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(0, 0, 2, 0)), 15)}; break;
	case 2: return {_mm256_alignr_epi8(u.v256[0], _mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(0, 0, 2, 0)), 14)}; break;
	case 3: return {_mm256_alignr_epi8(u.v256[0], _mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(0, 0, 2, 0)), 13)}; break;
	case 4: return {_mm256_alignr_epi8(u.v256[0], _mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(0, 0, 2, 0)), 12)}; break;
	case 5: return {_mm256_alignr_epi8(u.v256[0], _mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(0, 0, 2, 0)), 11)}; break;
	case 6: return {_mm256_alignr_epi8(u.v256[0], _mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(0, 0, 2, 0)), 10)}; break;
	case 7: return {_mm256_alignr_epi8(u.v256[0], _mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(0, 0, 2, 0)), 9)}; break;
	case 8: return {_mm256_alignr_epi8(u.v256[0], _mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(0, 0, 2, 0)), 8)}; break;
	case 9: return {_mm256_alignr_epi8(u.v256[0], _mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(0, 0, 2, 0)), 7)}; break;
	case 10: return {_mm256_alignr_epi8(u.v256[0], _mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(0, 0, 2, 0)), 6)}; break;
	case 11: return {_mm256_alignr_epi8(u.v256[0], _mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(0, 0, 2, 0)), 5)}; break;
	case 12: return {_mm256_alignr_epi8(u.v256[0], _mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(0, 0, 2, 0)), 4)}; break;
	case 13: return {_mm256_alignr_epi8(u.v256[0], _mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(0, 0, 2, 0)), 3)}; break;
	case 14: return {_mm256_alignr_epi8(u.v256[0], _mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(0, 0, 2, 0)), 2)}; break;
	case 15: return {_mm256_alignr_epi8(u.v256[0], _mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(0, 0, 2, 0)), 1)}; break;
	case 16: return {_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(0, 0, 2, 0))}; break;
	case 17: return {_mm256_slli_si256(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(0, 0, 2, 0)), 1)}; break;
	case 18: return {_mm256_slli_si256(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(0, 0, 2, 0)), 2)}; break;
	case 19: return {_mm256_slli_si256(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(0, 0, 2, 0)), 3)}; break;
	case 20: return {_mm256_slli_si256(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(0, 0, 2, 0)), 4)}; break;
	case 21: return {_mm256_slli_si256(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(0, 0, 2, 0)), 5)}; break;
	case 22: return {_mm256_slli_si256(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(0, 0, 2, 0)), 6)}; break;
	case 23: return {_mm256_slli_si256(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(0, 0, 2, 0)), 7)}; break;
	case 24: return {_mm256_slli_si256(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(0, 0, 2, 0)), 8)}; break;
	case 25: return {_mm256_slli_si256(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(0, 0, 2, 0)), 9)}; break;
	case 26: return {_mm256_slli_si256(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(0, 0, 2, 0)), 10)}; break;
	case 27: return {_mm256_slli_si256(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(0, 0, 2, 0)), 11)}; break;
	case 28: return {_mm256_slli_si256(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(0, 0, 2, 0)), 12)}; break;
	case 29: return {_mm256_slli_si256(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(0, 0, 2, 0)), 13)}; break;
	case 30: return {_mm256_slli_si256(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(0, 0, 2, 0)), 14)}; break;
	case 31: return {_mm256_slli_si256(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(0, 0, 2, 0)), 15)}; break;
	case 32: return Zeroes(); break;
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
really_inline SuperVector<32> SuperVector<32>::loadu_maskz(void const *ptr, uint8_t const len)
{
	SuperVector<32> mask = Ones() >> (32 - len);
	mask.print8("mask");
	SuperVector<32> v = _mm256_loadu_si256((const m256 *)ptr);
	v.print8("v");
    return mask & v;
}

#ifdef HS_OPTIMIZE
template<>
really_inline SuperVector<32> SuperVector<32>::alignr(SuperVector<32> &other, int8_t offset)
{
    return {_mm256_alignr_epi8(u.v256[0], other.u.v256[0], offset)};
}
#else
template<>
really_inline SuperVector<32> SuperVector<32>::alignr(SuperVector<32> &other, int8_t offset)
{
	switch(offset) {
	case 0: return other; break;
	case 1: return {_mm256_alignr_epi8(u.v256[0], other.u.v256[0], 1)}; break;
	case 2: return {_mm256_alignr_epi8(u.v256[0], other.u.v256[0], 2)}; break;
	case 3: return {_mm256_alignr_epi8(u.v256[0], other.u.v256[0], 3)}; break;
	case 4: return {_mm256_alignr_epi8(u.v256[0], other.u.v256[0], 4)}; break;
	case 5: return {_mm256_alignr_epi8(u.v256[0], other.u.v256[0], 5)}; break;
	case 6: return {_mm256_alignr_epi8(u.v256[0], other.u.v256[0], 6)}; break;
	case 7: return {_mm256_alignr_epi8(u.v256[0], other.u.v256[0], 7)}; break;
	case 8: return {_mm256_alignr_epi8(u.v256[0], other.u.v256[0], 8)}; break;
	case 9: return {_mm256_alignr_epi8(u.v256[0], other.u.v256[0], 9)}; break;
	case 10: return {_mm256_alignr_epi8(u.v256[0], other.u.v256[0], 10)}; break;
	case 11: return {_mm256_alignr_epi8(u.v256[0], other.u.v256[0], 11)}; break;
	case 12: return {_mm256_alignr_epi8(u.v256[0], other.u.v256[0], 12)}; break;
	case 13: return {_mm256_alignr_epi8(u.v256[0], other.u.v256[0], 13)}; break;
	case 14: return {_mm256_alignr_epi8(u.v256[0], other.u.v256[0], 14)}; break;
	case 15: return {_mm256_alignr_epi8(u.v256[0], other.u.v256[0], 15)}; break;
	default: break;
	}
	return *this;
}
#endif

template<>
really_inline SuperVector<32> SuperVector<32>::pshufb(SuperVector<32> b)
{
	return {_mm256_shuffle_epi8(u.v256[0], b.u.v256[0])};
}

#ifdef HS_OPTIMIZE
template<>
really_inline SuperVector<32> SuperVector<32>::lshift64(uint8_t const N)
{
	return {_mm256_slli_epi64(u.v256[0], N)};
}
#else
template<>
really_inline SuperVector<32> SuperVector<32>::lshift64(uint8_t const N)
{
	switch(N) {
	case 0: return *this; break;
	case 1: return {_mm256_slli_epi64(u.v256[0], 1)}; break;
	case 2: return {_mm256_slli_epi64(u.v256[0], 2)}; break;
	case 3: return {_mm256_slli_epi64(u.v256[0], 3)}; break;
	case 4: return {_mm256_slli_epi64(u.v256[0], 4)}; break;
	case 5: return {_mm256_slli_epi64(u.v256[0], 5)}; break;
	case 6: return {_mm256_slli_epi64(u.v256[0], 6)}; break;
	case 7: return {_mm256_slli_epi64(u.v256[0], 7)}; break;
	case 8: return {_mm256_slli_epi64(u.v256[0], 8)}; break;
	case 9: return {_mm256_slli_epi64(u.v256[0], 9)}; break;
	case 10: return {_mm256_slli_epi64(u.v256[0], 10)}; break;
	case 11: return {_mm256_slli_epi64(u.v256[0], 11)}; break;
	case 12: return {_mm256_slli_epi64(u.v256[0], 12)}; break;
	case 13: return {_mm256_slli_epi64(u.v256[0], 13)}; break;
	case 14: return {_mm256_slli_epi64(u.v256[0], 14)}; break;
	case 15: return {_mm256_slli_epi64(u.v256[0], 15)}; break;
	default: break;
	}
	return *this;
}
#endif

#ifdef HS_OPTIMIZE
template<>
really_inline SuperVector<32> SuperVector<32>::rshift64(uint8_t const N)
{
	return {_mm256_srli_epi64(u.v256[0], N)};
}
#else
template<>
really_inline SuperVector<32> SuperVector<32>::rshift64(uint8_t const N)
{
	switch(N) {
	case 0: return *this; break;
	case 1: return {_mm256_srli_epi64(u.v256[0], 1)}; break;
	case 2: return {_mm256_srli_epi64(u.v256[0], 2)}; break;
	case 3: return {_mm256_srli_epi64(u.v256[0], 3)}; break;
	case 4: return {_mm256_srli_epi64(u.v256[0], 4)}; break;
	case 5: return {_mm256_srli_epi64(u.v256[0], 5)}; break;
	case 6: return {_mm256_srli_epi64(u.v256[0], 6)}; break;
	case 7: return {_mm256_srli_epi64(u.v256[0], 7)}; break;
	case 8: return {_mm256_srli_epi64(u.v256[0], 8)}; break;
	case 9: return {_mm256_srli_epi64(u.v256[0], 9)}; break;
	case 10: return {_mm256_srli_epi64(u.v256[0], 10)}; break;
	case 11: return {_mm256_srli_epi64(u.v256[0], 11)}; break;
	case 12: return {_mm256_srli_epi64(u.v256[0], 12)}; break;
	case 13: return {_mm256_srli_epi64(u.v256[0], 13)}; break;
	case 14: return {_mm256_srli_epi64(u.v256[0], 14)}; break;
	case 15: return {_mm256_srli_epi64(u.v256[0], 15)}; break;
	default: break;
	}
	return *this;
}
#endif

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
really_inline SuperVector<64>::SuperVector(m256 const v)
{
    u.v512[0] = _mm512_broadcast_i64x4(v);
};

template<>
template<>
really_inline SuperVector<64>::SuperVector(m128 const v)
{
    u.v512[0] = _mm512_broadcast_i32x4(v);
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


// Methods
template <>
really_inline void SuperVector<64>::operator=(SuperVector<64> const &o)
{
    u.v512[0] = o.u.v512[0];
}

template <>
really_inline SuperVector<64> SuperVector<64>::operator&(SuperVector<64> const &b) const
{
    return {_mm512_and_si512(u.v512[0], b.u.v512[0])};
}

template <>
really_inline SuperVector<64> SuperVector<64>::operator|(SuperVector<64> const &b) const
{
    return {_mm512_or_si512(u.v512[0], b.u.v512[0])};
}

template <>
really_inline SuperVector<64> SuperVector<64>::operator^(SuperVector<64> const &b) const
{
    return {_mm512_xor_si512(u.v512[0], b.u.v512[0])};
}

template <>
really_inline SuperVector<64> SuperVector<64>::opandnot(SuperVector<64> const &b) const
{
    return {_mm512_andnot_si512(u.v512[0], b.u.v512[0])};
}

template <>
really_inline SuperVector<64> SuperVector<64>::eq(SuperVector<64> const &b) const
{
	m512_t sp = SuperVector<64>::Zeroes();
	sp.u.v256[0] = _mm256_cmpeq_epi8(u.v256[0], b.u.v256[0]);
	sp.u.v256[1] = _mm256_cmpeq_epi8(u.v256[1], b.u.v256[1]);
    return {sp.u.v512[0]};
}

template <>
really_inline typename SuperVector<64>::movemask_type SuperVector<64>::movemask(void)const
{   
	m512_t msb = SuperVector<64>::dup_u8(0x80);
	m512_t mask = msb & *this;
	return _mm512_cmpeq_epi8_mask(mask.u.v512[0],msb.u.v512[0]);
}


template <>
really_inline typename SuperVector<64>::movemask_type SuperVector<64>::eqmask(SuperVector<64> const b) const
{
    return _mm512_cmpeq_epi8_mask(u.v512[0], b.u.v512[0]);
}


#ifdef HS_OPTIMIZE
template <>
really_inline SuperVector<64> SuperVector<64>::operator>>(uint8_t const N) const
{
	// As found here: https://stackoverflow.com/questions/25248766/emulating-shifts-on-32-bytes-with-avx
	if (N < 16) {
		return {_mm256_alignr_epi8(_mm256_permute2x128_si256(A, A, _MM_SHUFFLE(2, 0, 0, 1)), A, N)};
	} else if (N == 16) {
		return {_mm256_permute2x128_si256(A, A, _MM_SHUFFLE(2, 0, 0, 1))};
	} else {
        return {_mm256_srli_si256(_mm256_permute2x128_si256(A, A, _MM_SHUFFLE(2, 0, 0, 1)), N - 16)};
    }
}
#else
template <>
really_inline SuperVector<64> SuperVector<64>::operator>>(uint8_t const N) const
{
	switch(N) {
	case 1: return {_mm256_alignr_epi8(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(2, 0, 0, 1)), u.v256[0], 1)}; break;
	case 2: return {_mm256_alignr_epi8(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(2, 0, 0, 1)), u.v256[0], 2)}; break;
	case 3: return {_mm256_alignr_epi8(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(2, 0, 0, 1)), u.v256[0], 3)}; break;
	case 4: return {_mm256_alignr_epi8(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(2, 0, 0, 1)), u.v256[0], 4)}; break;
	case 5: return {_mm256_alignr_epi8(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(2, 0, 0, 1)), u.v256[0], 5)}; break;
	case 6: return {_mm256_alignr_epi8(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(2, 0, 0, 1)), u.v256[0], 6)}; break;
	case 7: return {_mm256_alignr_epi8(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(2, 0, 0, 1)), u.v256[0], 7)}; break;
	case 8: return {_mm256_alignr_epi8(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(2, 0, 0, 1)), u.v256[0], 8)}; break;
	case 9: return {_mm256_alignr_epi8(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(2, 0, 0, 1)), u.v256[0], 9)}; break;
	case 10: return {_mm256_alignr_epi8(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(2, 0, 0, 1)), u.v256[0], 10)}; break;
	case 11: return {_mm256_alignr_epi8(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(2, 0, 0, 1)), u.v256[0], 11)}; break;
	case 12: return {_mm256_alignr_epi8(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(2, 0, 0, 1)), u.v256[0], 12)}; break;
	case 13: return {_mm256_alignr_epi8(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(2, 0, 0, 1)), u.v256[0], 13)}; break;
	case 14: return {_mm256_alignr_epi8(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(2, 0, 0, 1)), u.v256[0], 14)}; break;
	case 15: return {_mm256_alignr_epi8(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(2, 0, 0, 1)), u.v256[0], 15)}; break;
	case 16: return {_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(2, 0, 0, 1))}; break;
	case 17: return {_mm256_srli_si256(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(2, 0, 0, 1)), 1)}; break;
	case 18: return {_mm256_srli_si256(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(2, 0, 0, 1)), 2)}; break;
	case 19: return {_mm256_srli_si256(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(2, 0, 0, 1)), 3)}; break;
	case 20: return {_mm256_srli_si256(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(2, 0, 0, 1)), 4)}; break;
	case 21: return {_mm256_srli_si256(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(2, 0, 0, 1)), 5)}; break;
	case 22: return {_mm256_srli_si256(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(2, 0, 0, 1)), 6)}; break;
	case 23: return {_mm256_srli_si256(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(2, 0, 0, 1)), 7)}; break;
	case 24: return {_mm256_srli_si256(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(2, 0, 0, 1)), 8)}; break;
	case 25: return {_mm256_srli_si256(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(2, 0, 0, 1)), 9)}; break;
	case 26: return {_mm256_srli_si256(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(2, 0, 0, 1)), 10)}; break;
	case 27: return {_mm256_srli_si256(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(2, 0, 0, 1)), 11)}; break;
	case 28: return {_mm256_srli_si256(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(2, 0, 0, 1)), 12)}; break;
	case 29: return {_mm256_srli_si256(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(2, 0, 0, 1)), 13)}; break;
	case 30: return {_mm256_srli_si256(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(2, 0, 0, 1)), 14)}; break;
	case 31: return {_mm256_srli_si256(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(2, 0, 0, 1)), 15)}; break;
	case 32: return Zeroes(); break;
	default: break;
	}
	return *this;
}
#endif

#ifdef HS_OPTIMIZE
template <>
really_inline SuperVector<64> SuperVector<64>::operator<<(uint8_t const N) const
{
	// As found here: https://stackoverflow.com/questions/25248766/emulating-shifts-on-32-bytes-with-avx
    if (N < 16) {
		return {_mm256_alignr_epi8(A, _mm256_permute2x128_si256(A, A, _MM_SHUFFLE(0, 0, 2, 0)), 16 - N)};
	} else if (N == 16) {
		return {_mm256_permute2x128_si256(A, A, _MM_SHUFFLE(0, 0, 2, 0))};
	} else {
        return {_mm256_slli_si256(_mm256_permute2x128_si256(A, A, _MM_SHUFFLE(0, 0, 2, 0)), N - 16)};
    }
}
#else
template <>
really_inline SuperVector<64> SuperVector<64>::operator<<(uint8_t const N) const
{
	switch(N) {
	case 1: return {_mm256_alignr_epi8(u.v256[0], _mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(0, 0, 2, 0)), 15)}; break;
	case 2: return {_mm256_alignr_epi8(u.v256[0], _mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(0, 0, 2, 0)), 14)}; break;
	case 3: return {_mm256_alignr_epi8(u.v256[0], _mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(0, 0, 2, 0)), 13)}; break;
	case 4: return {_mm256_alignr_epi8(u.v256[0], _mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(0, 0, 2, 0)), 12)}; break;
	case 5: return {_mm256_alignr_epi8(u.v256[0], _mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(0, 0, 2, 0)), 11)}; break;
	case 6: return {_mm256_alignr_epi8(u.v256[0], _mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(0, 0, 2, 0)), 10)}; break;
	case 7: return {_mm256_alignr_epi8(u.v256[0], _mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(0, 0, 2, 0)), 9)}; break;
	case 8: return {_mm256_alignr_epi8(u.v256[0], _mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(0, 0, 2, 0)), 8)}; break;
	case 9: return {_mm256_alignr_epi8(u.v256[0], _mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(0, 0, 2, 0)), 7)}; break;
	case 10: return {_mm256_alignr_epi8(u.v256[0], _mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(0, 0, 2, 0)), 6)}; break;
	case 11: return {_mm256_alignr_epi8(u.v256[0], _mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(0, 0, 2, 0)), 5)}; break;
	case 12: return {_mm256_alignr_epi8(u.v256[0], _mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(0, 0, 2, 0)), 4)}; break;
	case 13: return {_mm256_alignr_epi8(u.v256[0], _mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(0, 0, 2, 0)), 3)}; break;
	case 14: return {_mm256_alignr_epi8(u.v256[0], _mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(0, 0, 2, 0)), 2)}; break;
	case 15: return {_mm256_alignr_epi8(u.v256[0], _mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(0, 0, 2, 0)), 1)}; break;
	case 16: return {_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(0, 0, 2, 0))}; break;
	case 17: return {_mm256_slli_si256(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(0, 0, 2, 0)), 1)}; break;
	case 18: return {_mm256_slli_si256(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(0, 0, 2, 0)), 2)}; break;
	case 19: return {_mm256_slli_si256(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(0, 0, 2, 0)), 3)}; break;
	case 20: return {_mm256_slli_si256(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(0, 0, 2, 0)), 4)}; break;
	case 21: return {_mm256_slli_si256(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(0, 0, 2, 0)), 5)}; break;
	case 22: return {_mm256_slli_si256(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(0, 0, 2, 0)), 6)}; break;
	case 23: return {_mm256_slli_si256(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(0, 0, 2, 0)), 7)}; break;
	case 24: return {_mm256_slli_si256(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(0, 0, 2, 0)), 8)}; break;
	case 25: return {_mm256_slli_si256(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(0, 0, 2, 0)), 9)}; break;
	case 26: return {_mm256_slli_si256(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(0, 0, 2, 0)), 10)}; break;
	case 27: return {_mm256_slli_si256(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(0, 0, 2, 0)), 11)}; break;
	case 28: return {_mm256_slli_si256(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(0, 0, 2, 0)), 12)}; break;
	case 29: return {_mm256_slli_si256(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(0, 0, 2, 0)), 13)}; break;
	case 30: return {_mm256_slli_si256(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(0, 0, 2, 0)), 14)}; break;
	case 31: return {_mm256_slli_si256(_mm256_permute2x128_si256(u.v256[0], u.v256[0], _MM_SHUFFLE(0, 0, 2, 0)), 15)}; break;
	case 32: return Zeroes(); break;
	default: break;
	}
	return *this;
}
#endif

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

template <>
really_inline SuperVector<64> SuperVector<64>::loadu_maskz(void const *ptr, uint8_t const len)
{
	SuperVector<64> mask = (~0UL) >> (64 - len);
	mask.print8("mask");
	SuperVector<64> v = _mm512_loadu_si512((const m512 *)ptr);
	v.print8("v");
    return mask & v;
}


template<>
really_inline SuperVector<64> SuperVector<64>::pshufb(SuperVector<64> b)
{
	return {_mm512_shuffle_epi8(u.v512[0], b.u.v512[0])};
}


#ifdef HS_OPTIMIZE
template<>
really_inline SuperVector<64> SuperVector<64>::alignr(SuperVector<64> &l, int8_t offset)
{
    return {_mm512_alignr_epi8(u.v512[0], l.u.v512[0], offset)};
}
#else
template<>
really_inline SuperVector<64> SuperVector<64>::alignr(SuperVector<64> &l, int8_t offset)
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


#ifdef HS_OPTIMIZE
template<>
really_inline SuperVector<64> SuperVector<64>::lshift64(uint8_t const N)
{
	return {_mm512_slli_epi64(u.v512[0], N)};
}
#else
template<>
really_inline SuperVector<64> SuperVector<64>::lshift64(uint8_t const N)
{
	switch(N) {
	case 0: return *this; break;
	case 1: return {_mm512_slli_epi64(u.v512[0], 1)}; break;
	case 2: return {_mm512_slli_epi64(u.v512[0], 2)}; break;
	case 3: return {_mm512_slli_epi64(u.v512[0], 3)}; break;
	case 4: return {_mm512_slli_epi64(u.v512[0], 4)}; break;
	case 5: return {_mm512_slli_epi64(u.v512[0], 5)}; break;
	case 6: return {_mm512_slli_epi64(u.v512[0], 6)}; break;
	case 7: return {_mm512_slli_epi64(u.v512[0], 7)}; break;
	case 8: return {_mm512_slli_epi64(u.v512[0], 8)}; break;
	case 9: return {_mm512_slli_epi64(u.v512[0], 9)}; break;
	case 10: return {_mm512_slli_epi64(u.v512[0], 10)}; break;
	case 11: return {_mm512_slli_epi64(u.v512[0], 11)}; break;
	case 12: return {_mm512_slli_epi64(u.v512[0], 12)}; break;
	case 13: return {_mm512_slli_epi64(u.v512[0], 13)}; break;
	case 14: return {_mm512_slli_epi64(u.v512[0], 14)}; break;
	case 15: return {_mm512_slli_epi64(u.v512[0], 15)}; break;
	default: break;
	}
	return *this;
}
#endif

#ifdef HS_OPTIMIZE
template<>
really_inline SuperVector<64> SuperVector<64>::rshift64(uint8_t const N)
{
	return {_mm512_srli_epi64(u.v512[0], N)};
}
#else
template<>
really_inline SuperVector<64> SuperVector<64>::rshift64(uint8_t const N)
{
	switch(N) {
	case 0: return *this; break;
	case 1: return {_mm512_srli_epi64(u.v512[0], 1)}; break;
	case 2: return {_mm512_srli_epi64(u.v512[0], 2)}; break;
	case 3: return {_mm512_srli_epi64(u.v512[0], 3)}; break;
	case 4: return {_mm512_srli_epi64(u.v512[0], 4)}; break;
	case 5: return {_mm512_srli_epi64(u.v512[0], 5)}; break;
	case 6: return {_mm512_srli_epi64(u.v512[0], 6)}; break;
	case 7: return {_mm512_srli_epi64(u.v512[0], 7)}; break;
	case 8: return {_mm512_srli_epi64(u.v512[0], 8)}; break;
	case 9: return {_mm512_srli_epi64(u.v512[0], 9)}; break;
	case 10: return {_mm512_srli_epi64(u.v512[0], 10)}; break;
	case 11: return {_mm512_srli_epi64(u.v512[0], 11)}; break;
	case 12: return {_mm512_srli_epi64(u.v512[0], 12)}; break;
	case 13: return {_mm512_srli_epi64(u.v512[0], 13)}; break;
	case 14: return {_mm512_srli_epi64(u.v512[0], 14)}; break;
	case 15: return {_mm512_srli_epi64(u.v512[0], 15)}; break;
	default: break;
	}
	return *this;
}
#endif


#endif // HAVE_AVX512

#endif // SIMD_IMPL_HPP
