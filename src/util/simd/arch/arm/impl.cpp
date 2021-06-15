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

#include "util/simd/arch/arm/types.hpp"

// 128-bit NEON implementation

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
really_inline SuperVector<16>::SuperVector<int8x16_t>(int8x16_t const o)
{
	u.v128[0] = static_cast<int32x4_t>(o);
}

template<>
template<>
really_inline SuperVector<16>::SuperVector<uint8x16_t>(uint8x16_t const o)
{
	u.v128[0] = static_cast<int32x4_t>(o);
}

template<>
template<>
really_inline SuperVector<16>::SuperVector<int8_t>(int8_t const o)
{
	u.v128[0] = vdupq_n_s8(o);
}

template<>
template<>
really_inline SuperVector<16>::SuperVector<uint8_t>(uint8_t const o)
{
	u.v128[0] = vdupq_n_u8(o);
}

template<>
template<>
really_inline SuperVector<16>::SuperVector<int16_t>(int16_t const o)
{
	u.v128[0] = vdupq_n_s16(o);
}

template<>
template<>
really_inline SuperVector<16>::SuperVector<uint16_t>(uint16_t const o)
{
	u.v128[0] = vdupq_n_u16(o);
}

template<>
template<>
really_inline SuperVector<16>::SuperVector<int32_t>(int32_t const o)
{
	u.v128[0] = vdupq_n_s32(o);
}

template<>
template<>
really_inline SuperVector<16>::SuperVector<uint32_t>(uint32_t const o)
{
	u.v128[0] = vdupq_n_u32(o);
}

template<>
template<>
really_inline SuperVector<16>::SuperVector<int64_t>(int64_t const o)
{
	u.v128[0] = vdupq_n_s64(o);
}

template<>
template<>
really_inline SuperVector<16>::SuperVector<uint64_t>(uint64_t const o)
{
	u.v128[0] = vdupq_n_u64(o);
}

// Constants
template<>
really_inline SuperVector<16> SuperVector<16>::Ones(void)
{
    return {vdupq_n_u8(0xFF)};
}

template<>
really_inline SuperVector<16> SuperVector<16>::Zeroes(void)
{
    return {vdupq_n_u8(0)};
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
    return {vandq_s8(u.v128[0], b.u.v128[0])};
}

template <>
really_inline SuperVector<16> SuperVector<16>::operator|(SuperVector<16> const b) const
{
    return {vandq_s8(u.v128[0], b.u.v128[0])};
}

template <>
really_inline SuperVector<16> SuperVector<16>::opand(SuperVector<16> const b) const
{
    return {vandq_s8(u.v128[0], b.u.v128[0])};
}

template <>
really_inline SuperVector<16> SuperVector<16>::opandnot(SuperVector<16> const b) const
{
    return {vandq_s8(u.v128[0], b.u.v128[0])};
}

template <>
really_inline SuperVector<16> SuperVector<16>::eq(SuperVector<16> const b) const
{
    return {vceqq_s8((int16x8_t)u.v128[0], (int16x8_t)b.u.v128[0])};
}

template <>
really_inline typename SuperVector<16>::movemask_type SuperVector<16>::movemask(void) const
{
    static const uint8x16_t powers{ 1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128 };

    // Compute the mask from the input
    uint64x2_t mask  = vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(vandq_u8((uint16x8_t)u.v128[0], powers))));
    uint64x2_t mask1 = (m128)vextq_s8(mask, zeroes128(), 7);
    mask = vorrq_u8(mask, mask1);

    // Get the resulting bytes
    uint16_t output;
    vst1q_lane_u16((uint16_t*)&output, (uint16x8_t)mask, 0);
    return static_cast<typename SuperVector<16>::movemask_type>(output);
}

template <>
really_inline typename SuperVector<16>::movemask_type SuperVector<16>::eqmask(SuperVector<16> const b) const
{
	return eq(b).movemask();
}

#ifndef HS_OPTIMIZE
template <>
really_inline SuperVector<16> SuperVector<16>::operator<<(uint8_t const N) const
{
	return {vshlq_n_s32(u.v128[0], N)};
}
#else
template <>
really_inline SuperVector<16> SuperVector<16>::operator<<(uint8_t const N) const
{
	switch(N) {
	case 0: return *this; break;
	case 1: return {vshlq_n_s32((int16x8_t) u.v128[0], 1)}; break;
	case 2: return {vshlq_n_s32((int16x8_t) u.v128[0], 2)}; break;
	case 3: return {vshlq_n_s32((int16x8_t) u.v128[0], 3)}; break;
	case 4: return {vshlq_n_s32((int16x8_t) u.v128[0], 4)}; break;
	case 5: return {vshlq_n_s32((int16x8_t) u.v128[0], 5)}; break;
	case 6: return {vshlq_n_s32((int16x8_t) u.v128[0], 6)}; break;
	case 7: return {vshlq_n_s32((int16x8_t) u.v128[0], 7)}; break;
	case 8: return {vshlq_n_s32((int16x8_t) u.v128[0], 8)}; break;
	case 9: return {vshlq_n_s32((int16x8_t) u.v128[0], 9)}; break;
	case 10: return {vshlq_n_s32((int16x8_t) u.v128[0], 10)}; break;
	case 11: return {vshlq_n_s32((int16x8_t) u.v128[0], 11)}; break;
	case 12: return {vshlq_n_s32((int16x8_t) u.v128[0], 12)}; break;
	case 13: return {vshlq_n_s32((int16x8_t) u.v128[0], 13)}; break;
	case 14: return {vshlq_n_s32((int16x8_t) u.v128[0], 14)}; break;
	case 15: return {vshlq_n_s32((int16x8_t) u.v128[0], 15)}; break;
	case 16: return Zeroes(); break;
	default: break;
	}
	return *this;
}
#endif

#ifdef HS_OPTIMIZE
template <>
really_inline SuperVector<16> SuperVector<16>::operator>>(uint8_t const N) const
{
	return {vshrq_n_s32(u.v128[0], N)};
}
#else
template <>
really_inline SuperVector<16> SuperVector<16>::operator>>(uint8_t const N) const
{
	switch(N) {
	case 0: return {vshrq_n_s32(u.v128[0], 0)}; break;
	case 1: return {vshrq_n_s32(u.v128[0], 1)}; break;
	case 2: return {vshrq_n_s32(u.v128[0], 2)}; break;
	case 3: return {vshrq_n_s32(u.v128[0], 3)}; break;
	case 4: return {vshrq_n_s32(u.v128[0], 4)}; break;
	case 5: return {vshrq_n_s32(u.v128[0], 5)}; break;
	case 6: return {vshrq_n_s32(u.v128[0], 6)}; break;
	case 7: return {vshrq_n_s32(u.v128[0], 7)}; break;
	case 8: return {vshrq_n_s32(u.v128[0], 8)}; break;
	case 9: return {vshrq_n_s32(u.v128[0], 9)}; break;
	case 10: return {vshrq_n_s32(u.v128[0], 10)}; break;
	case 11: return {vshrq_n_s32(u.v128[0], 11)}; break;
	case 12: return {vshrq_n_s32(u.v128[0], 12)}; break;
	case 13: return {vshrq_n_s32(u.v128[0], 13)}; break;
	case 14: return {vshrq_n_s32(u.v128[0], 14)}; break;
	case 15: return {vshrq_n_s32(u.v128[0], 15)}; break;
	default: break;
	}
	return *this;
}
#endif

template <>
really_inline SuperVector<16> SuperVector<16>::loadu(void const *ptr)
{
    return {vld1q_s32((const int32_t *)ptr)};
}

template <>
really_inline SuperVector<16> SuperVector<16>::load(void const *ptr)
{
    assert(ISALIGNED_N(ptr, alignof(SuperVector::size)));
    ptr = assume_aligned(ptr, SuperVector::size);
    return {vld1q_s32((const int32_t *)ptr)};
}

template <>
really_inline SuperVector<16> SuperVector<16>::loadu_maskz(void const *ptr, uint8_t const len)
{
    uint8_t alignment = (uintptr_t)(ptr) & 15;
    SuperVector<16> maskb = Ones() << alignment;
    SuperVector<16> maske = Ones() >> (16 -len - alignment);
    SuperVector<16> v = SuperVector<16>::loadu((const m128 *)ptr);
    return {maskb.u.v128[0] & maske.u.v128[0] & v.u.v128[0]};
}

#ifndef HS_OPTIMIZE
template<>
really_inline SuperVector<16> SuperVector<16>::alignr(SuperVector<16> r, int8_t offset)
{
    return {vextq_s8((int16x8_t)u.v128[0], (int16x8_t)r.u.v128[0], offset)};
}
#else
template<>
really_inline SuperVector<16> SuperVector<16>::alignr(SuperVector<16> l, int8_t offset)
{
	switch(offset) {
	case 0: return *this; break;
	case 1: return {vextq_s8((int16x8_t) u.v128[0], (int16x8_t) l.u.v128[0], 1)}; break;
	case 2: return {vextq_s8((int16x8_t) u.v128[0], (int16x8_t) l.u.v128[0], 2)}; break;
	case 3: return {vextq_s8((int16x8_t) u.v128[0], (int16x8_t) l.u.v128[0], 3)}; break;
	case 4: return {vextq_s8((int16x8_t) u.v128[0], (int16x8_t) l.u.v128[0], 4)}; break;
	case 5: return {vextq_s8((int16x8_t) u.v128[0], (int16x8_t) l.u.v128[0], 5)}; break;
	case 6: return {vextq_s8((int16x8_t) u.v128[0], (int16x8_t) l.u.v128[0], 6)}; break;
	case 7: return {vextq_s8((int16x8_t) u.v128[0], (int16x8_t) l.u.v128[0], 7)}; break;
	case 8: return {vextq_s8((int16x8_t) u.v128[0], (int16x8_t) l.u.v128[0], 8)}; break;
	case 9: return {vextq_s8((int16x8_t) u.v128[0], (int16x8_t) l.u.v128[0], 9)}; break;
	case 10: return {vextq_s8((int16x8_t) u.v128[0], (int16x8_t) l.u.v128[0], 10)}; break;
	case 11: return {vextq_s8((int16x8_t) u.v128[0], (int16x8_t) l.u.v128[0], 11)}; break;
	case 12: return {vextq_s8((int16x8_t) u.v128[0], (int16x8_t) l.u.v128[0], 12)}; break;
	case 13: return {vextq_s8((int16x8_t) u.v128[0], (int16x8_t) l.u.v128[0], 13)}; break;
	case 14: return {vextq_s8((int16x8_t) u.v128[0], (int16x8_t) l.u.v128[0], 14)}; break;
	case 15: return {vextq_s8((int16x8_t) u.v128[0], (int16x8_t) l.u.v128[0], 15)}; break;
	case 16: return l; break;
	default: break;
	}
	return *this;
}
#endif

template<>
really_inline SuperVector<16> SuperVector<16>::pshufb(SuperVector<16> b)
{
    /* On Intel, if bit 0x80 is set, then result is zero, otherwise which the lane it is &0xf.
       In NEON, if >=16, then the result is zero, otherwise it is that lane.
       btranslated is the version that is converted from Intel to NEON.  */
    int8x16_t btranslated = vandq_s8((int8x16_t)b.u.v128[0], vdupq_n_s8(0x8f));
    return {vqtbl1q_s8((int8x16_t)u.v128[0], (uint8x16_t)btranslated)};
}

#ifdef HS_OPTIMIZE
template<>
really_inline SuperVector<16> SuperVector<16>::lshift64(uint8_t const l)
{
	return {(m128)vshlq_n_s64(u.v128[0], l)};
}
#else
template<>
really_inline SuperVector<16> SuperVector<16>::lshift64(uint8_t const l)
{
	switch(l) {
	case 0: return {vshlq_n_s64(u.v128[0], 0)}; break;
	case 1: return {vshlq_n_s64(u.v128[0], 1)}; break;
	case 2: return {vshlq_n_s64(u.v128[0], 2)}; break;
	case 3: return {vshlq_n_s64(u.v128[0], 3)}; break;
	case 4: return {vshlq_n_s64(u.v128[0], 4)}; break;
	case 5: return {vshlq_n_s64(u.v128[0], 5)}; break;
	case 6: return {vshlq_n_s64(u.v128[0], 6)}; break;
	case 7: return {vshlq_n_s64(u.v128[0], 7)}; break;
	case 8: return {vshlq_n_s64(u.v128[0], 8)}; break;
	case 9: return {vshlq_n_s64(u.v128[0], 9)}; break;
	case 10: return {vshlq_n_s64(u.v128[0], 10)}; break;
	case 11: return {vshlq_n_s64(u.v128[0], 11)}; break;
	case 12: return {vshlq_n_s64(u.v128[0], 12)}; break;
	case 13: return {vshlq_n_s64(u.v128[0], 13)}; break;
	case 14: return {vshlq_n_s64(u.v128[0], 14)}; break;
	case 15: return {vshlq_n_s64(u.v128[0], 15)}; break;
	default: break;
	}
	return *this;
}
#endif

#ifdef HS_OPTIMIZE
template<>
really_inline SuperVector<16> SuperVector<16>::rshift64(uint8_t const l)
{
	return {(m128)vshrq_n_s64(u.v128[0], l)};
}
#else
template<>
really_inline SuperVector<16> SuperVector<16>::rshift64(uint8_t const l)
{
	switch(l) {
	case 0: return {vshrq_n_s64(u.v128[0], 0)}; break;
	case 1: return {vshrq_n_s64(u.v128[0], 1)}; break;
	case 2: return {vshrq_n_s64(u.v128[0], 2)}; break;
	case 3: return {vshrq_n_s64(u.v128[0], 3)}; break;
	case 4: return {vshrq_n_s64(u.v128[0], 4)}; break;
	case 5: return {vshrq_n_s64(u.v128[0], 5)}; break;
	case 6: return {vshrq_n_s64(u.v128[0], 6)}; break;
	case 7: return {vshrq_n_s64(u.v128[0], 7)}; break;
	case 8: return {vshrq_n_s64(u.v128[0], 8)}; break;
	case 9: return {vshrq_n_s64(u.v128[0], 9)}; break;
	case 10: return {vshrq_n_s64(u.v128[0], 10)}; break;
	case 11: return {vshrq_n_s64(u.v128[0], 11)}; break;
	case 12: return {vshrq_n_s64(u.v128[0], 12)}; break;
	case 13: return {vshrq_n_s64(u.v128[0], 13)}; break;
	case 14: return {vshrq_n_s64(u.v128[0], 14)}; break;
	case 15: return {vshrq_n_s64(u.v128[0], 15)}; break;
	default: break;
	}
	return *this;
}
#endif


#endif // SIMD_IMPL_HPP
