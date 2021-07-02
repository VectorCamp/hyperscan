/*
 * Copyright (c) 2015-2017, Intel Corporation
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

/** \file
 * \brief Shufti: character class acceleration.
 *
 * Utilises the SSSE3 pshufb shuffle instruction
 */

#include "truffle.h"
#include "ue2common.h"
#include "util/arch.h"
#include "util/bitutils.h"
#include "util/unaligned.h"

#include "util/simd/types.hpp"


template <uint16_t S>
const u8 *firstMatch(const u8 *buf, typename SuperVector<S>::movemask_type z);

template <uint16_t S>
const u8 *lastMatch(const u8 *buf, typename SuperVector<S>::movemask_type z);


template <>
really_inline
const u8 *firstMatch<16>(const u8 *buf, typename SuperVector<16>::movemask_type z) {
    DEBUG_PRINTF("buf %p z %08x \n", buf, z);
    DEBUG_PRINTF("z %08x\n", z);
    if (unlikely(z != 0xffff)) {
        u32 pos = ctz32(~z & 0xffff);
        DEBUG_PRINTF("~z %08x\n", ~z);
        DEBUG_PRINTF("match @ pos %u\n", pos);
        assert(pos < 16);
        return buf + pos;
    } else {
        return NULL; // no match
    }
}

template <>
really_inline
const u8 *firstMatch<64>(const u8 *buf, typename SuperVector<64>::movemask_type z) {
    DEBUG_PRINTF("z 0x%016llx\n", z);
    if (unlikely(z != ~0ULL)) {
        u32 pos = ctz64(~z);
        DEBUG_PRINTF("match @ pos %u\n", pos);
        assert(pos < 64);
        return buf + pos;
    } else {
        return NULL; // no match
    }
}


template <>
really_inline
const u8 *lastMatch<16>(const u8 *buf, typename SuperVector<16>::movemask_type z) {
    DEBUG_PRINTF("buf %p z %08x \n", buf, z);
    DEBUG_PRINTF("z %08x\n", z);
    if (unlikely(z != 0xffff)) {
        u32 pos = clz32(~z & 0xffff);
        DEBUG_PRINTF("~z %08x\n", ~z);
        DEBUG_PRINTF("match @ pos %u\n", pos);
        assert(pos >= 16 && pos < 32);
        return buf + (31 - pos);
    } else {
        return NULL; // no match
    }
}

template <>
really_inline
const u8 *lastMatch<64>(const u8 *buf, typename SuperVector<64>::movemask_type z) {
    DEBUG_PRINTF("z 0x%016llx\n", z);
    if (unlikely(z != ~0ULL)) {
        u32 pos = clz64(~z);
        DEBUG_PRINTF("match @ pos %u\n", pos);
        assert(pos < 64);
        return buf + pos;
    } else {
        return NULL; // no match
    }
}


template <uint16_t S>
static really_inline
typename SuperVector<S>::movemask_type block(SuperVector<S> shuf_mask_lo_highclear, SuperVector<S> shuf_mask_lo_highset,
            SuperVector<S> v){

    SuperVector<S> highconst = SuperVector<S>::set1_16x8(0x80);           
    print_m128_16x8("highconst", highconst.u.v128[0]);
    
    SuperVector<S> shuf_mask_hi = SuperVector<S>::set1_2x64(0x8040201008040201);
    print_m128_2x64("shuf_mask_hi", shuf_mask_hi.u.v128[0]);
    
    SuperVector<S> shuf1 = shuf_mask_lo_highclear.pshufb(v);
    SuperVector<S> t1 = v ^ highconst;
    SuperVector<S> shuf2 = shuf_mask_lo_highset.pshufb(t1);
    SuperVector<S> t2 = highconst.opandnot(v.rshift64(4));
    SuperVector<S> shuf3 = shuf_mask_hi.pshufb(t2);
    SuperVector<S> tmp = shuf3 & (shuf1 | shuf2);

    return tmp.eqmask(SuperVector<S>::Zeroes());
}


template <uint16_t S>
static really_inline const u8 *truffleMini(SuperVector<S> shuf_mask_lo_highclear, SuperVector<S> shuf_mask_lo_highset,
                       const u8 *buf, const u8 *buf_end){
    uintptr_t len = buf_end - buf;
    assert(len < 16);
    SuperVector<S> chars = SuperVector<S>::loadu(buf); 

    u32 mask = (0xffff >> (16 - len)) ^ 0xffff;
    typename SuperVector<S>::movemask_type z = block(shuf_mask_lo_highclear, shuf_mask_lo_highset, chars);
    const u8 *rv = firstMatch<S>(buf, z | mask);

    if (rv) {
        return rv;
    } else {
        return buf_end;
    }
}


template <uint16_t S>
static really_inline
const u8 *fwdBlock(SuperVector<S> shuf_mask_lo_highclear, SuperVector<S> shuf_mask_lo_highset, SuperVector<S> v, 
                    const u8 *buf) {
    typename SuperVector<S>::movemask_type z = block(shuf_mask_lo_highclear, shuf_mask_lo_highset, v);
    DEBUG_PRINTF("z %08x\n", z);
    return firstMatch<S>(buf, z);
}


template <uint16_t S>
const u8 *truffleExecReal(SuperVector<S> shuf_mask_lo_highclear, SuperVector<S> shuf_mask_lo_highset, const u8 *buf, const u8 *buf_end) {
    assert(buf && buf_end);
    assert(buf < buf_end);
    DEBUG_PRINTF("truffle %p len %zu\n", buf, buf_end - buf);
    DEBUG_PRINTF("b %s\n", buf);

    const SuperVector<S> wide_shuf_mask_lo_highclear(shuf_mask_lo_highclear);
    const SuperVector<S> wide_shuf_mask_lo_highset(shuf_mask_lo_highset);

    const u8 *d = buf;
    const u8 *rv;

    DEBUG_PRINTF("start %p end %p \n", d, buf_end);
    assert(d < buf_end);

    if (d + S <= buf_end) {
        // peel off first part to cacheline boundary
        const u8 *d1 = ROUNDUP_PTR(d, S);
        DEBUG_PRINTF("until aligned %p \n", d1);
        if (d1 != d) {
            rv = truffleMini(shuf_mask_lo_highclear, shuf_mask_lo_highset, d, d1);
            if (rv != d1) {
                return rv;
            }
            d = d1;
        }

        size_t loops = (buf_end - d) / S;
        DEBUG_PRINTF("loops %ld \n", loops);

        for (size_t i = 0; i < loops; i++, d+= S) {
            DEBUG_PRINTF("d %p \n", d);
            const u8 *base = ROUNDUP_PTR(d, S);
            // On large packet buffers, this prefetch appears to get us about 2%.
            __builtin_prefetch(base + 256);

            SuperVector<S> chars = SuperVector<S>::load(d);
            rv = fwdBlock(wide_shuf_mask_lo_highclear, wide_shuf_mask_lo_highset, chars, d);
            if (rv) return rv;
        }
    }

     DEBUG_PRINTF("d %p e %p \n", d, buf_end);
    // finish off tail

    rv = buf_end;
    if (d != buf_end) {
        rv = truffleMini(shuf_mask_lo_highclear, shuf_mask_lo_highset, d, buf_end);
        DEBUG_PRINTF("rv %p \n", rv);
    }
    
    return rv;

}


template <uint16_t S>
static really_inline const u8 *truffleRevMini(SuperVector<S> shuf_mask_lo_highclear, SuperVector<S> shuf_mask_lo_highset,
            const u8 *buf, const u8 *buf_end){
    uintptr_t len = buf_end - buf;
    assert(len < 16);
    
    SuperVector<S> chars = SuperVector<S>::loadu(buf); 


    u32 mask = (0xffff >> (16 - len)) ^ 0xffff;
    
    typename SuperVector<S>::movemask_type z = block(shuf_mask_lo_highclear, shuf_mask_lo_highset, chars);
    const u8 *rv = lastMatch<S>(buf,z | mask);

    if (rv) {
        return rv;
    }
    return buf - 1;            
}

template <uint16_t S>
static really_inline
const u8 *revBlock(SuperVector<S> shuf_mask_lo_highclear, SuperVector<S> shuf_mask_lo_highset, SuperVector<S> v, 
                    const u8 *buf) {
    typename SuperVector<S>::movemask_type z = block(shuf_mask_lo_highclear, shuf_mask_lo_highset, v);
    DEBUG_PRINTF("z %08x\n", z);
    return lastMatch<S>(buf, z);
}


template <uint16_t S>
const u8 *rtruffleExecReal(SuperVector<S> shuf_mask_lo_highclear, SuperVector<S> shuf_mask_lo_highset, const u8 *buf, const u8 *buf_end){
    assert(buf && buf_end);
    assert(buf < buf_end);
    DEBUG_PRINTF("trufle %p len %zu\n", buf, buf_end - buf);
    DEBUG_PRINTF("b %s\n", buf);

    const SuperVector<S> wide_shuf_mask_lo_highclear(shuf_mask_lo_highclear);
    const SuperVector<S> wide_shuf_mask_lo_highset(shuf_mask_lo_highset);

    const u8 *d = buf_end;
    const u8 *rv;

    DEBUG_PRINTF("start %p end %p \n", buf, d);
    assert(d > buf);
    if (d - S >= buf) {
        // peel off first part to cacheline boundary
        const u8 *d1 = ROUNDDOWN_PTR(d, S);
        DEBUG_PRINTF("until aligned %p \n", d1);
        if (d1 != d) {
            rv = truffleRevMini(shuf_mask_lo_highclear, shuf_mask_lo_highset, d1, d);
            if (rv != d1 - 1) return rv;
            d = d1;
        }

        while (d - S >= buf) {
            d -= S;
            DEBUG_PRINTF("d %p \n", d);
            const u8 *base = ROUNDDOWN_PTR(buf, S);
            // On large packet buffers, this prefetch appears to get us about 2%.
            __builtin_prefetch(base + 256);
    
            SuperVector<S> chars = SuperVector<S>::load(d);
            rv = revBlock(wide_shuf_mask_lo_highclear, wide_shuf_mask_lo_highset, chars, d);
            if (rv) return rv;
        }

        
    }

    DEBUG_PRINTF("d %p e %p \n", buf, d);
    // finish off tail

    if (d != buf) {
        rv = truffleRevMini(shuf_mask_lo_highclear, shuf_mask_lo_highset, buf, d);
        DEBUG_PRINTF("rv %p \n", rv);
        if (rv != d - 1) return rv;
    }
    
    return buf;

}


