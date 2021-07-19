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

#include "util/supervector/supervector.hpp"
#include "util/match.hpp"

template <uint16_t S>
static really_inline
typename SuperVector<S>::movemask_type block(SuperVector<S> shuf_mask_lo_highclear, SuperVector<S> shuf_mask_lo_highset,
            SuperVector<S> v){

    SuperVector<S> highconst = SuperVector<S>::dup_u8(0x80);
    SuperVector<S> shuf_mask_hi = SuperVector<S>::dup_u64(0x8040201008040201);
    
    SuperVector<S> shuf1 = shuf_mask_lo_highclear.pshufb(v);
    SuperVector<S> t1 = v ^ highconst;
    SuperVector<S> shuf2 = shuf_mask_lo_highset.pshufb(t1);
    SuperVector<S> t2 = highconst.opandnot(v.rshift64(4));
    SuperVector<S> shuf3 = shuf_mask_hi.pshufb(t2);
    SuperVector<S> tmp = (shuf1 | shuf2) & shuf3;

    shuf_mask_lo_highclear.print8("shuf_mask_lo_highclear");
    shuf_mask_lo_highset.print8("shuf_mask_lo_highset");
    v.print8("v");
    highconst.print8("highconst");
    shuf_mask_hi.print8("shuf_mask_hi");
    shuf1.print8("shuf1");
    t1.print8("t1");
    shuf2.print8("shuf2");
    t2.print8("t2");
    shuf3.print8("shuf3");
    tmp.print8("tmp");
    DEBUG_PRINTF(" z: 0x%016llx\n", (u64a)tmp.eqmask(SuperVector<S>::Zeroes()));

    return tmp.eqmask(SuperVector<S>::Zeroes());
}

template <uint16_t S>
static really_inline const u8 *truffleMini(SuperVector<S> shuf_mask_lo_highclear, SuperVector<S> shuf_mask_lo_highset,
                       const u8 *buf, const u8 *buf_end){
    uintptr_t len = buf_end - buf;
    assert(len < S);

    DEBUG_PRINTF("buf %p buf_end %p \n", buf, buf_end);
    SuperVector<S> chars = SuperVector<S>::loadu_maskz(buf, len);
    chars.print8("chars");

    typename SuperVector<S>::movemask_type z = block(shuf_mask_lo_highclear, shuf_mask_lo_highset, chars);
    const u8 *rv = firstMatch<S>(buf, z);
    DEBUG_PRINTF("rv %p buf+len %p \n", rv, buf+len);

    if (rv && rv < buf+len) {
        return rv;
    }
    return buf_end;
}

template <uint16_t S>
static really_inline
const u8 *fwdBlock(SuperVector<S> shuf_mask_lo_highclear, SuperVector<S> shuf_mask_lo_highset, SuperVector<S> v, 
                    const u8 *buf) {
    typename SuperVector<S>::movemask_type z = block(shuf_mask_lo_highclear, shuf_mask_lo_highset, v);
    DEBUG_PRINTF(" z: 0x%016llx\n", (u64a)z);
    return firstMatch<S>(buf, z);
}


template <uint16_t S>
const u8 *truffleExecReal(m128 &shuf_mask_lo_highclear, m128 shuf_mask_lo_highset, const u8 *buf, const u8 *buf_end) {
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
        if (!ISALIGNED_N(d, S)) {
            // peel off first part to cacheline boundary
            const u8 *d1 = ROUNDUP_PTR(d, S);
            DEBUG_PRINTF("until aligned %p \n", d1);
            if (d1 != d) {
                rv = truffleMini(wide_shuf_mask_lo_highclear, wide_shuf_mask_lo_highset, d, d1);
                if (rv != d1) {
                    return rv;
                }
                d = d1;
            }
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
        rv = truffleMini(wide_shuf_mask_lo_highclear, wide_shuf_mask_lo_highset, d, buf_end);
        DEBUG_PRINTF("rv %p \n", rv);
    }
    
    return rv;
}


template <uint16_t S>
static really_inline const u8 *truffleRevMini(SuperVector<S> shuf_mask_lo_highclear, SuperVector<S> shuf_mask_lo_highset,
            const u8 *buf, const u8 *buf_end){
    uintptr_t len = buf_end - buf;
    DEBUG_PRINTF("buf %p len %ld\n", buf, len);
    assert(len < S);
    
    SuperVector<S> chars = SuperVector<S>::loadu_maskz(buf, len);

    typename SuperVector<S>::movemask_type z = block(shuf_mask_lo_highclear, shuf_mask_lo_highset, chars);
    const u8 *rv = lastMatch<S>(buf, z);
    DEBUG_PRINTF("rv %p buf+len %p \n", rv, buf+len);

    if (rv && rv < buf+len) {
        return rv;
    }
    return buf - 1;            
}

template <uint16_t S>
static really_inline
const u8 *revBlock(SuperVector<S> shuf_mask_lo_highclear, SuperVector<S> shuf_mask_lo_highset, SuperVector<S> v, 
                    const u8 *buf) {
    typename SuperVector<S>::movemask_type z = block(shuf_mask_lo_highclear, shuf_mask_lo_highset, v);
    DEBUG_PRINTF(" z: 0x%016llx\n", (u64a)z);
    return lastMatch<S>(buf, z);
}


template <uint16_t S>
const u8 *rtruffleExecReal(m128 shuf_mask_lo_highclear, m128 shuf_mask_lo_highset, const u8 *buf, const u8 *buf_end){
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
        if (!ISALIGNED_N(d, S)) {
            // peel off first part to cacheline boundary
            const u8 *d1 = ROUNDDOWN_PTR(d, S);
            DEBUG_PRINTF("until aligned %p \n", d1);
            if (d1 != d) {
                rv = truffleRevMini(wide_shuf_mask_lo_highclear, wide_shuf_mask_lo_highset, d1, d);
                if (rv != d1 - 1) return rv;
                d = d1;
            }
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

    DEBUG_PRINTF("tail: d %p e %p \n", buf, d);
    // finish off tail

    if (d != buf) {
        rv = truffleRevMini(wide_shuf_mask_lo_highclear, wide_shuf_mask_lo_highset, buf, d);
        DEBUG_PRINTF("rv %p \n", rv);
        if (rv) return rv;
    }
    
    return buf - 1;
}


