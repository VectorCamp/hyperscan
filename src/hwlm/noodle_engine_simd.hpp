/*
 * Copyright (c) 2017, Intel Corporation
 * Copyright (c) 2020, 2021, VectorCamp PC
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

/* SIMD engine agnostic noodle scan parts */

#include "util/simd/types.hpp"

static u8 CASEMASK[] = { 0xff, 0xdf };

static really_inline
u8 caseClear8(u8 x, bool noCase)
{
    return static_cast<u8>(x & CASEMASK[(u8)noCase]);
}

template<uint16_t S>
static really_inline SuperVector<S> getMask(u8 c, bool noCase) {
    u8 k = caseClear8(c, noCase);
    return SuperVector<S>(k);
}

template<uint16_t S>
static really_inline SuperVector<S> getCaseMask(void) {
    return SuperVector<S>(CASEMASK[1]);
}


static really_really_inline
hwlm_error_t single_zscan(const struct noodTable *n,const u8 *d, const u8 *buf,
		Z_TYPE z, size_t len, const struct cb_info *cbi) {
    while (unlikely(z)) {
        Z_TYPE pos = JOIN(findAndClearLSB_, Z_BITS)(&z);
        size_t matchPos = d - buf + pos;
        DEBUG_PRINTF("match pos %zu\n", matchPos);
        hwlmcb_rv_t rv = final(n, buf, len, n->msk_len != 1, cbi, matchPos);
        RETURN_IF_TERMINATED(rv);
    }
    return HWLM_SUCCESS;
}

static really_really_inline
hwlm_error_t double_zscan(const struct noodTable *n,const u8 *d, const u8 *buf,
		Z_TYPE z, size_t len, const struct cb_info *cbi) {
    while (unlikely(z)) {
        Z_TYPE pos = JOIN(findAndClearLSB_, Z_BITS)(&z);
        size_t matchPos = d - buf + pos - 1;
        DEBUG_PRINTF("match pos %zu\n", matchPos);
        hwlmcb_rv_t rv = final(n, buf, len, true, cbi, matchPos);
        RETURN_IF_TERMINATED(rv);
    }
    return HWLM_SUCCESS;
}

// The short scan routine. It is used both to scan data up to an
// alignment boundary if needed and to finish off data that the aligned scan
// function can't handle (due to small/unaligned chunk at end)
template<uint16_t S>
static really_inline
hwlm_error_t scanSingleUnaligned(const struct noodTable *n, const u8 *buf,
                                 SuperVector<S> caseMask, SuperVector<S> mask1,
                                 const struct cb_info *cbi, size_t len, size_t start,
                                 size_t end) {
    const u8 *d = buf + start;
    DEBUG_PRINTF("start %zu end %zu\n", start, end);
    const size_t l = end - start;
    DEBUG_PRINTF("l = %ld\n", l);
    //assert(l <= 64);
    if (!l) {
        return HWLM_SUCCESS;
    }

    typename SuperVector<S>::movemask_type mask = SINGLE_LOAD_MASK(l);
    SuperVector<S> v = SuperVector<S>::loadu(d) & caseMask;
    typename SuperVector<S>::movemask_type z = mask & mask1.eqmask(v);

    return single_zscan(n, d, buf, z, len, cbi);
}

template<uint16_t S>
static really_inline
hwlm_error_t scanDoubleUnaligned(const struct noodTable *n, const u8 *buf,
                                 SuperVector<S> caseMask, SuperVector<S> mask1, SuperVector<S> mask2, typename SuperVector<S>::movemask_type *lastz1,
                                 const struct cb_info *cbi, size_t len, size_t start, size_t end) {
    const u8 *d = buf + start;
    DEBUG_PRINTF("start %zu end %zu", start, end);
    const size_t l = end - start;
    assert(l <= S);
    if (!l) {
        return HWLM_SUCCESS;
    }
    SuperVector<S> v = SuperVector<S>::loadu(d) & caseMask;

    typename SuperVector<S>::movemask_type mask = DOUBLE_LOAD_MASK(l);
    typename SuperVector<S>::movemask_type z1 = mask1.eqmask(v);
    typename SuperVector<S>::movemask_type z2 = mask2.eqmask(v);
    typename SuperVector<S>::movemask_type z = mask & (*lastz1 | z1 << 1) & z2;
    *lastz1 = z1 >> (l -1);

    return double_zscan(n, d, buf, z, len, cbi);
}

template <uint16_t S>
static really_inline
hwlm_error_t scanSingleMain(const struct noodTable *n, const u8 *buf,
                            size_t len, size_t offset,
                            SuperVector<S> caseMask, SuperVector<S> mask1,
                            const struct cb_info *cbi) {
    size_t start = offset + n->msk_len - 1;
    size_t end = len;

    const u8 *d = buf + start;
    const u8 *e = buf + end;
    DEBUG_PRINTF("start %p end %p \n", d, e);
    assert(d < e);
    if (d + S <= e) {
        // peel off first part to cacheline boundary
        const u8 *d1 = ROUNDUP_PTR(d, S);
        DEBUG_PRINTF("until aligned %p \n", d1);
        if (scanSingleUnaligned(n, buf, caseMask, mask1, cbi, len, start, d1 - buf) == HWLM_TERMINATED) {
            return HWLM_TERMINATED;
        }
        d = d1;

        size_t loops = (end - (d - buf)) / S;
        DEBUG_PRINTF("loops %ld \n", loops);

        for (size_t i = 0; i < loops; i++, d+= S) {
            DEBUG_PRINTF("d %p \n", d);
            const u8 *base = ROUNDUP_PTR(d, 64);
            // On large packet buffers, this prefetch appears to get us about 2%.
            __builtin_prefetch(base + 256);

            SuperVector<S> v = SuperVector<S>::load(d) & caseMask;
            typename SuperVector<S>::movemask_type z = mask1.eqmask(v);

            hwlm_error_t rv = single_zscan(n, d, buf, z, len, cbi);
            RETURN_IF_TERMINATED(rv);
        }
    }

    DEBUG_PRINTF("d %p e %p \n", d, e);
    // finish off tail

    return scanSingleUnaligned(n, buf, caseMask, mask1, cbi, len, d - buf, end);
}

template <uint16_t S>
static really_inline
hwlm_error_t scanDoubleMain(const struct noodTable *n, const u8 *buf,
                            size_t len, size_t offset,
                            SuperVector<S> caseMask, SuperVector<S> mask1, SuperVector<S> mask2,
                            const struct cb_info *cbi) {
    // we stop scanning for the key-fragment when the rest of the key can't
    // possibly fit in the remaining buffer
    size_t end = len - n->key_offset + 2;

    size_t start = offset + n->msk_len - n->key_offset;

    typename SuperVector<S>::movemask_type lastz1{0};

    const u8 *d = buf + start;
    const u8 *e = buf + end;
    DEBUG_PRINTF("start %p end %p \n", d, e);
    assert(d < e);
    if (d + S <= e) {
        // peel off first part to cacheline boundary
        const u8 *d1 = ROUNDUP_PTR(d, S);
        DEBUG_PRINTF("until aligned %p \n", d1);
        if (scanDoubleUnaligned(n, buf, caseMask, mask1, mask2, &lastz1, cbi, len, start, d1 - buf) == HWLM_TERMINATED) {
            return HWLM_TERMINATED;
        }
        d = d1;

        size_t loops = (end - (d - buf)) / S;
        DEBUG_PRINTF("loops %ld \n", loops);

        for (size_t i = 0; i < loops; i++, d+= S) {
            DEBUG_PRINTF("d %p \n", d);
            const u8 *base = ROUNDUP_PTR(d, 64);
            // On large packet buffers, this prefetch appears to get us about 2%.
            __builtin_prefetch(base + 256);

            SuperVector<S> v = SuperVector<S>::load(d) & caseMask;
            typename SuperVector<S>::movemask_type z1 = mask1.eqmask(v);
            typename SuperVector<S>::movemask_type z2 = mask2.eqmask(v);
            typename SuperVector<S>::movemask_type z = (z1 << 1 | lastz1) & z2;
            lastz1 = z1 >> Z_SHIFT;

            hwlm_error_t rv = double_zscan(n, d, buf, z, len, cbi);
            RETURN_IF_TERMINATED(rv);
        }
    }

    DEBUG_PRINTF("d %p e %p \n", d, e);
    // finish off tail

    return scanDoubleUnaligned(n, buf, caseMask, mask1, mask2, &lastz1, cbi, len, d - buf, end);
}

// Single-character specialisation, used when keyLen = 1
static really_inline
hwlm_error_t scanSingle(const struct noodTable *n, const u8 *buf, size_t len,
                        size_t start, bool noCase, const struct cb_info *cbi) {
    if (!ourisalpha(n->key0)) {
        noCase = 0; // force noCase off if we don't have an alphabetic char
    }

    const SuperVector<VECTORSIZE> caseMask{noCase ? getCaseMask<VECTORSIZE>() : SuperVector<VECTORSIZE>::Ones()};
    const SuperVector<VECTORSIZE> mask1{getMask<VECTORSIZE>(n->key0, noCase)};

    return scanSingleMain(n, buf, len, start, caseMask, mask1, cbi);
}


static really_inline
hwlm_error_t scanDouble(const struct noodTable *n, const u8 *buf, size_t len,
                        size_t start, bool noCase, const struct cb_info *cbi) {

    const SuperVector<VECTORSIZE> caseMask{noCase ? getCaseMask<VECTORSIZE>() : SuperVector<VECTORSIZE>::Ones()};
    const SuperVector<VECTORSIZE> mask1{getMask<VECTORSIZE>(n->key0, noCase)};
    const SuperVector<VECTORSIZE> mask2{getMask<VECTORSIZE>(n->key1, noCase)};

    return scanDoubleMain(n, buf, len, start, caseMask, mask1, mask2, cbi);
}