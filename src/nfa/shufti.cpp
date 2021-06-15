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

#include "shufti.h"
#include "ue2common.h"
#include "util/arch.h"
#include "util/bitutils.h"

#ifdef DEBUG
#include <ctype.h>

#define DUMP_MSK(_t)                                \
static UNUSED                                       \
void dumpMsk##_t(m##_t msk) {                       \
    u8 * mskAsU8 = (u8 *)&msk;                      \
    for (unsigned i = 0; i < sizeof(msk); i++) {    \
        u8 c = mskAsU8[i];                          \
        for (int j = 0; j < 8; j++) {               \
            if ((c >> (7-j)) & 0x1)                 \
                printf("1");                        \
            else                                    \
                printf("0");                        \
        }                                           \
        printf(" ");                                \
    }                                               \
}                                                   \
static UNUSED                                       \
void dumpMsk##_t##AsChars(m##_t msk) {              \
    u8 * mskAsU8 = (u8 *)&msk;                      \
    for (unsigned i = 0; i < sizeof(msk); i++) {    \
        u8 c = mskAsU8[i];                          \
        if (isprint(c))                             \
            printf("%c",c);                         \
        else                                        \
            printf(".");                            \
    }                                               \
}

#endif

#ifdef DEBUG
DUMP_MSK(128)
#endif



/** \brief Naive byte-by-byte implementation. */
static really_inline
const u8 *shuftiFwdSlow(const u8 *lo, const u8 *hi, const u8 *buf,
                        const u8 *buf_end) {
    assert(buf < buf_end);

    DEBUG_PRINTF("buf %p end %p \n", buf, buf_end);
    for (; buf < buf_end; ++buf) {
        u8 c = *buf;
        if (lo[c & 0xf] & hi[c >> 4]) {
            break;
        }
    }
    return buf;
}

/** \brief Naive byte-by-byte implementation. */
static really_inline
const u8 *shuftiRevSlow(const u8 *lo, const u8 *hi, const u8 *buf,
                        const u8 *buf_end) {
    assert(buf < buf_end);

    for (buf_end--; buf_end >= buf; buf_end--) {
        u8 c = *buf_end;
        if (lo[c & 0xf] & hi[c >> 4]) {
            break;
        }
    }
    return buf_end;
}

#if !defined(HAVE_SVE)
#include "shufti_simd.hpp"

const u8 *shuftiExec(m128 mask_lo, m128 mask_hi, const u8 *buf,
                      const u8 *buf_end) {
    return shuftiExecReal<VECTORSIZE>(mask_lo, mask_hi, buf, buf_end);
}

const u8 *rshuftiExec(m128 mask_lo, m128 mask_hi, const u8 *buf,
                       const u8 *buf_end) {
    return rshuftiExecReal<VECTORSIZE>(mask_lo, mask_hi, buf, buf_end);
}

const u8 *shuftiDoubleExec(m128 mask1_lo, m128 mask1_hi,
                            m128 mask2_lo, m128 mask2_hi,
                            const u8 *buf, const u8 *buf_end) {
    return shuftiDoubleExecReal<VECTORSIZE>(mask1_lo, mask1_hi, mask2_lo, mask2_hi, buf, buf_end);
}
#endif