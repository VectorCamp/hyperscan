/*
 * Copyright (c) 2021, Arm Limited
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
 * \brief Vermicelli: AArch64 SVE implementation.
 *
 * (users should include vermicelli.h instead of this)
 */

static really_inline
uint64_t vermSearchGetOffset(svbool_t matched) {
    return svcntp_b8(svptrue_b8(), svbrkb_z(svptrue_b8(), matched));
}

static really_inline
int dvermSearchGetOffset(svbool_t matched, svbool_t matched_rot) {
    int offset = vermSearchGetOffset(matched);
    int offset_rot = vermSearchGetOffset(matched_rot) - 1;
    return (offset_rot < offset) ? offset_rot : offset;
}

static really_inline
uint64_t rdvermSearchGetSingleOffset(svbool_t matched) {
    return svcntp_b8(svptrue_b8(), svbrkb_z(svptrue_b8(), svrev_b8(matched)));
}

static really_inline
uint64_t rdvermSearchGetOffset(svbool_t matched, svbool_t matched_rot) {
    uint64_t offset = rdvermSearchGetSingleOffset(matched);
    uint64_t offset_rot = rdvermSearchGetSingleOffset(matched_rot) - 1;
    return (offset_rot < offset) ? offset_rot : offset;
}

static really_inline
const u8 *vermSearchCheckMatched(const u8 *buf, svbool_t matched) {
    if (unlikely(svptest_any(svptrue_b8(), matched))) {
        const u8 *matchPos = buf + vermSearchGetOffset(matched);
        DEBUG_PRINTF("match pos %p\n", matchPos);
        return matchPos;
    }
    return NULL;
}

static really_inline
const u8 *rvermSearchCheckMatched(const u8 *buf, svbool_t matched) {
    if (unlikely(svptest_any(svptrue_b8(), matched))) {
        const u8 *matchPos = buf + (svcntb() -
            svcntp_b8(svptrue_b8(), svbrka_z(svptrue_b8(), svrev_b8(matched))));
        DEBUG_PRINTF("match pos %p\n", matchPos);
        return matchPos;
    }
    return NULL;
}

static really_inline
const u8 *dvermSearchCheckMatched(const u8 *buf, svbool_t matched,
                                  svbool_t matched_rot, svbool_t any) {
    if (unlikely(svptest_any(svptrue_b8(), any))) {
        const u8 *matchPos = buf + dvermSearchGetOffset(matched, matched_rot);
        DEBUG_PRINTF("match pos %p\n", matchPos);
        return matchPos;
    }
    return NULL;
}

static really_inline
const u8 *rdvermSearchCheckMatched(const u8 *buf, svbool_t matched,
                                   svbool_t matched_rot, svbool_t any) {
    if (unlikely(svptest_any(svptrue_b8(), any))) {
        const u8 *matchPos = buf + (svcntb() -
                                rdvermSearchGetOffset(matched, matched_rot));
        DEBUG_PRINTF("match pos %p\n", matchPos);
        return matchPos;
    }
    return NULL;
}

static really_inline
svbool_t singleMatched(svuint8_t chars, const u8 *buf, svbool_t pg,
                       bool negate, const int64_t vnum) {
    svuint8_t vec = svld1_vnum_u8(pg, buf, vnum);
    if (negate) {
        return svnmatch(pg, vec, chars);
    } else {
        return svmatch(pg, vec, chars);
    }
}

static really_inline
svbool_t doubleMatched(svuint16_t chars, const u8 *buf, const u8 *buf_rot,
                       svbool_t pg, svbool_t pg_rot, svbool_t * const matched,
                       svbool_t * const matched_rot) {
    svuint16_t vec = svreinterpret_u16(svld1_u8(pg, buf));
    svuint16_t vec_rot = svreinterpret_u16(svld1_u8(pg_rot, buf_rot));
    *matched = svmatch(pg, vec, chars);
    *matched_rot = svmatch(pg_rot, vec_rot, chars);
    return svorr_z(svptrue_b8(), *matched, *matched_rot);
}

static really_inline
const u8 *vermSearchOnce(svuint8_t chars, const u8 *buf, const u8 *buf_end,
                         bool negate) {
    DEBUG_PRINTF("start %p end %p\n", buf, buf_end);
    assert(buf <= buf_end);
    DEBUG_PRINTF("l = %td\n", buf_end - buf);
    svbool_t pg = svwhilelt_b8_s64(0, buf_end - buf);
    svbool_t matched = singleMatched(chars, buf, pg, negate, 0);
    return vermSearchCheckMatched(buf, matched);
}

static really_inline
const u8 *vermSearchLoopBody(svuint8_t chars, const u8 *buf, bool negate) {
    DEBUG_PRINTF("start %p end %p\n", buf, buf + svcntb());
    svbool_t matched = singleMatched(chars, buf, svptrue_b8(), negate, 0);
    return vermSearchCheckMatched(buf, matched);
}

static really_inline
const u8 *vermSearchLoopBodyUnrolled(svuint8_t chars, const u8 *buf,
                                     bool negate) {
    DEBUG_PRINTF("start %p end %p\n", buf, buf + (2 * svcntb()));
    svbool_t matched0 = singleMatched(chars, buf, svptrue_b8(), negate, 0);
    svbool_t matched1 = singleMatched(chars, buf, svptrue_b8(), negate, 1);
    svbool_t any = svorr_z(svptrue_b8(), matched0, matched1);
    if (unlikely(svptest_any(svptrue_b8(), any))) {
        if (svptest_any(svptrue_b8(), matched0)) {
            return buf + vermSearchGetOffset(matched0);
        } else {
            return buf + svcntb() + vermSearchGetOffset(matched1);
        }
    }
    return NULL;
}

static really_inline
const u8 *rvermSearchOnce(svuint8_t chars, const u8 *buf, const u8 *buf_end,
                          bool negate) {
    DEBUG_PRINTF("start %p end %p\n", buf, buf_end);
    assert(buf <= buf_end);
    DEBUG_PRINTF("l = %td\n", buf_end - buf);
    svbool_t pg = svwhilelt_b8_s64(0, buf_end - buf);
    svbool_t matched = singleMatched(chars, buf, pg, negate, 0);
    return rvermSearchCheckMatched(buf, matched);
}

static really_inline
const u8 *rvermSearchLoopBody(svuint8_t chars, const u8 *buf, bool negate) {
    DEBUG_PRINTF("start %p end %p\n", buf, buf + svcntb());
    svbool_t matched = singleMatched(chars, buf, svptrue_b8(), negate, 0);
    return rvermSearchCheckMatched(buf, matched);
}

static really_inline
const u8 *dvermSearchOnce(svuint16_t chars, const u8 *buf, const u8 *buf_end) {
    DEBUG_PRINTF("start %p end %p\n", buf, buf_end);
    assert(buf < buf_end);
    DEBUG_PRINTF("l = %td\n", buf_end - buf);
    svbool_t pg = svwhilelt_b8_s64(0, buf_end - buf);
    svbool_t pg_rot = svwhilele_b8_s64(0, buf_end - buf);
    svbool_t matched, matched_rot;
    // buf - 1 won't underflow as the first position in the buffer has been
    // dealt with meaning that buf - 1 is within the buffer.
    svbool_t any = doubleMatched(chars, buf, buf - 1, pg, pg_rot,
                                 &matched, &matched_rot);
    return dvermSearchCheckMatched(buf, matched, matched_rot, any);
}

static really_inline
const u8 *dvermSearchLoopBody(svuint16_t chars, const u8 *buf) {
    DEBUG_PRINTF("start %p end %p\n", buf, buf + svcntb());
    svbool_t matched, matched_rot;
    // buf - 1 won't underflow as the first position in the buffer has been
    // dealt with meaning that buf - 1 is within the buffer.
    svbool_t any = doubleMatched(chars, buf, buf - 1, svptrue_b8(),
                                 svptrue_b8(), &matched, &matched_rot);
    return dvermSearchCheckMatched(buf, matched, matched_rot, any);
}

static really_inline
const u8 *rdvermSearchOnce(svuint16_t chars, const u8 *buf, const u8 *buf_end) {
    DEBUG_PRINTF("start %p end %p\n", buf, buf_end);
    assert(buf < buf_end);

    DEBUG_PRINTF("l = %td\n", buf_end - buf);
    // buf_end can be read as the last position in the buffer has been
    // dealt with meaning that buf_end is within the buffer.
    // buf_end needs to be read by both the buf load and the buf + 1 load,
    // this is because buf_end must be the upper 8 bits of the 16 bit element
    // to be matched.
    svbool_t pg = svwhilele_b8_s64(0, buf_end - buf);
    svbool_t pg_rot = svwhilelt_b8_s64(0, buf_end - buf);
    svbool_t matched, matched_rot;
    svbool_t any = doubleMatched(chars, buf, buf + 1, pg, pg_rot,
                                 &matched, &matched_rot);
    return rdvermSearchCheckMatched(buf, matched, matched_rot, any);
}

static really_inline
const u8 *rdvermSearchLoopBody(svuint16_t chars, const u8 *buf) {
    DEBUG_PRINTF("start %p end %p\n", buf, buf + svcntb());
    svbool_t matched, matched_rot;
    // buf + svcntb() can be read as the last position in the buffer has
    // been dealt with meaning that buf + svcntb() is within the buffer.
    svbool_t any = doubleMatched(chars, buf, buf + 1, svptrue_b8(),
                                 svptrue_b8(), &matched, &matched_rot);
    return rdvermSearchCheckMatched(buf, matched, matched_rot, any);
}

static really_inline
const u8 *vermSearch(char c, bool nocase, const u8 *buf, const u8 *buf_end,
                     bool negate) {
    assert(buf < buf_end);
    svuint8_t chars = getCharMaskSingle(c, nocase);
    size_t len = buf_end - buf;
    if (len <= svcntb()) {
        return vermSearchOnce(chars, buf, buf_end, negate);
    }
    // peel off first part to align to the vector size
    const u8 *aligned_buf = ROUNDUP_PTR(buf, svcntb_pat(SV_POW2));
    assert(aligned_buf < buf_end);
    if (buf != aligned_buf) {
        const u8 *ptr = vermSearchLoopBody(chars, buf, negate);
        if (ptr) return ptr;
    }
    buf = aligned_buf;
    uint64_t unrolled_cntb = 2 * svcntb();
    size_t unrolled_loops = (buf_end - buf) / unrolled_cntb;
    DEBUG_PRINTF("unrolled_loops %zu \n", unrolled_loops);
    for (size_t i = 0; i < unrolled_loops; i++, buf += unrolled_cntb) {
        const u8 *ptr = vermSearchLoopBodyUnrolled(chars, buf, negate);
        if (ptr) return ptr;
    }
    size_t loops = (buf_end - buf) / svcntb();
    DEBUG_PRINTF("loops %zu \n", loops);
    for (size_t i = 0; i < loops; i++, buf += svcntb()) {
        const u8 *ptr = vermSearchLoopBody(chars, buf, negate);
        if (ptr) return ptr;
    }
    DEBUG_PRINTF("buf %p buf_end %p \n", buf, buf_end);
    return buf == buf_end ? NULL : vermSearchLoopBody(chars, buf_end - svcntb(),
                                                      negate);
}

static really_inline
const u8 *rvermSearch(char c, bool nocase, const u8 *buf, const u8 *buf_end,
                      bool negate) {
    assert(buf < buf_end);
    svuint8_t chars = getCharMaskSingle(c, nocase);
    size_t len = buf_end - buf;
    if (len <= svcntb()) {
        return rvermSearchOnce(chars, buf, buf_end, negate);
    }
    // peel off first part to align to the vector size
    const u8 *aligned_buf_end = ROUNDDOWN_PTR(buf_end, svcntb_pat(SV_POW2));
    assert(buf < aligned_buf_end);
    if (buf_end != aligned_buf_end) {
        const u8 *ptr = rvermSearchLoopBody(chars, buf_end - svcntb(), negate);
        if (ptr) return ptr;
    }
    buf_end = aligned_buf_end;
    size_t loops = (buf_end - buf) / svcntb();
    DEBUG_PRINTF("loops %zu \n", loops);
    for (size_t i = 0; i < loops; i++) {
        buf_end -= svcntb();
        const u8 *ptr = rvermSearchLoopBody(chars, buf_end, negate);
        if (ptr) return ptr;
    }
    DEBUG_PRINTF("buf %p buf_end %p \n", buf, buf_end);
    return buf == buf_end ? NULL : rvermSearchLoopBody(chars, buf, negate);
}

static really_inline
const u8 *dvermSearch(char c1, char c2, bool nocase, const u8 *buf,
                      const u8 *buf_end) {
    svuint16_t chars = getCharMaskDouble(c1, c2, nocase);
    size_t len = buf_end - buf;
    if (len <= svcntb()) {
        return dvermSearchOnce(chars, buf, buf_end);
    }
    // peel off first part to align to the vector size
    const u8 *aligned_buf = ROUNDUP_PTR(buf, svcntb_pat(SV_POW2));
    assert(aligned_buf < buf_end);
    if (buf != aligned_buf) {
        const u8 *ptr = dvermSearchLoopBody(chars, buf);
        if (ptr) return ptr;
    }
    buf = aligned_buf;
    size_t loops = (buf_end - buf) / svcntb();
    DEBUG_PRINTF("loops %zu \n", loops);
    for (size_t i = 0; i < loops; i++, buf += svcntb()) {
        const u8 *ptr = dvermSearchLoopBody(chars, buf);
        if (ptr) return ptr;
    }
    DEBUG_PRINTF("buf %p buf_end %p \n", buf, buf_end);
    return buf == buf_end ? NULL : dvermSearchLoopBody(chars,
                                                       buf_end - svcntb());
}

static really_inline
const u8 *rdvermSearch(char c1, char c2, bool nocase, const u8 *buf,
                       const u8 *buf_end) {
    svuint16_t chars = getCharMaskDouble(c1, c2, nocase);
    size_t len = buf_end - buf;
    if (len <= svcntb()) {
        return rdvermSearchOnce(chars, buf, buf_end);
    }
    // peel off first part to align to the vector size
    const u8 *aligned_buf_end = ROUNDDOWN_PTR(buf_end, svcntb_pat(SV_POW2));
    assert(buf < aligned_buf_end);
    if (buf_end != aligned_buf_end) {
        const u8 *rv = rdvermSearchLoopBody(chars, buf_end - svcntb());
        if (rv) return rv;
    }
    buf_end = aligned_buf_end;
    size_t loops = (buf_end - buf) / svcntb();
    DEBUG_PRINTF("loops %zu \n", loops);
    for (size_t i = 0; i < loops; i++) {
        buf_end -= svcntb();
        const u8 *rv = rdvermSearchLoopBody(chars, buf_end);
        if (rv) return rv;
    }
    DEBUG_PRINTF("buf %p buf_end %p \n", buf, buf_end);
    return buf == buf_end ? NULL : rdvermSearchLoopBody(chars, buf);
}

static really_inline
const u8 *vermicelliExec(char c, bool nocase, const u8 *buf,
                         const u8 *buf_end) {
    DEBUG_PRINTF("verm scan %s\\x%02hhx over %td bytes\n",
                 nocase ? "nocase " : "", c, buf_end - buf);
    const u8 *ptr = vermSearch(c, nocase, buf, buf_end, false);
    return ptr ? ptr : buf_end;
}

/* like vermicelliExec except returns the address of the first character which
 * is not c */
static really_inline
const u8 *nvermicelliExec(char c, bool nocase, const u8 *buf,
                         const u8 *buf_end) {
    DEBUG_PRINTF("nverm scan %s\\x%02hhx over %td bytes\n",
                 nocase ? "nocase " : "", c, buf_end - buf);
    const u8 *ptr = vermSearch(c, nocase, buf, buf_end, true);
    return ptr ? ptr : buf_end;
}

// Reverse vermicelli scan. Provides exact semantics and returns (buf - 1) if
// character not found.
static really_inline
const u8 *rvermicelliExec(char c, bool nocase, const u8 *buf,
                          const u8 *buf_end) {
    DEBUG_PRINTF("rev verm scan %s\\x%02hhx over %td bytes\n",
                 nocase ? "nocase " : "", c, buf_end - buf);
    const u8 *ptr = rvermSearch(c, nocase, buf, buf_end, false);
    return ptr ? ptr : buf - 1;
}

/* like rvermicelliExec except returns the address of the last character which
 * is not c */
static really_inline
const u8 *rnvermicelliExec(char c, bool nocase, const u8 *buf,
                           const u8 *buf_end) {
    DEBUG_PRINTF("rev verm scan %s\\x%02hhx over %td bytes\n",
                 nocase ? "nocase " : "", c, buf_end - buf);
    const u8 *ptr = rvermSearch(c, nocase, buf, buf_end, true);
    return ptr ? ptr : buf - 1;
}

static really_inline
const u8 *vermicelliDoubleExec(char c1, char c2, bool nocase, const u8 *buf,
                               const u8 *buf_end) {
    DEBUG_PRINTF("double verm scan %s\\x%02hhx%02hhx over %td bytes\n",
                 nocase ? "nocase " : "", c1, c2, buf_end - buf);
    assert(buf < buf_end);
    if (buf_end - buf > 1) {
        ++buf;
        const u8 *ptr = dvermSearch(c1, c2, nocase, buf, buf_end);
        if (ptr) {
            return ptr;
        }
    }
    /* check for partial match at end */
    u8 mask = nocase ? CASE_CLEAR : 0xff;
    if ((buf_end[-1] & mask) == (u8)c1) {
        DEBUG_PRINTF("partial!!!\n");
        return buf_end - 1;
    }
    return buf_end;
}

/* returns highest offset of c2 (NOTE: not c1) */
static really_inline
const u8 *rvermicelliDoubleExec(char c1, char c2, bool nocase, const u8 *buf,
                                const u8 *buf_end) {
    DEBUG_PRINTF("rev double verm scan %s\\x%02hhx%02hhx over %td bytes\n",
                 nocase ? "nocase " : "", c1, c2, buf_end - buf);
    assert(buf < buf_end);
    if (buf_end - buf > 1) {
        --buf_end;
        const u8 *ptr = rdvermSearch(c1, c2, nocase, buf, buf_end);
        if (ptr) {
            return ptr;
        }
    }
    return buf - 1;
}