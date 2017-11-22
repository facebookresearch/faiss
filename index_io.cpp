/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.

#include "index_io.h"

#include <cstdio>
#include <cstdlib>

#include <sys/mman.h>

#include "FaissAssert.h"

#include "IndexFlat.h"
#include "VectorTransform.h"
#include "IndexLSH.h"
#include "IndexPQ.h"
#include "IndexIVF.h"
#include "IndexIVFPQ.h"
#include "MetaIndexes.h"
#include "IndexScalarQuantizer.h"

/*************************************************************
 * The I/O format is the content of the class. For objects that are
 * inherited, like Index, a 4-character-code (fourcc) indicates which
 * child class this is an instance of.
 *
 * In this case, the fields of the parent class are written first,
 * then the ones for the child classes. Note that this requires
 * classes to be serialized to have a constructor without parameters,
 * so that the fields can be filled in later. The default constructor
 * should set reasonable defaults for all fields.
 *
 * The fourccs are assigned arbitrarily. When the class changed (added
 * or deprecated fields), the fourcc can be replaced. New code should
 * be able to read the old fourcc and fill in new classes.
 *
 * TODO: serialization to strings for use in Python pickle or Torch
 * serialization.
 *
 * TODO: in this file, the read functions that encouter errors may
 * leak memory.
 **************************************************************/



namespace faiss {

static uint32_t fourcc (const char sx[4]) {
    const unsigned char *x = (unsigned char*)sx;
    return x[0] | x[1] << 8 | x[2] << 16 | x[3] << 24;
}

/*************************************************************
 * I/O macros
 *
 * we use macros so that we have a line number to report in
 * abort (). This makes debugging a lot easier.
 **************************************************************/


#define WRITEANDCHECK(ptr, n) {                                 \
        size_t ret = fwrite (ptr, sizeof (* (ptr)), n, f);      \
        FAISS_THROW_IF_NOT_MSG (ret == (n), "write error");     \
    }

#define READANDCHECK(ptr, n) {                                  \
        size_t ret = fread (ptr, sizeof (* (ptr)), n, f);       \
        FAISS_THROW_IF_NOT_MSG (ret == (n), "read error");      \
    }

#define WRITE1(x) WRITEANDCHECK(&(x), 1)
#define READ1(x)  READANDCHECK(&(x), 1)

#define WRITEVECTOR(vec) {                      \
        size_t size = (vec).size ();            \
        WRITEANDCHECK (&size, 1);               \
        WRITEANDCHECK ((vec).data (), size);    \
    }

#define READVECTOR(vec) {                       \
        long size;                            \
        READANDCHECK (&size, 1);                \
        FAISS_THROW_IF_NOT (size >= 0 && size < (1L << 40));  \
        (vec).resize (size);                    \
        READANDCHECK ((vec).data (), size);     \
    }

struct ScopeFileCloser {
    FILE *f;
    ScopeFileCloser (FILE *f): f (f) {}
    ~ScopeFileCloser () {fclose (f); }
};

// Macros for read/write arrays aligned to 16 bytes in the
// file. Useful when mmapped.

#define WRITETABPAD16(tab, size_in) {                                   \
    size_t size = (size_in);                                            \
    WRITEANDCHECK (&size, 1);                                           \
    uint8_t padding[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}; \
    int idx = ftell(f) % 16;                                            \
    padding [idx] = 15 - idx;                                           \
    WRITEANDCHECK (padding + idx, 16 - idx);                            \
    WRITEANDCHECK ((tab), size);                                        \
}

#define READTABPAD16(tab, basetype, expected_size) {     \
    size_t size;                                \
    READANDCHECK (&size, 1);                    \
    FAISS_THROW_IF_NOT ((expected_size) == size);           \
    uint8_t padding[16], npad;                  \
    READ1(npad);                                \
    FAISS_THROW_IF_NOT (npad < 16);                         \
    READANDCHECK (padding, npad);               \
    (tab) = new basetype [size];       \
    READANDCHECK ((tab), size);                 \
}

// read only the array header, return its offset and skip over it
#define TABOFFSETPAD16(taboffset, basetype, expected_size) {    \
    size_t size;                                                \
    READANDCHECK (&size, 1);                                    \
    FAISS_THROW_IF_NOT ((expected_size) == size);                           \
    uint8_t padding[16], npad;                                  \
    READ1(npad);                                                \
    FAISS_THROW_IF_NOT (npad < 16);                                         \
    READANDCHECK (padding, npad);                               \
    taboffset = ftell(f);                                       \
    fseek (f, sizeof(basetype) * size, SEEK_CUR);               \
}




/*************************************************************
 * Write
 **************************************************************/

static void write_index_header (const Index *idx, FILE *f) {
    WRITE1 (idx->d);
    WRITE1 (idx->ntotal);
    Index::idx_t dummy = 1 << 20;
    WRITE1 (dummy);
    WRITE1 (dummy);
    WRITE1 (idx->is_trained);
    WRITE1 (idx->metric_type);
}



void write_VectorTransform (const VectorTransform *vt, FILE *f) {
    if (const LinearTransform * lt =
           dynamic_cast < const LinearTransform *> (vt)) {
        if (dynamic_cast<const RandomRotationMatrix *>(lt)) {
            uint32_t h = fourcc ("rrot");
            WRITE1 (h);
        } else if (const PCAMatrix * pca =
                   dynamic_cast<const PCAMatrix *>(lt)) {
            uint32_t h = fourcc ("PcAm");
            WRITE1 (h);
            WRITE1 (pca->eigen_power);
            WRITE1 (pca->random_rotation);
            WRITE1 (pca->balanced_bins);
            WRITEVECTOR (pca->mean);
            WRITEVECTOR (pca->eigenvalues);
            WRITEVECTOR (pca->PCAMat);
        } else {
            // generic LinearTransform (includes OPQ)
            uint32_t h = fourcc ("LTra");
            WRITE1 (h);
        }
        WRITE1 (lt->have_bias);
        WRITEVECTOR (lt->A);
        WRITEVECTOR (lt->b);
    } else if (const RemapDimensionsTransform *rdt =
               dynamic_cast<const RemapDimensionsTransform *>(vt)) {
        uint32_t h = fourcc ("RmDT");
        WRITE1 (h);
        WRITEVECTOR (rdt->map);
    } else if (const NormalizationTransform *nt =
               dynamic_cast<const NormalizationTransform *>(vt)) {
        uint32_t h = fourcc ("VNrm");
        WRITE1 (h);
        WRITE1 (nt->norm);
    } else {
        FAISS_THROW_MSG ("cannot serialize this");
    }
    // common fields
    WRITE1 (vt->d_in);
    WRITE1 (vt->d_out);
    WRITE1 (vt->is_trained);
}

static void write_ProductQuantizer (const ProductQuantizer *pq, FILE *f) {
    WRITE1 (pq->d);
    WRITE1 (pq->M);
    WRITE1 (pq->nbits);
    WRITEVECTOR (pq->centroids);
}

static void write_ScalarQuantizer (const ScalarQuantizer *ivsc, FILE *f) {
    WRITE1 (ivsc->qtype);
    WRITE1 (ivsc->rangestat);
    WRITE1 (ivsc->rangestat_arg);
    WRITE1 (ivsc->d);
    WRITE1 (ivsc->code_size);
    WRITEVECTOR (ivsc->trained);
}

void write_ProductQuantizer (const ProductQuantizer*pq, const char *fname) {
    FILE *f = fopen (fname, "w");
    FAISS_THROW_IF_NOT_FMT (f, "cannot open %s for writing", fname);
    ScopeFileCloser closer(f);
    write_ProductQuantizer (pq, f);
}


static void write_ivf_header (const IndexIVF * ivf, FILE *f,
                              bool include_ids = true) {
    write_index_header (ivf, f);
    WRITE1 (ivf->nlist);
    WRITE1 (ivf->nprobe);
    write_index (ivf->quantizer, f);
    if (include_ids) {
        for (size_t i = 0; i < ivf->nlist; i++)
            WRITEVECTOR (ivf->ids[i]);
    }
    WRITE1 (ivf->maintain_direct_map);
    WRITEVECTOR (ivf->direct_map);
}

void write_index (const Index *idx, FILE *f) {
    if (const IndexFlat * idxf = dynamic_cast<const IndexFlat *> (idx)) {
        uint32_t h = fourcc (
              idxf->metric_type == METRIC_INNER_PRODUCT ? "IxFI" :
              idxf->metric_type == METRIC_L2 ? "IxF2" : nullptr);
        WRITE1 (h);
        write_index_header (idx, f);
        WRITEVECTOR (idxf->xb);
    } else if(const IndexLSH * idxl = dynamic_cast<const IndexLSH *> (idx)) {
        uint32_t h = fourcc ("IxHe");
        WRITE1 (h);
        write_index_header (idx, f);
        WRITE1 (idxl->nbits);
        WRITE1 (idxl->rotate_data);
        WRITE1 (idxl->train_thresholds);
        WRITEVECTOR (idxl->thresholds);
        WRITE1 (idxl->bytes_per_vec);
        write_VectorTransform (&idxl->rrot, f);
        WRITEVECTOR (idxl->codes);
    } else if(const IndexPQ * idxp = dynamic_cast<const IndexPQ *> (idx)) {
        uint32_t h = fourcc ("IxPq");
        WRITE1 (h);
        write_index_header (idx, f);
        write_ProductQuantizer (&idxp->pq, f);
        WRITEVECTOR (idxp->codes);
        // search params -- maybe not useful to store?
        WRITE1 (idxp->search_type);
        WRITE1 (idxp->encode_signs);
        WRITE1 (idxp->polysemous_ht);
    } else if(const IndexScalarQuantizer * idxs =
              dynamic_cast<const IndexScalarQuantizer *> (idx)) {
        uint32_t h = fourcc ("IxSQ");
        WRITE1 (h);
        write_index_header (idx, f);
        write_ScalarQuantizer (&idxs->sq, f);
        WRITEVECTOR (idxs->codes);
    } else if(const IndexIVFFlat * ivfl =
              dynamic_cast<const IndexIVFFlat *> (idx)) {
        uint32_t h = fourcc ("IvFL");
        WRITE1 (h);
        write_ivf_header (ivfl, f);
        for(int i = 0; i < ivfl->nlist; i++)
            WRITEVECTOR (ivfl->codes[i]);
    } else if(const IndexIVFScalarQuantizer * ivsc =
              dynamic_cast<const IndexIVFScalarQuantizer *> (idx)) {
        uint32_t h = fourcc ("IvSQ");
        WRITE1 (h);
        write_ivf_header (ivsc, f);
        write_ScalarQuantizer (&ivsc->sq, f);
        WRITE1 (ivsc->code_size);
        for(int i = 0; i < ivsc->nlist; i++)
            WRITEVECTOR (ivsc->codes[i]);
    } else if(const IndexIVFPQ * ivpq =
              dynamic_cast<const IndexIVFPQ *> (idx)) {
        const IndexIVFPQR * ivfpqr = dynamic_cast<const IndexIVFPQR *> (idx);
        const IndexIVFPQCompact * ivfpqc =
            dynamic_cast<const IndexIVFPQCompact *> (idx);
        uint32_t h = fourcc (ivfpqr ? "IvQR" : ivfpqc ? "IvPC" : "IvPQ");
        WRITE1 (h);
        write_ivf_header (ivpq, f, !ivfpqc);
        WRITE1 (ivpq->by_residual);
        WRITE1 (ivpq->code_size);
        write_ProductQuantizer (&ivpq->pq, f);
        if (!ivfpqc) {
            for(int i = 0; i < ivpq->codes.size(); i++)
                WRITEVECTOR (ivpq->codes[i]);
        }
        if (ivfpqr) {
            write_ProductQuantizer (&ivfpqr->refine_pq, f);
            WRITEVECTOR (ivfpqr->refine_codes);
            WRITE1 (ivfpqr->k_factor);
        }
        if (ivfpqc) {
            WRITETABPAD16 (ivfpqc->limits, ivfpqc->nlist + 1);
            WRITETABPAD16 (ivfpqc->compact_ids, ivfpqc->ntotal);
            WRITETABPAD16 (ivfpqc->compact_codes,
                           ivfpqc->ntotal * ivfpqc->code_size);
        }
    } else if(const IndexPreTransform * ixpt =
              dynamic_cast<const IndexPreTransform *> (idx)) {
        uint32_t h = fourcc ("IxPT");
        WRITE1 (h);
        write_index_header (ixpt, f);
        int nt = ixpt->chain.size();
        WRITE1 (nt);
        for (int i = 0; i < nt; i++)
            write_VectorTransform (ixpt->chain[i], f);
        write_index (ixpt->index, f);
    } else if(const MultiIndexQuantizer * imiq =
              dynamic_cast<const MultiIndexQuantizer *> (idx)) {
        uint32_t h = fourcc ("Imiq");
        WRITE1 (h);
        write_index_header (imiq, f);
        write_ProductQuantizer (&imiq->pq, f);
    } else if(const IndexRefineFlat * idxrf =
              dynamic_cast<const IndexRefineFlat *> (idx)) {
        uint32_t h = fourcc ("IxRF");
        WRITE1 (h);
        write_index_header (idxrf, f);
        write_index (idxrf->base_index, f);
        write_index (&idxrf->refine_index, f);
        WRITE1 (idxrf->k_factor);
    } else if(const IndexIDMap * idxmap =
              dynamic_cast<const IndexIDMap *> (idx)) {
        uint32_t h =
            dynamic_cast<const IndexIDMap2 *> (idx) ? fourcc ("IxM2") :
            fourcc ("IxMp");
        // no need to store additional info for IndexIDMap2
        WRITE1 (h);
        write_index_header (idxmap, f);
        write_index (idxmap->index, f);
        WRITEVECTOR (idxmap->id_map);
    } else {
      FAISS_THROW_MSG ("don't know how to serialize this type of index");
    }
}

void write_index (const Index *idx, const char *fname) {
    FILE *f = fopen (fname, "w");
    FAISS_THROW_IF_NOT_FMT (f, "cannot open %s for writing", fname);
    ScopeFileCloser closer(f);
    write_index (idx, f);
}

void write_VectorTransform (const VectorTransform *vt, const char *fname) {
    FILE *f = fopen (fname, "w");
    FAISS_THROW_IF_NOT_FMT (f, "cannot open %s for writing", fname);
    ScopeFileCloser closer(f);
    write_VectorTransform (vt, f);
}

/*************************************************************
 * Read
 **************************************************************/

static void read_index_header (Index *idx, FILE *f) {
    READ1 (idx->d);
    READ1 (idx->ntotal);
    Index::idx_t dummy;
    READ1 (dummy);
    READ1 (dummy);
    READ1 (idx->is_trained);
    READ1 (idx->metric_type);
    idx->verbose = false;
}

VectorTransform* read_VectorTransform (FILE *f) {
    uint32_t h;
    READ1 (h);
    VectorTransform *vt = nullptr;

    if (h == fourcc ("rrot") || h == fourcc ("PCAm") ||
        h == fourcc ("LTra") || h == fourcc ("PcAm")) {
        LinearTransform *lt = nullptr;
        if (h == fourcc ("rrot")) {
            lt = new RandomRotationMatrix ();
        } else if (h == fourcc ("PCAm") ||
                   h == fourcc ("PcAm")) {
            PCAMatrix * pca = new PCAMatrix ();
            READ1 (pca->eigen_power);
            READ1 (pca->random_rotation);
            if (h == fourcc ("PcAm"))
                READ1 (pca->balanced_bins);
            READVECTOR (pca->mean);
            READVECTOR (pca->eigenvalues);
            READVECTOR (pca->PCAMat);
            lt = pca;
        } else if (h == fourcc ("LTra")) {
            lt = new LinearTransform ();
        }
        READ1 (lt->have_bias);
        READVECTOR (lt->A);
        READVECTOR (lt->b);
        vt = lt;
    } else if (h == fourcc ("RmDT")) {
        RemapDimensionsTransform *rdt = new RemapDimensionsTransform ();
        READVECTOR (rdt->map);
        vt = rdt;
    } else if (h == fourcc ("VNrm")) {
        NormalizationTransform *nt = new NormalizationTransform ();
        READ1 (nt->norm);
        vt = nt;
    } else {
        FAISS_THROW_MSG("fourcc not recognized");
    }
    READ1 (vt->d_in);
    READ1 (vt->d_out);
    READ1 (vt->is_trained);
    return vt;
}

static void read_ProductQuantizer (ProductQuantizer *pq, FILE *f) {
    READ1 (pq->d);
    READ1 (pq->M);
    READ1 (pq->nbits);
    pq->set_derived_values ();
    READVECTOR (pq->centroids);
}

static void read_ScalarQuantizer (ScalarQuantizer *ivsc, FILE *f) {
    READ1 (ivsc->qtype);
    READ1 (ivsc->rangestat);
    READ1 (ivsc->rangestat_arg);
    READ1 (ivsc->d);
    READ1 (ivsc->code_size);
    READVECTOR (ivsc->trained);
}


ProductQuantizer * read_ProductQuantizer (const char*fname) {
    FILE *f = fopen (fname, "r");
    FAISS_THROW_IF_NOT_FMT (f, "cannot open %s for writing", fname);
    ScopeFileCloser closer(f);
    ProductQuantizer *pq = new ProductQuantizer();
    ScopeDeleter1<ProductQuantizer> del (pq);
    read_ProductQuantizer(pq, f);
    del.release ();
    return pq;
}

static void read_ivf_header (IndexIVF * ivf, FILE *f,
                             bool include_ids = true) {
    read_index_header (ivf, f);
    READ1 (ivf->nlist);
    READ1 (ivf->nprobe);
    ivf->quantizer = read_index (f);
    ivf->own_fields = true;
    if (include_ids) {
        ivf->ids.resize (ivf->nlist);
        for (size_t i = 0; i < ivf->nlist; i++)
            READVECTOR (ivf->ids[i]);
    }
    READ1 (ivf->maintain_direct_map);
    READVECTOR (ivf->direct_map);
}

static IndexIVFPQ *read_ivfpq (FILE *f, uint32_t h, bool try_mmap)
{

    IndexIVFPQR *ivfpqr =
        h == fourcc ("IvQR") ? new IndexIVFPQR () : nullptr;
    IndexIVFPQCompact *ivfpqc =
        h == fourcc ("IvPC") ? new IndexIVFPQCompact () : nullptr;
    IndexIVFPQ * ivpq = ivfpqr ? ivfpqr : ivfpqc ? ivfpqc : new IndexIVFPQ ();
    read_ivf_header (ivpq, f, !ivfpqc);
    READ1 (ivpq->by_residual);
    READ1 (ivpq->code_size);
    read_ProductQuantizer (&ivpq->pq, f);
    if (!ivfpqc) {
        ivpq->codes.resize (ivpq->nlist);
        for (size_t i = 0; i < ivpq->nlist; i++)
            READVECTOR (ivpq->codes[i]);
    }
    // precomputed table not stored. It is cheaper to recompute it
    ivpq->use_precomputed_table = 0;
    if (ivpq->by_residual)
        ivpq->precompute_table ();
    if (ivfpqr) {
        read_ProductQuantizer (&ivfpqr->refine_pq, f);
        READVECTOR (ivfpqr->refine_codes);
        READ1 (ivfpqr->k_factor);
    }
    if (ivfpqc) {
        if (!try_mmap) {
            READTABPAD16 (ivfpqc->limits, uint32_t, ivfpqc->nlist + 1);
            READTABPAD16 (ivfpqc->compact_ids, uint32_t, ivfpqc->ntotal);
            READTABPAD16 (ivfpqc->compact_codes, uint8_t,
                          ivfpqc->ntotal * ivfpqc->code_size);
        } else {
            long offset_limits, offset_compact_ids, offset_compact_codes;
            TABOFFSETPAD16 (offset_limits, uint32_t, ivfpqc->nlist + 1);
            TABOFFSETPAD16 (offset_compact_ids, uint32_t, ivfpqc->ntotal);
            TABOFFSETPAD16 (offset_compact_codes, uint8_t,
                            ivfpqc->ntotal * ivfpqc->code_size);
            ivfpqc->mmap_length = ftell (f);
            // mmap the whole file
            ivfpqc->mmap_buffer = (char*)mmap (
                    nullptr, ivfpqc->mmap_length,
                    PROT_READ, MAP_SHARED, fileno (f), 0);
            if (!ivfpqc->mmap_buffer) {
                perror ("mmap failed");
                abort ();
            }
            // at this point the file can be closed, it does not
            // invalidate the mapping
            ivfpqc->limits = (uint32_t*)(ivfpqc->mmap_buffer + offset_limits);
            ivfpqc->compact_ids = (uint32_t*)(ivfpqc->mmap_buffer +
                                              offset_compact_ids);
            ivfpqc->compact_codes = (uint8_t*)(ivfpqc->mmap_buffer +
                                               offset_compact_codes);
        }
    }
    return ivpq;
}

int read_old_fmt_hack = 0;

Index *read_index (FILE * f, bool try_mmap) {
    Index * idx = nullptr;
    uint32_t h;
    READ1 (h);
    if (h == fourcc ("IxFI") || h == fourcc ("IxF2")) {
        IndexFlat *idxf;
        if (h == fourcc ("IxFI")) idxf = new IndexFlatIP ();
        else                      idxf = new IndexFlatL2 ();
        read_index_header (idxf, f);
        READVECTOR (idxf->xb);
        FAISS_THROW_IF_NOT (idxf->xb.size() == idxf->ntotal * idxf->d);
        // leak!
        idx = idxf;
    } else if (h == fourcc("IxHE") || h == fourcc("IxHe")) {
        IndexLSH * idxl = new IndexLSH ();
        read_index_header (idxl, f);
        READ1 (idxl->nbits);
        READ1 (idxl->rotate_data);
        READ1 (idxl->train_thresholds);
        READVECTOR (idxl->thresholds);
        READ1 (idxl->bytes_per_vec);
        if (h == fourcc("IxHE")) {
            FAISS_THROW_IF_NOT_FMT (idxl->nbits % 64 == 0,
                            "can only read old format IndexLSH with "
                            "nbits multiple of 64 (got %d)",
                            (int) idxl->nbits);
            // leak
            idxl->bytes_per_vec *= 8;
        }
        {
            RandomRotationMatrix *rrot = dynamic_cast<RandomRotationMatrix *>
                (read_VectorTransform (f));
            FAISS_THROW_IF_NOT_MSG(rrot, "expected a random rotation");
            idxl->rrot = *rrot;
            delete rrot;
        }
        READVECTOR (idxl->codes);
        FAISS_THROW_IF_NOT (idxl->rrot.d_in == idxl->d &&
                      idxl->rrot.d_out == idxl->nbits);
        FAISS_THROW_IF_NOT (
               idxl->codes.size() == idxl->ntotal * idxl->bytes_per_vec);
        idx = idxl;
    } else if (h == fourcc ("IxPQ") || h == fourcc ("IxPo") ||
               h == fourcc ("IxPq")) {
        // IxPQ and IxPo were merged into the same IndexPQ object
        IndexPQ * idxp =new IndexPQ ();
        read_index_header (idxp, f);
        read_ProductQuantizer (&idxp->pq, f);
        READVECTOR (idxp->codes);
        if (h == fourcc ("IxPo") || h == fourcc ("IxPq")) {
            READ1 (idxp->search_type);
            READ1 (idxp->encode_signs);
            READ1 (idxp->polysemous_ht);
        }
        // Old versoins of PQ all had metric_type set to INNER_PRODUCT
        // when they were in fact using L2. Therefore, we force metric type
        // to L2 when the old format is detected
        if (h == fourcc ("IxPQ") || h == fourcc ("IxPo")) {
            idxp->metric_type = METRIC_L2;
        }
        idx = idxp;
    } else if (h == fourcc ("IvFl") || h == fourcc("IvFL")) {
        IndexIVFFlat * ivfl = new IndexIVFFlat ();
        read_ivf_header (ivfl, f);
        ivfl->code_size = ivfl->d * sizeof(float);
        ivfl->codes.resize (ivfl->nlist);
        if (h == fourcc ("IvFL")) {
            for (size_t i = 0; i < ivfl->nlist; i++) {
                READVECTOR (ivfl->codes[i]);
            }
        } else { // old format
            for (size_t i = 0; i < ivfl->nlist; i++) {
                std::vector<float> vec;
                READVECTOR (vec);
                ivfl->codes[i].resize(vec.size() * sizeof(float));
                memcpy(ivfl->codes[i].data(), vec.data(),
                       ivfl->codes[i].size());
            }
        }
        idx = ivfl;
    } else if (h == fourcc ("IxSQ")) {
        IndexScalarQuantizer * idxs = new IndexScalarQuantizer ();
        read_index_header (idxs, f);
        read_ScalarQuantizer (&idxs->sq, f);
        READVECTOR (idxs->codes);
        idxs->code_size = idxs->sq.code_size;
        idx = idxs;
    } else if(h == fourcc ("IvSQ")) {
        IndexIVFScalarQuantizer * ivsc = new IndexIVFScalarQuantizer();
        read_ivf_header (ivsc, f);
        ivsc->codes.resize(ivsc->nlist);
        read_ScalarQuantizer (&ivsc->sq, f);
        READ1 (ivsc->code_size);
        for(int i = 0; i < ivsc->nlist; i++)
            READVECTOR (ivsc->codes[i]);
        idx = ivsc;
    } else if(h == fourcc ("IvPQ") || h == fourcc ("IvQR") ||
              h == fourcc ("IvPC")) {

        idx = read_ivfpq (f, h, try_mmap);

    } else if(h == fourcc ("IxPT")) {
        IndexPreTransform * ixpt = new IndexPreTransform();
        ixpt->own_fields = true;
        read_index_header (ixpt, f);
        int nt;
        if (read_old_fmt_hack == 2) {
            nt = 1;
        } else {
            READ1 (nt);
        }
        for (int i = 0; i < nt; i++) {
            ixpt->chain.push_back (read_VectorTransform (f));
        }
        ixpt->index = read_index (f);
        idx = ixpt;
    } else if(h == fourcc ("Imiq")) {
        MultiIndexQuantizer * imiq = new MultiIndexQuantizer ();
        read_index_header (imiq, f);
        read_ProductQuantizer (&imiq->pq, f);
        idx = imiq;
    } else if(h == fourcc ("IxRF")) {
        IndexRefineFlat *idxrf = new IndexRefineFlat ();
        read_index_header (idxrf, f);
        idxrf->base_index = read_index(f);
        idxrf->own_fields = true;
        IndexFlat *rf = dynamic_cast<IndexFlat*> (read_index (f));
        std::swap (*rf, idxrf->refine_index);
        delete rf;
        READ1 (idxrf->k_factor);
        idx = idxrf;
    } else if(h == fourcc ("IxMp") || h == fourcc ("IxM2")) {
        bool is_map2 = h == fourcc ("IxM2");
        IndexIDMap * idxmap = is_map2 ? new IndexIDMap2 () : new IndexIDMap ();
        read_index_header (idxmap, f);
        idxmap->index = read_index (f);
        idxmap->own_fields = true;
        READVECTOR (idxmap->id_map);
        if (is_map2) {
            static_cast<IndexIDMap2*>(idxmap)->construct_rev_map ();
        }
        idx = idxmap;
    } else {
        FAISS_THROW_FMT("Index type 0x%08x not supported\n", h);
        idx = nullptr;
    }
    return idx;
}



Index *read_index (const char *fname, bool try_mmap) {
    FILE *f = fopen (fname, "r");
    FAISS_THROW_IF_NOT_FMT (f, "cannot open %s for reading:", fname);
    Index *idx = read_index (f, try_mmap);
    fclose (f);
    return idx;
}

VectorTransform *read_VectorTransform (const char *fname) {
    FILE *f = fopen (fname, "r");
    if (!f) {
        fprintf (stderr, "cannot open %s for reading:", fname);
        perror ("");
        abort ();
    }
    VectorTransform *vt = read_VectorTransform (f);
    fclose (f);
    return vt;
}

/*************************************************************
 * cloning functions
 **************************************************************/



Index * clone_index (const Index *index)
{
    Cloner cl;
    return cl.clone_Index (index);
}

// assumes there is a copy constructor ready. Always try from most
// specific to most general
#define TRYCLONE(classname, obj) \
    if (const classname *clo = dynamic_cast<const classname *>(obj)) { \
        return new classname(*clo); \
    } else

VectorTransform *Cloner::clone_VectorTransform (const VectorTransform *vt)
{
    TRYCLONE (RemapDimensionsTransform, vt)
    TRYCLONE (OPQMatrix, vt)
    TRYCLONE (PCAMatrix, vt)
    TRYCLONE (RandomRotationMatrix, vt)
    TRYCLONE (LinearTransform, vt)
    {
      FAISS_THROW_MSG("clone not supported for this type of VectorTransform");
    }
    return nullptr;
}

IndexIVF * Cloner::clone_IndexIVF (const IndexIVF *ivf)
{
    TRYCLONE (IndexIVFPQR, ivf)
    TRYCLONE (IndexIVFPQ, ivf)
    TRYCLONE (IndexIVFFlat, ivf)
    TRYCLONE (IndexIVFScalarQuantizer, ivf)
    {
      FAISS_THROW_MSG("clone not supported for this type of IndexIVF");
    }
    return nullptr;
}

Index *Cloner::clone_Index (const Index *index)
{
    TRYCLONE (IndexPQ, index)
    TRYCLONE (IndexLSH, index)
    TRYCLONE (IndexFlatL2, index)
    TRYCLONE (IndexFlatIP, index)
    TRYCLONE (IndexFlat, index)
    TRYCLONE (IndexScalarQuantizer, index)
    TRYCLONE (MultiIndexQuantizer, index)
    if (const IndexIVF * ivf = dynamic_cast<const IndexIVF*>(index)) {
        IndexIVF *res = clone_IndexIVF (ivf);
        res->own_fields = true;
        res->quantizer = clone_Index (ivf->quantizer);
        return res;
    } else if (const IndexPreTransform * ipt =
               dynamic_cast<const IndexPreTransform*> (index)) {
        IndexPreTransform *res = new IndexPreTransform ();
        res->d = ipt->d;
        res->index = clone_Index (ipt->index);
        for (int i = 0; i < ipt->chain.size(); i++)
            res->chain.push_back (clone_VectorTransform (ipt->chain[i]));
        res->own_fields = true;
        return res;
    } else {
        FAISS_THROW_MSG( "clone not supported for this type of Index");
    }
    return nullptr;
}


} // namespace faiss
