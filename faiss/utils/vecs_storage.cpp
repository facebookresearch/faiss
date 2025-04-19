#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/vecs_storage.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <cstdio>
#include <cstdlib>

namespace faiss {

template <typename TVec, typename TLoad>
TVec* vecs_read(const char* fileName, size_t num, size_t numOffset, int* dim) {
    FILE* f = fopen(fileName, "rb");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fileName);
        perror("");
        abort();
    }

    size_t numElementsRead;
    int currentDimension;
    numElementsRead = fread(&currentDimension, sizeof(int), 1, f);
    FAISS_THROW_IF_NOT_FMT(
            numElementsRead == 1,
            "could not read vector dimension: %zu",
            numElementsRead);
    FAISS_THROW_IF_NOT_FMT(
            (currentDimension > 0 && currentDimension < 1000000),
            "unreasonable dimension: %d",
            currentDimension);
    struct stat st;
    fstat(fileno(f), &st);
    size_t fileSize = st.st_size;
    size_t rowSize = currentDimension * sizeof(TLoad) + sizeof(int);
    fseek(f, numOffset * rowSize, SEEK_SET);
    FAISS_THROW_IF_NOT_FMT(
            fileSize % rowSize == 0,
            "%s - weird file size: %zu, %zu",
            fileName,
            fileSize,
            rowSize);
    FAISS_THROW_IF_NOT_FMT(
            num <= fileSize / rowSize - numOffset,
            "%s - invalid number of vectors: %zu, %zu, %zu, %zu",
            fileName,
            num,
            fileSize,
            rowSize,
            numOffset);

    TVec* vecs = new TVec[num * currentDimension];
    *dim = currentDimension;

    numElementsRead = 0;
    TLoad buffer[currentDimension];
    for (size_t i = 0; i < num; i++) {
        numElementsRead += fread(&currentDimension, sizeof(int), 1, f);
        FAISS_THROW_IF_NOT_FMT(
                (currentDimension == *dim),
                "weird dimension: %d, %d",
                currentDimension,
                *dim);
        TVec* currentVec = vecs + i * currentDimension;

        if (sizeof(TVec) == sizeof(TLoad)) {
            numElementsRead += fread(
                    currentVec, sizeof(TLoad), (size_t)currentDimension, f);
        } else {
            numElementsRead +=
                    fread(buffer, sizeof(TLoad), (size_t)currentDimension, f);
            for (int j = 0; j < currentDimension; j++) {
                currentVec[j] = (TVec)buffer[j];
            }
        }
    }
    fclose(f);
    FAISS_THROW_IF_NOT_FMT(
            numElementsRead == num * (currentDimension + 1),
            "could not read whole file: %zu, %zu, %d",
            numElementsRead,
            num,
            currentDimension);
    return vecs;
}

float* bvecs_read(
        const char* fileName,
        size_t num,
        size_t numOffset,
        int* dim) {
    return vecs_read<float, unsigned char>(fileName, num, numOffset, dim);
}

int* ivecs_read(const char* fileName, size_t num, size_t numOffset, int* dim) {
    return vecs_read<int, int>(fileName, num, numOffset, dim);
}

float* fvecs_read(
        const char* fileName,
        size_t num,
        size_t numOffset,
        int* dim) {
    return vecs_read<float, float>(fileName, num, numOffset, dim);
}

template <typename TVec, typename TStore>
void vecs_write(const char* fileName, size_t num, int dim, TVec* vecs) {
    FILE* f = fopen(fileName, "ab");

    size_t vecsPos = 0;
    TStore buffer[dim];
    for (size_t i = 0; i < num; i++) {
        fwrite(&dim, sizeof(int), 1, f);
        if (sizeof(TVec) == sizeof(TStore)) {
            fwrite(&vecs[vecsPos], sizeof(TStore), (size_t)dim, f);
        } else {
            for (size_t j = 0; j < dim; j++) {
                buffer[j] = (TStore)vecs[vecsPos + j];
            }
            fwrite(&buffer, sizeof(TStore), (size_t)dim, f);
        }
        vecsPos += dim;
    }
    fclose(f);
} // namespace faiss

void bvecs_write(const char* fileName, size_t num, int dim, float* vecs) {
    vecs_write<float, unsigned char>(fileName, num, dim, vecs);
}

void ivecs_write(const char* fileName, size_t num, int dim, int* vecs) {
    vecs_write<int, int>(fileName, num, dim, vecs);
}

void fvecs_write(const char* fileName, size_t num, int dim, float* vecs) {
    vecs_write<float, float>(fileName, num, dim, vecs);
}

} // namespace faiss
