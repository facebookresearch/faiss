#include <faiss/perf_tests/utils.h>
namespace faiss::perf_tests {
std::map<std::string, faiss::ScalarQuantizer::QuantizerType> sq_types() {
    static std::map<std::string, faiss::ScalarQuantizer::QuantizerType>
            sq_types = {
                    {"QT_8bit", faiss::ScalarQuantizer::QT_8bit},
                    {"QT_4bit", faiss::ScalarQuantizer::QT_4bit},
                    {"QT_8bit_uniform",
                     faiss::ScalarQuantizer::QT_8bit_uniform},
                    {"QT_4bit_uniform",
                     faiss::ScalarQuantizer::QT_4bit_uniform},
                    {"QT_fp16", faiss::ScalarQuantizer::QT_fp16},
                    {"QT_8bit_direct", faiss::ScalarQuantizer::QT_8bit_direct},
                    {"QT_6bit", faiss::ScalarQuantizer::QT_6bit},
                    {"QT_bf16", faiss::ScalarQuantizer::QT_bf16},
                    {"QT_8bit_direct_signed",
                     faiss::ScalarQuantizer::QT_8bit_direct_signed}};
    return sq_types;
}
} // namespace faiss::perf_tests
