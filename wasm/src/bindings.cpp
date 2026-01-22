#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include <vector>

using namespace emscripten;

// Wrapper class to handle memory management and simplify JS interaction
class FaissIndexFlatL2 {
   public:
    FaissIndexFlatL2(int d) : index(d) {}

    void add(const val& vectors) {
        std::vector<float> v = vecFromJSArray(vectors);
        int n = v.size() / index.d;
        index.add(n, v.data());
    }

    val search(const val& query, int k) {
        std::vector<float> q = vecFromJSArray(query);
        int n = q.size() / index.d;

        std::vector<float> distances(n * k);
        std::vector<faiss::idx_t> labels(n * k);

        index.search(n, q.data(), k, distances.data(), labels.data());

        val result = val::object();
        result.set("distances", valFromStdVector(distances));
        result.set("labels", valFromStdVector(labels));
        return result;
    }

    int ntotal() const {
        return index.ntotal;
    }

    int d() const {
        return index.d;
    }

   private:
    faiss::IndexFlatL2 index;

    std::vector<float> vecFromJSArray(const val& v) {
        std::vector<float> rv;
        unsigned int length = v["length"].as<unsigned int>();
        rv.resize(length);
        emscripten::val memoryView{
                emscripten::typed_memory_view(length, rv.data())};
        memoryView.call<void>("set", v);
        return rv;
    }

    val valFromStdVector(const std::vector<float>& v) {
        val Float32Array = val::global("Float32Array");
        val result = Float32Array.new_(v.size());
        val memoryView = val(typed_memory_view(v.size(), v.data()));
        result.call<void>("set", memoryView);
        return result;
    }

    val valFromStdVector(const std::vector<faiss::idx_t>& v) {
        // faiss::idx_t is usually int64_t, but JS only supports doubles or
        // BigInt. For simplicity in this demo, we'll cast to double (safe for
        // small indices) or Int32 if possible. Let's convert to a vector of
        // doubles for JS compatibility if indices are small enough. Or better,
        // return a Float64Array or Int32Array if we assume 32-bit indices.
        // Standard FAISS uses 64-bit indices.
        // Let's return a BigInt64Array view if supported, or just copy to a JS
        // array. For now, let's copy to a JS array of numbers.
        val result = val::array();
        for (size_t i = 0; i < v.size(); ++i) {
            result.call<void>("push", (double)v[i]);
        }
        return result;
    }
};

EMSCRIPTEN_BINDINGS(faiss_module) {
    class_<FaissIndexFlatL2>("IndexFlatL2")
            .constructor<int>()
            .function("add", &FaissIndexFlatL2::add)
            .function("search", &FaissIndexFlatL2::search)
            .property("ntotal", &FaissIndexFlatL2::ntotal)
            .property("d", &FaissIndexFlatL2::d);
}
