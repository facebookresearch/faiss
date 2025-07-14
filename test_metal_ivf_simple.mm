#include <iostream>
#include <vector>
#include <random>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/metal/MetalIndexIVFFlat.h>
#include <faiss/metal/MetalResources.h>

using namespace std;

int main() {
    cout << "Simple IVF test" << endl;
    
    const int d = 64;
    const int nb = 1000;
    const int nlist = 10;
    
    // Generate random data
    vector<float> database(nb * d);
    for (int i = 0; i < nb * d; ++i) {
        database[i] = drand48();
    }
    
    try {
        cout << "Creating quantizer..." << endl;
        faiss::IndexFlatL2 quantizer(d);
        
        cout << "Creating CPU IVF..." << endl;
        faiss::IndexIVFFlat cpu_index(&quantizer, d, nlist);
        cout << "Training CPU IVF..." << endl;
        cpu_index.train(nb, database.data());
        cout << "Adding to CPU IVF..." << endl;
        cpu_index.add(nb, database.data());
        cout << "CPU IVF created successfully" << endl;
        
        cout << "\nCreating Metal resources..." << endl;
        auto resources = faiss::metal::get_default_metal_resources();
        
        cout << "Creating Metal quantizer..." << endl;
        faiss::IndexFlatL2 metal_quantizer(d);
        
        cout << "Creating Metal IVF..." << endl;
        faiss::metal::MetalIndexIVFFlat metal_index(resources, &metal_quantizer, d, nlist);
        
        cout << "Training Metal IVF..." << endl;
        metal_index.train(nb, database.data());
        
        cout << "Adding to Metal IVF..." << endl;
        metal_index.add(nb, database.data());
        
        cout << "Metal IVF created successfully" << endl;
        
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    
    cout << "Test completed successfully!" << endl;
    return 0;
}