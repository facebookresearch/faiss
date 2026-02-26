const { FaissIndex } = require('../dist/index.js');
const factory = require('../demo/faiss-wasm.js');

async function run() {
    console.log("Loading WASM module...");
    const module = await factory();
    console.log("Module loaded.");

    const d = 128;
    const index = new module.IndexFlatL2(d);
    console.log(`Index created. d=${index.d}, ntotal=${index.ntotal}`);

    // Create some random vectors
    const n = 100;
    const vectors = [];
    for (let i = 0; i < n * d; i++) {
        vectors.push(Math.random());
    }

    console.log(`Adding ${n} vectors...`);
    index.add(vectors);
    console.log(`Added. ntotal=${index.ntotal}`);

    if (index.ntotal !== n) {
        console.error("Error: ntotal mismatch");
        process.exit(1);
    }

    // Search
    const k = 5;
    const query = vectors.slice(0, d); // First vector
    console.log("Searching for the first vector...");
    const results = index.search(query, k);

    console.log("Results:", results);

    const labels = results.labels;
    const distances = results.distances;

    console.log(`Top 1: Label ${labels[0]}, Distance ${distances[0]}`);

    if (labels[0] !== 0) {
        console.error("Error: Top result is not the query vector");
        process.exit(1);
    }

    if (Math.abs(distances[0]) > 1e-5) {
        console.error("Error: Distance to self is not 0");
        process.exit(1);
    }

    console.log("Test PASSED");
}

run().catch(e => {
    console.error("Test FAILED:", e);
    process.exit(1);
});
