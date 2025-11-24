const { FaissIndex } = require('../dist/index.js');
const factory = require('../demo/faiss-wasm.js');

// Helper to generate random data
function generateData(d, n) {
    const data = new Float32Array(n * d);
    for (let i = 0; i < n * d; i++) {
        data[i] = Math.random();
    }
    return data;
}

// Helper to compute L2 distance between two vectors
function l2Distance(v1, v2) {
    let sum = 0;
    for (let i = 0; i < v1.length; i++) {
        const diff = v1[i] - v2[i];
        sum += diff * diff;
    }
    return sum;
}

// Naive exact search for verification
function naiveSearch(xb, xq, d, k) {
    const nq = xq.length / d;
    const nb = xb.length / d;

    const allDistances = [];
    const allLabels = [];

    for (let i = 0; i < nq; i++) {
        const q = xq.subarray(i * d, (i + 1) * d);
        const distances = [];
        for (let j = 0; j < nb; j++) {
            const b = xb.subarray(j * d, (j + 1) * d);
            distances.push({
                id: j,
                dist: l2Distance(q, b)
            });
        }
        // Sort by distance
        distances.sort((a, b) => a.dist - b.dist);

        // Keep top k
        const topK = distances.slice(0, k);
        allDistances.push(topK.map(r => r.dist));
        allLabels.push(topK.map(r => r.id));
    }

    return { distances: allDistances, labels: allLabels };
}

async function run() {
    console.log("Loading WASM module...");
    const module = await factory();

    const d = 32;
    const nb = 1000;
    const nq = 10;
    const k = 10;

    console.log(`Generating data: d=${d}, nb=${nb}, nq=${nq}`);
    const xb = generateData(d, nb);
    const xq = generateData(d, nq);

    console.log("Building IndexFlatL2...");
    const index = new module.IndexFlatL2(d);

    index.add(xb);

    console.log(`Index ntotal: ${index.ntotal}`);
    if (index.ntotal !== nb) {
        throw new Error(`Expected ntotal=${nb}, got ${index.ntotal}`);
    }

    console.log("Searching...");
    const results = index.search(xq, k);

    console.log("Verifying results...");
    const groundTruth = naiveSearch(xb, xq, d, k);

    let n_ok = 0;
    for (let i = 0; i < nq; i++) {
        const gtLabels = groundTruth.labels[i];
        const resLabels = results.labels.slice(i * k, (i + 1) * k);

        const gtDist = groundTruth.distances[i];
        const resDist = results.distances.slice(i * k, (i + 1) * k);

        let intersection = 0;
        for (let j = 0; j < k; j++) {
            if (gtLabels.includes(resLabels[j])) {
                intersection++;
            }
        }

        for (let j = 0; j < k; j++) {
            if (Math.abs(gtDist[j] - resDist[j]) > 1e-4) {
                console.error(`Distance mismatch at query ${i}, rank ${j}: expected ${gtDist[j]}, got ${resDist[j]}`);
            }
        }

        if (intersection === k) n_ok++;
    }

    console.log(`Accuracy: ${n_ok}/${nq}`);
    if (n_ok < nq * 0.9) {
        throw new Error("Accuracy too low");
    }

    console.log("TestIndexFlat PASSED");
}

run().catch(e => {
    console.error("Test FAILED:", e);
    process.exit(1);
});
