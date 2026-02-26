# FAISS WASM

WebAssembly bindings for the FAISS library, enabling efficient similarity search directly in the browser or Node.js.

## Installation

```bash
npm install faiss-wasm
```

## Building

To build the WASM module and TypeScript bindings from source:

1.  **Prerequisites**:
    *   Emscripten SDK (emsdk) installed and active.
    *   Node.js and NPM.
    *   CMake.

2.  **Build**:
    ```bash
    npm install
    npm run build
    ```
    This command runs both the WASM compilation (using Emscripten) and the TypeScript compilation.

    *   `npm run build:wasm`: Compiles C++ to WASM.
    *   `npm run build:ts`: Compiles TypeScript to JavaScript.

## Testing

To run the automated tests (Node.js):

```bash
npm test
```

This runs:
*   `tests/smoke.test.js`: A basic sanity check.
*   `tests/index_flat.test.js`: A ported test verifying correctness against a naive JS implementation.

## Demo

A browser-based demo is available to verify functionality visually.

To run the demo:

```bash
npm run demo
```

Then open [http://localhost:8080](http://localhost:8080) in your browser.

## Usage

### Node.js

```javascript
const { FaissIndex } = require('faiss-wasm');
const FaissModuleFactory = require('faiss-wasm/demo/faiss-wasm.js');

async function main() {
    const factory = await FaissModuleFactory();
    const index = new factory.IndexFlatL2(128); // Dimension 128
    
    // Add vectors (flat array)
    const vectors = new Float32Array(100 * 128); // 100 vectors
    // ... fill vectors ...
    index.add(vectors);
    
    // Search
    const query = new Float32Array(128);
    // ... fill query ...
    const results = index.search(query, 5); // Top 5
    
    console.log(results.distances);
    console.log(results.labels);
}
main();
```

### Browser

Include the generated `faiss-wasm.js` and `faiss-wasm.wasm` in your project.

```javascript
// Load the module
FaissModule().then(module => {
    const index = new module.IndexFlatL2(128);
    // ... usage is similar to Node.js
});
```
