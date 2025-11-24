export interface FaissModule {
    IndexFlatL2: new (d: number) => FaissIndexFlatL2;
}

export interface FaissIndexFlatL2 {
    add(vectors: Float32Array): void;
    search(query: Float32Array, k: number): { distances: number[], labels: number[] };
    ntotal: number;
    d: number;
    delete(): void;
}

declare var FaissModule: (options?: any) => Promise<FaissModule>;

export class FaissIndex {
    private index: FaissIndexFlatL2 | null = null;
    private module: FaissModule | null = null;

    constructor(private d: number) {}

    async init() {
        // @ts-ignore
        const factory = await FaissModule();
        this.module = factory;
        this.index = new factory.IndexFlatL2(this.d);
    }

    add(vectors: number[][]) {
        if (!this.index) throw new Error("Index not initialized");
        const flat = new Float32Array(vectors.flat());
        this.index.add(flat);
    }

    search(query: number[], k: number) {
        if (!this.index) throw new Error("Index not initialized");
        const flat = new Float32Array(query);
        return this.index.search(flat, k);
    }

    free() {
        if (this.index) {
            this.index.delete();
            this.index = null;
        }
    }
}
