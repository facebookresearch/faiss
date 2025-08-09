import numpy as np
import faiss

d = 32
np.random.seed(0)
xb = np.random.rand(2000, d).astype('float32')

# Crea index con IVF+PQ
index = faiss.index_factory(d, "IVF64,PQ16")

# Imposta metric_type a INNER_PRODUCT (1)
index.metric_type = faiss.METRIC_INNER_PRODUCT

index.train(xb)
index.add(xb)

print("CPU metric_type =", index.metric_type)  # 1 = IP, 2 = L2
