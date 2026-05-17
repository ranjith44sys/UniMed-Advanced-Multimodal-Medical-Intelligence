import faiss
import numpy as np
import pickle
import os

class FAISSIndex:
    def __init__(self, embedding_dim=512, index_path="../faiss_db/medical_index.faiss", metadata_path="../faiss_db/metadata.pkl"):
        self.embedding_dim = embedding_dim
        self.index_path = index_path
        self.metadata_path = metadata_path
        
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            with open(metadata_path, "rb") as f:
                self.metadata = pickle.load(f)
            print(f"Loaded FAISS index with {self.index.ntotal} vectors.")
        else:
            self.index = faiss.IndexFlatL2(embedding_dim)
            self.metadata = []
            print("Created new FAISS index.")

    def add_vectors(self, vectors, metadata_list):
        """
        Add vectors and their associated metadata (diagnosis, findings, image paths).
        """
        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors).astype('float32')
        
        self.index.add(vectors)
        self.metadata.extend(metadata_list)
        self.save()

    def search(self, query_vector, k=5, modality=None):
        if not isinstance(query_vector, np.ndarray):
            query_vector = np.array(query_vector).astype('float32')
        
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)
            
        # Search for more candidates if we are filtering
        search_k = k * 10 if modality else k
        distances, indices = self.index.search(query_vector, min(search_k, self.index.ntotal))
        
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx == -1: continue
            
            meta = self.metadata[idx]
            if modality and meta.get('modality') != modality:
                continue
                
            results.append({
                "distance": float(distances[0][i]),
                "metadata": meta
            })
            
            if len(results) >= k:
                break
                
        return results

    def save(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)
