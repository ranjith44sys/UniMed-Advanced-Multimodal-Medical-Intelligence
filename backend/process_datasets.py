import os
# Redirect Kaggle Cache to D drive (MUST BE BEFORE KAGGLEHUB IMPORT)
os.environ["KAGGLEHUB_CACHE_DIR"] = "d:/KaggleCache"
import kagglehub
import json
import torch
import numpy as np
from PIL import Image

# Local imports
from models.biomed_clip import BioMedCLIPEncoder
from retrieval.faiss_engine import FAISSIndex

def process_your_datasets():
    print("Initializing Multi-Modal Dataset Processing with ULTRA-LIGHTWEIGHT datasets...")
    
    # Updated mapping to use the smallest possible professional datasets
    dataset_map = {
        "Chest X-ray": "paultimothymooney/chest-xray-pneumonia",
        "MRI Scan": "awsaf49/brats2020-training-data",
        "Pathology Slide": "vbookshelf/breast-cancer-cell-segmentation",
        "Ultrasound": "mahnurrahman/echonet-dynamic",
        "Endoscopy": "nipunarora8/kvasir-dataset-classification"
    }

    encoder = BioMedCLIPEncoder()
    faiss_db = FAISSIndex(index_path="../faiss_db/medical_index.faiss", metadata_path="../faiss_db/metadata.pkl")
    
    all_vectors = []
    all_metadata = []

    for name, slug in dataset_map.items():
        print(f"\n--- Processing {name} ({slug}) ---")
        try:
            print(f"Downloading/Verifying {name}...")
            output_subdir = os.path.join("d:/KaggleCache", slug.replace("/", "_"))
            os.makedirs(output_subdir, exist_ok=True)
            path = kagglehub.dataset_download(slug, output_dir=output_subdir)
            print(f"Path: {path}")
            
            # Step 2: Sample images from the path
            image_paths = []
            for root, dirs, files in os.walk(path):
                for file in files:
                    # Look for images in subfolders
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_paths.append(os.path.join(root, file))
                        if len(image_paths) >= 10:
                            break
                if len(image_paths) >= 10:
                    break
            
            if not image_paths:
                print(f"No images found in {path}")
                continue
                
            # Step 3: Extract features and add to FAISS
            for img_path in image_paths:
                print(f"  Encoding {os.path.basename(img_path)}...")
                features = encoder.extract_image_features(img_path)
                all_vectors.append(features.cpu().numpy()[0])
                all_metadata.append({
                    "label": name,
                    "modality": name,
                    "findings": f"Clinical sample from {name} dataset.",
                    "path": img_path
                })
                
        except Exception as e:
            print(f"  Error processing {name}: {e}")

    if all_vectors:
        faiss_db.add_vectors(np.array(all_vectors), all_metadata)
        print(f"\nSUCCESS: FAISS Index populated with {len(all_vectors)} samples from ULTRA-LIGHTWEIGHT datasets.")
        print("You can now run 'python main.py' to start the analysis system.")
    else:
        print("\nFAILURE: No data was processed.")

if __name__ == "__main__":
    process_your_datasets()
