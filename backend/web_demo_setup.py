import os
import requests
import torch
import numpy as np

# Local imports (since we are inside backend/)
from models.biomed_clip import BioMedCLIPEncoder
from retrieval.faiss_engine import FAISSIndex

def setup_web_demo():
    print("Setting up Web-Based Demo (Lite Mode)...")
    
    # Curated public medical images for the demo
    samples = [
        {
            "url": "https://upload.wikimedia.org/wikipedia/commons/d/d1/Pneumonia_on_Chest_X-Ray.jpg",
            "label": "Pneumonia",
            "modality": "Chest X-ray",
            "findings": "Focal opacification in the right lower lobe consistent with pneumonia."
        },
        {
            "url": "https://upload.wikimedia.org/wikipedia/commons/a/a1/Normal_posteroanterior_chest_x-ray.jpg",
            "label": "Normal",
            "modality": "Chest X-ray",
            "findings": "Clear lung fields with no focal consolidation or pleural effusion."
        },
        {
            "url": "https://upload.wikimedia.org/wikipedia/commons/5/5f/T1w_MRI_of_a_human_brain_with_a_glioblastoma_multiforme.jpg",
            "label": "Brain Tumor",
            "modality": "MRI Scan",
            "findings": "Large contrast-enhancing mass in the right hemisphere with significant perilesional edema."
        },
        {
            "url": "https://upload.wikimedia.org/wikipedia/commons/4/4e/CT_scan_of_the_human_brain.jpg",
            "label": "Normal",
            "modality": "CT Scan",
            "findings": "Intracranial structures are normal. No evidence of hemorrhage or mass effect."
        },
        {
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e8/Adenocarcinoma_of_the_lung_-_histopathology.jpg/800px-Adenocarcinoma_of_the_lung_-_histopathology.jpg",
            "label": "Adenocarcinoma",
            "modality": "Pathology Slide",
            "findings": "Malignant epithelial cells forming glandular structures. High mitotic activity noted."
        },
        {
            "url": "https://upload.wikimedia.org/wikipedia/commons/d/df/Ultrasound_of_a_human_fetus.jpg",
            "label": "Normal Development",
            "modality": "Ultrasound",
            "findings": "Routine fetal ultrasound showing normal growth parameters and cardiac activity."
        },
        {
            "url": "https://upload.wikimedia.org/wikipedia/commons/0/05/Endoscopy_of_the_stomach.jpg",
            "label": "Gastritis",
            "modality": "Endoscopy",
            "findings": "Diffuse erythema of the gastric mucosa suggestive of moderate gastritis."
        }
    ]

    # Use paths relative to backend/
    os.makedirs("../datasets/demo_samples", exist_ok=True)
    encoder = BioMedCLIPEncoder()
    # Path relative to backend/
    faiss_db = FAISSIndex(index_path="../faiss_db/medical_index.faiss", metadata_path="../faiss_db/metadata.pkl")
    
    vectors = []
    metadata = []

    for i, sample in enumerate(samples):
        print(f"Processing {sample['modality']} sample ({i+1}/{len(samples)})...")
        try:
            path = f"../datasets/demo_samples/sample_{i}.jpg"
            if not os.path.exists(path):
                response = requests.get(sample['url'], timeout=10)
                with open(path, "wb") as f:
                    f.write(response.content)
            
            # Extract features
            features = encoder.extract_image_features(path)
            vectors.append(features.cpu().numpy()[0])
            metadata.append({
                "label": sample['label'],
                "modality": sample['modality'],
                "findings": sample['findings'],
                "path": path
            })
            
        except Exception as e:
            print(f"Failed to process sample {i}: {e}")

    if vectors:
        faiss_db.add_vectors(np.array(vectors), metadata)
        print(f"\nDemo setup complete! FAISS index populated with {len(vectors)} web-sourced samples.")
    else:
        print("\nSetup failed: No samples were processed.")

if __name__ == "__main__":
    setup_web_demo()
