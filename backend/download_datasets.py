import kagglehub
import os
import json

def download_and_register():
    """
    Downloads ULTRA-LIGHTWEIGHT datasets (sub-100MB for Path/Endo).
    """
    datasets = {
        "Radiology_Xray": "paultimothymooney/chest-xray-pneumonia", # 1.15GB
        "RSNA_Pneumonia": "iamtapendu/rsna-pneumonia-processed-dataset", # 1GB
        "BraTS_MRI": "awsaf49/brats2020-training-data", # 1.5GB
        "Camelyon_Pathology": "vbookshelf/breast-cancer-cell-segmentation", # 20MB
        "EchoNet_Ultrasound": "mahnurrahman/echonet-dynamic",
        "Kvasir_Endoscopy": "debeshjha1/kvasirseg" # 50MB
    }
    
    paths = {}
    print("Accessing Ultra-Lightweight Kaggle Datasets...")
    
    for name, slug in datasets.items():
        print(f"Checking {name} ({slug})...")
        try:
            path = kagglehub.dataset_download(slug)
            paths[name] = path
            print(f"Verified: {path}")
        except Exception as e:
            print(f"Error: {e}")

    config_path = "../datasets/dataset_config.json"
    os.makedirs("../datasets", exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(paths, f, indent=4)
        
    print(f"\nConfiguration saved to {config_path}")

if __name__ == "__main__":
    download_and_register()
