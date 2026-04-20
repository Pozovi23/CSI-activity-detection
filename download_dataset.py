import kagglehub

# Download latest version
path = kagglehub.dataset_download("shuokanghuang/wimans")

print("Path to dataset files:", path)