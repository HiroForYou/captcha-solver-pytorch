import tarfile

tar = tarfile.open("model.tar.gz", "w:gz")
for name in ["encoder.pkl", "best.bin", "inference.py"]:
    tar.add(name)
tar.close()
