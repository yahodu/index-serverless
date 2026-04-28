Run on atleast 500 GB storage

git clone https://github.com/yahodu/index-serverless.git

pip install -r requirements.txt


MS1M-Retinaface Dataset
gdown "https://drive.google.com/uc?id=1JgmzL9OLTqDAZE86pBgETtSQL4USKTFy"


Antelopev2
wget https://huggingface.co/MonsterMMORPG/tools/resolve/main/antelopev2.zip


Uploaded 1M distilled data to storj

curl -L https://github.com/storj/storj/releases/latest/download/uplink_linux_amd64.zip -o uplink.zip
unzip uplink.zip
sudo install uplink /usr/local/bin/uplink
uplink setup
uplink cp /path/to/bigfile.zip sj://your-bucket/ --parallelism 16 --progress


apt-get install pv
tar --xattrs --acls -cf - distill_data_5M | pv -s $(du -sb distill_data_5M | cut -f1) > distill_data_5M.tar

pv distill_data_5M.tar | tar --xattrs --acls -xf -



watch -n 1 nvidia-smi


# Basic — runs both models, compares results
python 0_model_comparison.py --dir ./photos --query query.jpg

# Save index so you don't re-index every time
python 0_model_comparison.py --dir ./photos --query query.jpg --save-index index.npz

# Load saved index for a new query (instant)
python 0_model_comparison.py --load-index index_buffalo_l.npz --query new_query.jpg

# Only run one model
python 0_model_comparison.py --dir ./photos --query query.jpg --models buffalo_l

# Tune threshold (lower = more matches, higher = stricter)
python 0_model_comparison.py --dir ./photos --query query.jpg --threshold 0.35


distilled_25M_1M, distilled_25M_5M, distilled_62M_1M, distilled_62M_5M
