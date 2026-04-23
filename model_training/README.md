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
