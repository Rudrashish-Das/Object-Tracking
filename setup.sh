git clone https://github.com/ifzhang/ByteTrack.git
sudo apt-get install ffmpeg libsm6 libxext6  -y
pip install protobuf==3.20.*
pip install numpy==1.23.4
pip install ultralytics --no-cache-dir
pip install -r requirements.txt --no-cache-dir
(cd ByteTrack; python3 setup.py -q develop)
pip install develop cython_bbox onemetric loguru lap thop --no-cache-dir
pip install supervision==0.1.0
