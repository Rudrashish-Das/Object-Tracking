git clone https://github.com/ifzhang/ByteTrack.git
pip install numpy --no-cache-dir
pip install ultralytics --no-cache-dir
pip install -r requirements.txt --no-cache-dir
(cd ByteTrack; python3 setup.py -q develop)
pip install develop cython_bbox onemetric loguru lap thop --no-cache-dir
pip install protobuf==3.20.*
pip install numpy==1.23.4
pip install supervision==0.1.0
