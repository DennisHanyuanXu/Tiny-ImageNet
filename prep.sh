# Download data
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip
rm tiny-imagenet-200.zip

# Setup virtual env
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt

