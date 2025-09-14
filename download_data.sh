if [ -x curl ]; then
  echo "curl not installed."
fi

if [ -x tar ]; then
  echo "tar not installed."
fi

if [ -x unzip ]; then
  echo "unzip not installed."
fi

DATA=data/
DATA1=BSD300.tgz
DATA2=train400.zip
DATA2_DIR=train400

mkdir $DATA -p

if ! [ -f $DATA1 ]; then
  curl -L https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz \
    -o $DATA1
fi

tar -xzf $DATA1 -C $DATA

mkdir $DATA/$DATA2_DIR -p
unzip $DATA2 -d $DATA/$DATA2_DIR

echo "Data downloaded and extracted to $DATA"
