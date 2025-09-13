if [ -x curl ]; then
  echo "curl not installed."
fi

if [ -x tar ]; then
  echo "tar not installed."
fi

DATA=data/
DATA1=BSD300.tgz

mkdir $DATA

if ! [ -f $DATA1 ]; then
  curl -L https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz \
    -o $DATA1
fi

tar -xvzf $DATA1 -C $DATA
