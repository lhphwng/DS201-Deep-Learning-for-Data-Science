DATA_DIR="../data"
mkdir -p "$DATA_DIR"

# download training set
curl -L https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz -o "$DATA_DIR/train-images-idx3-ubyte.gz"
curl -L https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz -o "$DATA_DIR/train-labels-idx1-ubyte.gz"

# download test set
curl -L https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz -o "$DATA_DIR/t10k-images-idx3-ubyte.gz"
curl -L https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz -o "$DATA_DIR/t10k-labels-idx1-ubyte.gz"

# unzip gz files
gzip -d "$DATA_DIR"/*.gz