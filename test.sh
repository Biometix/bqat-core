docker build -t bqat-core .

docker run --rm -it \
    -v "$(pwd)"/data:/app/data \
    bqat-core
