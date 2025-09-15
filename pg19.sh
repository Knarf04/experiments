PG19_PATH=/gpfs/hshen/dataset/pg19
mkdir -p PG19_PATH
git clone https://huggingface.co/datasets/deepmind/pg19.git $PG19_PATH
head -1 $PG19_PATH/data/train_files.txt > $PG19_PATH/data/train_files.txt