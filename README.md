

# Detector simulation with generative models

## Quick start
First, install required packages.
```shell
cd rla_simulation
pip3 install -r requirements.txt
```

Second, set up the environment
```shell
source env.sh
```

Download the data root file -- for instance to data folder after creating it
-- and then use the following command to run a training.
```shell
mkdir data
cd data
wget https://example.com/example/training_LARGE.root
cd ..
```

Adjust `configs/vae_1.yaml` file accordingly and then run the training as:
```shell
python3 rlasim/bin/train_1.py
```

Generate data with:
```shell
python3 rlasim/bin/train_1.py --infer
```