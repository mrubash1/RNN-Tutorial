# Recurrent Neural Networks  - A Short TensorFlow Tutorial

### Setup
Clone this repo to your local machine, and add the RNN-Tutorial directory as a system variable to your `~/.profile`. Instructions given for bash shell:

```bash
git clone https://github.com/silicon-valley-data-science/RNN-Tutorial
cd RNN-Tutorial
echo "export RNN_TUTORIAL=${PWD}" >> ~/.profile
echo "export PYTHONPATH=$RNN_TUTORIAL/src:${PYTHONPATH}" >> ~/.profile
source ~/.profile
```

Create a Conda environment (You will need to [Install Conda](https://conda.io/docs/install/quick.html) first)

```bash
conda create --name tf-rnn python=3
source activate tf-rnn
cd $RNN_TUTORIAL
pip install -r requirements.txt
```

### Install TensorFlow

If you have a NVIDIA GPU with [CUDA](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/#package-manager-installation) already installed

```bash
pip install tensorflow-gpu==1.0.1
```

If you will be running TensorFlow on CPU only (i.e. a MacBook Pro), use the following command (if you get an error the first time you run this command read below):

```bash
pip install --upgrade \
 https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.0.1-py3-none-any.whl
```

**Error note** (if you did not get an error skip this paragraph): At this point, due to how people have installed pip, we've seen people with different outcomes. If you get an error the first time, rerunning it might show that it installs without error (this is false hope). Try running with `pip install --upgrade  https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.0.1-py3-none-any.whl --force-reinstall`. The `--force-reinstall` flag tells it to reinstall the package. If that still doesn't work, please open an [issue](https://github.com/silicon-valley-data-science/RNN-Tutorial/issues), or you can try to follow the advice [here](https://www.tensorflow.org/install/install_mac).


### Run RNN training
All configurations for the RNN training script can be found in `$RNN_TUTORIAL/configs/neural_network.ini`

```bash
python $RNN_TUTORIAL/src/train_framework/tf_train_ctc.py
```

_NOTE: If you have a GPU available, the code will run faster if you set `tf_device = /gpu:0` in `configs/neural_network.ini`_


### TensorBoard configuration
To visualize your results via tensorboard:

```bash
tensorboard --logdir=$RNN_TUTORIAL/models/nn/debug_models/summary/
```

- TensorBoard can be found in your browser at [http://localhost:6006](http://localhost:6006).
- `tf.name_scope` is used to define parts of the network for visualization in TensorBoard. TensorBoard automatically finds any similarly structured network parts, such as identical fully connected layers and groups them in the graph visualization.
- Related to this are the `tf.summary.* methods` that log values of network parts, such as distributions of layer activations or error rate across epochs. These summaries are grouped within the `tf.name_scope`.
- See the official TensorFlow documentation for more details.


### Run unittests
We have included example unittests for the `tf_train_ctc.py` script

```bash
python $RNN_TUTORIAL/src/tests/train_framework/tf_train_ctc_test.py
```

### Add data
We have included example data from the [LibriVox corpus](https://librivox.org) in `data/raw/librivox/LibriSpeech/`. The data is separated into folders:
    - Train: train-clean-100-wav (5 examples)
    - Test: test-clean-wav (2 examples)
    - Dev: dev-clean-wav (2 examples)
If you would like to train a performant model, you can add additional wave and txt files to these folders, or create a new folder and update `configs/neural_network.ini` with the folder locations  


### Remove additions

We made a few additions to your `.profile` -- remove those additions if you want, or if you want to keep the system variables, add it to your `.bash_profile` by running:

```bash
echo "source ~/.profile" >> .bash_profile
```

