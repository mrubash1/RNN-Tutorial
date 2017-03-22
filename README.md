# Recurrent Neural Networks  - A Short TensorFlow Tutorial

### Setup
Clone this repo to your local machine
~~~~
git clone https://github.com/silicon-valley-data-science/RNN-Tutorial
~~~~

In your `~/.profile`, Add home directory of RNN-Tutorial as a system variable (or modify if located elsewhere)
~~~~
echo 'export RNN_TUTORIAL=[absolute file path]/RNN-Tutorial' >> ~/.profile
echo 'export PYTHONPATH=$RNN_TUTORIAL/src:${PYTHONPATH}' >> ~/.profile
source ~/.profile
~~~~
_Note: [absolute file path] is often $HOME_

Create a Conda environment (You will need to [Install Conda](https://conda.io/docs/install/quick.html) first)
~~~~
conda create --name tf-rnn python=3
source activate tf-rnn
cd $RNN_TUTORIAL
pip install -r requirements.txt 
~~~~


### Install TensorFlow
If you have a NVIDIA GPU with [CUDA](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/#package-manager-installation) already installed
~~~~
pip install tensorflow-gpu==1.0.1
~~~~
If you will be running TensorFlow on CPU only (i.e. a MacBook Pro), follow these [instructions](https://www.tensorflow.org/install/install_mac)
~~~~
pip3 install --upgrade \
 https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.0.1-py3-none-any.whl
~~~~


### Run RNN training
All configurations for the RNN training script can be found in `configs/neural_network.ini`
~~~~
python $RNN_TUTORIAL/src/train_framework/tf_train_ctc.py
~~~~
_NOTE: If you have a GPU available, the code will run faster if you set `tf_device = /gpu:0` in `configs/neural_network.ini`_


### TensorBoard configuration
To visualize your results via tensorboard:
~~~~
tensorboard --logdir=$RNN_TUTORIAL/models/nn/debug_models/summary/
~~~~
- TensorBoard can be found in your browser at `http://127.0.1.1:6006`
- `tf.name_scope` is used to define parts of the network for visualization in TensorBoard. TensorBoard automatically finds any similarly structured network parts, such as identical fully connected layers and groups them in the graph visualization.
- Related to this are the `tf.summary.* methods` that log values of network parts, such as distributions of layer activations or error rate across epochs. These summaries are grouped within the `tf.name_scope`.
- See the official TensorFlow documentation for more details.


### Run unittests
We have included example unittests for the `tf_train_ctc.py` script
~~~~
python $RNN_TUTORIAL/src/tests/train_framework/tf_train_ctc_test.py
~~~~


### Add data
We have included example data from the [LibriVox corpus](https://librivox.org) in `data/raw/librivox/LibriSpeech/`. The data is separated into folders:
    - Train: train-clean-100-wav (5 examples)
    - Test: test-clean-wav (2 examples)
    - Dev: dev-clean-wav (2 examples)
If you would like to train a performant model, you can add additional wave and txt files to these folders, or create a new folder and update `configs/neural_network.ini` with the folder locations  
