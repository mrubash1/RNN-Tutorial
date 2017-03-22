# notes

```bash
# ~/github/RNN-Tutorial
# jonathan-svds (master)*$ pip3 install --upgrade \
>  https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.0.1-py3-none-any.whl
Collecting tensorflow==1.0.1 from https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.0.1-py3-none-any.whl
  Downloading https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.0.1-py3-none-any.whl (39.3MB)
Collecting six>=1.10.0 (from tensorflow==1.0.1)
  Using cached six-1.10.0-py2.py3-none-any.whl
Collecting protobuf>=3.1.0 (from tensorflow==1.0.1)
  Using cached protobuf-3.2.0-py2.py3-none-any.whl
Requirement already up-to-date: wheel>=0.26 in /usr/local/lib/python3.6/site-packages (from tensorflow==1.0.1)
Collecting numpy>=1.11.0 (from tensorflow==1.0.1)
  Downloading numpy-1.12.1-cp36-cp36m-macosx_10_6_intel.macosx_10_9_intel.macosx_10_9_x86_64.macosx_10_10_intel.macosx_10_10_x86_64.whl (4.4MB)
    100% |████████████████████████████████| 4.4MB 282kB/s
Collecting setuptools (from protobuf>=3.1.0->tensorflow==1.0.1)
  Downloading setuptools-34.3.2-py2.py3-none-any.whl (389kB)
    100% |████████████████████████████████| 399kB 2.7MB/s
Collecting appdirs>=1.4.0 (from setuptools->protobuf>=3.1.0->tensorflow==1.0.1)
  Downloading appdirs-1.4.3-py2.py3-none-any.whl
Collecting packaging>=16.8 (from setuptools->protobuf>=3.1.0->tensorflow==1.0.1)
  Downloading packaging-16.8-py2.py3-none-any.whl
Collecting pyparsing (from packaging>=16.8->setuptools->protobuf>=3.1.0->tensorflow==1.0.1)
  Using cached pyparsing-2.2.0-py2.py3-none-any.whl
Installing collected packages: six, appdirs, pyparsing, packaging, setuptools, protobuf, numpy, tensorflow
  Found existing installation: setuptools 32.2.0
    Uninstalling setuptools-32.2.0:
      Successfully uninstalled setuptools-32.2.0
Exception:
Traceback (most recent call last):
  File "/usr/local/lib/python3.6/site-packages/pip/basecommand.py", line 215, in main
    status = self.run(options, args)
  File "/usr/local/lib/python3.6/site-packages/pip/commands/install.py", line 342, in run
    prefix=options.prefix_path,
  File "/usr/local/lib/python3.6/site-packages/pip/req/req_set.py", line 784, in install
    **kwargs
  File "/usr/local/lib/python3.6/site-packages/pip/req/req_install.py", line 851, in install
    self.move_wheel_files(self.source_dir, root=root, prefix=prefix)
  File "/usr/local/lib/python3.6/site-packages/pip/req/req_install.py", line 1064, in move_wheel_files
    isolated=self.isolated,
  File "/usr/local/lib/python3.6/site-packages/pip/wheel.py", line 377, in move_wheel_files
    clobber(source, dest, False, fixer=fixer, filter=filter)
  File "/usr/local/lib/python3.6/site-packages/pip/wheel.py", line 323, in clobber
    shutil.copyfile(srcfile, destfile)
  File "/usr/local/Cellar/python3/3.6.0/Frameworks/Python.framework/Versions/3.6/lib/python3.6/shutil.py", line 115, in copyfile
    with open(dst, 'wb') as fdst:
PermissionError: [Errno 13] Permission denied: '/usr/local/bin/f2py'
(tf-rnn)
# ~/github/RNN-Tutorial
# jonathan-svds (master)*$ pip3 install --upgrade  https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.0.1-py3-none-any.whl^C
(tf-rnn)
# ~/github/RNN-Tutorial
# jonathan-svds (master)*$ pip3 install --upgrade \
>  https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.0.1-py3-none-any.whl
Collecting tensorflow==1.0.1 from https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.0.1-py3-none-any.whl
  Using cached https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.0.1-py3-none-any.whl
Requirement already up-to-date: protobuf>=3.1.0 in /usr/local/lib/python3.6/site-packages (from tensorflow==1.0.1)
Requirement already up-to-date: numpy>=1.11.0 in /usr/local/lib/python3.6/site-packages (from tensorflow==1.0.1)
Requirement already up-to-date: six>=1.10.0 in /usr/local/lib/python3.6/site-packages (from tensorflow==1.0.1)
Requirement already up-to-date: wheel>=0.26 in /usr/local/lib/python3.6/site-packages (from tensorflow==1.0.1)
Requirement already up-to-date: setuptools in /usr/local/lib/python3.6/site-packages (from protobuf>=3.1.0->tensorflow==1.0.1)
Requirement already up-to-date: packaging>=16.8 in /usr/local/lib/python3.6/site-packages (from setuptools->protobuf>=3.1.0->tensorflow==1.0.1)
Requirement already up-to-date: appdirs>=1.4.0 in /usr/local/lib/python3.6/site-packages (from setuptools->protobuf>=3.1.0->tensorflow==1.0.1)
Requirement already up-to-date: pyparsing in /usr/local/lib/python3.6/site-packages (from packaging>=16.8->setuptools->protobuf>=3.1.0->tensorflow==1.0.1)
Installing collected packages: tensorflow
Successfully installed tensorflow-1.0.1
```

```bash
# ~/github/RNN-Tutorial
# jonathan-svds (master)*$ tensorboard --logdir=$RNN_TUTORIAL/models/nn/debug_models/summary/
Traceback (most recent call last):
  File "/Users/jonathan/miniconda3/envs/tf-rnn/lib/python3.6/site-packages/tensorflow/python/__init__.py", line 61, in <module>
    from tensorflow.python import pywrap_tensorflow
  File "/Users/jonathan/miniconda3/envs/tf-rnn/lib/python3.6/site-packages/tensorflow/python/pywrap_tensorflow.py", line 28, in <module>
    _pywrap_tensorflow = swig_import_helper()
  File "/Users/jonathan/miniconda3/envs/tf-rnn/lib/python3.6/site-packages/tensorflow/python/pywrap_tensorflow.py", line 24, in swig_import_helper
    _mod = imp.load_module('_pywrap_tensorflow', fp, pathname, description)
  File "/Users/jonathan/miniconda3/envs/tf-rnn/lib/python3.6/imp.py", line 242, in load_module
    return load_dynamic(name, filename, file)
  File "/Users/jonathan/miniconda3/envs/tf-rnn/lib/python3.6/imp.py", line 342, in load_dynamic
    return _load(spec)
ImportError: dlopen(/Users/jonathan/miniconda3/envs/tf-rnn/lib/python3.6/site-packages/tensorflow/python/_pywrap_tensorflow.so, 10): Library not loaded: @rpath/libcudart.8.0.dylib
  Referenced from: /Users/jonathan/miniconda3/envs/tf-rnn/lib/python3.6/site-packages/tensorflow/python/_pywrap_tensorflow.so
  Reason: image not found

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/Users/jonathan/miniconda3/envs/tf-rnn/bin/tensorboard", line 7, in <module>
    from tensorflow.tensorboard.tensorboard import main
  File "/Users/jonathan/miniconda3/envs/tf-rnn/lib/python3.6/site-packages/tensorflow/__init__.py", line 24, in <module>
    from tensorflow.python import *
  File "/Users/jonathan/miniconda3/envs/tf-rnn/lib/python3.6/site-packages/tensorflow/python/__init__.py", line 72, in <module>
    raise ImportError(msg)
ImportError: Traceback (most recent call last):
  File "/Users/jonathan/miniconda3/envs/tf-rnn/lib/python3.6/site-packages/tensorflow/python/__init__.py", line 61, in <module>
    from tensorflow.python import pywrap_tensorflow
  File "/Users/jonathan/miniconda3/envs/tf-rnn/lib/python3.6/site-packages/tensorflow/python/pywrap_tensorflow.py", line 28, in <module>
    _pywrap_tensorflow = swig_import_helper()
  File "/Users/jonathan/miniconda3/envs/tf-rnn/lib/python3.6/site-packages/tensorflow/python/pywrap_tensorflow.py", line 24, in swig_import_helper
    _mod = imp.load_module('_pywrap_tensorflow', fp, pathname, description)
  File "/Users/jonathan/miniconda3/envs/tf-rnn/lib/python3.6/imp.py", line 242, in load_module
    return load_dynamic(name, filename, file)
  File "/Users/jonathan/miniconda3/envs/tf-rnn/lib/python3.6/imp.py", line 342, in load_dynamic
    return _load(spec)
ImportError: dlopen(/Users/jonathan/miniconda3/envs/tf-rnn/lib/python3.6/site-packages/tensorflow/python/_pywrap_tensorflow.so, 10): Library not loaded: @rpath/libcudart.8.0.dylib
  Referenced from: /Users/jonathan/miniconda3/envs/tf-rnn/lib/python3.6/site-packages/tensorflow/python/_pywrap_tensorflow.so
  Reason: image not found


Failed to load the native TensorFlow runtime.

See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/get_started/os_setup.md#import_error

for some common reasons and solutions.  Include the entire stack trace
above this error message when asking for help.
```