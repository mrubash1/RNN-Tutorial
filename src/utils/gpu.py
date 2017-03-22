from tensorflow.python.client import device_lib


def get_available_gpus():
    """
    Returns the number of GPUs available on this system.
    """
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def check_if_gpu_available(gpu_name):
    """
    Returns boolean of if a specific gpu_name (string) is available
    On the system
    """
    list_of_gpus = get_available_gpus()
    if gpu_name not in list_of_gpus:
        return False
    else:
        return True
