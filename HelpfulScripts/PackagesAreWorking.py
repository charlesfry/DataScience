def main() :
    print("This script verifies that certain packages are working")
    print("including: Python itself, numpy, pandas, seaborn, matplotlib,\n"
          "tensorFlow, and tensorFlow's integration with CUDA")

    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    import sys
    print("Python version:",sys.version)
    print("Python Version info:",sys.version_info,'\n')
    error_packages = []

    try:
        import numpy as np
        print("\nNumpy version:", np.version.version)
    except ModuleNotFoundError:
        print("numpy not responsive")
        error_packages.append('numpy')

    # try:
    #     import tpot
    #     print("\nTPOT version:",tpot.__version__)
    # except:
    #     print('TPOT not responsive')
    #     error_packages.append('TPOT')

    try :
        import pandas as pd
        print('\npandas version:',pd.__version__)
    except ModuleNotFoundError:
        print("pandas not responsive")
        error_packages.append('pandas')

    try :
        import seaborn as sns
        print('\nSeaborn version:',sns.__version__)
    except ModuleNotFoundError:
        print("seaborn not responsive")
        error_packages.append('seaborn')

    try :
        import matplotlib
        print('\nmatplotlib version:',matplotlib.__version__)
    except ModuleNotFoundError:
        print("matplotlib not responsive")
        error_packages.append('matplotlib')

    # separate TensorFlow from the rest of the information
    print('\n-----------------------------')
    try :
        import tensorflow as tf
        print('\nTensorFlow version: {}\n'.format(tf.__version__))
        has_cuda = len(tf.config.list_physical_devices('GPU'))
    except ModuleNotFoundError:
        print("TensorFlow not responsive")
        error_packages.append('TensorFlow')
    try :
        if len(tf.config.list_physical_devices('GPU')):
            print("-----------------------------\n\ntensorflow uses CUDA")
            for i,device in enumerate(tf.config.list_physical_devices('GPU')) :
                print(f'Device {i}: {device}')
        else:
            error_packages.append('CUDA for TensorFlow')
            print("tensorflow does not use CUDA")
    except ModuleNotFoundError:
        print("error loading physical devices. \nTensorFlow does not use CUDA")
        error_packages.append('CUDA for TensorFlow')

    try:
        import torch
        print("\nPyTorch version:", torch.__version__)
        device_count = torch.cuda.device_count()
        if device_count < 1 :print('PyTorch does not detect a GPU')
        else : print(f'{device_count} Devices found.')
        for i in range(device_count) :
            print(f'Device {i}: {torch.cuda.get_device_name(i)}')
    except ModuleNotFoundError:
        print("PyTorch not responsive")
        error_packages.append('torch')

    try:
        import torchvision
        print("\ntorchvision version:", torchvision.__version__)
    except:
        print('torchvision not responsive')
        error_packages.append('torchvision')
    try:
        import torchaudio
        torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE = False
        print("\ntorchvision version:",torchaudio.__version__)
    except:
        print('torchvision not responsive')
        error_packages.append('torchaudio')
    print('-----------------------------\n')

    # try :
    #     from mlagents_envs.environment import UnityEnvironment
    #     poo = UnityEnvironment.behavior_specs
    #     print("ML-Agents imported successfully")
    # except ModuleNotFoundError:
    #     print("Unity ML-Agents not installed")
    #     error_packages.append('ML-Agents')

    print("\nTesting complete")
    if len(error_packages) > 0 :
        print(f'Failed to access: \n{error_packages}')
    else :
        import torch
        if len(tf.config.list_physical_devices('GPU')) > 0 and torch.cuda.device_count() > 0 :
            print('\nAll packages installed successfully with gpu support')
        else :
            print('\nPackages installed, but gpu support not successful')

    from datetime import datetime
    os.system(f'python -m pip freeze > requirements.{datetime.now().date().strftime("%Y%m%d")}.txt')

if __name__ == '__main__' :
    main()