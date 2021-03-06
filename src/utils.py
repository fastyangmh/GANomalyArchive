# import
from ruamel.yaml import safe_load
from torchvision import transforms
from os.path import isfile
import torch
from glob import glob
from os.path import join

# def


def load_yaml(filepath):
    with open(filepath, 'r') as f:
        config = safe_load(f)
    return config


def get_transform_from_file(filepath):
    """Load transform from file path for image transformations.

    Args:
        filepath (str): the file path of the backbone model.

    Returns:
        dict: the dictionary contains the transforms of train, validation, test, and predict.
    """
    if filepath is None:
        return {}.fromkeys(['train', 'val', 'test', 'predict'], None)
    elif isfile(filepath):
        transform_dict = {}
        transform_config = load_yaml(filepath=filepath)
        for stage in transform_config.keys():
            transform_dict[stage] = []
            if type(transform_config[stage]) != dict:
                transform_dict[stage] = None
                continue
            for name, value in transform_config[stage].items():
                if value is None:
                    transform_dict[stage].append(
                        eval('transforms.{}()'.format(name)))
                else:
                    if type(value) is dict:
                        value = ('{},'*len(value)).format(*
                                                          ['{}={}'.format(a, b) for a, b in value.items()])
                    transform_dict[stage].append(
                        eval('transforms.{}({})'.format(name, value)))
            transform_dict[stage] = transforms.Compose(transform_dict[stage])
        return transform_dict
    else:
        assert False, 'please check the transform config path: {}'.format(
            filepath)


def load_checkpoint(model, use_cuda, checkpoint_path):
    map_location = torch.device(
        device='cuda') if use_cuda else torch.device(device='cpu')
    checkpoint = torch.load(f=checkpoint_path, map_location=map_location)
    model.load_state_dict(checkpoint['state_dict'])
    return model


def get_files(filepath, file_type):
    files = []
    if type(file_type) != list:
        file_type = [file_type]
    for v in file_type:
        files += sorted(glob(join(filepath, '*.{}'.format(v))))
    return files
