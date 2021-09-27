# import
from os.path import join
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torch.utils.data.dataset import random_split
from torchvision.datasets import ImageFolder
from src.project_parameters import ProjectParameters
from pytorch_lightning import LightningDataModule
from src.utils import get_transform_from_file
from typing import Optional, Callable
from PIL import Image


# class


class ImageFolder(ImageFolder):
    def __init__(self, root: str, transform: Optional[Callable], in_chans):
        super().__init__(root, transform=transform, loader=Image.open)
        self.in_chans = in_chans

    def __getitem__(self, index: int):
        path, _ = self.samples[index]
        data = self.loader(path)
        assert self.in_chans == len(
            data.getbands()), 'please check the channels of image.'
        if self.transform is not None:
            data = self.transform(data)
        return data


class MNIST(MNIST):
    def __init__(self, root: str, train: bool, transform: Optional[Callable], download: bool):
        super().__init__(root, train=train, transform=transform, download=download)
        self.__get_normal_data_and_label__()

    def __get_normal_data_and_label__(self):
        self.data = self.data[self.targets == 0]
        self.targets = self.targets[self.targets == 0]

    def __getitem__(self, index: int):
        data, _ = super().__getitem__(index)
        return data


class DataModule(LightningDataModule):
    def __init__(self, project_parameters):
        super().__init__()
        self.project_parameters = project_parameters
        self.transform_dict = get_transform_from_file(
            filepath=project_parameters.transform_config_path)

    def prepare_data(self):
        if self.project_parameters.predefined_dataset is None:
            self.dataset = {}
            for stage in ['train', 'val', 'test']:
                self.dataset[stage] = ImageFolder(root=join(self.project_parameters.data_path, stage),
                                                  transform=self.transform_dict[stage], in_chans=self.project_parameters.in_chans)
                # modify the maximum number of files
                if self.project_parameters.max_files is not None:
                    lengths = (self.project_parameters.max_files, len(
                        self.dataset[stage])-self.project_parameters.max_files)
                    self.dataset[stage] = random_split(
                        dataset=self.dataset[stage], lengths=lengths)[0]
        else:
            train_set = eval('{}(root=self.project_parameters.data_path, train=True, download=True, transform=self.transform_dict["train"])'.format(
                self.project_parameters.predefined_dataset))
            test_set = eval('{}(root=self.project_parameters.data_path, train=False, download=True, transform=self.transform_dict["test"])'.format(
                self.project_parameters.predefined_dataset))
            # modify the maximum number of files
            if self.project_parameters.max_files is not None:
                for v in [train_set, test_set]:
                    v.data = v.data[:self.project_parameters.max_files]
                    v.targets = v.targets[:self.project_parameters.max_files]
            train_val_lengths = [round((1-self.project_parameters.val_size)*len(train_set)),
                                 round(self.project_parameters.val_size*len(train_set))]
            train_set, val_set = random_split(
                dataset=train_set, lengths=train_val_lengths)
            self.dataset = {'train': train_set,
                            'val': val_set, 'test': test_set}

    def train_dataloader(self):
        return DataLoader(dataset=self.dataset['train'], batch_size=self.project_parameters.batch_size, shuffle=True, pin_memory=self.project_parameters.use_cuda, num_workers=self.project_parameters.num_workers)

    def val_dataloader(self):
        return DataLoader(dataset=self.dataset['val'], batch_size=self.project_parameters.batch_size, shuffle=False, pin_memory=self.project_parameters.use_cuda, num_workers=self.project_parameters.num_workers)

    def test_dataloader(self):
        return DataLoader(dataset=self.dataset['test'], batch_size=self.project_parameters.batch_size, shuffle=False, pin_memory=self.project_parameters.use_cuda, num_workers=self.project_parameters.num_workers)

    def get_data_loaders(self):
        return {'train': self.train_dataloader(),
                'val': self.val_dataloader(),
                'test': self.test_dataloader()}


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # get data_module
    data_module = DataModule(project_parameters=project_parameters)
    data_module.prepare_data()

    # display the dataset information
    for stage in ['train', 'val', 'test']:
        print(stage, data_module.dataset[stage])

    # get data loaders
    data_loaders = data_module.get_data_loaders()
