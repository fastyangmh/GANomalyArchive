# import
from glob import glob
from os import makedirs
from os.path import join
from src.train import train
import numpy as np
from src.utils import get_files
from src.project_parameters import ProjectParameters
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from shutil import copy2, copytree, rmtree

# def


def _train_val_dataset_from_data_path(project_parameters):
    data = []
    for stage in ['train', 'val']:
        files = get_files(filepath=join(
            project_parameters.data_path, '{}/*/'.format(stage)), file_type=['jpg', 'png'])
        data += sorted(files)
    return np.array(data)


def _copy_files_to_destination_path(files, destination_path):
    makedirs(name=join(destination_path, 'normal'), exist_ok=True)
    for f in files:
        copy2(src=f, dst=join(destination_path, 'normal'))


def _create_k_fold_data(project_parameters, dataset):
    skf = StratifiedKFold(n_splits=project_parameters.n_splits, shuffle=True)
    for idx, (train_index, val_index) in tqdm(enumerate(skf.split(X=dataset, y=np.zeros(len(dataset)))), total=project_parameters.n_splits):
        x_train = dataset[train_index]
        x_val = dataset[val_index]
        destination_path = join(
            project_parameters.k_fold_data_path, 'k_fold_{}'.format(idx+1))
        makedirs(name=join(destination_path, 'train'), exist_ok=True)
        makedirs(name=join(destination_path, 'val'), exist_ok=True)
        _copy_files_to_destination_path(
            files=x_train, destination_path=join(destination_path, 'train'))
        _copy_files_to_destination_path(
            files=x_val, destination_path=join(destination_path, 'val'))
        copytree(src=join(project_parameters.data_path, 'test'),
                 dst=join(destination_path, 'test'))


def _get_k_fold_result(project_parameters):
    print('start k-fold cross-validation')
    results = {}
    directories = sorted(glob(join(project_parameters.k_fold_data_path, '*/')))
    for idx, directory in enumerate(directories):
        print('-'*30)
        print('\nk-fold cross-validation: {}/{}'.format(idx +
                                                        1, project_parameters.n_splits))
        project_parameters.data_path = directory
        results[idx+1] = train(project_parameters=project_parameters)
    return results


def _parse_k_fold_result(results):
    train_generator_loss, val_generator_loss, test_generator_loss = [], [], []
    train_discriminator_loss, val_discriminator_loss, test_discriminator_loss = [], [], []
    for result in results.values():
        each_stage_result = {stage: list(result[stage][0].values()) for stage in [
            'train', 'val', 'test']}
        train_generator_loss.append(each_stage_result['train'][0])
        train_discriminator_loss.append(each_stage_result['train'][1])
        val_generator_loss.append(each_stage_result['val'][0])
        val_discriminator_loss.append(each_stage_result['val'][1])
        test_generator_loss.append(each_stage_result['test'][0])
        test_discriminator_loss.append(each_stage_result['test'][1])
    return {'train': (train_generator_loss, train_discriminator_loss),
            'val': (val_generator_loss, val_discriminator_loss),
            'test': (test_generator_loss, test_discriminator_loss)}


def _calculate_mean_and_error(arrays):
    return np.mean(arrays), (max(arrays)-min(arrays))/2


def evaluate(project_parameters):
    train_val_dataset = _train_val_dataset_from_data_path(
        project_parameters=project_parameters)
    _create_k_fold_data(project_parameters=project_parameters,
                        dataset=train_val_dataset)
    results = _get_k_fold_result(project_parameters=project_parameters)
    results = _parse_k_fold_result(results=results)
    print('-'*30)
    print('k-fold cross-validation training generator loss mean:\t{} ± {}'.format(*
                                                                                  _calculate_mean_and_error(arrays=results['train'][0])))
    print('k-fold cross-validation training discriminator loss mean:\t{} ± {}'.format(*
                                                                                      _calculate_mean_and_error(arrays=results['train'][1])))
    print('k-fold cross-validation validation generator loss mean:\t{} ± {}'.format(*
                                                                                    _calculate_mean_and_error(arrays=results['val'][0])))
    print('k-fold cross-validation validation discriminator loss mean:\t{} ± {}'.format(*
                                                                                        _calculate_mean_and_error(arrays=results['val'][1])))
    print('k-fold cross-validation test generator loss mean:\t{} ± {}'.format(*
                                                                              _calculate_mean_and_error(arrays=results['test'][0])))
    print('k-fold cross-validation test discriminator loss mean:\t{} ± {}'.format(*
                                                                                  _calculate_mean_and_error(arrays=results['test'][1])))
    rmtree(path=project_parameters.k_fold_data_path)
    return results


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # k-fold cross validation
    if project_parameters.predefined_dataset is not None:
        print('temporarily does not support predefined dataset.')
    else:
        result = evaluate(project_parameters=project_parameters)
