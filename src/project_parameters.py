# import
import argparse
from os.path import abspath, join
import torch
from src.utils import load_yaml
from datetime import datetime

# class


class ProjectParameters:
    def __init__(self):
        self._parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        # base
        self._parser.add_argument('--mode', type=str, choices=['train', 'predict', 'tune', 'evaluate'], required=True,
                                  help='if the mode equals train, will train the model. if the mode equals predict, will use the pre-trained model to predict. if the mode equals tune, will hyperparameter tuning the model. if the mode equals evaluate, will evaluate the model by the k-fold validation.')
        self._parser.add_argument(
            '--data_path', type=str, required=True, help='the data path.')
        self._parser.add_argument('--predefined_dataset', type=str, default=None, choices=[
                                  'MNIST'], help='the predefined dataset that provided the MNIST datasets.')
        self._parser.add_argument(
            '--random_seed', type=self._str_to_int, default=0, help='the random seed.')
        self._parser.add_argument(
            '--save_path', type=str, default='save/', help='the path which stores the checkpoint of PyTorch lightning.')
        self._parser.add_argument('--no_cuda', action='store_true', default=False,
                                  help='whether to use Cuda to train the model. if True which will train the model on CPU. if False which will train on GPU.')
        self._parser.add_argument('--gpus', type=self._str_to_int_list, default=-1,
                                  help='number of GPUs to train on (int) or which GPUs to train on (list or str) applied per node. if give -1 will use all available GPUs.')
        self._parser.add_argument(
            '--parameters_config_path', type=str, default=None, help='the parameters config path.')

        # data preparation
        self._parser.add_argument(
            '--batch_size', type=int, default=32, help='how many samples per batch to load.')
        self._parser.add_argument('--val_size', type=float, default=0.2,
                                  help='the validation data size used for the predefined dataset.')
        self._parser.add_argument('--num_workers', type=int, default=torch.get_num_threads(
        ), help='how many subprocesses to use for data loading.')
        self._parser.add_argument('--transform_config_path', type=self._str_to_str,
                                  default='config/transform.yaml', help='the transform config path.')
        self._parser.add_argument(
            '--max_files', type=self._str_to_int, default=None, help='the maximum number of files for loading files.')
        self._parser.add_argument('--in_chans', type=int, default=3,
                                  help='number of input channels / colors (default: 3).')
        self._parser.add_argument('--image_size', type=int, default=256, required=True,
                                  help='the input image size. if you have set resize transform, please set the image size to the same value as resize in transform.yaml. note that the input image must be equal in height and width.')

        # model
        self._parser.add_argument('--checkpoint_path', type=str, default=None,
                                  help='the path of the pre-trained model checkpoint.')
        self._parser.add_argument('--optimizer_config_path', type=str,
                                  default='config/optimizer.yaml', help='the optimizer config path.')
        self._parser.add_argument(
            '--latent_size', type=int, default=100, help='size of the latent z vector.')
        self._parser.add_argument(
            '--discriminator_features', type=int, default=64, help='number of features of the discriminator network.')
        self._parser.add_argument(
            '--generator_features', type=int, default=64, help='number of features of the generator network.')
        self._parser.add_argument(
            '--adversarial_weight', type=float, default=1, help='adversarial loss weight.')
        self._parser.add_argument(
            '--reconstruction_weight', type=float, default=50, help='reconstruction loss weight.')
        self._parser.add_argument(
            '--encoder_weight', type=float, default=1, help='encoder loss weight.')

        # train
        self._parser.add_argument('--val_iter', type=self._str_to_int, default=1,
                                  help='the number of validation iteration. if set None, the val_iter will set the same as train_iter.')
        self._parser.add_argument(
            '--lr', type=float, default=1e-3, help='the learning rate.')
        self._parser.add_argument(
            '--train_iter', type=int, default=100, help='the number of training iteration.')
        self._parser.add_argument('--lr_scheduler', type=str, default='CosineAnnealingLR', choices=[
                                  'StepLR', 'CosineAnnealingLR'], help='the lr scheduler while training model.')
        self._parser.add_argument(
            '--step_size', type=int, default=10, help='period of learning rate decay.')
        self._parser.add_argument('--gamma', type=int, default=0.1,
                                  help='multiplicative factor of learning rate decay.')
        self._parser.add_argument('--precision', type=int, default=32, choices=[
                                  16, 32], help='full precision (32) or half precision (16). Can be used on CPU, GPU or TPUs.')
        self._parser.add_argument('--profiler', type=str, default=None, choices=[
            'simple', 'advanced'], help='to profile individual steps during training and assist in identifying bottlenecks.')
        self._parser.add_argument('--weights_summary', type=str, default=None, choices=[
                                  'top', 'full'], help='prints a summary of the weights when training begins.')

        # predict
        self._parser.add_argument('--gui', action='store_true', default=False,
                                  help='whether to use the gui window while predicting.')
        self._parser.add_argument(
            '--threshold', type=float, default=0.2, help='the anomaly score threshold.')

        # evaluate
        self._parser.add_argument(
            '--n_splits', type=int, default=5, help='number of folds. must be at least 2.')

        # tune
        self._parser.add_argument(
            '--tune_iter', type=int, default=100, help='the number of tuning iteration.')
        self._parser.add_argument('--tune_cpu', type=int, default=1,
                                  help='CPU resources to allocate per trial in hyperparameter tuning.')
        self._parser.add_argument('--tune_gpu', type=float, default=None,
                                  help='GPU resources to allocate per trial in hyperparameter tuning.')
        self._parser.add_argument('--hyperparameter_config_path', type=str,
                                  default='config/hyperparameter.yaml', help='the hyperparameter config path.')
        self._parser.add_argument('--tune_debug', action='store_true',
                                  default=False, help='whether to use debug mode while tuning.')

    def _str_to_str(self, s):
        return None if s == 'None' or s == 'none' else s

    def _str_to_int(self, s):
        return None if s == 'None' or s == 'none' else int(s)

    def _str_to_int_list(self, s):
        return [int(v) for v in s.split(',') if len(v) > 0]

    def _get_new_dict(self, old_dict, yaml_dict):
        for k in yaml_dict.keys():
            del old_dict[k]
        return {**old_dict, **yaml_dict}

    def parse(self):
        project_parameters = self._parser.parse_args()
        if project_parameters.parameters_config_path is not None:
            project_parameters = argparse.Namespace(**self._get_new_dict(old_dict=vars(
                project_parameters), yaml_dict=load_yaml(filepath=abspath(project_parameters.parameters_config_path))))
        else:
            del project_parameters.parameters_config_path

        # base
        project_parameters.data_path = abspath(
            path=project_parameters.data_path)
        if project_parameters.predefined_dataset is not None and project_parameters.mode != 'predict':
            project_parameters.data_path = join(
                project_parameters.data_path, project_parameters.predefined_dataset)
        project_parameters.use_cuda = torch.cuda.is_available(
        ) and not project_parameters.no_cuda
        project_parameters.gpus = project_parameters.gpus if project_parameters.use_cuda else 0

        # data preparation
        project_parameters.classes = sorted(['abnormal', 'normal'])
        project_parameters.class_to_idx = {
            c: idx for idx, c in enumerate(project_parameters.classes)}
        project_parameters.num_classes = len(project_parameters.classes)
        if project_parameters.transform_config_path is not None:
            project_parameters.transform_config_path = abspath(
                project_parameters.transform_config_path)

        # model
        project_parameters.optimizer_config_path = abspath(
            project_parameters.optimizer_config_path)

        # train
        if project_parameters.val_iter is None:
            project_parameters.val_iter = project_parameters.train_iter

        # predict
        project_parameters.use_gui = project_parameters.gui

        # evaluate
        if project_parameters.mode == 'evaluate':
            project_parameters.k_fold_data_path = './k_fold_dataset{}'.format(
                datetime.now().strftime('%Y%m%d%H%M%S'))

        # tune
        if project_parameters.tune_gpu is None:
            project_parameters.tune_gpu = torch.cuda.device_count()/project_parameters.tune_cpu
        if project_parameters.mode == 'tune':
            project_parameters.num_workers = project_parameters.tune_cpu
        project_parameters.hyperparameter_config_path = abspath(
            project_parameters.hyperparameter_config_path)

        return project_parameters


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # display each parameter
    for name, value in vars(project_parameters).items():
        print('{:<20}= {}'.format(name, value))
