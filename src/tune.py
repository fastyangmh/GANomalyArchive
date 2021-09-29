# import
from os.path import join
from src.utils import load_yaml
from src.project_parameters import ProjectParameters
import ray.tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
from ray import init, shutdown
from functools import partial
from copy import copy
from src.train import train
import numpy as np

# def


def _get_hyperparameter_space(project_parameters):
    hyperparameter_space_config = load_yaml(
        filepath=project_parameters.hyperparameter_config_path)
    assert hyperparameter_space_config is not None, 'the hyperparameter space config has not any content.'
    hyperparameter_space = {}
    for parameter_type in hyperparameter_space_config.keys():
        assert parameter_type in ['int', 'float', 'choice'], 'the type is wrong, please check it. the type: {}'.format(
            parameter_type)
        for parameter_name, parameter_value in hyperparameter_space_config[parameter_type].items():
            assert parameter_name in project_parameters, 'the parameter name is wrong, please check it. the parameter name: {}'.format(
                parameter_name)
            if parameter_type == 'int':
                hyperparameter_space[parameter_name] = ray.tune.randint(
                    lower=min(parameter_value), upper=max(parameter_value))
            elif parameter_type == 'float':
                hyperparameter_space[parameter_name] = ray.tune.uniform(
                    lower=min(parameter_value), upper=max(parameter_value))
            elif parameter_type == 'choice':
                hyperparameter_space[parameter_name] = ray.tune.choice(
                    categories=parameter_value)
    return hyperparameter_space


def _set_tune_project_parameters(hyperparameter, project_parameters):
    for k, v in hyperparameter.items():
        if type(v) == str:
            exec('project_parameters.{}="{}"'.format(k, v))
        else:
            exec('project_parameters.{}={}'.format(k, v))
        if k == 'train_iter' and not project_parameters.use_early_stopping:
            exec('project_parameters.val_iter={}'.format(v))
    return project_parameters


def _parse_tune_result(result):
    generator_loss_dict, discriminator_loss_dict = {}, {}
    for stage in ['train', 'val', 'test']:
        generator_loss, discriminator_loss = result[stage][0].values()
        generator_loss_dict[stage] = generator_loss
        discriminator_loss_dict[stage] = discriminator_loss
    return generator_loss_dict, discriminator_loss_dict


def _tune_function(hyperparameter, project_parameters):
    if project_parameters.tune_debug:
        ray.tune.report(sum_of_generator_loss=sum(
            [value for value in hyperparameter.values() if type(value) is not str]))
    else:
        tune_project_parameters = _set_tune_project_parameters(
            hyperparameter=hyperparameter, project_parameters=copy(project_parameters))
        result = train(project_parameters=tune_project_parameters)
        generator_loss_dict, discriminator_loss_dict = _parse_tune_result(
            result=result)
        sum_of_generator_loss = sum(generator_loss_dict.values())
        ray.tune.report(train_generator_loss=generator_loss_dict['train'],
                        train_discriminator_loss=discriminator_loss_dict['train'],
                        val_generator_loss=generator_loss_dict['val'],
                        val_discriminator_loss=discriminator_loss_dict['val'],
                        test_generator_loss=generator_loss_dict['test'],
                        test_discriminator_loss=discriminator_loss_dict['test'],
                        sum_of_generator_loss=sum_of_generator_loss)


def tune(project_parameters):
    project_parameters.mode = 'train'
    hyperparameter_space = _get_hyperparameter_space(
        project_parameters=project_parameters)
    tune_scheduler = ASHAScheduler(metric='sum_of_generator_loss', mode='min')
    reporter = CLIReporter(metric_columns=['train_generator_loss', 'train_discriminator_loss', 'val_generator_loss',
                                           'val_discriminator_loss', 'test_generator_loss', 'test_discriminator_loss', 'sum_of_generator_loss'])
    init(dashboard_host='0.0.0.0')
    tuning_result = ray.tune.run(run_or_experiment=partial(
        _tune_function, project_parameters=project_parameters),
        resources_per_trial={
            'cpu': project_parameters.tune_cpu, 'gpu': project_parameters.tune_gpu},
        config=hyperparameter_space,
        num_samples=project_parameters.tune_iter,
        scheduler=tune_scheduler,
        local_dir=join(project_parameters.save_path, 'tuning_logs'),
        progress_reporter=reporter)
    best_trial = tuning_result.get_best_trial(
        'sum_of_generator_loss', 'min', 'last')
    if not project_parameters.tune_debug:
        project_parameters = _set_tune_project_parameters(
            hyperparameter=best_trial.config, project_parameters=project_parameters)
        result = train(project_parameters=project_parameters)
        result['tune'] = tuning_result
    else:
        result = {'tune': tuning_result}
    print('best trial name: {}'.format(best_trial))
    print('best trial result: {}'.format(
        best_trial.last_result['sum_of_generator_loss']))
    print('best trial config: {}'.format(best_trial.config))
    if 'parameters_config_path' in project_parameters:
        output = 'num_workers: {}'.format(project_parameters.num_workers)
        for k, v in best_trial.config.items():
            output += '\n{}: {}'.format(k, v)
        print('best trial config command:\n{}'.format(output))
    else:
        print('best trial config command: --num_workers {}{}'.format(project_parameters.num_workers, (' --{} {}' *
                                                                                                      len(best_trial.config)).format(*np.concatenate(list(zip(best_trial.config.keys(), best_trial.config.values()))))))
    shutdown()
    return result


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # tune
    result = tune(project_parameters=project_parameters)
