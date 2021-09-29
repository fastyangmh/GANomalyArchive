# import
from src.project_parameters import ProjectParameters
from src.train import train

# def


def main(project_parameters):
    result = None
    if project_parameters.mode == 'train':
        result = train(project_parameters=project_parameters)
    return result


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # main
    result = main(project_parameters=project_parameters)
