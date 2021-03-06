# import
from src.project_parameters import ProjectParameters
from src.train import train
from src.predict import Predict
from src.gui import GUI
from src.evaluate import evaluate
from src.tune import tune

# def


def main(project_parameters):
    result = None
    if project_parameters.mode == 'train':
        result = train(project_parameters=project_parameters)
    elif project_parameters.mode == 'predict':
        if project_parameters.use_gui:
            gui = GUI(project_parameters=project_parameters)
            gui.run()
        else:
            result, _ = Predict(project_parameters=project_parameters)(
                data_path=project_parameters.data_path)
            # use [:-1] to remove the latest comma
            print(('{},'*project_parameters.num_classes).format(*
                                                                project_parameters.classes)[:-1])
            print(result)
    elif project_parameters.mode == 'evaluate':
        if project_parameters.predefined_dataset is not None:
            print('temporarily does not support predefined dataset.')
        else:
            evaluate(project_parameters=project_parameters)
    elif project_parameters.mode == 'tune':
        result = tune(project_parameters=project_parameters)
    return result


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # main
    result = main(project_parameters=project_parameters)
