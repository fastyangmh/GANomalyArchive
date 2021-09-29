# import
from src.project_parameters import ProjectParameters
from src.train import train
from src.predict import Predict
from src.gui import GUI

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
            result = Predict(project_parameters=project_parameters)(
                data_path=project_parameters.data_path)
            print(result)

    return result


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # main
    result = main(project_parameters=project_parameters)
