# import
from src.project_parameters import ProjectParameters
from src.predict import Predict
import tkinter as tk
from tkinter import Tk, Button, Label, filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from src.utils import get_transform_from_file

# class


class GUI:
    def __init__(self, project_parameters):
        self.project_parameters = project_parameters
        self.predict_object = Predict(project_parameters=project_parameters)
        self.data_path = None
        self.transform = get_transform_from_file(
            filepath=project_parameters.transform_config_path)['predict']

        # window
        self.window = Tk()
        self.window.geometry('{}x{}'.format(
            self.window.winfo_screenwidth(), self.window.winfo_screenheight()))
        self.window.title('Demo GUI')

        # button
        self.load_image_button = Button(
            self.window, text='load image', fg='black', bg='white', command=self._load_image)
        self.recognize_button = Button(
            self.window, text='recognize', fg='black', bg='white', command=self._recognize)

        # label
        self.data_path_label = Label(self.window, text='', fg='black')
        self.gallery_image_label = Label(self.window, text='', fg='black')
        self.abnormal_score_label = Label(self.window, text='', fg='black')
        self.result_label = Label(
            self.window, text='', fg='black', font=(None, 50))

        # matplotlib canvas
        # this is Tkinter default background-color
        facecolor = (0.9254760742, 0.9254760742, 0.9254760742)
        self.gallery_image_canvas = FigureCanvasTkAgg(
            Figure(figsize=(5, 5), facecolor=facecolor), master=self.window)

    def _resize_image(self, image):
        width, height = image.size
        ratio = max(self.window.winfo_height(),
                    self.window.winfo_width())/max(width, height)
        ratio *= 0.25
        image = image.resize((int(width*ratio), int(height*ratio)))
        return image

    def _load_image(self):
        self.gallery_image_canvas.figure.clear()
        color_mode = 'RGB' if self.project_parameters.in_chans == 3 else 'L'
        self.data_path = filedialog.askopenfilename(
            initialdir='./', title='Select image file', filetypes=(('png files', '*.png'), ('jpeg files', '*.jpg')))
        resized_image = self._resize_image(
            image=Image.open(fp=self.data_path).convert(color_mode))
        imageTk = ImageTk.PhotoImage(resized_image)
        self.gallery_image_label.config(image=imageTk)
        self.gallery_image_label.image = imageTk
        self.data_path_label.config(
            text='image path: {}'.format(self.data_path))

    def _recognize(self):
        if self.data_path is not None:
            abnormal_score, fake_image = self.predict_object(
                data_path=self.data_path)
            self.gallery_image_label.config(image='')
            fake_image = fake_image[0]
            if fake_image.shape[0] == 1:
                fake_image = fake_image[0]
            else:
                fake_image = np.transpose(fake_image, (1, 2, 0))
            color_mode = 'RGB' if self.project_parameters.in_chans == 3 else 'L'
            cmap = 'gray' if color_mode == 'L' else None
            real_image = Image.open(self.data_path).convert(color_mode)
            if self.transform is not None:
                real_image = self.transform(real_image).cpu().data.numpy()
                real_image = real_image.transpose(1, 2, 0)
            else:
                real_image = np.array(real_image)
            if real_image.shape[-1] == 1:
                real_image = real_image[..., 0]
            self.gallery_image_canvas.figure.clear()
            subplot1 = self.gallery_image_canvas.figure.add_subplot(131)
            subplot1.title.set_text('real')
            subplot1.imshow(real_image, cmap=cmap)
            subplot1.axis('off')
            subplot2 = self.gallery_image_canvas.figure.add_subplot(132)
            subplot2.title.set_text('fake')
            subplot2.imshow(fake_image, cmap=cmap)
            subplot2.axis('off')
            subplot3 = self.gallery_image_canvas.figure.add_subplot(133)
            subplot3.title.set_text('difference')
            subplot3.imshow(np.abs(real_image-fake_image), cmap=cmap)
            subplot3.axis('off')
            self.gallery_image_canvas.figure.tight_layout()
            self.gallery_image_canvas.draw()
            self.abnormal_score_label.config(
                text='abnormal score: {}'.format(abnormal_score))
            result = 'normal' if abnormal_score < self.project_parameters.threshold else 'abnormal'
            self.result_label.config(text=result)
        else:
            messagebox.showerror(
                title='Error!', message='please select an image!')

    def run(self):
        # button
        self.load_image_button.pack(anchor=tk.NW)
        self.recognize_button.pack(anchor=tk.NW)

        # label
        self.data_path_label.pack(anchor=tk.N)
        self.gallery_image_label.pack(anchor=tk.N)
        self.gallery_image_canvas.get_tk_widget().pack(anchor=tk.N)
        self.abnormal_score_label.pack(anchor=tk.N)
        self.result_label.pack(anchor=tk.N)

        # run
        self.window.mainloop()


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # GUI
    gui = GUI(project_parameters=project_parameters)
    gui.run()
