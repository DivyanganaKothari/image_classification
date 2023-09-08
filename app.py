import os
import tkinter as tk
from tkinter import simpledialog, messagebox
from tkinter import Label
import PIL.Image
import PIL.ImageTk
import cv2 as cv
from PIL import Image, ImageTk
import camera
import model

LOGO_PATH= "logo.jpg"


class App:

    def __init__(self, window=tk.Tk(), window_title="Camera Classifier", num_classes=2):

        self.window = window
        self.window_title = window_title
        self.window.title("Fraunhofer")

        self.window.configure(bg="light gray")  # Change the background color

        # self.window.attributes("-fullscreen", True)

        # self.counters = [1] * num_classes
        self.num_classes = num_classes
        self.model = model.Model(num_classes=self.num_classes)
        self.is_model_trained = False
        self.class_name_labels = []

        # Create a frame for the logo label
        logo_frame = tk.Frame(self.window)
        logo_frame.pack(side=tk.LEFT, padx=10, pady=10, anchor=tk.NW)

        # Load and display the logo image
        logo_image = Image.open(LOGO_PATH)
        logo_image = ImageTk.PhotoImage(logo_image)
        self.logo_label = tk.Label(logo_frame, image=logo_image)
        self.logo_label.image = logo_image  # Keep a reference to prevent image from being garbage collected
        self.logo_label.pack()

        self.auto_predict = False

        self.camera = camera.Camera()

        self.delay = 15
        self.status_label = tk.Label(self.window, text="Status: Ready")
        self.init_gui()
        self.window.attributes("-topmost", True)
        self.update()

        self.window.mainloop()

    def class_name_frame(self):
        #  a frame to hold the class names if it doesn't already exist
        if not hasattr(self, 'class_names_frame'):
            self.class_names_frame = tk.LabelFrame(self.window, text="Class Names", font=("Arial", 12))
            self.class_names_frame.pack(side=tk.RIGHT, padx=10, pady=10, anchor=tk.NE, fill=tk.Y)

        self.class_names = []
        self.class_labels = []
        self.class_buttons = []

        for i in range(self.num_classes):
            class_name = simpledialog.askstring(f"Class {i + 1} Name", f"Enter the name of class {i + 1}:",
                                                parent=self.window)
            self.class_names.append(class_name)
            class_label = tk.Label(self.class_names_frame, text=class_name)
            class_label.config(font=("Arial", 12))
            class_label.pack(anchor=tk.W, padx=10, pady=5)
            self.class_labels.append(class_label)

            class_button = tk.Button(self.window, text=f"Add {class_name}", bg="#D0D0E9", width=50,
                                     command=lambda num=i: self.save_for_class(num))
            class_button.pack(anchor=tk.CENTER, expand=True)
            self.class_buttons.append(class_button)

    def init_gui(self):
        self.num_classes = simpledialog.askinteger("Number of Classes", "Enter the number of classes:")
        self.counters = [1] * self.num_classes
        if self.num_classes is None:
            self.num_classes = 2  # Default value

        # a frame for the camera feed
        camera_frame = tk.Frame(self.window)
        camera_frame.pack(side=tk.LEFT, padx=10, pady=10)

        # a label for the camera feed section
        camera_label = tk.Label(camera_frame, text="Camera Feed", font=("Arial", 14))
        camera_label.pack()

        self.camera = camera.Camera()

        self.canvas = tk.Canvas(camera_frame, width=self.camera.width, height=self.camera.height)
        self.canvas.pack()

        self.class_name_frame()


        self.btn_train = tk.Button(self.window, text="Train Model", bg="#D0D0E9", width=50, command=self.train_model)
        self.btn_train.pack(anchor=tk.CENTER, expand=True)

        self.btn_predict = tk.Button(self.window, text="Predict",bg="#D0D0E9", width=50, command=self.predict)
        self.btn_predict.pack(anchor=tk.CENTER, expand=True)

        self.btn_auto_predict = tk.Button(self.window, text="Auto Predict", bg="#D0D0E9",width=50, command=self.auto_predict_toggle)
        self.btn_auto_predict.pack(anchor=tk.CENTER, expand=True)

        self.btn_reset = tk.Button(self.window, text="Reset", bg="#D0D0E9",width=50, command=self.reset)
        self.btn_reset.pack(anchor=tk.CENTER, expand=True)

        self.class_label = tk.Label(self.window, text="CLASS")
        self.class_label.config(font=("Arial", 20))
        self.class_label.pack(anchor=tk.CENTER, expand=True)

    def auto_predict_toggle(self):
        self.auto_predict = not self.auto_predict

    def update(self):
        if self.auto_predict:
            print(self.predict())

        ret, frame = self.camera.get_frame()

        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(self.delay, self.update)

    def train_model(self):

        class_counters = [1] * self.num_classes
        self.model.train_model(class_counters)
        self.is_model_trained = True

    def predict(self):
        if not self.is_model_trained:
            messagebox.showerror("Error", "Model is not trained yet.")
            return

        frame = self.camera.get_frame()
        prediction = self.model.predict(frame)

        if prediction >= 0 and prediction <= self.num_classes - 1:
            predicted_class = self.class_names[prediction]
            self.class_label.config(text=predicted_class)
            return predicted_class
        else:
            # Clear the class label if no valid prediction is made
            self.class_label.config(text="CLASS")

    def reset(self):
        result = messagebox.askyesno("Confirmation", "Reset will delete all saved images. Continue?")
        if result:
            for class_num in range(self.num_classes):
                folder = str(class_num)
                if os.path.exists(folder):
                    for file in os.listdir(folder):
                        file_path = os.path.join(folder, file)
                        if os.path.isfile(file_path):
                            os.unlink(file_path)

            self.counters = [1] * self.num_classes
            self.model = model.Model(num_classes=self.num_classes)
            self.is_model_trained = False
            self.class_label.config(text="CLASS")

            self.class_label.config(text="CLASS")

            # Clear the class names as well
            self.class_names = []

            # Clear the existing class labels and buttons
            for class_label in self.class_labels:
                class_label.destroy()
            for class_button in self.class_buttons:
                class_button.destroy()

            self.class_name_frame()
            self.btn_train.config(state=tk.NORMAL)
            self.btn_predict.config(state=tk.NORMAL)

    def save_for_class(self, class_num):
        ret, frame = self.camera.get_frame()
        if not os.path.exists(str(class_num)):
            os.mkdir(str(class_num))

        file_path = f'{class_num}/frame{self.counters[class_num - 1]}.jpg'
        print("Saving image to:", file_path)

        cv.imwrite(file_path, frame)

        img = PIL.Image.open(file_path)
        img.thumbnail((150, 150), PIL.Image.ANTIALIAS)
        img.save(file_path)

        self.counters[class_num - 1] += 1
