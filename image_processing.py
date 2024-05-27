import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QSlider, QGridLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from collections import deque
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
original_image = None  # Define original_image globally
processed_images = deque()  # Store processed images history
redo_images = deque()  # Store images for redo

class ImageProcessingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Processing")
        self.init_ui()
    
    def init_ui(self):
        # Set minimum window size
        self.setMinimumSize(1200, 800)

        # Create buttons
        load_button = QPushButton("Load Image")
        load_button.clicked.connect(self.load_image)

        undo_button = QPushButton("Undo")
        undo_button.clicked.connect(self.undo)

        redo_button = QPushButton("Redo")
        redo_button.clicked.connect(self.redo)

        reset_button = QPushButton("Reset")
        reset_button.clicked.connect(self.reset)

        save_button = QPushButton("Save Image")
        save_button.clicked.connect(self.save_image)

        # Create sliders and apply buttons side by side
        noise_layout = self.create_slider_with_button("Noise Reduction", self.apply_noise_reduction)
        smoothing_layout = self.create_slider_with_button("Smoothing", self.apply_smoothing)
        low_pass_layout = self.create_slider_with_button("Low Pass Filtering", self.apply_low_pass_filter)
        high_pass_layout = self.create_button("High Pass Filtering", self.apply_high_pass_filter)
        contrast_stretching_layout = self.create_double_slider_with_button("Contrast Stretching", self.apply_contrast_stretching)
        hist_equal_layout = self.create_button("Histogram Equalization (All)", self.apply_histogram_equalization)
        hist_equal_rgb_layout = self.create_button("Histogram Equalization (RGB)", self.apply_histogram_equalization_rgb)

        # Create image panel
        self.panel = QLabel()
        self.panel.setAlignment(Qt.AlignCenter)

        # Create histogram panels
        self.hist_panel_r = QLabel()
        self.hist_panel_g = QLabel()
        self.hist_panel_b = QLabel()

        self.hist_panel_r.setAlignment(Qt.AlignCenter)
        self.hist_panel_g.setAlignment(Qt.AlignCenter)
        self.hist_panel_b.setAlignment(Qt.AlignCenter)

        # Layout setup
        button_layout = QHBoxLayout()
        button_layout.addWidget(load_button)
        button_layout.addWidget(undo_button)
        button_layout.addWidget(redo_button)
        button_layout.addWidget(reset_button)
        button_layout.addWidget(save_button)

        filter_button_layout = QVBoxLayout()
        filter_button_layout.addLayout(noise_layout)
        filter_button_layout.addLayout(smoothing_layout)
        filter_button_layout.addLayout(low_pass_layout)
        filter_button_layout.addLayout(high_pass_layout)
        filter_button_layout.addLayout(contrast_stretching_layout)
        filter_button_layout.addLayout(hist_equal_layout)
        filter_button_layout.addLayout(hist_equal_rgb_layout)

        hist_layout = QGridLayout()
        hist_layout.addWidget(QLabel("Red Channel Histogram"), 0, 0)
        hist_layout.addWidget(QLabel("Green Channel Histogram"), 0, 1)
        hist_layout.addWidget(QLabel("Blue Channel Histogram"), 0, 2)
        hist_layout.addWidget(self.hist_panel_r, 1, 0)
        hist_layout.addWidget(self.hist_panel_g, 1, 1)
        hist_layout.addWidget(self.hist_panel_b, 1, 2)

        main_layout = QVBoxLayout()
        main_layout.addLayout(button_layout)
        main_layout.addLayout(filter_button_layout)
        main_layout.addWidget(self.panel)
        main_layout.addLayout(hist_layout)

        self.setLayout(main_layout)

        # Apply dark mode stylesheet
        self.setStyleSheet("""
            QWidget {
                background-color: #333;
                color: white;
            }
            QPushButton {
                background-color: #666;
                border: 1px solid #555;
                color: white;
            }
            QSlider {
                background-color: #666;
                color: white;
            }
            QLabel {
                color: white;
            }
        """)

    def create_slider_with_button(self, label, apply_function):
        layout = QHBoxLayout()
        layout.addWidget(QLabel(label))
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(1)
        slider.setMaximum(100)
        slider.setValue(50)
        layout.addWidget(slider)
        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(apply_function)
        layout.addWidget(apply_button)
        return layout

    def create_double_slider_with_button(self, label, apply_function):
        layout = QHBoxLayout()
        layout.addWidget(QLabel(label))
        slider1 = QSlider(Qt.Horizontal)
        slider1.setMinimum(0)
        slider1.setMaximum(1000)
        slider1.setValue(0)
        layout.addWidget(slider1)
        slider2 = QSlider(Qt.Horizontal)
        slider2.setMinimum(0)
        slider2.setMaximum(1000)
        slider2.setValue(100)
        layout.addWidget(slider2)
        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(apply_function)
        layout.addWidget(apply_button)
        return layout

    def create_button(self, label, apply_function):
        layout = QHBoxLayout()
        apply_button = QPushButton(label)
        apply_button.clicked.connect(apply_function)
        layout.addWidget(apply_button)
        return layout

    def load_image(self):
        global original_image, processed_images
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if path:
            original_image = cv2.imread(path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            processed_images.append(original_image.copy())
            self.display_image(original_image)
            self.display_histograms(original_image)

    def display_image(self, image):
        image = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        pixmap = pixmap.scaled(800, 800, Qt.KeepAspectRatio)
        self.panel.setPixmap(pixmap)

    def display_histograms(self, image):
        channels = ('r', 'g', 'b')
        for i, col in enumerate(channels):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            plt.figure()
            plt.title(f'{col.upper()} Channel Histogram')
            plt.xlabel('Intensity Value')
            plt.ylabel('Count')
            plt.plot(hist, color=col)
            plt.xlim([0, 256])

            # Save the histogram as a QPixmap
            plt.savefig(f'{col}_hist.png')
            plt.close()

            pixmap = QPixmap(f'{col}_hist.png')
            if col == 'r':
                self.hist_panel_r.setPixmap(pixmap.scaled(350, 350, Qt.KeepAspectRatio))
            elif col == 'g':
                self.hist_panel_g.setPixmap(pixmap.scaled(350, 350, Qt.KeepAspectRatio))
            elif col == 'b':
                self.hist_panel_b.setPixmap(pixmap.scaled(350, 350, Qt.KeepAspectRatio))

    def apply_noise_reduction(self):
        kernel_size = self.sender().parent().findChildren(QSlider)[0].value()
        self.apply_image_processing(self.process_noise_reduction, kernel_size)

    def apply_smoothing(self):
        kernel_size = self.sender().parent().findChildren(QSlider)[0].value()
        self.apply_image_processing(self.process_smoothing, kernel_size)

    def apply_low_pass_filter(self):
        kernel_size = self.sender().parent().findChildren(QSlider)[0].value()
        self.apply_image_processing(self.process_low_pass_filter, kernel_size)

    def apply_high_pass_filter(self):
        kernel_size = self.sender().parent().findChildren(QSlider)[0].value()
        self.apply_image_processing(self.process_high_pass_filter, kernel_size)

    def apply_contrast_stretching(self):
        sliders = self.sender().parent().findChildren(QSlider)
        lower_percentile = sliders[0].value()
        upper_percentile = sliders[1].value()
        self.apply_image_processing(self.process_contrast_stretching, lower_percentile, upper_percentile)

    def apply_histogram_equalization(self):
        self.apply_image_processing(self.process_histogram_equalization)

    def apply_histogram_equalization_rgb(self):
        equalize_level = self.sender().parent().findChildren(QSlider)[0].value()
        self.apply_image_processing(self.process_histogram_equalization_rgb, equalize_level)

    def apply_image_processing(self, process_function, *args):
        global original_image, processed_images
        if original_image is not None:
            processed_image = processed_images[-1].copy()
            processed_image = process_function(processed_image, *args)
            processed_images.append(processed_image.copy())
            self.display_image(processed_image)
            self.display_histograms(processed_image)

    def process_noise_reduction(self, image, kernel_size):
        kernel_size = max(1, kernel_size)
        kernel_size = kernel_size if kernel_size % 2 != 0 else kernel_size + 1
        return cv2.medianBlur(image, kernel_size)

    def process_smoothing(self, image, kernel_size):
        kernel_size = max(1, kernel_size)
        kernel_size = kernel_size if kernel_size % 2 != 0 else kernel_size + 1
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    def process_low_pass_filter(self, image, kernel_size):
        kernel_size = max(1, kernel_size)
        kernel_size = kernel_size if kernel_size % 2 != 0 else kernel_size + 1
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        return cv2.filter2D(image, -1, kernel)

    def process_high_pass_filter(self, image, kernel_size):
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        return cv2.convertScaleAbs(laplacian)

    def process_contrast_stretching(self, image, lower_percentile, upper_percentile):
        lower_percentile = max(0, min(100, lower_percentile))
        upper_percentile = max(lower_percentile, min(100, upper_percentile))

        channels = cv2.split(image)
        stretched_channels = []
        for channel in channels:
            lower_val = np.percentile(channel, lower_percentile)
            upper_val = np.percentile(channel, upper_percentile)
            if upper_val - lower_val == 0:
                stretched_channels.append(channel)
            else:
                stretched_channel = np.clip((channel - lower_val) * (255 / (upper_val - lower_val)), 0, 255)
                stretched_channels.append(stretched_channel.astype(np.uint8))

        return cv2.merge(stretched_channels)

    def process_histogram_equalization(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        equalized = cv2.equalizeHist(gray_image)
        return cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)

    def process_histogram_equalization_rgb(self, image, equalize_level):
        equalize_level = max(1, equalize_level)
        channels = cv2.split(image)
        equalized_channels = [cv2.equalizeHist(channel) for channel in channels]
        return cv2.merge(equalized_channels)

    def undo(self):
        global processed_images, redo_images
        if len(processed_images) > 1:
            redo_images.append(processed_images.pop())
            self.display_image(processed_images[-1])
            self.display_histograms(processed_images[-1])

    def redo(self):
        global processed_images, redo_images
        if redo_images:
            processed_images.append(redo_images.pop())
            self.display_image(processed_images[-1])
            self.display_histograms(processed_images[-1])

    def reset(self):
        global original_image, processed_images, redo_images
        if processed_images:
            original_image = processed_images[0].copy()
            processed_images.clear()
            redo_images.clear()
            processed_images.append(original_image.copy())
            self.display_image(original_image)
            self.display_histograms(original_image)

    def save_image(self):
        global processed_images
        if processed_images:
            processed_image = processed_images[-1]
            path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "JPEG Image Files (*.jpg)")
            if path:
                cv2.imwrite(path, cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    app = QApplication([])
    window = ImageProcessingApp()
    window.show()
    app.exec_()
