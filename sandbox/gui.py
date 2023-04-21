import sys
import time

import numpy as np
import PIL
from PyQt5.QtWidgets import QApplication, QMainWindow, QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class ImageViewer(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()

        self.last_draw_time = 0
        self.draw_interval = 0.1

        image = PIL.Image.open("/Users/nrahaman/Python/segment-anything/test_images/airport.png")
        image = image.convert("RGB")
        image = np.asarray(image)
        self.image = image
        self.mask = np.random.uniform(0, 1, size=image.shape)

        self.ax.imshow(self.image)
        self.ax.imshow(self.mask, cmap='jet', alpha=0.5)
        self.ax.axis('off')

        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)

        self.panning = False
        self.x0, self.y0 = 0, 0

    def on_mouse_press(self, event):
        if event.button == 1:  # Left mouse button
            self.panning = True
            self.x0, self.y0 = event.xdata, event.ydata

    def on_mouse_release(self, event):
        if event.button == 1:  # Left mouse button
            self.panning = False

    def on_mouse_move(self, event):
        if self.panning and event.xdata is not None and event.ydata is not None:
            dx = event.xdata - self.x0
            dy = self.y0 - event.ydata
            self.x0, self.y0 = event.xdata, event.ydata
            self.ax.set_xlim(self.ax.get_xlim() - dx)
            self.ax.set_ylim(self.ax.get_ylim() + dy)

            current_time = time.time()
            if current_time - self.last_draw_time >= self.draw_interval:
                self.draw()
                self.last_draw_time = current_time

    def on_scroll(self, event):
        zoom_factor = 1.1 if event.button == "up" else 1 / 1.1
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        x_range = (xlim[1] - xlim[0]) * zoom_factor
        y_range = (ylim[1] - ylim[0]) * zoom_factor

        x_center = (xlim[0] + xlim[1]) / 2
        y_center = (ylim[0] + ylim[1]) / 2

        new_xlim = (x_center - x_range / 2, x_center + x_range / 2)
        new_ylim = (y_center - y_range / 2, y_center + y_range / 2)

        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        self.draw()


app = QApplication(sys.argv)
main_window = QMainWindow()
image_viewer = ImageViewer()
main_window.setCentralWidget(image_viewer)
main_window.show()
sys.exit(app.exec_())
