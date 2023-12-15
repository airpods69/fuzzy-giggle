import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class LivePlotWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        layout = QGridLayout(self)

        # Create a Matplotlib figure and a canvas
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas, 0, 0, 1, 2)

        # Set up the initial plot
        self.x_values = np.linspace(0, 2 * np.pi, 100)
        self.y_sin = np.sin(self.x_values)
        self.y_cos = np.cos(self.x_values)

        self.sin_line, = self.ax.plot(self.x_values, self.y_sin, label='sin(x)')
        self.cos_line, = self.ax.plot(self.x_values, self.y_cos, label='cos(x)')

        self.ax.legend()

        # Set up animation
        self.animation = FuncAnimation(self.figure, self.update_plot, blit=False)
        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(20)  # Adjust the update interval as needed (e.g., 20 milliseconds)

        self.setGeometry(100, 100, 800, 600)
        self.setWindowTitle('Live Plot of Sin and Cos')
        self.show()

    def update_plot(self, frame):
        # Update data for live plotting
        self.x_values += 0.1
        self.y_sin = np.sin(self.x_values)
        self.y_cos = np.cos(self.x_values)

        # Update plot data
        self.sin_line.set_ydata(self.y_sin)
        self.cos_line.set_ydata(self.y_cos)

        # Adjust the plot limits if needed
        self.ax.relim()
        self.ax.autoscale_view()

        # Redraw the canvas
        self.canvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = LivePlotWidget()
    sys.exit(app.exec_())
