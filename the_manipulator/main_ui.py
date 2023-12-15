import random
import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QPushButton, QLineEdit
import pyqtgraph as pg

class LivePlotWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()
        self.count = 1

    def init_ui(self):
        layout = QGridLayout(self)
        self.script = [0, 0, 1, 3, 2, 1]

        # Create a PyQtGraph plot widget
        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)

        # Set up the initial plot
        self.x_values = np.linspace(0, 1, 1000)

        self.channels = []
        for i in range(0,14):
            # self.channels.append(np.random.rand(self.x_values.shape[0]))
            self.channels.append(np.zeros(self.x_values.shape[0]))


        colours = ['b', 'g', 'r', 'y']
        self.ind = 0
        self.direction_count = {"Left" : 0, "Forward" : 0, "Backward" : 0, "Right" : 0}


        self.plots = []
        for i in range(0, 14):
            self.plots.append(self.plot_widget.plot(self.x_values, i + self.channels[i], pen=colours[i % 4], name='random'))


        # Set up animation with a faster update interval
        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        # self.timer.start(40)  # Adjust the update interval as needed (e.g., 20 milliseconds)

        # Add a button to start/stop live plot
        self.start_stop_button = QPushButton('Start Live Plot')
        self.start_stop_button.clicked.connect(self.toggle_live_plot)
        layout.addWidget(self.start_stop_button, 1, 0, 1, 1)

        self.text_box = QLineEdit('Hello, World!')
        layout.addWidget(self.text_box, 1, 1, 1, 1)


        self.setGeometry(100, 100, 800, 600)
        self.setWindowTitle('Live Plot of EEG')
        self.show()

    def stop_plot(self):
        self.timer.stop()
        self.start_stop_button.setText('Start')
        self.check = False
        self.count += 1

    def start_plot(self):
        self.timer.start(20)  # Adjust the update interval as needed (e.g., 20 milliseconds)
        self.start_stop_button.setText('Stop')


    def toggle_live_plot(self):
        if not self.timer.isActive():
            self.start_plot()
            # self.timer.start(20)  # Adjust the update interval as needed (e.g., 20 milliseconds)
            # self.start_stop_button.setText('Stop')
        else:
            self.stop_plot()
            # self.timer.stop()
            # self.start_stop_button.setText('Start')

    def direction_text(self, direction):
        """
        0 -> Forward
        1 -> Right
        2 -> Left
        3 -> Backward
        """
        direc = ["Forward", "Right", "Left", "Backward"]
        return direc[random.choices([direction, random.choices([0, 1, 2, 3])[0]], weights=(0.80, 0.30))[0]]

    def update_text(self):
        direction = max(self.direction_count, key= lambda x: self.direction_count[x]) 
        self.text_box.setText(f"{direction}")
        self.direction_count = {"Left" : 0, "Forward" : 0, "Backward" : 0, "Right" : 0}

    def update_plot(self):
        # Update data for live plotting
        self.count += 0.1
        self.x_values += self.count
        self.check = True if int(self.count) % 8 == 0 else False

        for i in range(0,14):
            self.channels[i] = np.random.rand(self.x_values.shape[0]) + i

        for i in range(0, 14):
            self.plots[i].setData(self.x_values, self.channels[i])

        self.direction_count[self.direction_text(self.script[self.ind])] += 1
        self.update_text()

        if self.check:
            self.stop_plot()
            self.ind += 1

        # Update plot data
        # self.rand_curve.setData(self.x_values, self.y_cos1)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = LivePlotWidget()
    sys.exit(app.exec_())
