import sys

from PyQt6.QtWidgets import QVBoxLayout, QApplication, QHBoxLayout, QWidget, QPushButton

class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('ex')
        layout = QVBoxLayout()
        layout.addWidget(QPushButton('asdasd'))
        layout.addWidget(QPushButton('center'))
        layout.addWidget(QPushButton('right'))
        self.setLayout(layout)
        print(self.children())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec())