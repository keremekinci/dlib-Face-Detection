import sys
import cv2
import dlib
import os
import time
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QFileDialog, QLineEdit, QLabel, \
    QPushButton, QProgressBar, QTextEdit
from PyQt5.QtCore import Qt, QThread, pyqtSignal

class VideoProcessorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Processor")
        self.setGeometry(100, 100, 800, 600)

        self.video_path = ""
        self.output_path = ""
        self.model_output_path = ""

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()

        self.video_label = QLabel("Video Path:")
        self.video_line_edit = QLineEdit(self)
        self.layout.addWidget(self.video_label)
        self.layout.addWidget(self.video_line_edit)

        self.browse_input_button = QPushButton("Browse Input")
        self.browse_input_button.clicked.connect(self.browse_input)
        self.layout.addWidget(self.browse_input_button)

        self.saniye_label = QLabel("Saniye (Default: 1):")
        self.saniye_line_edit = QLineEdit(self)
        self.saniye_line_edit.setText("1")
        self.layout.addWidget(self.saniye_label)
        self.layout.addWidget(self.saniye_line_edit)

        self.frame_label = QLabel("Frame (Default: 1):")
        self.frame_line_edit = QLineEdit(self)
        self.frame_line_edit.setText("1")
        self.layout.addWidget(self.frame_label)
        self.layout.addWidget(self.frame_line_edit)

        self.start_button = QPushButton("Start Processing")
        self.start_button.clicked.connect(self.start_processing)
        self.layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop Processing")
        self.stop_button.clicked.connect(self.stop_processing)
        self.layout.addWidget(self.stop_button)
        self.stop_button.setEnabled(False)

        self.browse_output_button = QPushButton("Browse Output")
        self.browse_output_button.clicked.connect(self.browse_output)
        self.layout.addWidget(self.browse_output_button)

        self.elapsed_time_label = QLabel("Elapsed Time (s):")
        self.layout.addWidget(self.elapsed_time_label)

        self.face_count_label = QLabel("Total Face Count:")
        self.layout.addWidget(self.face_count_label)

        self.result_text = QTextEdit(self)
        self.layout.addWidget(self.result_text)

        self.central_widget.setLayout(self.layout)

        self.worker_thread = None
        self.model_thread = None
        self.stop_flag = False

    def browse_input(self):
        options = QFileDialog.Options()
        self.video_path, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mov *.avi *.mp4 *.mkv);;All Files (*)", options=options)
        self.video_line_edit.setText(self.video_path)

    def browse_output(self):
        options = QFileDialog.Options()
        self.output_path = QFileDialog.getExistingDirectory(self, "Select Output Directory", options=options)
        if self.output_path:
            self.result_text.clear()
            self.result_text.append(f"Output directory: {self.output_path}")

    def start_processing(self):
        self.video_path = self.video_line_edit.text()
        self.saniye = int(self.saniye_line_edit.text())
        self.frame = int(self.frame_line_edit.text())

        if not self.video_path:
            self.result_text.clear()
            self.result_text.append("Please select a video file.")
            return

        if not self.output_path:
            self.result_text.clear()
            self.result_text.append("Please select an output directory.")
            return

        self.face_count_label.setText("Total Face Count: Calculating...")

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.stop_flag = False

        self.worker_thread = VideoProcessingThread(self.video_path, self.saniye, self.frame, self.output_path)
        self.worker_thread.finished.connect(self.video_processing_finished)
        self.worker_thread.start()

    def stop_processing(self):
        if self.worker_thread:
            self.worker_thread.stop()
            self.stop_flag = True

    def video_processing_finished(self, elapsed_time, face_count):
        self.elapsed_time_label.setText(f"Elapsed Time (s): {elapsed_time:.2f}")
        self.face_count_label.setText(f"Total Face Count: {face_count}")

        if not self.stop_flag:
            self.model_thread = ModelProcessingThread(self.output_path)
            self.model_thread.finished.connect(self.model_processing_finished)
            self.model_thread.start()
        else:
            self.result_text.clear()
            self.result_text.append("Video processing stopped.")
            self.result_text.append(f"Total time: {elapsed_time:.2f} seconds.")

    def model_processing_finished(self, face_counts):
        self.result_text.clear()
        self.result_text.append("Face Counts:")
        self.result_text.append("Saniye - Dlib - OpenCV")
        for saniye, dlib_count, opencv_count in face_counts:
            self.result_text.append(f"{saniye} - {dlib_count}")

class VideoProcessingThread(QThread):
    finished = pyqtSignal(float, int)

    def __init__(self, video_path, saniye, frame, output_path):
        super().__init__()
        self.video_path = video_path
        self.saniye = saniye
        self.frame = frame
        self.output_path = output_path
        self.stop_flag = False

    def run(self):
        start_time = time.time()
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Error: Video file could not be opened.")
            return

        detector_dlib = dlib.get_frontal_face_detector()
        face_count = 0

        frame_rate = int(cap.get(5))
        frame_interval = self.saniye * frame_rate
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                if frame_count % frame_interval == 0:
                    faces_dlib = detector_dlib(frame)

                    saniye = frame_count // frame_rate
                    dlib_output_folder = os.path.join(self.output_path, f"Dlib-saniye{saniye:04d}")
                    os.makedirs(dlib_output_folder, exist_ok=True)

                    for i, face in enumerate(faces_dlib):
                        x, y, w, h = face.left(), face.top(), face.width(), face.height()
                        face_roi = frame[y:y + h, x:x + w]
                        cv2.imwrite(os.path.join(dlib_output_folder, f"face-{i + 1:03d}.jpg"), face_roi)
                        face_count += 1

            frame_count += 1

            if self.stop_flag:
                break

        cap.release()
        cv2.destroyAllWindows()
        end_time = time.time()
        elapsed_time = end_time - start_time
        self.finished.emit(elapsed_time, face_count)

    def stop(self):
        self.stop_flag = True

class ModelProcessingThread(QThread):
    finished = pyqtSignal(list)

    def __init__(self, output_path):
        super().__init__()
        self.output_path = output_path

    def run(self):
        face_counts = []

        for folder in os.listdir(self.output_path):
            if folder.startswith("Dlib-saniye"):
                saniye = int(folder.split("saniye")[1])
                dlib_count = len([name for name in os.listdir(os.path.join(self.output_path, folder)) if name.startswith("face-")])
                face_counts.append((saniye, dlib_count))

        self.finished.emit(face_counts)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VideoProcessorApp()
    window.show()
    sys.exit(app.exec_())
