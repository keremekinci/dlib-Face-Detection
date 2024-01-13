import cv2
import dlib
import os
import time

def detect_faces_in_images(input_folder,output_folder_path,faces_folder_path):
    detector_dlib = dlib.get_frontal_face_detector()
    start = time.time()
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_path = os.path.join(root, file)
                frame = cv2.imread(image_path)
                frame_copy = frame.copy()

                faces_dlib = detector_dlib(frame)
                if len(faces_dlib) > 0:
                    output_folder = os.path.join(output_folder_path, os.path.splitext(file)[0])
                    os.makedirs(output_folder, exist_ok=True)
                    faces_folder = os.path.join(faces_folder_path, os.path.splitext(file)[0]+" Faces "+str(len(faces_dlib)))
                    os.makedirs(faces_folder, exist_ok=True)

                    for i, face in enumerate(faces_dlib):
                        x, y, w, h = face.left(), face.top(), face.width(), face.height()
                        face_roi = frame[y:y + h, x:x + w]
                        output_filename= os.path.join(output_folder, f"{os.path.splitext(file)[0]}.png")
                        face_filename = os.path.join(faces_folder, f"face-{i + 1}.png")
                        cv2.imwrite(face_filename, face_roi)  # Save the modified image with rectangles drawn around faces
                        cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0),2)
                    cv2.imwrite(output_filename, frame_copy)
    print(time.time() - start)
    print("Face detection and annotation completed.")

if __name__ == "__main__":
    input_folder_path = 'inputs'  # Replace 'path_to_input_folder' with the path to your input folder
    output_folder_path = "outputs dlib"
    faces_folder_path = "faces dlib"
    detect_faces_in_images(input_folder_path,output_folder_path,faces_folder_path)
