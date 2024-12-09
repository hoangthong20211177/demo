import os
import cv2
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from tkinter import filedialog, Tk

# Đường dẫn tới thư mục chứa ảnh
image_dir = "C:\\iris\\dongvat"

def load_images_and_labels(image_dir):
    images = []
    labels = []
    label_map = {label: idx for idx, label in enumerate(os.listdir(image_dir))}  # Ánh xạ nhãn với số nguyên
    
    for label_name, label_idx in label_map.items():
        label_path = os.path.join(image_dir, label_name)
        if os.path.isdir(label_path):
            for filename in os.listdir(label_path):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    img_path = os.path.join(label_path, filename)
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Lỗi khi đọc ảnh: {filename}")
                        continue
                    img = cv2.resize(img, (64, 64))  # Resize ảnh về kích thước chuẩn
                    images.append(img)
                    labels.append(label_idx)
    
    if not images:
        raise ValueError("Không tìm thấy ảnh hợp lệ trong thư mục.")
    
    images = np.array(images)
    labels = np.array(labels)
    return images, labels


# Đọc dữ liệu
X, y = load_images_and_labels(image_dir)
X = X / 255.0  # Chuẩn hóa ảnh
X = X.reshape(X.shape[0], -1)  # Chuyển ảnh thành vector 1D

# Chia tập dữ liệu thành train và test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN Classifier
start_time = time.time()
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
knn_accuracy = accuracy_score(y_test, y_pred_knn)
knn_time = time.time() - start_time
print(f"KNN Accuracy: {knn_accuracy:.2f}, Time: {knn_time:.2f} seconds")

# SVM Classifier
start_time = time.time()
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
svm_accuracy = accuracy_score(y_test, y_pred_svm)
svm_time = time.time() - start_time
print(f"SVM Accuracy: {svm_accuracy:.2f}, Time: {svm_time:.2f} seconds")

# ANN Classifier
X_train_ann = X_train.reshape(-1, 64, 64, 3)
X_test_ann = X_test.reshape(-1, 64, 64, 3)
y_train_cat = to_categorical(y_train, num_classes=len(np.unique(y)))
y_test_cat = to_categorical(y_test, num_classes=len(np.unique(y)))

start_time = time.time()
ann = Sequential([
    Flatten(input_shape=(64, 64, 3)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(len(np.unique(y)), activation='softmax')  # Số lớp phải khớp với số nhãn
])
ann.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
ann.fit(X_train_ann, y_train_cat, epochs=10, verbose=0)

_, ann_accuracy = ann.evaluate(X_test_ann, y_test_cat, verbose=0)
ann_time = time.time() - start_time
print(f"ANN Accuracy: {ann_accuracy:.2f}, Time: {ann_time:.2f} seconds")

# Chức năng chọn ảnh và nhận dạng
def choose_and_predict_image(model_choice='knn'):
    # Khởi tạo cửa sổ chọn ảnh
    root = Tk()
    root.withdraw()  # Ẩn cửa sổ chính của Tkinter
    img_path = filedialog.askopenfilename(title="Chọn ảnh", filetypes=[("Image files", "*.jpg;*.png")])
    
    if img_path:
        # Đọc và tiền xử lý ảnh
        img = cv2.imread(img_path)
        if img is None:
            print("Lỗi khi đọc ảnh.")
            return
        img_resized = cv2.resize(img, (64, 64))  # Resize ảnh
        img_normalized = img_resized / 255.0  # Chuẩn hóa
        img_vector = img_normalized.reshape(1, -1)  # Chuyển ảnh thành vector 1D

        # Dự đoán với các mô hình
        if model_choice == 'knn':
            pred = knn.predict(img_vector)
        elif model_choice == 'svm':
            pred = svm.predict(img_vector)
        elif model_choice == 'ann':
            img_reshaped = img_resized.reshape(1, 64, 64, 3)
            pred = ann.predict(img_reshaped)
            pred = np.argmax(pred, axis=1)  # Chuyển kết quả dự đoán thành nhãn

        # Hiển thị kết quả
        label_map = {label: idx for idx, label in enumerate(os.listdir(image_dir))}
        reverse_label_map = {v: k for k, v in label_map.items()}
        print(f"Dự đoán: {reverse_label_map[pred[0]]}")
    else:
        print("Không chọn ảnh.")

# Gọi hàm để chọn ảnh và nhận dạng
choose_and_predict_image('knn')  # Bạn có thể thay 'knn' bằng 'svm' hoặc 'ann' để thử các mô hình khác
