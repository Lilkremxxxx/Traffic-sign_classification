# Traffic-sign_classification
This project involves building a deep learning model using Convolutional Neural Networks (CNN) to recognize Vietnamese traffic signs. Developed with Python and TensorFlow, it processes both images and live camera input. The system displays the traffic sign’s name, the corresponding penalty based on vehicle type, and the model’s confidence level.

Dự án này xây dựng một mô hình học sâu sử dụng mạng nơ-ron tích chập (CNN) để nhận diện biển báo giao thông Việt Nam. Hệ thống được phát triển bằng Python và TensorFlow, có khả năng xử lý ảnh tĩnh và dữ liệu từ camera trực tiếp. Ứng dụng hiển thị tên biển báo, mức phạt theo loại phương tiện và độ tin cậy của mô hình.

Dự án này sử dụng Dataset: **GTSRB - German Traffic Sign Recognition Benchmark**, available on Kaggle:
[GTSRB - German Traffic Sign Recognition Benchmark](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)

**Provided by user**: meowmeowmeowmeowmeow.


# 🚦 Traffic Sign Recognition using CNN

Ứng dụng nhận diện **biển báo giao thông Việt Nam** bằng **mạng nơ-ron tích chập (CNN)**, hiển thị tên biển báo, mức phạt theo phương tiện (ô tô/xe máy) và độ tin cậy. Hỗ trợ cả ảnh tĩnh và nhận diện qua camera.

---

## 🧠 1. Mô tả dự án

- Huấn luyện mô hình CNN với 43 loại biển báo.
- Giao diện đồ họa (GUI) thân thiện bằng Tkinter.
- Hiển thị thông tin mức phạt tương ứng theo luật giao thông Việt Nam.
- Tùy chọn ảnh đầu vào hoặc sử dụng webcam.

---

## 🗂️ 2. Dataset

- Dataset: [GTSRB German Traffic Sign – Kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
- Gồm 43 lớp biển báo, định dạng ảnh `.ppm`, được resize về 30×30 px.

🖼️ **Hình minh họa 1: Một số ảnh từ tập dữ liệu**

![00006](https://github.com/user-attachments/assets/b2473d03-1478-4548-9cbf-8fec38e9b508) ![00009](https://github.com/user-attachments/assets/5be14d24-645f-420f-bb77-63646c3a934a)  ![00044](https://github.com/user-attachments/assets/6b39127c-a992-4731-8233-0db4cde5d214)



---

## 🧪 3. Cài đặt & Huấn luyện

### Yêu cầu cài đặt
```bash
pip install -r requirements.txt
```
### Huấn luyện mô hình
Chạy file huấn luyện:

```bash
python traffic_sign.py
```
Sau khi huấn luyện, mô hình được lưu tại my_model.h5.
---

## 🖥️ 4. Giao diện người dùng (GUI)

Chạy ứng dụng GUI:

```bash
python gui.py
```

Tính năng:
 + Tải ảnh hoặc sử dụng webcam.

 + Chọn loại phương tiện: Ô tô / Xe máy

 + Hiển thị: tên biển báo, mức phạt và độ chính xác.

🖼️ Hình minh họa 2: Giao diện chính khi sử dụng

![Screenshot 2025-06-30 131849](https://github.com/user-attachments/assets/51e01450-80bf-4b7a-b59d-693570d3350b)

![Screenshot 2025-06-30 132221](https://github.com/user-attachments/assets/2d7d0c51-eec1-4958-b663-a65593a4f9af)


---
## 📌 5. Ví dụ kết quả

**Biển báo 1: Giới hạn tốc độ (30km/h)**

Phạt ô tô  : 4–6 triệu VNĐ nếu vi phạm

Phạt xe máy: 800k–1 triệu VNĐ nếu vi phạm

Độ tin cậy: 100%



**Biển báo 2: Vòng xuyến**

Phạt oto - xe máy: 400-600k VNĐ nếu không tuân thủ biển

Độ tin cậy: 100%

---
## 📝Ghi chú thêm

Dễ dàng mở rộng thêm loại biển báo Việt Nam.


Có thể tích hợp thêm camera giám sát hoặc thiết bị nhúng.


Mọi mức phạt mang tính tham khảo từ luật GT Việt Nam (2024–2025).


---
## 👏 Cảm ơn

Đây là dự án học thuật mang tính minh họa. Dataset sử dụng từ nguồn công khai trên Kaggle:
[GTSRB - German Traffic Sign Recognition Benchmark](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)



