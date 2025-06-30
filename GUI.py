import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import ImageTk, Image
import cv2
import numpy as np
from keras.models import load_model

# Load model và danh sách biển báo
model = load_model('my_model.h5')
classes = {
    1: 'Giới hạn tốc độ (20km/h)', 
    2: 'Giới hạn tốc độ (30km/h)', 
    3: 'Giới hạn tốc độ (50km/h)',
    4: 'Giới hạn tốc độ (60km/h)', 
    5: 'Giới hạn tốc độ (70km/h)', 
    6: 'Giới hạn tốc độ (80km/h)',
    7: 'Hết giới hạn tốc độ (80km/h)', 
    8: 'Giới hạn tốc độ (100km/h)', 
    9: 'Giới hạn tốc độ (120km/h)',
    10: 'Cấm vượt', 
    11: 'Cấm vượt xe trên 3.5 tấn', 
    12: 'Giao nhau với đường không ưu tiên',
    13: 'Đường ưu tiên', 
    14: 'Nhường đường', 
    15: 'Dừng lại', 
    16: 'Cấm xe', 
    17: 'Cấm xe trên 3.5 tấn',
    18: 'Cấm vào', 
    19: 'Chú ý chung', 
    20: 'Khúc cua nguy hiểm bên trái', 
    21: 'Khúc cua nguy hiểm bên phải',
    22: 'Khúc cua đôi', 
    23: 'Đường gồ ghề', 
    24: 'Đường trơn trượt', 
    25: 'Đường hẹp bên phải',
    26: 'Công trường đang thi công', 
    27: 'Tín hiệu giao thông', 
    28: 'Người đi bộ', 
    29: 'Trẻ em qua đường',
    30: 'Xe đạp qua đường', 
    31: 'Cảnh báo băng/tuyết', 
    32: 'Động vật hoang dã qua đường',
    33: 'Hết giới hạn tốc độ + Hạn chế vượt', 
    34: 'Rẽ phải phía trước', 
    35: 'Rẽ trái phía trước',
    36: 'Chỉ đi thẳng', 
    37: 'Đi thẳng hoặc rẽ phải', 
    38: 'Đi thẳng hoặc rẽ trái', 
    39: 'Đi tiếp bên phải',
    40: 'Đi tiếp bên trái', 
    41: 'Vòng xuyến', 
    42: 'Hết cấm vượt', 
    43: 'Hết cấm vượt đối với xe trên 3,5'
}

# Định nghĩa mức phạt dưới dạng danh sách nhiều chuỗi
fines = {
    'Giới hạn tốc độ (20km/h)': {'car': ['Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng nếu vượt quá tốc độ quy định. Nếu gây tai nạn giao thông, mức phạt sẽ tăng lên từ 20.000.000 đồng đến 22.000.000 đồng'], 'motorbike': ['Phạt tiền từ 800.000 đồng đến 1.000.000 đồng nếu vượt quá tốc độ quy định. Nếu gây tai nạn giao thông, mức phạt sẽ tăng lên từ 10.000.000 đồng đến 14.000.000 đồng']},
    'Giới hạn tốc độ (30km/h)': {'car': ['Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng nếu vượt quá tốc độ quy định. Nếu gây tai nạn giao thông, mức phạt sẽ tăng lên từ 20.000.000 đồng đến 22.000.000 đồng'], 'motorbike': ['Phạt tiền từ 800.000 đồng đến 1.000.000 đồng nếu vượt quá tốc độ quy định. Nếu gây tai nạn giao thông, mức phạt sẽ tăng lên từ 10.000.000 đồng đến 14.000.000 đồng']},
    'Giới hạn tốc độ (50km/h)': {'car': ['Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng nếu vượt quá tốc độ quy định. Nếu gây tai nạn giao thông, mức phạt sẽ tăng lên từ 20.000.000 đồng đến 22.000.000 đồng'], 'motorbike': ['Phạt tiền từ 800.000 đồng đến 1.000.000 đồng nếu vượt quá tốc độ quy định. Nếu gây tai nạn giao thông, mức phạt sẽ tăng lên từ 10.000.000 đồng đến 14.000.000 đồng']},
    'Giới hạn tốc độ (60km/h)': {'car': ['Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng nếu vượt quá tốc độ quy định. Nếu gây tai nạn giao thông, mức phạt sẽ tăng lên từ 20.000.000 đồng đến 22.000.000 đồng'], 'motorbike': ['Phạt tiền từ 800.000 đồng đến 1.000.000 đồng nếu vượt quá tốc độ quy định. Nếu gây tai nạn giao thông, mức phạt sẽ tăng lên từ 10.000.000 đồng đến 14.000.000 đồng']},
    'Giới hạn tốc độ (70km/h)': {'car': ['Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng nếu vượt quá tốc độ quy định. Nếu gây tai nạn giao thông, mức phạt sẽ tăng lên từ 20.000.000 đồng đến 22.000.000 đồng'], 'motorbike': ['Phạt tiền từ 800.000 đồng đến 1.000.000 đồng nếu vượt quá tốc độ quy định. Nếu gây tai nạn giao thông, mức phạt sẽ tăng lên từ 10.000.000 đồng đến 14.000.000 đồng']},
    'Giới hạn tốc độ (80km/h)': {'car': ['Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng nếu vượt quá tốc độ quy định. Nếu gây tai nạn giao thông, mức phạt sẽ tăng lên từ 20.000.000 đồng đến 22.000.000 đồng'], 'motorbike': ['Phạt tiền từ 800.000 đồng đến 1.000.000 đồng nếu vượt quá tốc độ quy định. Nếu gây tai nạn giao thông, mức phạt sẽ tăng lên từ 10.000.000 đồng đến 14.000.000 đồng']},
    'Hết giới hạn tốc độ (80km/h)': {'car': ['Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng nếu vượt quá tốc độ quy định. Nếu gây tai nạn giao thông, mức phạt sẽ tăng lên từ 20.000.000 đồng đến 22.000.000 đồng'], 'motorbike': ['Phạt tiền từ 800.000 đồng đến 1.000.000 đồng nếu vượt quá tốc độ quy định. Nếu gây tai nạn giao thông, mức phạt sẽ tăng lên từ 10.000.000 đồng đến 14.000.000 đồng']},
    'Giới hạn tốc độ (100km/h)': {'car': ['Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng nếu vượt quá tốc độ quy định. Nếu gây tai nạn giao thông, mức phạt sẽ tăng lên từ 20.000.000 đồng đến 22.000.000 đồng'], 'motorbike': ['Phạt tiền từ 800.000 đồng đến 1.000.000 đồng nếu vượt quá tốc độ quy định. Nếu gây tai nạn giao thông, mức phạt sẽ tăng lên từ 10.000.000 đồng đến 14.000.000 đồng']},
    'Giới hạn tốc độ (120km/h)': {'car': ['Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng nếu vượt quá tốc độ quy định. Nếu gây tai nạn giao thông, mức phạt sẽ tăng lên từ 20.000.000 đồng đến 22.000.000 đồng'], 'motorbike': ['Phạt tiền từ 800.000 đồng đến 1.000.000 đồng nếu vượt quá tốc độ quy định. Nếu gây tai nạn giao thông, mức phạt sẽ tăng lên từ 10.000.000 đồng đến 14.000.000 đồng']},
    'Cấm vượt': {'car': ['Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng. Nếu gây tai nạn giao thông, mức phạt sẽ tăng lên từ 20.000.000 đồng đến 22.000.000 đồng'], 'motorbike': ['Phạt tiền từ 800.000 đồng đến 1.000.000 đồng. Nếu gây tai nạn giao thông, mức phạt sẽ tăng lên từ 10.000.000 đồng đến 14.000.000 đồng']},
    'Cấm vượt đối với xe trên 3.5 tấn':{'car': ['Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng nếu vượt quá tốc độ quy định. Nếu gây tai nạn giao thông, mức phạt sẽ tăng lên từ 20.000.000 đồng đến 22.000.000 đồng'], 'motorbike': ['Không áp dụng']},
    'Giao nhau với đường không ưu tiên': {'car': ['Chú ý'], 'motorbike': ['Chú ý']},
    'Đường ưu tiên': {'car': ['Chú ý'], 'motorbike': ['Chú ý']},
    'Nhường đường': {'car': ['Chú ý'], 'motorbike': ['Chú ý']},
    'Dừng lại': {'car': ['Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng nếu đi vào khu vực cấm'], 'motorbike': ['Phạt tiền từ 800.000 đồng đến 1.000.000 đồng nếu không chấp nhận hiệu lệnh dừng']},
    'Cấm xe': {'car': ['Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng. Nếu gây tai nạn giao thông, mức phạt sẽ tăng lên từ 20.000.000 đồng đến 22.000.000 đồng'], 'motorbike': ['Phạt tiền từ 800.000 đồng đến 1.000.000 đồng. Nếu gây tai nạn giao thông, mức phạt sẽ tăng lên từ 10.000.000 đồng đến 14.000.000 đồng']},
    'Cấm xe trên 3.5 tấn vượt ': {'car': ['Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng. Nếu gây tai nạn giao thông, mức phạt sẽ tăng lên từ 20.000.000 đồng đến 22.000.000 đồng'], 'motorbike': ['Phạt tiền từ 800.000 đồng đến 1.000.000 đồng. Nếu gây tai nạn giao thông, mức phạt sẽ tăng lên từ 10.000.000 đồng đến 14.000.000 đồng']},
    'Cấm vào': {'car': ['Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng nếu đi vào khu vực cấm'], 'motorbike': ['Phạt tiền từ 800.000 đồng đến 1.000.000 đồng nếu không chấp nhận hiệu lệnh dừng']},
    'Chú ý chung': {'car': ['Chú ý'], 'motorbike': ['Chú ý']},
    'Khúc cua nguy hiểm bên trái': {'car': ['Chú ý'], 'motorbike': ['Chú ý']},
    'Khúc cua nguy hiểm bên phải': {'car': ['Chú ý'], 'motorbike': ['Chú ý']},
    'Khúc cua đôi': {'car': ['Chú ý'], 'motorbike': ['Chú ý']},
    'Đường gồ ghề': {'car': ['Chú ý'], 'motorbike': ['Chú ý']},
    'Đường trơn trượt': {'car': ['Chú ý'], 'motorbike': ['Chú ý']},
    'Đường hẹp bên phải': {'car': ['Chú ý'], 'motorbike': ['Chú ý']},
    'Công trường đang thi công': {'car': ['Chú ý'], 'motorbike': ['Chú ý']},
    'Tín hiệu giao thông': {'car': ['Chú ý'], 'motorbike': ['Chú ý']},
    'Người đi bộ': {'car': ['Chú ý'], 'motorbike': ['Chú ý']},
    'Trẻ em qua đường': {'car': ['Chú ý'], 'motorbike': ['Chú ý']},
    'Xe đạp qua đường': {'car': ['Chú ý'], 'motorbike': ['Chú ý']},
    'Cảnh báo băng/tuyết': {'car': ['Chú ý'], 'motorbike': ['Chú ý']},
    'Động vật hoang dã qua đường': {'car': ['Chú ý'], 'motorbike': ['Chú ý']},
    'Hết giới hạn tốc độ + Hạn chế vượt': {'car': ['Nếu vượt quá giới hạn tốc độ: phạt tiền từ 600.000 đồng đến 1.200.000 đồng'], 'motorbike': ['Nếu vượt quá giới hạn tốc độ: phạt tiền từ 400.000 đồng đến 800.000 đồng' ]},
    'Rẽ phải phía trước': {'car': ['Phạt tiền từ 400.000 đồng đến 600.000 đồng nếu không tuân thủ biển báo'], 'motorbike': ['Phạt tiền từ 200.000 đồng đến 400.000 đồng nếu không tuân thủ biển báo']},
    'Rẽ trái phía trước': {'car': ['Phạt tiền từ 400.000 đồng đến 600.000 đồng nếu không tuân thủ biển báo'], 'motorbike': ['Phạt tiền từ 200.000 đồng đến 400.000 đồng nếu không tuân thủ biển báo']},
    'Chỉ đi thẳng': {'car': ['Phạt tiền từ 400.000 đồng đến 600.000 đồng nếu không tuân thủ biển báo'], 'motorbike': ['Phạt tiền từ 200.000 đồng đến 400.000 đồng nếu không tuân thủ biển báo']},
    'Đi thẳng hoặc rẽ phải': {'car': ['Phạt tiền từ 400.000 đồng đến 600.000 đồng nếu không tuân thủ biển báo'], 'motorbike': ['Phạt tiền từ 200.000 đồng đến 400.000 đồng nếu không tuân thủ biển báo']},
    'Đi thẳng hoặc rẽ trái': {'car': ['Phạt tiền từ 400.000 đồng đến 600.000 đồng nếu không tuân thủ biển báo'], 'motorbike': ['Phạt tiền từ 200.000 đồng đến 400.000 đồng nếu không tuân thủ biển báo']},
    'Đi tiếp bên phải': {'car': ['Phạt tiền từ 400.000 đồng đến 600.000 đồng nếu không tuân thủ biển báo'], 'motorbike': ['Phạt tiền từ 200.000 đồng đến 400.000 đồng nếu không tuân thủ biển báo']},
    'Đi tiếp bên trái': {'car': ['Phạt tiền từ 400.000 đồng đến 600.000 đồng nếu không tuân thủ biển báo'], 'motorbike': ['Phạt tiền từ 200.000 đồng đến 400.000 đồng nếu không tuân thủ biển báo']},
    'Vòng xuyến': {'car': ['Phạt tiền từ 400.000 đồng đến 600.000 đồng nếu không tuân thủ biển báo'], 'motorbike': ['Phạt tiền từ 200.000 đồng đến 400.000 đồng nếu không tuân thủ biển báo']},
    'Hết cấm vượt': {'car': ['Chú ý'], 'motorbike': ['Chú ý']},
    'Hết cấm vượt đối với xe trên 3,5': {'car': ['Chú ý'], 'motorbike': ['Không áp dụng']},
}

# Khởi tạo GUI
root = tk.Tk()
root.geometry('900x700')
root.title('Nhận diện biển báo giao thông')
root.configure(bg='#f0f0f0')

# Frame chứa các nút và radio button
control_frame = tk.Frame(root, bg='#f0f0f0')
control_frame.pack(pady=10)

# Frame hiển thị ảnh với kích thước 800x600
image_frame = tk.Frame(root, bg='white', width=800, height=600)
image_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)

# Khu vực hiển thị ảnh
img_label = tk.Label(image_frame, bg='white')
img_label.pack(expand=True)

# Frame thông tin kết quả
info_frame = tk.Frame(root, bg='#f0f0f0')
info_frame.pack(pady=10)

# Radio button chọn phương tiện (chỉ chọn 1)
vehicle_frame = tk.Frame(control_frame, bg='#f0f0f0')
vehicle_frame.grid(row=0, column=0, padx=20)

vehicle_type = tk.StringVar(value="car")  # Mặc định chọn "car"
ttk.Radiobutton(vehicle_frame, text="Oto", variable=vehicle_type, value="car").grid(row=0, column=0, padx=10)
ttk.Radiobutton(vehicle_frame, text="Xe Máy", variable=vehicle_type, value="motorbike").grid(row=0, column=1, padx=10)

# Các nút chức năng
btn_style = {'background': '#c71b20', 'foreground': 'white', 'font': ('Arial', 10, 'bold'), 'borderwidth': 0}
tk.Button(control_frame, text="Tải ảnh lên", command=lambda: upload_image(), **btn_style).grid(row=0, column=1, padx=5)
tk.Button(control_frame, text="Mở camera", command=lambda: open_camera(), **btn_style).grid(row=0, column=2, padx=5)
tk.Button(control_frame, text="Tắt Camera", command=lambda: close_camera(), **btn_style).grid(row=0, column=4, padx=5)
tk.Button(control_frame, text="Phân loại", command=lambda: process_capture(), **btn_style).grid(row=0, column=3, padx=5)

# Thông tin kết quả
result_style = {'bg': '#f0f0f0', 'font': ('Arial', 12)}
tk.Label(info_frame, text="Tên biển báo:", **result_style).grid(row=0, column=0, sticky='w')
name_label = tk.Label(info_frame, text="", **result_style, fg='#2c3e50')
name_label.grid(row=0, column=1, padx=10)

tk.Label(info_frame, text="Thông báo:", **result_style).grid(row=1, column=0, sticky='w')
fine_label = tk.Label(info_frame, text="", **result_style, fg='#2c3e50', justify='left')  # justify='left' để căn trái
fine_label.grid(row=1, column=1, padx=10)

tk.Label(info_frame, text="Độ chính xác:", **result_style).grid(row=2, column=0, sticky='w')
confidence_label = tk.Label(info_frame, text="", **result_style, fg='#2c3e50')
confidence_label.grid(row=2, column=1, padx=10)

# Biến lưu trữ
current_image_path = None
cap = None  # Biến toàn cục cho camera

# Hàm xử lý
def update_display(image):
    img = ImageTk.PhotoImage(image)
    img_label.configure(image=img)
    img_label.image = img

def classify(image):
    try:
        img_array = np.expand_dims(np.array(image.resize((30, 30))), axis=0)
        probabilities = model.predict(img_array)[0]  # Lấy mảng xác suất
        max_prob = np.max(probabilities)  # Xác suất cao nhất
        pred = np.argmax(probabilities)  # Lớp có xác suất cao nhất
        
        # Đặt ngưỡng xác suất
        confidence_threshold = 0.9
        if max_prob < confidence_threshold:
            name_label.config(text="None")
            fine_label.config(text="Không xác định")
            confidence_label.config(text=f"{max_prob * 100:.2f}%")
        else:
            sign_name = classes.get(pred + 1, None)
            if sign_name is None:
                name_label.config(text="None")
                fine_label.config(text="Không xác định")
                confidence_label.config(text=f"{max_prob * 100:.2f}%")
            else:
                # Lấy danh sách các mức phạt và nối chúng thành chuỗi nhiều dòng
                fine_list = fines.get(sign_name, {'car': ['Không xác định'], 'motorbike': ['Không xác định']})[vehicle_type.get()]
                fine_text = "\n".join(fine_list)  # Nối các dòng bằng ký tự xuống dòng
                name_label.config(text=sign_name)
                fine_label.config(text=fine_text)  # Hiển thị nhiều dòng
                confidence_label.config(text=f"{max_prob * 100:.2f}%")
    except Exception as e:
        print("Error:", e)
        name_label.config(text="None")
        fine_label.config(text="Không xác định")
        confidence_label.config(text="0.00%")

def upload_image():
    global current_image_path
    path = filedialog.askopenfilename()
    if path:
        current_image_path = path
        image = Image.open(path).resize((800, 600))
        update_display(image)
        # Xóa kết quả cũ khi chỉ tải ảnh lên
        name_label.config(text="")
        fine_label.config(text="")
        confidence_label.config(text="")

def draw_crop_box(frame):
    """Vẽ ô vuông crop ở giữa frame"""
    height, width = frame.shape[:2]
    crop_size = 200  # Kích thước ô vuông crop (200x200)
    x_start = (width - crop_size) // 2
    y_start = (height - crop_size) // 2
    x_end = x_start + crop_size
    y_end = y_start + crop_size
    
    # Vẽ ô vuông màu xanh lá
    cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
    return x_start, y_start, x_end, y_end

def open_camera():
    global cap
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            tk.messagebox.showerror("Lỗi", "Không thể mở camera!")
            cap = None
            return
        show_camera_feed()

def show_camera_feed():
    global cap
    if cap is not None and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Vẽ ô crop ở giữa
            x_start, y_start, x_end, y_end = draw_crop_box(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame).resize((800, 600))
            update_display(img)
            root.after(10, show_camera_feed)
        else:
            tk.messagebox.showerror("Lỗi", "Không thể đọc frame từ camera!")
            close_camera()

def process_capture():
    global cap, current_image_path
    if cap and cap.isOpened():  # Nếu đang dùng camera
        ret, frame = cap.read()
        if ret:
            # Lấy tọa độ ô crop
            height, width = frame.shape[:2]
            crop_size = 200
            x_start = (width - crop_size) // 2
            y_start = (height - crop_size) // 2
            x_end = x_start + crop_size
            y_end = y_start + crop_size
            
            # Crop vùng ảnh trong ô vuông
            cropped_frame = frame[y_start:y_end, x_start:x_end]
            cv2.imwrite('capture.jpg', cropped_frame)
            current_image_path = 'capture.jpg'
            
            # Phân loại ảnh crop
            cropped_image = Image.fromarray(cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB))
            classify(cropped_image)
        else:
            tk.messagebox.showerror("Lỗi", "Không thể chụp ảnh từ camera!")
    elif current_image_path:  # Nếu đã tải ảnh lên
        image = Image.open(current_image_path)
        classify(image)

def close_camera():
    global cap
    if cap is not None and cap.isOpened():
        cap.release()
        cap = None
        img_label.config(image='')
        tk.messagebox.showinfo("Thông báo", "Camera đã được tắt.")

# Chạy ứng dụng
root.mainloop()