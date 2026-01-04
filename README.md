Repo này chứa bản cài đặt của LassoNet, một kiến trúc mạng nơ-ron được thiết kế để lựa chọn đặc trưng, như được mô tả trong bài báo "LassoNet: A Neural Network with Feature Sparsity".

## Cấu trúc file

- `lassonet/`
  - `model.py`: Định nghĩa kiến trúc `LassoNet`.
  - `prox.py`: Cài đặt các toán tử proximal (Hier-Prox).
  - `trainer.py`: Chứa `LassoNetClassifier` tương thích với Scikit-learn.
  - `data_utils.py`: Các tiện ích để tải các bộ dữ liệu khác nhau (MNIST, ISOLET, Mushroom, v.v.).
  - `utils.py`: Các hàm hỗ trợ đánh giá và vẽ biểu đồ.
  - `main.py`: Điểm nhập chính để chạy các thí nghiệm.

## Yêu cầu

Đảm bảo bạn đã cài đặt các thư viện sau:

```bash
pip install torch numpy scipy scikit-learn matplotlib pandas pillow
```

## Hướng dẫn sử dụng

### Cấu hình

Các thí nghiệm được cấu hình trực tiếp trong `lassonet/main.py`. Bạn có thể sửa đổi các biến toàn cục ở đầu tệp để thay đổi bộ dữ liệu, siêu tham số và cài đặt huấn luyện.

Mở `lassonet/main.py` và điều chỉnh các tham số sau:

```python
# Các tham số thí nghiệm
BATCH_SIZE = 256
EPOCHS = 1000       # Số epoch tối đa (đã bật dừng sớm)
LR = 1e-3           # Tốc độ học
PATIENCE = 10       # Số epoch early stopping
dataset = "MNIST"   # Các tùy chọn: "MNIST", "ISOLET", "Mushroom", "COIL", "Activity"

# Số lượng đặc trưng mục tiêu cần chọn
K = 50              

# Cài đặt Đường dẫn Điều chuẩn
# Đặt là 'auto' để mô hình tự xác định lambda khởi đầu, 
# hoặc cung cấp một giá trị float dựa trên các lần chạy trước.
LAMBDA_START = "auto" 
PATH_MULTIPLIER = 0.02
```

### Chạy thực nghiệm

Để chạy toàn bộ quá trình huấn luyện đường dẫn điều chuẩn, chọn lọc đặc trưng và đánh giá hạ nguồn, thực thi lệnh sau từ thư mục gốc:

```bash
python lassonet/main.py
```

### Flow

1.  **Tải dữ liệu**: Bộ dữ liệu được chỉ định sẽ được tải và tiền xử lý.
2.  **Huấn luyện đường di điều chuẩn**: Mô hình LassoNet huấn luyện với các giá trị lambda tăng dần (phạt L1) để làm thưa các trọng số lớp skip ($\theta$).
3.  **Lưu kết quả**:
    - Lưu đường dẫn huấn luyện vào tệp `.pkl`.
    - Tạo biểu đồ `{dataset}_regularization_path.png` hiển thị Độ chính xác so với Số lượng Đặc trưng.
4.  **Huấn luyện lại (Decoder)**: Một mạng mới được huấn luyện lại chỉ sử dụng các đặc trưng đã chọn (không có phạt L1) để đo lường hiệu năng.
5.  **So sánh downstream learner**:
    - Huấn luyện `ExtraTreesClassifier` và `SVC` trên các đặc trưng đã chọn để so sánh.
    - Tạo biểu đồ `{dataset}_comparison_roc_curve.png` (nếu `EVAL_BINARY` là True).

## Tune siêu tham số

Để tinh chỉnh hệ số phân cấp ($M$), bạn có thể bỏ chú thích lời gọi hàm `tune_M()` trong khối `if __name__ == "__main__":` của `lassonet/main.py`:

```python
if __name__ == "__main__":
    # main()
    tune_M() # Chạy GridSearchCV cho M = [5, 10, 15]
```

## Các Bộ dữ liệu được Hỗ trợ

- **MNIST**: Chữ số viết tay.
- **ISOLET**: Nhận dạng giọng nói các chữ cái cô lập.
- **MICE**: Biểu hiện protein ở chuột.
- **COIL-20**: Danh mục đối tượng.
- **Activity**: Nhận dạng hoạt động con người.
- **Mushroom**: Phân loại nấm (Dữ liệu dạng bảng).
