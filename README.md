## Đồ án cuối kì CS231 -- Phân loại biển báo giao thông

Repository này chứa source code và các tài nguyên cho bài toán phân loại biển báo giao thông sử dụng các kỹ thuật học máy. Mục tiêu chính là phát triển một mô hình có khả năng phân loại biển báo giao thông vào một trong bốn loại: Biển báo cấm, Biển báo nguy hiểm, Biển báo hiệu lệnh, và Biển báo chỉ dẫn. Dự án bao gồm các bước tiền xử lý dữ liệu, trích xuất đặc trưng sử dụng HOG, và huấn luyện các mô hình phân loại như k-Nearest Neighbors (kNN) và Random Forest.

### Các tính năng

- **Tiền xử lý dữ liệu**: Tải, cắt, và trực quan hóa dữ liệu.
- **Trích xuất đặc trưng**: Sử dụng Histogram of Oriented Gradients (HOG) để nắm bắt các đặc trưng quan trọng.
- **Huấn luyện mô hình**: Triển khai các mô hình kNN và Random Forest.

### Hướng dẫn sử dụng

### Yêu Cầu Hệ Thống

- Python 3.11
- Các thư viện: numpy, pandas, scikit-learn, matplotlib, opencv-python, scikit-image
- Xem thêm ở file `requirements.txt`

### Cấu Trúc Thư Mục

- `data/`: Chứa bộ dữ liệu huấn luyện và kiểm tra.
- `scripts/`: Các script để tiền xử lý dữ liệu và trích xuất đặc trưng.
- `notebooks/`: Các notebook Jupyter để huấn luyện và đánh giá mô hình.
- `models/`: Các mô hình đã huấn luyện.

---
