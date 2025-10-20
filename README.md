<p align="center">
  <a href="https://www.uit.edu.vn/" title="Trường Đại học Công nghệ Thông tin" style="border: 5;">
    <img src="https://i.imgur.com/WmMnSRt.png" alt="Trường Đại học Công nghệ Thông tin | University of Information Technology">
  </a>
</p>

<!-- Title -->
<h1 align="center"><b>CS419 - TRUY XUẤT THÔNG TIN</b></h1>

## GIỚI THIỆU MÔN HỌC

<a name="gioithieumonhoc"></a>

- **Mã môn học**: CS419
- **Lớp học**: CS419.Q11
- **Năm học**: 2025-2026

## GIẢNG VIÊN HƯỚNG DẪN

<a name="giangvien"></a>

- TS. **Nguyễn Trọng Chỉnh** - *chinhnt@uit.edu.vn*

## THÀNH VIÊN NHÓM

<a name="thanhvien"></a>
| STT | MSSV | Họ và Tên | Github | Email |
| ------ |:-------------:| ----------------------:|-----------------------------------------------------:|-------------------------:
| 1 | 22520880 | Nguyễn Viết Anh Minh |[vamnguyen](https://github.com/vamnguyen) |22520880@gm.uit.edu.vn |
| 2 | 22520967 | Hồng Khải Nguyên |[Kevinzzz2004](https://github.com/Kevinzzz2004) |22520967@gm.uit.edu.vn |

# 🔍 Đồ án Truy xuất Thông tin (Information Retrieval System)

## 🧠 Giới thiệu

Đây là một **ứng dụng web mô phỏng hệ thống truy xuất thông tin (IR System)**, cho phép người dùng nhập truy vấn (query) và tìm kiếm trong tập dữ liệu tin tức (VNExpress).
Hệ thống được xây dựng bằng **Flask (Python)**, kết hợp nhiều mô hình xếp hạng phổ biến như:

- **TF–IDF Cosine Similarity**
- **BM25 (Okapi)**
- **Query Likelihood (Dirichlet smoothing)**
- **Odds Likelihood Ratio**

Ứng dụng hỗ trợ:

- Giao diện web để tìm kiếm, đánh dấu tài liệu liên quan.
- Đánh giá hiệu năng truy xuất bằng các chỉ số: Precision, Recall, F1, MAP, nDCG.
- Cơ chế cache (pickle + npz) giúp tăng tốc độ chạy ở lần kế tiếp.

---

## 🏗️ Cấu trúc thư mục

```
📦 do-an-ir/
│
├── app.py                     # Flask web app chính
├── indexer.py                 # Xây dựng inverted index + TF-IDF + cache
├── ranker.py                  # Các mô hình xếp hạng (Cosine, BM25, QL, Odds)
├── evaluator.py               # Đánh giá hiệu năng truy xuất
├── utils.py                   # Các hàm tiện ích: tokenize, stopwords, v.v.
│
├── templates/                 # HTML templates cho Flask
│   ├── index.html             # Trang tìm kiếm
│   ├── results.html           # Trang hiển thị kết quả
│   └── eval.html              # Trang đánh giá
│
├── static/                    # (tùy chọn) chứa CSS / JS / ảnh tĩnh
│
├── news_dataset.json          # Tập dữ liệu tin tức (nguồn VNExpress)
├── ground_truth.json          # Dữ liệu đánh giá do người dùng đánh dấu
│
├── cache_index.pkl            # Cache inverted index (tự động sinh)
├── tfidf_vectorizer.pkl       # Cache vectorizer TF-IDF (tự động sinh)
├── tfidf_matrix.npz           # Cache ma trận TF-IDF (tự động sinh)
├── news_tokenized.json        # Phiên bản tokenized của dữ liệu (tự động sinh)
│
├── requirements.txt           # Danh sách thư viện Python cần cài
├── .gitignore                 # Bỏ qua các file cache / temp
└── README.md                  # (file này)
```

---

## ⚙️ Cài đặt môi trường

### 1️⃣ Tạo môi trường ảo (khuyên dùng)

```bash
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
# hoặc
.venv\Scripts\activate         # Windows
```

### 2️⃣ Cài các thư viện cần thiết

```bash
pip install -r requirements.txt
```

_(Nếu bạn chưa có `requirements.txt`, tạo bằng:)_

```bash
pip freeze > requirements.txt
```

---

## 🚀 Cách chạy project

### 🔹 Lần đầu tiên (chưa có cache)

1. Đảm bảo file `news_dataset.json` có trong thư mục gốc.
2. Chạy ứng dụng:

```bash
python app.py
```

3. Lần đầu chạy, hệ thống sẽ:

   - Tokenize toàn bộ văn bản.
   - Xây dựng inverted index, TF–IDF matrix.
   - Lưu cache (mất vài phút tùy dataset).

4. Sau khi hoàn tất, mở trình duyệt tại:

   ```
   http://127.0.0.1:5000/
   ```

### 🔹 Các lần sau (đã có cache)

Cache sẽ tự động được load lại, chạy gần như ngay lập tức.

---

## 💡 Hướng dẫn sử dụng giao diện

| Chức năng                  | Mô tả                                                                         |
| -------------------------- | ----------------------------------------------------------------------------- |
| **Trang chủ** (`/`)        | Nhập truy vấn (VD: _"chính sách giáo dục"_) và chọn phương pháp xếp hạng      |
| **Kết quả tìm kiếm**       | Hiển thị danh sách tài liệu, điểm số, chủ đề (topic), nguồn (source), tác giả |
| **Mark Relevant / Unmark** | Ghi nhận tài liệu nào là liên quan để phục vụ đánh giá                        |
| **Evaluation**             | Tính toán các chỉ số Precision, Recall, F1, MAP, nDCG dựa trên ground truth   |

---

## 📊 Đánh giá mô hình

Hệ thống hỗ trợ các chỉ số IR phổ biến:

- Precision@K
- Recall@K
- F1-score
- MAP@K
- nDCG@K

File `ground_truth.json` lưu các tài liệu mà người dùng đánh dấu là **relevant** cho từng query.

---

## ⚡ Ghi chú

- Các file cache (`*.pkl`, `*.npz`, `news_tokenized.json`) có thể rất lớn → **không nên commit lên GitHub.**
- Đã thêm sẵn `.gitignore` để bỏ qua chúng.
- Nếu cần chia sẻ cho người khác chạy nhanh, có thể upload cache lên Drive và hướng dẫn tải về.

---

## 📚 Tham khảo

- _Introduction to Information Retrieval_ – Manning, Raghavan, Schütze (Cambridge, 2008)
- VNExpress Dataset: https://www.kaggle.com/datasets/haitranquangofficial/vietnamese-online-news-dataset/data
- Scikit-learn, Flask, Pandas, TQDM documentation

---

🧭 _© 2025 – Đồ án CS419: Information Retrieval System._
