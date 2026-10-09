# Agri Behavioral RecSys: Causal Debiasing in Recommender Systems

Dự án nghiên cứu, thực nghiệm và đối sánh các mô hình Khử độ lệch Nhân quả (Causal Debiasing) cho Hệ thống Gợi ý (Recommender Systems). Dự án tập trung triển khai và đánh giá 3 kiến trúc chính: **Matrix Factorization (MF Baseline)**, **IViDR (Instrumental Variable & Identifiable VAE)**, và **DR-BIAS + CDR (Doubly Robust with Counterfactual Disentangled Representations)** trên 2 bộ dữ liệu chuẩn: **Coat** và **KuaiRand-Pure**.

Toàn bộ mã nguồn thực nghiệm được quản lý tập trung trong thư mục [`collaborative-filtering/`](collaborative-filtering/).

---

## ⚡ Bảng Lệnh Nhanh (Cheat Sheet)

Tất cả các lệnh đều chạy từ thư mục `collaborative-filtering/` thông qua entrypoint [`experiments/run.py`](collaborative-filtering/experiments/run.py):

| Mục Đích | Câu Lệnh Thực Thi |
| :--- | :--- |
| **Kiểm thử toàn bộ hệ thống** | `.\venv\Scripts\python.exe experiments/run.py test` |
| **Smoke test nhanh (~10s)** | `.\venv\Scripts\python.exe experiments/run.py smoke` |
| 🏆 **Chạy ALL mô hình trên Coat** | `.\venv\Scripts\python.exe experiments/run.py benchmark --dataset coat` |
| 🏆 **Chạy ALL mô hình trên KuaiRand** | `.\venv\Scripts\python.exe experiments/run.py benchmark --dataset kuairand` |
| 🏆 **Chạy ALL mô hình trên CẢ 2 DATASET** | `.\venv\Scripts\python.exe experiments/run.py benchmark --dataset all` |
| **Tuning ALL mô hình trên Coat** | `.\venv\Scripts\python.exe experiments/run.py tune --model all --epochs 50` |
| **Tuning ALL mô hình trên KuaiRand** | `.\venv\Scripts\python.exe experiments/run.py tune-kr --model all --epochs 25` |
| **So sánh Adam vs SGD** | `.\venv\Scripts\python.exe experiments/run.py optimizers --dataset coat --epochs 50` |
| **Dọn dẹp cache tạm thời** | `.\venv\Scripts\python.exe experiments/run.py clean --apply` |

---

## 🚀 Hướng Dẫn Cài Đặt Môi Trường

### 1. Yêu cầu hệ thống
- Python 3.11+
- Hệ điều hành: Windows / Linux / macOS (Khuyên dùng Windows PowerShell hoặc Linux Bash)
- Phần cứng: CPU hoặc GPU (PyTorch tự động phát hiện thiết bị phù hợp)

### 2. Thiết lập Virtual Environment & Cài đặt Thư viện
Mở terminal và chuyển vào thư mục `collaborative-filtering`:

```powershell
cd collaborative-filtering

# Khởi tạo môi trường ảo Python 3.11
py -3.11 -m venv venv

# Kích hoạt môi trường (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Cài đặt toàn bộ dependencies
pip install -r requirements.txt
```

### 3. Cấu trúc Dữ liệu Đầu Vào Bắt Buộc
Dữ liệu thô được đặt trong thư mục `../data/raw/` (hoặc `data/raw/`):

```text
data/raw/
├── coat/
│   ├── train.ascii                             # Dữ liệu huấn luyện có bias (6.960 ratings)
│   ├── test.ascii                              # Dữ liệu ngẫu nhiên unbiased (4.640 ratings)
│   ├── propensities.ascii                      # Xác suất hiển thị (Propensity scores)
│   └── user_item_features/
│       ├── user_features.ascii                 # Thuộc tính người dùng (Dùng cho IViDR)
│       └── item_features.ascii                 # Thuộc tính sản phẩm
└── kuairand_pure/data/
    ├── log_standard_4_08_to_4_21_pure.csv       # Standard exposure log (tuần 1)
    ├── log_standard_4_22_to_5_08_pure.csv       # Standard exposure log (tuần 2)
    ├── log_random_4_22_to_5_08_pure.csv         # Random unbiased test log
    └── user_features_pure.csv                  # Thuộc tính người dùng
```

---

## 💻 Hướng Dẫn Chạy Chi Tiết (CLI Reference)

Tất cả các tác vụ (kiểm thử, chạy thử, benchmark, tinh chỉnh tham số, so sánh bộ tối ưu) được quản lý thống nhất qua entrypoint [`experiments/run.py`](collaborative-filtering/experiments/run.py).

> [!NOTE]
> Mọi lệnh dưới đây đều thực thi từ thư mục `collaborative-filtering/` bằng Python của `venv`.

---

### 1. Kiểm thử Tự Động Toàn Diện (Unit Tests)
Chạy toàn bộ 40 test cases (kiểm tra rò rỉ dữ liệu, tính đúng đắn của metric ranking, gradient các mô hình causal, tính toàn vẹn của checkpoint):
```powershell
.\venv\Scripts\python.exe experiments/run.py test
```

---

### 2. Hướng Dẫn Chạy TẤT CẢ Các Mô Hình (Full Benchmark & Smoke Test)

Dự án cung cấp luồng chạy benchmark đồng thời cả 3 mô hình (**MF Baseline**, **IViDR**, và **DR-BIAS + CDR**) với các siêu tham số tối ưu đã được hiệu chỉnh:

#### a. Chạy ALL mô hình trên Dataset **Coat**:
```powershell
# Chạy đầy đủ 3 mô hình trên Coat (mặc định tối đa 100 epochs, early stopping)
.\venv\Scripts\python.exe experiments/run.py benchmark --dataset coat
```

#### b. Chạy ALL mô hình trên Dataset **KuaiRand-Pure** (1.4M interactions):
```powershell
# Chạy đầy đủ 3 mô hình trên KuaiRand-Pure
.\venv\Scripts\python.exe experiments/run.py benchmark --dataset kuairand
```

#### c. Chạy ALL mô hình trên CẢ 2 DATASET trong một lệnh duy nhất:
```powershell
.\venv\Scripts\python.exe experiments/run.py benchmark --dataset all
```

#### d. Chạy thử nghiệm nhanh ALL mô hình (Smoke Test / Quick Run):
```powershell
# Smoke test siêu tốc: chạy 1 epoch cho 3 mô hình trên Coat + mẫu nhỏ 64 user KuaiRand
.\venv\Scripts\python.exe experiments/run.py smoke

# Quick benchmark: chạy cả 3 mô hình trên Coat với 5 epochs
.\venv\Scripts\python.exe experiments/run.py benchmark --dataset coat --quick --epochs 5
```

---

### 3. Chạy Từng Mô Hình Riêng Lẻ (Individual Models)

Nếu bạn chỉ muốn huấn luyện hoặc đánh giá một mô hình cụ thể:

```powershell
# 1. Chạy riêng IViDR trên Coat
.\venv\Scripts\python.exe experiments/run.py benchmark --dataset coat --model ividr

# 2. Chạy riêng DR-BIAS + CDR trên Coat
.\venv\Scripts\python.exe experiments/run.py benchmark --dataset coat --model cdr

# 3. Chạy riêng MF Baseline trên Coat
.\venv\Scripts\python.exe experiments/run.py benchmark --dataset coat --model mf

# 4. Chạy riêng IViDR trên KuaiRand
.\venv\Scripts\python.exe experiments/run.py benchmark --dataset kuairand --model ividr
```

---

### 4. Tinh Chỉnh Siêu Tham Số (Hyperparameter Tuning)

Quy trình tuning tự động quét qua không gian tham số, lựa chọn cấu hình có **Validation NDCG@5** cao nhất và đánh giá DUY NHẤT 1 LẦN trên tập Unbiased Test.

#### a. Tinh chỉnh trên Coat:
```powershell
# Tinh chỉnh TOÀN BỘ mô hình trên Coat (mặc định 50 epochs mỗi trial)
.\venv\Scripts\python.exe experiments/run.py tune --model all --epochs 50

# Tinh chỉnh riêng DR-BIAS + CDR trên Coat (quét ngưỡng uncertainty và learning rate)
.\venv\Scripts\python.exe experiments/run.py tune --model cdr --epochs 35

# Tinh chỉnh riêng IViDR trên Coat
.\venv\Scripts\python.exe experiments/run.py tune --model ividr --epochs 50
```

#### b. Tinh chỉnh trên KuaiRand-Pure:
```powershell
# Tinh chỉnh TOÀN BỘ mô hình trên KuaiRand (25 epochs mỗi trial)
.\venv\Scripts\python.exe experiments/run.py tune-kr --model all --epochs 25

# Tinh chỉnh riêng IViDR trên KuaiRand
.\venv\Scripts\python.exe experiments/run.py tune-kr --model ividr --epochs 25
```

---

### 5. So Sánh Bộ Tối Ưu Hóa (Adam vs SGD Analysis)

Trong RecSys, **Adam cho chỉ số đánh giá cao hơn, tốc độ hội tụ nhanh hơn và độ ổn định tốt hơn rõ rệt so với SGD**:
- **Adam (Adaptive Moment Estimation)**: Tự động điều chỉnh learning rate thích nghi theo từng trọng số embedding. Giúp các item/user ít xuất hiện (gradient thưa) vẫn nhận được bước cập nhật đủ lớn để hội tụ.
- **SGD thuần**: Dùng learning rate cố định cho toàn bộ embedding, cực kỳ nhạy cảm với dải learning rate và hội tụ rất chậm trên dữ liệu thưa.

| Tiêu chí | **Adam** | **SGD** |
| :--- | :--- | :--- |
| **Chỉ số Đánh giá (Test NDCG@5)** | 🏆 **Cao hơn rõ rệt (~0.46 - 0.49)** | ⚠️ **Thấp hơn (~0.40 - 0.44)** |
| **Tốc độ hội tụ** | 🚀 Đạt điểm cực trị sau 15–25 epochs | ⏳ Cần số epoch lớn hơn nhiều |
| **Độ ổn định qua các Seed ($\sigma$)** | Ổn định, độ lệch chuẩn nhỏ | Biến động mạnh giữa các seed |
| **Xử lý Sparse Embeddings** | 🌟 Rất tốt (Momentum + Adaptive Step) | Kém trên cold items / sparse logs |

#### Câu lệnh thực hiện đối chứng thực nghiệm công bằng giữa Adam và SGD:
```powershell
# So sánh trên Coat (50 epochs)
.\venv\Scripts\python.exe experiments/run.py optimizers --dataset coat --epochs 50

# So sánh trên KuaiRand (50 epochs)
.\venv\Scripts\python.exe experiments/run.py optimizers --dataset kuairand --epochs 50
```

---

### 6. Huấn Luyện MF Tùy Biến Nâng Cao
Chạy trực tiếp module MF với cấu hình tùy chỉnh (hỗ trợ BCE/MSE loss, scheduler, weight decay scope):
```powershell
# Adam + BCE mặc định
.\venv\Scripts\python.exe experiments/run.py mf --dataset coat --optimizer adam --loss bce --lr 0.001 --dim 64 --epochs 50

# SGD + BCE
.\venv\Scripts\python.exe experiments/run.py mf --dataset coat --optimizer sgd --loss bce --lr 0.1 --dim 64 --epochs 50
```

---

### 7. Đánh Giá Checkpoint Có Sẵn (Không Cần Train Lại)
Chấm điểm trực tiếp một checkpoint `.pt` đã huấn luyện trước đó:
```powershell
.\venv\Scripts\python.exe experiments/run.py evaluate --dataset coat --checkpoint checkpoints/coat_ividr_best.pt
```

---

### 8. Dọn Dẹp Artifacts & Cache An Toàn
Công cụ dọn dẹp giữ nguyên dữ liệu thô (`data/raw/`), môi trường ảo (`venv`), mã nguồn và các báo cáo kết quả thực nghiệm trong `results/`:
```powershell
# Xem trước danh sách các file tạm sẽ được xóa (Preview mode)
.\venv\Scripts\python.exe experiments/run.py clean

# Thực hiện xóa cache và file bytecode tạm thời
.\venv\Scripts\python.exe experiments/run.py clean --apply
```

---

## 📊 Bảng Kết Quả Thực Nghiệm & Siêu Tham Số Tối Ưu

### 1. Dataset Coat (iDCF Protocol: 16 candidate items/user, 30% Val / 70% Test)
| Mô hình | Siêu tham số tối ưu | Val NDCG@5 | Test NDCG@5 | Test Recall@5 | Test AUC |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **MF Baseline** | `dim=64, lr=1e-3, weight_decay=1e-5, batch_size=512` | 0.4940 | 0.4629 | 0.4898 | 0.6460 |
| **DR-BIAS + CDR** | `dim=32, lr=2e-3, un_thres=0.01, max_ips=20.0, batch_size=128` (Warm-start MF) | **0.5517** | **0.4841** | **0.4932** | **0.6826** |
| **IViDR** | `dim=64, latent_dim=32, lr=3e-3, phi=2.0, beta_elbo=0.01, batch_size=512` | **0.5847** | **0.5454** | **0.5714** | **0.7064** |

> **Nhận xét**: Cả hai mô hình Nhân quả (CDR và IViDR) đều vượt trội rõ rệt so với MF Baseline trên mọi chỉ số đánh giá Unbiased Test. IViDR đạt hiệu năng cao nhất nhờ khả năng tách biệt biểu diễn sở thích bất biến ($Z$) và biến thiên hiển thị ($C$) qua mạng biến phân iVAE.

---

### 2. Dataset KuaiRand-Pure (1.4M Interactions, 23.5k Users)
| Mô hình | Siêu tham số tối ưu | Val NDCG@5 | Test NDCG@5 | Test Recall@5 | Test AUC |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **MF Baseline** | `dim=128, lr=5e-4, weight_decay=1e-5, batch_size=4096` | 0.3428 | 0.3434 | 0.2948 | 0.5560 |
| **DR-BIAS + CDR** | `dim=128, lr=1e-3, un_thres=0.005, batch_size=8192` | **0.3452** | **0.3459** | **0.2990** | **0.5609** |
| **IViDR** | `dim=128, latent_dim=32, lr=2e-3, phi=0.1, beta_elbo=0.001, batch_size=4096` | **0.3510** | **0.3484** | **0.3049** | **0.5606** |

---

## 🔬 Giao Thức Đánh Giá Khoa Học (Evaluation Protocol)

1. **Phân chia Dữ liệu (Data Splitting)**:
   - **Tập huấn luyện (Training)**: Toàn bộ tương tác logged có bias từ standard logs (`train.ascii` trên Coat, `log_standard_pure` trên KuaiRand).
   - **Tập kiểm định (Validation)** và **Tập kiểm thử (Test)**: Được chia từ tập dữ liệu ngẫu nhiên không thiên lệch (Random log) theo tỷ lệ **30% Validation / 70% Test** theo `user_id` với cố định `seed = 1234`.
2. **Nguyên tắc Đánh giá Khách quan (Strict Scientific Rigor)**:
   - Toàn bộ quá trình chọn epoch (Early Stopping) và lựa chọn siêu tham số (Hyperparameter Tuning) **chỉ được phép quan sát tập Validation**.
   - Tập **Held-out Unbiased Test** chỉ được nạp và đánh giá **DUY NHẤT 1 LẦN** sau khi mô hình tốt nhất đã được chọn và đóng băng trọng số.
3. **Không gian Ứng viên (Candidate Pool)**:
   - Đánh giá xếp hạng trên tập các ứng viên được hiển thị thực tế trong random log (16 items/user trên Coat; candidate pool trên KuaiRand), không tự gán nhãn negative cho các item chưa quan sát.
4. **Hệ thống Chỉ số (Metrics)**:
   - **NDCG@5** và **Recall@5**: Tính toán theo thang macro trên tất cả người dùng có ít nhất 1 tương tác dương (positive interaction).
   - **HitRate@5** và **AUC**: Đánh giá khả năng phân loại và trúng mục tiêu trên danh sách gợi ý Top-5.

---

## 📁 Cấu Trúc Báo Cáo & Kết Quả Lưu Trữ

Sau mỗi lần chạy, kết quả sẽ tự động được ghi nhận tại thư mục [`collaborative-filtering/results/`](collaborative-filtering/results/):
- **Tệp Markdown (`.md`)**: Bảng so sánh trực quan các chỉ số giữa các mô hình.
- **Tệp JSON (`.json`)**: Chứa toàn bộ cấu hình, learning curves qua từng epoch, signature mã nguồn và dữ liệu để phục vụ kiểm chứng khoa học.
- **Tệp Log (`.log`)**: Lưu lại toàn bộ stdout/stderr chi tiết của quá trình huấn luyện.

---

## 📑 Tài Liệu Tham Khảo Trong Dự Án

- [`collaborative-filtering/docs/README.md`](collaborative-filtering/docs/README.md): **Trung tâm tài liệu tổng hợp (Documentation Hub)**.
- [`collaborative-filtering/docs/colab_recsys_rest_api_plan.md`](collaborative-filtering/docs/colab_recsys_rest_api_plan.md): Kế hoạch chuyển đổi CF thành RESTful API trên Google Colab.
- [`collaborative-filtering/README.md`](collaborative-filtering/README.md): Hướng dẫn kỹ thuật chuyên sâu và chi tiết pipeline.
- [`collaborative-filtering/docs/evaluation_protocol.md`](collaborative-filtering/docs/evaluation_protocol.md): Chi tiết toán học và công thức các chỉ số ranking.
- [`collaborative-filtering/docs/mf_optimizer_audit.md`](collaborative-filtering/docs/mf_optimizer_audit.md): Phân tích so sánh Adam vs SGD và kiểm tra kiến trúc MF.
- [`collaborative-filtering/docs/paper_reference_comparison.md`](collaborative-filtering/docs/paper_reference_comparison.md): Đối chiếu kết quả thực nghiệm với các bài báo gốc (iDCF, IViDR, CDR).
- [`collaborative-filtering/results/README.md`](collaborative-filtering/results/README.md): Lưu trữ và giải thích các báo cáo kết quả thực nghiệm mới nhất.
