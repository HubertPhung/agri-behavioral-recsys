# 📚 Trung Tâm Tài Liệu Nghiên Cứu & Kỹ Thuật (Documentation Hub)

Thư mục này quản lý tập trung toàn bộ tài liệu nghiên cứu lý thuyết, bài báo khoa học gốc, tóm tắt phân tích, kế hoạch triển khai kỹ thuật và báo cáo thực nghiệm của hệ thống gợi ý **TopTop RecSys** (`agri-behavioral-recsys`).

---

## 🗂️ Cấu Trúc Thư Mục

```text
collaborative-filtering/docs/
├── README.md                           # Bản chỉ mục này
├── presentation_slides.md              # Nội dung trình chiếu slide báo cáo dự án
│
├── 🚀 Kế Hoạch Kỹ Thuật & Triển Khai
│   └── colab_recsys_rest_api_plan.md   # Kế hoạch chuyển đổi CF thành RESTful API (Colab + Short Video)
│
├── 📑 Bài Báo Gốc & Phân Tích Lý Thuyết
│   ├── papers/                         # Tệp PDF bài báo khoa học gốc
│   │   ├── CDR.pdf                     # CIKM 2023 - Conservative Doubly Robust
│   │   ├── iDCF.pdf                    # KDD 2023 - Identifiable Debiasing in Collaborative Filtering
│   │   ├── IViDR.pdf                   # KDD 2023 - Instrumental Variable & Identifiable VAE
│   │   └── ReCRec.pdf                  # WSDM 2023 - Representation Counterfactual Debiasing
│   │
│   └── paper_summaries/                # Tóm tắt lý thuyết & phân tích chi tiết bằng Markdown
│       ├── CDR.md                      # Phân tích toán học & cơ chế DR-BIAS + CDR
│       ├── iDCF.md                     # Phân tích giao thức iDCF và định lý khả định danh
│       ├── IViDR.md                    # Phân tích kiến trúc iVAE & biến công cụ (IV)
│       └── ReCRec.md                   # Phân tích phản thực nghiệm & biểu diễn độc lập
│
└── 🔬 Giao Thức Đánh Giá & Báo Cáo Thực Nghiệm
    ├── evaluation_protocol.md          # Giao thức đánh giá khoa học (Metric NDCG@5, Recall@5, Split)
    ├── mf_optimizer_audit.md           # Kiểm toán thuật toán MF & so sánh Adam vs SGD
    ├── paper_reference_comparison.md   # Đối chiếu kết quả thực nghiệm với các công bố gốc
    ├── table3_run.md                   # Nhật ký kiểm chứng tái lập Table 3
    ├── benchmark_v2_plan_and_error_analysis.md # Phân tích lỗi và kế hoạch benchmark v2
    └── kuairand_tuning_2026-09-29.md   # Báo cáo tinh chỉnh siêu tham số trên KuaiRand
```

---

## 📖 Hướng Dẫn Truy Cập Nhanh

### 1. Dành cho Nghiên Cứu & Thuyết Minh
- **Báo cáo slide:** Đọc [`presentation_slides.md`](presentation_slides.md) để lấy nội dung thuyết trình.
- **Lý thuyết bài báo:** Đọc các file tóm tắt trong thư mục [`paper_summaries/`](paper_summaries/) kèm đối chiếu file PDF gốc trong [`papers/`](papers/).
- **Đối chiếu kết quả thực nghiệm:** Xem [`paper_reference_comparison.md`](paper_reference_comparison.md).

### 2. Dành cho Kỹ Thuật & Tích Hợp Ứng Dụng
- **Chuyển đổi RESTful API & Google Colab:** Xem [`colab_recsys_rest_api_plan.md`](colab_recsys_rest_api_plan.md) để xem đặc tả API, nhãn tương tác video ngắn (`play_time_ms >= 3s` hoặc thả tim) và cách khởi chạy.
- **Chuẩn hóa đo lường:** Xem [`evaluation_protocol.md`](evaluation_protocol.md).
