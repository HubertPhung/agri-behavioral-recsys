# Số tham chiếu bài báo và điều kiện so sánh

Bảng tham chiếu được giữ từ bảng người dùng cung cấp và tài liệu nghiên cứu của dự án. Đây là **số tham chiếu bên ngoài**, không phải kết quả chạy v4. Yahoo hiện chưa có pipeline trong dự án.

| Mô hình | Coat NDCG@5 | Coat Recall@5 | Yahoo NDCG@5 | Yahoo Recall@5 | Kuai NDCG@5 | Kuai Recall@5 |
| :-- | --: | --: | --: | --: | --: | --: |
| MF | 0.5524 | 0.5294 | 0.5629 | 0.7129 | 0.3748 | 0.3247 |
| DeepDCF-MF | 0.5373 | 0.5141 | 0.6395 | 0.7729 | 0.4078 | 0.3491 |
| iDCF | 0.5744 | 0.5504 | 0.6455 | 0.7837 | 0.4093 | 0.3513 |
| ReCRec | 0.5946 | 0.5787 | 0.6872 | 0.8124 | 0.3979 | 0.3285 |
| CDR, biến thể tốt nhất theo dataset | 0.6567 | 0.6678 | 0.6565 | 0.7323 | 0.3167 | 0.3078 |
| IViDR | 0.5903 | 0.5783 | 0.6602 | 0.7901 | 0.4161 | 0.3549 |

Dòng CDR ghép biến thể tốt nhất theo dataset: DR-BIAS+CDR cho Coat/Yahoo; MRDR+CDR cho KuaiRand. Không dùng dòng ghép này làm kết quả của một mô hình DR-BIAS+CDR duy nhất. Tài liệu dự án ghi riêng DR-BIAS+CDR KuaiRand là 0.3098 / 0.3048; cần giữ đúng nhãn biến thể khi đối chiếu.

## Nguồn

- [iDCF, Debiasing Recommendation by Learning Identifiable Latent Confounders (2023)](https://arxiv.org/pdf/2302.05052): nguồn tham khảo nhóm baseline MF, DeepDCF-MF và iDCF. [Notebook KuaiRand](https://github.com/BgmLover/iDCF/blob/main/data_process/kuairand/build_dataset.ipynb) và [mã split/evaluation](https://github.com/BgmLover/iDCF/blob/main/utils.py) là căn cứ cho preprocessing/split hiện tại của dự án.
- [CDR, CIKM 2023](https://jiawei-chen.github.io/paper/CIKM23-CDR.pdf), [mã nguồn tác giả](https://github.com/CrazyDumpling/CDR_CIKM2023): lưu ý biến thể DR-BIAS/MRDR, tỷ lệ validation/test và dữ liệu sau lọc khác giao thức iDCF.
- [IViDR, arXiv 2024](https://arxiv.org/pdf/2410.12451): tài liệu cơ chế IV và iVAE. Bản v5 sửa ridge chung thành pinv theo item; lịch joint training/proxy vẫn là giả định địa phương, chưa xác nhận tái lập nguyên bản. Xem [cấu hình hiện tại](table3_run.md).
- [ReCRec (2024)](https://zhoushengisnoob.github.io/papers/TOIS2024.pdf): kiểm tra đúng biến thể, split và ngân sách tuning trước khi đặt cạnh điểm địa phương.

## Điều kiện cho bảng so sánh trực tiếp

Phải thống nhất phiên bản dữ liệu, nhãn, quy tắc lọc/gộp exposure, user split, overlap train/test, candidate pool, user không positive, công thức Recall, backbone và ngân sách tuning. Sau đó cố định cấu hình bằng validation, chạy nhiều seed và báo mean ± std. Hiện chưa đủ điều kiện kết luận thắng/thua giữa các dòng trên và benchmark dự án.

[Kết quả địa phương có artifact](../results/README.md) · [Phân tích sai số](benchmark_v2_plan_and_error_analysis.md)
