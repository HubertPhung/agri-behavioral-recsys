# Giao thức đánh giá chung

Nguồn thực thi duy nhất: `src/evaluation/protocol.py`; evaluator dùng chung:
`src/evaluation/evaluator.py`. Xem cấu hình bằng `python experiments/run.py protocol`.
MF, IViDR và CDR dùng cùng evaluator ở validation và test.

## Quy tắc được cố định

| Thành phần | Quy tắc |
| :-- | :-- |
| Train | Biased log; giữ nhãn dương và âm đã exposure |
| Nhãn | Coat rating >= 4; KuaiRand chỉ is_click |
| Validation/test | Random log, user split 30/70, seed 1234 |
| Candidate | Những item xuất hiện trong random log của user; không sampled negatives |
| NDCG@5 | DCG nhị phân, discount 1/log2(rank+1); IDCG đến min(5, số positive) |
| Recall@5 | Số positive trúng top 5 / toàn bộ positive trong candidate pool |
| Macro average | Trung bình theo user có ít nhất một positive |
| Chọn model | Validation NDCG@5; test chỉ chấm checkpoint/config đã chọn |
| Nhiều seed | Config cố định; mặc định 42–51; mean và sample std (ddof=1) |
| Một seed | Std không xác định (null/N/A), không ghi 0 để biểu thị độ ổn định |
| Auxiliary | Precision, HitRate, AUC macro theo user có hai lớp; AUC ties tính 0,5 |

`Recall@5_bounded` vẫn có trong JSON để chẩn đoán; **không dùng thay Recall@5**.
AUC không có user đủ hai lớp trả null và auc_users=0. CI theo user của một lượt
chạy không thay thế std giữa seed.

[iDCF paper, mục 5.1 và phụ lục B](https://reijz.github.io/papers/2023-debiasing-recommendation-by-learning-ide.pdf)
là căn cứ chọn tỷ lệ split, nhãn và báo cáo 10 seed.
[Mã iDCF utils.py](https://github.com/BgmLover/iDCF/blob/main/utils.py) là căn cứ
cho chia user, Recall chia tổng positive và loại user không positive khỏi macro metric.

## Phạm vi tương đồng với paper

Đây là **giao thức địa phương bám theo iDCF**, không phải cam kết tái lập mọi bài.
[CDR mục 4.1](https://jiawei-chen.github.io/paper/CIKM23-CDR.pdf) dùng 10% random data
validation và 90% test, đồng thời tập KuaiRand khác kích thước. Điểm CDR chạy ở
đây chỉ so sánh được với các mô hình khác trong cùng thí nghiệm; không đặt ngang
điểm gốc CDR như một benchmark có điều kiện giống nhau.

Khác biệt cần công bố:

- Local gộp exposure lặp thành item duy nhất, positive nếu có ít nhất một click.
  Mã iDCF đưa trực tiếp các dòng vào CSR và cộng giá trị trùng; vì vậy tương đương
  metric chỉ được kiểm chứng trên nhãn nhị phân, cặp duy nhất và score không hòa.
- Local xử lý score hòa bằng item ID tăng dần để tái lập; tác giả dùng argpartition
  không quy định tie-break tương đương. Không khẳng định khớp bitwise khi score hòa.
- Local chỉ xếp hạng candidate đã log, kể cả khi ít hơn 5. Không thêm item ngoài
  log để đủ 5. Có kiểm thử riêng cho trường hợp này.
- Coat giữ các cặp train/random trùng và báo số overlap; không tự mask thành
  bài toán unseen-item. Full-catalog evaluator là API khác, không dùng trong runner.
- MF mặc định BCE, kiến trúc causal và ngân sách tuning khác mã tác giả.
  Giống metric không đồng nghĩa tái lập được điểm hoặc kết luận nhân quả của paper.

Report và checkpoint ghi `metric_revision`, công thức/aggregation, candidate pool,
split, quy tắc tie, data/source hash và cờ `paper_scores_directly_comparable=false`.
Checkpoint cũ thiếu revision này không được âm thầm chấm lại bằng metric mới.

## Kiểm thử

`python experiments/run.py test` gom toàn bộ suite: kiểm tra công thức NDCG/Recall
bằng tính tay, so với dense-ranking độc lập theo quy ước iDCF, positive > k,
candidate < k, zero-positive user, candidate sai, score NaN/Inf, tie-break,
seed std, split rời nhau và chọn config trước khi đọc điểm test.

`python experiments/run.py smoke` kiểm tra đường chạy ba model trên cả hai dataset.
Coat dùng toàn bộ dữ liệu; KuaiRand mặc định lấy 64 user. Report luôn ghi smoke;
điểm này không dùng so sánh chất lượng với bài báo. Muốn đánh giá thực nghiệm,
chạy benchmark đầy đủ, tune bằng validation và lặp seed trên cấu hình cố định.
