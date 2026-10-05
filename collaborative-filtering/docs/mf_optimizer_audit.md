# Kiểm tra MF và bộ tối ưu — 2026-09-27

## Kết luận

Code hiện tại thực hiện đúng **biased matrix factorization**:

`s_ui = p_u^T q_i + b_u + b_i + mu`.

Mặc định tối ưu `mean(c_ui * BCEWithLogits(s_ui, y_ui))` trên exposure đã quan sát,
cộng gradient L2 trên toàn bộ hai bảng embedding qua Adam weight decay.
Với Adam mặc định ở đây, weight decay là L2 ghép vào gradient, không phải AdamW.
Khởi tạo embedding Normal(0, .01²), bias=0. Forward trả logits; không sigmoid trước
BCEWithLogitsLoss. Score ranking dùng logits cũng cho thứ tự như sigmoid.
`alpha=0` cho weight=1; alpha>0 ở KuaiRand đặt weight positive=1+alpha.

MF là **mô hình**; Adam, SGD và ALS là **cách giải bài toán tối ưu**. Dùng Adam
không làm MF biến thành mô hình khác. Loss/sampling/regularization mới quyết định
MF đang học bài toán nào. Không có căn cứ kết luận một optimizer tốt nhất cho mọi dataset.

## Đối chiếu nguồn gốc

| Phương án | Mục tiêu và ý nghĩa với dự án |
| :-- | :-- |
| Adam | Lựa chọn có căn cứ nếu đối chiếu iDCF: paper dùng Adam cho mọi mô hình. |
| SGD | Tối ưu được cùng BCE/MSE; cần tune learning rate riêng, kiểm tra hội tụ và seed. |
| ALS cổ điển | Giải luân phiên các bài toán bình phương nhỏ nhất; không phải phép thay trực tiếp cho Adam trong BCE. |

[iDCF, KDD 2023, phụ lục B](https://reijz.github.io/papers/2023-debiasing-recommendation-by-learning-ide.pdf)
dùng validation để chọn learning rate {1e-3, 5e-4, 1e-4, 5e-5, 1e-5},
weight decay {1e-5, 1e-6}, Adam, và báo NDCG/Recall mean/std qua 10 seed.
Đây là căn cứ chọn Adam khi tái lập iDCF, không phải bằng chứng Adam thắng SGD/ALS.

[Trainer MF tác giả](https://github.com/BgmLover/iDCF/blob/main/MF.py) dùng **MSE**, Adam
và weight decay trên toàn bộ tham số. [Kiến trúc tác giả](https://github.com/BgmLover/iDCF/blob/main/models/mf.py)
khởi tạo uniform, có user/item bias và mean cố định bằng 0.
MF địa phương dùng BCE mặc định, Normal initialization, global bias học được,
embedding-only decay và ReduceLROnPlateau. Thêm tùy chọn MSE/all-decay/no-scheduler
chỉ thu hẹp một phần khác biệt; chưa được gọi là bản tái lập paper.

[Hu, Koren, Volinsky (ICDM 2008)](https://yifanhu.net/PUB/cf.pdf) dùng weighted squared
error trên toàn bộ user–item với confidence và L2. Khi cố định một phía, ALS giải
hệ tuyến tính cho phía còn lại. Unknown pairs có vai trò trong mục tiêu này;
khác với BCE trên click/non-click đã exposure của dự án. Nếu thêm implicit ALS,
cần đặt tên baseline riêng, tune confidence/L2 và dùng chung evaluator.

[Adam (Kingma & Ba)](https://arxiv.org/abs/1412.6980) là phương pháp gradient dùng
ước lượng moment thích nghi. Nó không cung cấp bảo đảm đứng đầu NDCG trên hai
dataset đang xét. Khuyến nghị thực hành ở đây: giữ Adam làm baseline chính khi
đối chiếu iDCF, dùng so sánh validation có kiểm soát để chọn cho ứng dụng cụ thể.

## Đánh giá và các giới hạn

- Train: Coat biased ratings; KuaiRand hai standard log, chỉ `is_click`.
- Random log chia user 30/70 validation/test, seed 1234; tập user hai split rời nhau.
- Chọn epoch/config bằng validation NDCG@5, nạp checkpoint đã chọn để chấm test.
- Candidate là logged exposures; user không positive bị loại khỏi macro metric.
- Coat vẫn có 247 cặp train/test trùng. Đây không phải phép đo unseen-item thuần túy.
- Gộp exposure lặp, công thức Recall, split, loss, regularization và seed phải khớp
  trước khi so điểm paper. Adam giống nhau không đủ để bảo đảm công bằng.
- IViDR/CDR địa phương còn khác kiến trúc và propensity với tác giả; thay optimizer
  không khắc phục những khác biệt đó.

## Thay đổi và kiểm chứng

- Trainer MF hỗ trợ Adam/SGD, BCE/MSE, decay embeddings/all và scheduler plateau/none.
- MSE dùng score thô, không sigmoid; SGD dùng momentum=0 để định nghĩa rõ phép so sánh.
- Metadata lưu các lựa chọn trên, phân biệt checkpoint theo identity.
- Sửa thống kê loss theo số mẫu thay vì trung bình đều các batch có kích thước khác nhau.
- Sửa wrapper MF không còn KeyError khi chỉ cung cấp `n_users`/`n_items` chuẩn.
- Công cụ so sánh chọn tất cả config trên validation trước khi đọc điểm test.
- Kiểm thử mới đối chiếu một bước SGD với gradient của mục tiêu BCE/MSE + L2,
  và kiểm tra test không tham gia chọn cấu hình.
- Dọn 161 file sinh ra cũ, 687.794.208 byte (~655,93 MiB), gồm 96 checkpoint,
  2 cache và 26 report/log. Giữ raw datasets, venv, source, papers.

[Báo cáo chạy mới](../results/README.md) ghi rõ đâu là phép so sánh, đâu là kiểm tra
pipeline chưa hội tụ. Báo cáo/slide lịch sử chỉ dùng tham khảo diễn tiến dự án.
