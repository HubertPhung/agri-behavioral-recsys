# Lượt chạy theo thông số công bố của Table 3 IViDR

Phạm vi người dùng xác nhận: **MF và IViDR trên Coat/KuaiRand**; không bao gồm Yahoo
hay bảy baseline còn lại. Không dùng số trong paper làm kết quả mô hình địa phương.

## Thông số khóa trước khi chấm test

- Adam; grid LR = {1e-3, 5e-4, 1e-4, 5e-5, 1e-5}, decay = {1e-5, 1e-6}.
- 10 cấu hình/dataset; chọn bằng validation NDCG@5 ở seed 42.
- Cố định cấu hình rồi chạy 10 seed 42–51; mỗi seed chọn checkpoint bằng validation.
- Full Coat và Full KuaiRand-Pure; không smoke/subsample user/item.
- NDCG@5 và Recall@5; báo mean ± sample std.

Nguồn grid, optimizer và số lượt: [IViDR mục 5.1](https://arxiv.org/html/2410.12451v1#S5.SS1).
Các lựa chọn không được bài báo xác định rõ được ghi ở metadata, không coi là thông số tác giả:

| Lựa chọn triển khai | Giá trị dùng cho MF |
| :-- | :-- |
| Epoch tối đa / patience | 100 / 20; có early stopping, báo epoch thực tế và chạm trần |
| Batch Coat / KuaiRand | 512 / 4096 |
| Dim Coat / KuaiRand | 32 / 128, theo kích thước IViDR; dim riêng MF chưa được công bố rõ |
| Loss / regularization | BCE / L2 trên tham số; không scheduler |
| MF score | Dot product + user bias + item bias; global intercept cố định 0 theo Eq. 23 |
| Khởi tạo | Embedding Normal(0, .01²), bias 0 |
| Seed / tuning | Dãy 42–51; tune bằng validation seed 42 |
| Thiết bị | CPU; fused Adam, hai worker độc lập, hai thread mỗi worker |

Fused Adam giữ phương trình cập nhật Adam; đã kiểm tra 10 bước với L2 ở hai mức
decay so với Adam chuẩn, sai số trong atol/rtol 1e-6. Nó không đổi thành AdamW.
Không dùng thời gian chạy song song làm benchmark tốc độ so bài báo.

## Ranh giới tái lập IViDR

MF có thể chạy với score đã công bố và các lựa chọn trên. **IViDR chưa đủ điều kiện
xác nhận tái lập nguyên bản**, do chưa tìm được implementation/config tác giả qua
paper arXiv v1, tìm kiếm web và GitHub repository search.

Phiên bản `ividr_v5` đã sửa phép chiếu thành `Z_j @ pinv(Z_j) @ MLP0(T_j)` riêng
theo item, với user context chỉ lấy từ biased train. Dùng cơ sở không gian đặc trưng
để giảm kích thước SVD; các item có cùng không gian đầy đủ dùng chung cơ sở.
Kiểm thử đối chiếu với phép nhân pseudoinverse trực tiếp, gồm cả item không có lịch sử.

| Hạng mục | Paper | Code IViDR hiện tại | Kết luận |
| :-- | :-- | :-- | :-- |
| Tái tạo treatment | Eq. 3–7: `Z_j` riêng theo item, Moore–Penrose pseudoinverse | `ItemIVTreatmentReconstruction`: pinv riêng theo item, embedding features tuyến tính | Đã sửa phép chiếu; mean pooling cho gate là giả định địa phương |
| Input iVAE | Posterior điều kiện trên exposure `A`, proxy `W`; kết hợp `X` và treatment | `ividr_model.py`: embedding MF của user, cộng mean treatment trong lịch sử cho nhánh thứ nhất | Cần xác định biểu diễn `X` và ánh xạ exposure theo implementation tác giả |
| ELBO | Eq. 40–44: likelihood cộng trên item và KL Gaussian | Sum KL theo latent, sum likelihood; sample uniform 128 item nhân N/K | Đã sửa hệ số; lấy mẫu là estimator của full sum |
| Quy trình train | Appendix B: tái tạo treatment, học confounder, sau đó train recommender | `train_ividr.py`: joint BCE + beta ELBO, KL warmup | Khác lịch tối ưu |
| Proxy | Paper đưa ví dụ proxy, chưa chỉ rõ toàn bộ cột cho mỗi dataset | Coat ghép features với số interaction; KuaiRand trích các cột feature | Lựa chọn địa phương, chưa xác nhận tương đương |
| Các bổ sung | Chưa công bố LayerNorm, clipping, scheduler | LayerNorm của C, clip gradient 1.0, scheduler mặc định none | LayerNorm/clipping vẫn là lựa chọn địa phương |

Vì vậy, không có đủ bằng chứng để gọi code IViDR hiện tại là implementation chính
xác của Table 3. Có thể tự xây dựng bản tham chiếu từ phương trình, nhưng các quyết
định còn thiếu phải được công bố là giả định; chạy nhiều seed không tự khắc phục
khác biệt thuật toán. Cần mã/config hoặc mô tả bổ sung của tác giả để xác nhận
tái lập nguyên bản, đặc biệt cách tạo `X`, `Z_j`, `W` và lịch huấn luyện từng giai đoạn.

Không chặn mọi thử nghiệm khi thiếu code tác giả: workflow đã hỗ trợ chạy bản
tham chiếu có khai báo giả định. Trạng thái là `local_reference_not_author_implementation`.
Không suy p-value từ mean/std của paper; hoàn tất lượt chạy vẫn có
`paper_reproduction_complete=false`.

## Chạy và tiếp tục MF + IViDR tham chiếu

```powershell
.\venv\Scripts\python.exe experiments\run.py table3 --dataset all --model all --threads 2 --workers 2
```

Lệnh này chạy **MF và IViDR tham chiếu**. Chọn riêng bằng `--model mf` hoặc `--model ividr`.
File `table3_MODEL_DATASET_HASH.json` được cập nhật sau
mỗi trial/seed, có source/data hash và chỉ rõ trạng thái; Markdown cùng tên để đọc.
Chạy lại đúng lệnh sẽ bỏ qua trial/seed đã hoàn tất, chỉ train lại job đang dở.
Không sửa source hoặc dọn checkpoint trong lúc chạy. Runner có khóa tiến trình;
resume từ chối báo cáo mất checkpoint, và source hash được kiểm tra trước/sau train.

`mf_reference_complete` hoặc `ividr_reference_complete` chỉ hoàn tất riêng một
mô hình/dataset. Checkpoint IViDR v4 bị từ chối khi dùng kiến trúc v5; phải train mới.
Artifact các lượt cũ đã không còn trên đĩa tại lần kiểm tra tiếp theo; không dùng
các con số từng báo trước đó làm kết quả cho mã mới.

## MF và optimizer

MF là kiến trúc phân rã ma trận, còn Adam/SGD là cách cập nhật tham số. Lượt MF này
tính `p_u dot q_i + b_u + b_i` rồi tối ưu BCE bằng Adam. ALS dạng bình phương tối
thiểu luân phiên thông thường dùng objective khác, nên thay Adam bằng ALS sẽ
không còn giữ thiết lập tối ưu mà Table 3 công bố. Việc bài báo chọn Adam không
chứng minh Adam luôn tốt nhất cho mọi dữ liệu và objective; các phép thử Adam/SGD
ngắn trước đây cũng không đủ để kết luận điều đó.
