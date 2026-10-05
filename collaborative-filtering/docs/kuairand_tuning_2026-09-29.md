# Kiểm tra siêu tham số KuaiRand-Pure (29-09-2026)

Thử nghiệm trên toàn bộ KuaiRand-Pure, seed 42, kiến trúc IViDR v5 địa phương.
Chọn checkpoint theo **validation NDCG@5**. Không dùng test để chọn siêu tham số.

| Mô hình | Cấu hình | Validation NDCG@5 tốt nhất | Epoch tốt nhất | Ghi chú |
| :-- | :-- | --: | --: | :-- |
| MF | LR 0,001 | 0,342165 | 4 | Checkpoint hiện có; giữ mặc định |
| DR-BIAS + CDR | LR 0,001; ngưỡng 0,002 | 0,343254 | 6 | Checkpoint hiện có; giữ mặc định |
| IViDR | LR 0,002; phi 0,2 | 0,313152 | 1 | Checkpoint hiện có; tối đa 25 epoch, patience 7 |
| IViDR | LR 0,001; phi 0,2 | **0,314394** | 1 | Thử 2 epoch trên dữ liệu đầy đủ: 0,314394 rồi 0,3102 |

Hai cấu hình IViDR cùng dim 128, latent dim 64, beta ELBO 0,001, batch 4096,
weight decay 1e-5 và cùng seed. LR 0,001 cao hơn LR 0,002 đúng 0,001243
validation NDCG@5 (~0,4% tương đối). Lượt LR 0,001 được chạy lại với 2 luồng và
cho cùng điểm epoch 1. Vì epoch 2 giảm, checkpoint chọn vẫn là epoch 1; lượt được
dừng sau epoch 2 do thời gian CPU. Đây **chưa phải grid hoàn chỉnh hoặc bằng chứng
về cải thiện ổn định nhiều seed**. Mặc định benchmark IViDR KuaiRand đổi sang LR
0,001 như một điều chỉnh nhỏ; MF và CDR giữ cấu hình cũ.

Sau khi chọn bằng validation, checkpoint LR 0,001 được chấm held-out test đúng một
lần: NDCG@5 **0,313292**, Recall@5 **0,281983**, Precision@5 **0,246749**,
HitRate@5 **0,681381**, AUC macro-user **0,543884**. Số test này không dùng để
chọn hay so tiếp các siêu tham số. Không suy p-value từ lượt một seed này.

Lệnh chạy lại benchmark với mặc định mới từ thư mục `collaborative-filtering`:

```powershell
.\venv\Scripts\python.exe experiments\run.py benchmark --dataset kuairand --model ividr
```

Grid validation rộng hơn có thể chạy bằng:

```powershell
.\venv\Scripts\python.exe experiments\run.py tune-kr --model ividr --epochs 25
```

Grid này tốn nhiều thời gian trên CPU. Các checkpoint/cache của thử nghiệm nêu
trên đã được xóa theo yêu cầu; chạy lại sẽ tạo artifact mới từ dữ liệu raw.
