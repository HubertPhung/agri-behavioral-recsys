# KuaiRand-Pure: Báo Cáo Tinh Chỉnh Siêu Tham Số (Validation Tuning)

- **Số Epoch Tối Đa**: 25 epochs (kèm Early Stopping)
- **Giao thức thực nghiệm**: Chọn cấu hình tốt nhất hoàn toàn bằng tập Validation (30% Random log).
- **Đánh giá khách quan**: Chỉ chấm điểm Held-Out Test DUY NHẤT 1 LẦN cho mô hình được chọn.

| Mô hình | Cấu hình tối ưu | Val NDCG@5 | Test NDCG@5 | Test Recall@5 | Test Precision@5 | Test HitRate@5 | Test AUC |
| :-- | :-- | --: | --: | --: | --: | --: | --: |
| MF | `{"batch_size": 4096, "embedding_dim": 128, "lr": 0.0005, "weight_decay": 1e-05}` | 0.3428 | 0.3434 | 0.2948 | 0.2677 | 0.7017 | 0.5560 |
| IVIDR | `{"batch_size": 4096, "beta_elbo": 0.001, "embedding_dim": 128, "latent_dim": 32, "lr": 0.002, "phi": 0.1}` | 0.3510 | 0.3484 | 0.3049 | 0.2695 | 0.7170 | 0.5606 |
| CDR | `{"batch_size": 8192, "embedding_dim": 128, "lr": 0.001, "un_thres": 0.005}` | 0.3452 | 0.3459 | 0.2990 | 0.2694 | 0.7102 | 0.5609 |

### Chi Tiết Toàn Bộ Các Trials Trong Grid Search

| Mô hình | Cấu hình thử nghiệm | Best Epoch | Val NDCG@5 | Thời gian (s) |
| :-- | :-- | --: | --: | --: |
| MF | `{"batch_size": 4096, "embedding_dim": 128, "lr": 0.001, "weight_decay": 1e-05}` | 4 | 0.3422 | 136.3s |
| MF | `{"batch_size": 4096, "embedding_dim": 128, "lr": 0.0005, "weight_decay": 1e-05}` | 6 | 0.3428 | 162.2s |
| MF | `{"batch_size": 4096, "embedding_dim": 128, "lr": 0.002, "weight_decay": 1e-05}` | 2 | 0.3409 | 102.2s |
| MF | `{"batch_size": 4096, "embedding_dim": 64, "lr": 0.001, "weight_decay": 1e-05}` | 4 | 0.3419 | 62.1s |
| IVIDR | `{"batch_size": 4096, "beta_elbo": 0.001, "embedding_dim": 128, "latent_dim": 64, "lr": 0.002, "phi": 0.1}` | 12 | 0.3466 | 4857.2s |
| IVIDR | `{"batch_size": 4096, "beta_elbo": 0.001, "embedding_dim": 128, "latent_dim": 64, "lr": 0.002, "phi": 0.05}` | 12 | 0.3492 | 4761.1s |
| IVIDR | `{"batch_size": 4096, "beta_elbo": 0.001, "embedding_dim": 128, "latent_dim": 64, "lr": 0.002, "phi": 0.2}` | 6 | 0.3398 | 3329.1s |
| IVIDR | `{"batch_size": 4096, "beta_elbo": 0.001, "embedding_dim": 128, "latent_dim": 64, "lr": 0.001, "phi": 0.1}` | 15 | 0.3434 | 5685.9s |
| IVIDR | `{"batch_size": 4096, "beta_elbo": 0.001, "embedding_dim": 128, "latent_dim": 32, "lr": 0.002, "phi": 0.1}` | 13 | 0.3510 | 4915.4s |
| CDR | `{"batch_size": 8192, "embedding_dim": 128, "lr": 0.001, "un_thres": 0.005}` | 8 | 0.3452 | 1657.4s |
| CDR | `{"batch_size": 8192, "embedding_dim": 128, "lr": 0.001, "un_thres": 0.002}` | 8 | 0.3438 | 1636.4s |
| CDR | `{"batch_size": 8192, "embedding_dim": 128, "lr": 0.001, "un_thres": 0.01}` | 8 | 0.3450 | 1629.2s |
| CDR | `{"batch_size": 8192, "embedding_dim": 128, "lr": 0.0005, "un_thres": 0.005}` | 17 | 0.3450 | 2710.9s |
| CDR | `{"batch_size": 8192, "embedding_dim": 128, "lr": 0.002, "un_thres": 0.005}` | 5 | 0.3428 | 1356.9s |

*(Báo cáo tự động tạo bởi TopTop Collaborative Filtering Pipeline)*
