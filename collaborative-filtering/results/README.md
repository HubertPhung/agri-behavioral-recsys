# Trạng thái kết quả

Các checkpoint, cache dữ liệu đã xử lý và log thử nghiệm tạm đã được dọn theo
yêu cầu. Thư mục này được giữ để các lượt chạy sau ghi báo cáo mới.

Xem [kiểm tra siêu tham số KuaiRand-Pure](../docs/kuairand_tuning_2026-09-29.md)
và [giới hạn tái lập Table 3](../docs/table3_run.md).

Từ thư mục `collaborative-filtering`, chạy:

```powershell
.\venv\Scripts\python.exe experiments\run.py benchmark --dataset kuairand
```

Lệnh `experiments\run.py clean` xem trước các file sinh ra;
`experiments\run.py clean --apply` xóa chúng sau khi các tiến trình train dừng.
