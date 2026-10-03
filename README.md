# Repo này thực hiện Nghiên cứu và thực nghiệm các phương pháp Contrastive Decoding nhằm giảm Object Hallucination cho Vision-Language Models. #

**Nhắc lại về VCD:** Cho một model VLMs $\theta$ với một ảnh $v$ và một prompt $x$. Ta cho mô hình sinh ra hai outputs riêng biệt, một với ảnh gốc $v$ và một với ảnh đã qua Gaussian noise $v'$. Quá trình sinh một token mới sử dụng VCD được mô tả bằng công thức sau

$$p_{vcd}(y|v, v', x) = \text{softmax}[(1+\alpha)\,\text{logit}_\theta(y|v,x) - \alpha\, \text{logit} _\theta(y|v',x)],$$

Trong đó $\alpha$ là một siêu tham số để điều chỉnh logits của các token với $alpha$ lớn thể hiện nhấn mạnh các logit của các token được sinh ra cùng với ảnh gốc, loại trừ các token sinh ra thông qua ảnh đã qua Gaussian noise.



Cấu trúc repo đề xuất 

```text
cd-vlm/
├── README.md
├── requirements.txt
├── .gitignore
├── src/cdvlm/
│   ├── __init__.py
│   ├── models/
│   │   ├── llava.py            # load model llava, tạo inputs, forward
│   │   └── qwen.py             # 
│   ├── transforms/
│   │   └── gaussian_noise.py   
│   └── decoding/
│       └── contrastive.py      # vòng lặp giải mã: regular và vcd
├── scripts/
│   └── run_demo.py             # chạy thử 1 ảnh từ dòng lệnh
├── tests/
│   └── test_regular_matches_generate.py
├── notebooks/
│   └── kaggle_llava.ipynb      # notebook mỏng, chỉ clone repo và gọi hàm
└── results/                    # (làm sau) kết quả đánh giá, file nhỏ dạng json
```