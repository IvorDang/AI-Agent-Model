# A7 — Intelligent System: Knowledge Graph + RAG + Agent

## 1. Phạm vi triển khai
Bản demo dùng Python, scikit-learn TF-IDF retrieval, NetworkX Knowledge Graph và agent dựa trên luật. Nó chạy không cần API key, phù hợp để thực hành trên Anaconda/Jupyter và triển khai giao diện Streamlit. Đây là prototype có thể chạy, không phải LLM production-grade: câu trả lời được tổng hợp từ bằng chứng truy hồi, chưa gọi LLM sinh tự do và chưa có mô hình DL huấn luyện riêng.

## 2. Chạy trong Anaconda (Windows)
Mở Anaconda Prompt, chuyển tới thư mục giải nén dự án:
```bash
conda create -n a7_rag python=3.11 -y
conda activate a7_rag
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter notebook
```
Mở `A7_Knowledge_Graph_RAG_Agent.ipynb`, chọn kernel `a7_rag`, chạy lần lượt từ trên xuống.

## 3. Deploy giao diện web local
Trong Anaconda Prompt, tại thư mục dự án:
```bash
streamlit run app.py
```
Mở URL local Streamlit được in trong terminal (thường là http://localhost:8501). Để chia sẻ trên mạng cần triển khai lên máy chủ/hosting và cấu hình bảo mật phù hợp; lệnh trên chỉ deploy local.

## 4. Dữ liệu
- `data/knowledge_base.json`: các đoạn tài liệu mẫu.
- `data/knowledge_graph.json`: triples (subject, relation, object).
- `evaluation/questions.csv`: bộ câu hỏi kiểm thử mẫu.

Thay nội dung mẫu bằng tài liệu môn học của bạn. Với PDF thật, có thể trích xuất văn bản bằng pypdf; cần kiểm tra kết quả OCR với PDF scan.

## 5. Chụp minh chứng
Chạy notebook và app thực tế rồi chụp:
1. `01_environment.png`: cell cài đặt/import thành công.
2. `02_knowledge_graph.png`: số node/cạnh và danh sách triples.
3. `03_retrieval.png`: kết quả truy hồi top-k, score và source.
4. `04_hybrid_answer.png`: câu hỏi + câu trả lời + KG facts + chunks nguồn.
5. `05_agent.png`: chế độ Intelligent Agent và kế hoạch công cụ.
6. `06_streamlit.png`: giao diện đang chạy trên trình duyệt.
Lưu vào thư mục `screenshots/`. Không nên dùng ảnh minh họa giả làm bằng chứng chạy thực tế.

## 6. Nâng cấp LLM (tùy chọn)
Kiến trúc được tách để có thể nối API LLM hoặc mô hình cục bộ. Chỉ thêm API khi được phép, không commit API key lên Git. Bản mặc định không yêu cầu API key để tránh lỗi cấu hình và chi phí.

## 7. Giới hạn
- TF-IDF mạnh ở từ khóa, không tương đương embedding neural hiện đại.
- Knowledge Graph được khai báo thủ công, không tự trích xuất entity/relationship từ mọi PDF.
- Agent dùng luật xác định, không có planning tự chủ bằng LLM.
- RAG không đảm bảo câu trả lời đúng; phải kiểm tra nguồn.
- Dataset nhỏ dùng cho minh họa; cần thêm dữ liệu và đánh giá có hệ thống trước khi dùng thật.
