import os
import cv2
import time
import json
import socket
import uvicorn
import onnxruntime
import numpy as np
from typing import Optional
from datetime import datetime
from fastapi import Header, Query
from insightface.app import FaceAnalysis
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse
from fastapi import FastAPI, Request, UploadFile, Form, File, HTTPException

app = FastAPI()

# ================= CẤU HÌNH THƯ MỤC =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACE_FOLDER = os.path.join(BASE_DIR, "face_data")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
LOG_FOLDER = os.path.join(BASE_DIR, "logs")
os.makedirs(FACE_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(LOG_FOLDER, exist_ok=True)

app.mount("/uploads", StaticFiles(directory=UPLOAD_FOLDER), name="uploads")
app.mount("/face_data", StaticFiles(directory=FACE_FOLDER), name="face_data")

# ================= CACHE & GLOBAL VARS =================
uid_encoding_cache = {}  # { uid: embedding_vector }
active_sessions = {}     # { uid: {"status": "pending"/"yess"/"noo", "ts": epoch_seconds} }
SESSION_TTL_SEC = 45
THRESHOLD = 0.45  # Ngưỡng tương đồng
UPLOAD_PASSWORD = "nhom1" # Password trang web

# ================= KHỞI TẠO INSIGHTFACE =================
print("[InsightFace] Đang khởi tạo model...")
face_app = FaceAnalysis(
    name='buffalo_l', 
    providers=['CPUExecutionProvider']
)
face_app.prepare(ctx_id=0, det_size=(640, 640))
print("[InsightFace] Khởi tạo hoàn tất!")

# Danh sách nhận diện
known_face_embeddings = []
known_face_names = []

# ================= CÁC HÀM HỖ TRỢ (UTILS) =================
def now_ts() -> int:
    return int(time.time())

def cleanup_sessions():
    expired = [uid for uid, s in active_sessions.items() if now_ts() - s["ts"] > SESSION_TTL_SEC]
    for uid in expired:
        del active_sessions[uid]

def find_uid_image_path(uid: str) -> Optional[str]:
    targets = [f"{uid}.jpg", f"{uid}.jpeg", f"{uid}.png"]
    lower_targets = {t.lower() for t in targets}
    for fn in os.listdir(FACE_FOLDER):
        if fn.lower() in lower_targets:
            return os.path.join(FACE_FOLDER, fn)
    return None

def extract_embedding(image_path: str):
    try:
        img = cv2.imread(image_path)
        if img is None: return None
        faces = face_app.get(img)
        if len(faces) == 0:
            print(f"[Warning] Không phát hiện khuôn mặt trong {image_path}")
            return None
        embedding = faces[0].embedding
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    except Exception as e:
        print(f"[Error] {e}")
        return None

def load_uid_encoding(uid: str):
    if uid in uid_encoding_cache: return uid_encoding_cache[uid]
    path = find_uid_image_path(uid)
    if not path: return None
    embedding = extract_embedding(path)
    if embedding is not None:
        uid_encoding_cache[uid] = embedding
    return embedding

def load_known_faces():
    global known_face_embeddings, known_face_names
    known_face_embeddings = []
    known_face_names = []
    print("[Load] Đang load face database...")
    for file in os.listdir(FACE_FOLDER):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            uid = os.path.splitext(file)[0]
            path = os.path.join(FACE_FOLDER, file)
            embedding = extract_embedding(path)
            if embedding is not None:
                known_face_embeddings.append(embedding)
                known_face_names.append(uid)
                print(f"  ✓ Loaded: {uid}")
    print(f"[Load] Hoàn tất! Tổng {len(known_face_names)} khuôn mặt")

load_known_faces()

def load_uids():
    return [os.path.splitext(f)[0] for f in os.listdir(FACE_FOLDER) 
            if f.lower().endswith((".jpg", ".jpeg", ".png"))]

def delete_uid_file(uid: str):
    deleted = False
    for ext in [".jpg", ".jpeg", ".png"]:
        path = os.path.join(FACE_FOLDER, f"{uid}{ext}")
        if os.path.exists(path):
            os.remove(path)
            deleted = True
    if deleted:
        if uid in uid_encoding_cache: del uid_encoding_cache[uid]
        try:
            idx = known_face_names.index(uid)
            known_face_names.pop(idx)
            known_face_embeddings.pop(idx)
        except ValueError: pass
    return deleted

def cosine_similarity(emb1, emb2):
    return np.dot(emb1, emb2)

# Hàm tạo HTML cho Gallery (Tách riêng để dùng chung cho load trang và API cập nhật)
def generate_gallery_html_content():
    all_files = sorted(os.listdir(UPLOAD_FOLDER), reverse=True)
    gallery_files = [f for f in all_files if f.lower().endswith((".jpg", ".jpeg", ".png"))][:50]
    
    html = ""
    if gallery_files:
        for f in gallery_files:
            url = f"/uploads/{f}"
            html += f"""
            <div class="gallery-item">
                <a href="{url}" target="_blank"><img src='{url}' loading="lazy"></a>
                <p>{f}</p>
            </div>
            """
    else:
        html = "<p style='text-align:center; color:#666;'>Chưa có lịch sử nhận diện nào.</p>"
    return html

# ================= GIAO DIỆN GỘP (DASHBOARD) =================

@app.get("/dashboard/gallery-content", response_class=HTMLResponse)
async def get_gallery_content():
    """API trả về HTML của phần Gallery để AJAX cập nhật"""
    return generate_gallery_html_content()

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_view():
    # 1. Lấy dữ liệu cho Tab "Quản lý Face Data"
    uids = load_uids()
    uid_rows = ""
    if uids:
        for uid in uids:
            img_path = find_uid_image_path(uid)
            if img_path:
                ext = os.path.splitext(img_path)[1]
                display_path = f"/face_data/{uid}{ext}"
                uid_rows += f"""
                <tr>
                    <td><strong>{uid}</strong></td>
                    <td style="text-align:center;">
                        <img src='{display_path}' class="face-thumb">
                    </td>
                    <td>
                        <form method="POST" action="/dashboard/delete" class="action-form">
                            <input type="hidden" name="delete_uid" value="{uid}">
                            <input type="password" name="password" placeholder="Mật khẩu xóa" class="pw-input" required>
                            <button type="submit" class="btn btn-danger">Xóa</button>
                        </form>
                    </td>
                </tr>
                """
    else:
        uid_rows = "<tr><td colspan='3' style='text-align:center;'>Chưa có dữ liệu khuôn mặt.</td></tr>"

    # 2. Lấy dữ liệu cho Tab "Lịch sử nhận diện" (Gallery) -> Gọi hàm helper
    gallery_html = generate_gallery_html_content()

    # 3. HTML Template
    html = f"""
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>HỆ THỐNG QUẢN LÝ PHÒNG HỌC (DESIGN BY NHOM1) - Dashboard</title>
        <style>
            :root {{ --primary: #0078ff; --danger: #ff4444; --bg: #f4f6f9; --white: #ffffff; }}
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: var(--bg); margin: 0; padding: 20px; }}
            .container {{ max-width: 1000px; margin: auto; background: var(--white); border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); overflow: hidden; }}
            
            /* Header */
            .header {{ background: var(--primary); color: var(--white); padding: 20px; text-align: center; }}
            .header h2 {{ margin: 0; font-size: 24px; }}
            .badge {{ background: rgba(255,255,255,0.2); padding: 4px 10px; border-radius: 20px; font-size: 14px; margin-left: 10px; }}

            /* Tabs */
            .tabs {{ display: flex; border-bottom: 2px solid #eee; }}
            .tab-btn {{ flex: 1; padding: 15px; background: #f9f9f9; border: none; outline: none; cursor: pointer; font-size: 16px; font-weight: 600; color: #555; transition: 0.3s; }}
            .tab-btn:hover {{ background: #eee; }}
            .tab-btn.active {{ background: var(--white); color: var(--primary); border-bottom: 3px solid var(--primary); }}

            /* Content Area */
            .tab-content {{ display: none; padding: 30px; animation: fadeIn 0.4s; }}
            .tab-content.active {{ display: block; }}
            @keyframes fadeIn {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}

            /* Form Styles */
            label {{ display: block; margin: 10px 0 5px; font-weight: 600; }}
            input[type="text"], input[type="password"], input[type="file"] {{ width: 100%; padding: 10px; margin-bottom: 15px; border: 1px solid #ddd; border-radius: 6px; box-sizing: border-box; }}
            .btn {{ padding: 10px 20px; border: none; border-radius: 6px; cursor: pointer; color: white; font-weight: 600; }}
            .btn-primary {{ background: var(--primary); width: 100%; }}
            .btn-primary:hover {{ background: #005fcc; }}
            .btn-danger {{ background: var(--danger); padding: 6px 12px; font-size: 13px; }}
            .btn-danger:hover {{ background: #cc0000; }}

            /* Table Styles */
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            th, td {{ padding: 12px; border-bottom: 1px solid #eee; text-align: left; }}
            th {{ background: #f8f9fa; color: #333; }}
            .face-thumb {{ width: 60px; height: 60px; object-fit: cover; border-radius: 50%; border: 2px solid #ddd; }}
            .action-form {{ display: flex; align-items: center; gap: 5px; }}
            .pw-input {{ width: 100px !important; margin-bottom: 0 !important; font-size: 12px; }}

            /* Gallery Grid */
            .gallery-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 15px; }}
            .gallery-item {{ text-align: center; background: #fff; padding: 10px; border: 1px solid #eee; border-radius: 8px; transition: transform 0.2s; }}
            .gallery-item:hover {{ transform: translateY(-3px); box-shadow: 0 4px 10px rgba(0,0,0,0.1); }}
            .gallery-item img {{ width: 100%; height: 120px; object-fit: cover; border-radius: 6px; }}
            .gallery-item p {{ margin: 8px 0 0; font-size: 12px; color: #666; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
            
            /* Loading indicator */
            .live-indicator {{ font-size: 12px; color: green; display: none; margin-left: 10px; animation: blink 1s infinite; }}
            @keyframes blink {{ 50% {{ opacity: 0; }} }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h2>HỆ THỐNG QUẢN LÝ PHÒNG HỌC (DESIGN BY NHOM1) <span class="badge">InsightFace</span></h2>
            </div>
            
            <div class="tabs">
                <button class="tab-btn active" onclick="openTab(event, 'Manage')">Face data Giảng viên</button>
                <button class="tab-btn" onclick="openTab(event, 'History')">Lịch sử Nhận diện</button>
            </div>

            <div id="Manage" class="tab-content active">
                <div style="display: flex; gap: 30px; flex-wrap: wrap;">
                    <div style="flex: 1; min-width: 300px; background: #f9f9f9; padding: 20px; border-radius: 8px;">
                        <h3 style="margin-top:0; color:#444;">Thêm khuôn mặt mới</h3>
                        <form method="POST" action="/dashboard/upload" enctype="multipart/form-data">
                            <label>Mật khẩu quản trị:</label>
                            <input type="password" name="password" required placeholder="Nhập password">
                            
                            <label>UID (Mã thẻ):</label>
                            <input type="text" name="uid" required placeholder="">
                            
                            <label>Ảnh mẫu (JPG/JPEG/PNG):</label>
                            <input type="file" name="file" accept=".jpg,.jpeg,.png" required>
                            
                            <button type="submit" class="btn btn-primary">Tải lên</button>
                        </form>
                    </div>

                    <div style="flex: 2; min-width: 300px;">
                        <h3 style="margin-top:0; color:#444;">Danh sách đã đăng ký ({len(uids)})</h3>
                        <table>
                            <thead>
                                <tr>
                                    <th>UID</th>
                                    <th style="text-align:center">Face data</th>
                                    <th>Password</th>
                                </tr>
                            </thead>
                            <tbody>
                                {uid_rows}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>

            <div id="History" class="tab-content">
                <h3 style="margin-top:0; color:#444; display:flex; align-items:center;">
                    Ảnh chụp gần nhất (50 ảnh)
                    <span class="live-indicator" id="liveTag">● Đang cập nhật tự động</span>
                </h3>
                <div class="gallery-grid" id="galleryGrid">
                    {gallery_html}
                </div>
            </div>
        </div>

        <script>
            function openTab(evt, tabName) {{
                var i, tabcontent, tablinks;
                tabcontent = document.getElementsByClassName("tab-content");
                for (i = 0; i < tabcontent.length; i++) {{
                    tabcontent[i].style.display = "none";
                    tabcontent[i].classList.remove("active");
                }}
                tablinks = document.getElementsByClassName("tab-btn");
                for (i = 0; i < tablinks.length; i++) {{
                    tablinks[i].className = tablinks[i].className.replace(" active", "");
                }}
                document.getElementById(tabName).style.display = "block";
                document.getElementById(tabName).classList.add("active");
                evt.currentTarget.className += " active";

                // Hiển thị indicator nếu mở tab History
                if(tabName === 'History') {{
                    document.getElementById('liveTag').style.display = 'inline-block';
                }} else {{
                    document.getElementById('liveTag').style.display = 'none';
                }}
            }}
            
            // Auto Refresh Logic
            setInterval(function() {{
                // Chỉ fetch data nếu đang ở tab History
                if(document.getElementById('History').classList.contains('active')) {{
                    fetch('/dashboard/gallery-content')
                        .then(response => response.text())
                        .then(html => {{
                            document.getElementById('galleryGrid').innerHTML = html;
                        }})
                        .catch(err => console.error('Lỗi cập nhật ảnh:', err));
                }}
            }}, 2000); // Cập nhật mỗi 2 giây

            // Mở tab dựa trên URL hash
            if(window.location.hash === "#History") {{
                 document.querySelector("button[onclick*='History']").click();
            }}
        </script>
    </body>
    </html>
    """
    return HTMLResponse(html)

@app.post("/dashboard/upload", response_class=HTMLResponse)
async def dashboard_upload(password: str = Form(...), uid: str = Form(...), file: UploadFile = File(...)):
    if password != UPLOAD_PASSWORD:
        return HTMLResponse(f"<script>alert('Sai mật khẩu!'); window.history.back();</script>")
    
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        return HTMLResponse(f"<script>alert('Chỉ hỗ trợ file ảnh!'); window.history.back();</script>")
    
    ext = os.path.splitext(file.filename)[1]
    save_path = os.path.join(FACE_FOLDER, f"{uid}{ext}")
    
    with open(save_path, "wb") as f:
        f.write(await file.read())
    
    embedding = extract_embedding(save_path)
    if embedding is not None:
        if uid in known_face_names:
            idx = known_face_names.index(uid)
            known_face_names.pop(idx)
            known_face_embeddings.pop(idx)
        known_face_embeddings.append(embedding)
        known_face_names.append(uid)
        uid_encoding_cache[uid] = embedding
        msg = f"Đã thêm {uid} thành công!"
    else:
        msg = f"Cảnh báo: Không tìm thấy mặt trong ảnh của {uid}!"

    # Quay lại trang dashboard tab Manage
    return HTMLResponse(f"<script>alert('{msg}'); window.location.href='/dashboard';</script>")

@app.post("/dashboard/delete", response_class=HTMLResponse)
async def dashboard_delete(password: str = Form(...), delete_uid: str = Form(...)):
    if password != UPLOAD_PASSWORD:
        return HTMLResponse(f"<script>alert('Sai mật khẩu!'); window.history.back();</script>")
    
    success = delete_uid_file(delete_uid)
    msg = f"Đã xóa {delete_uid}" if success else "Không tìm thấy UID"
    
    # Quay lại trang dashboard
    return HTMLResponse(f"<script>alert('{msg}'); window.location.href='/dashboard';</script>")

# Redirect root to dashboard
@app.get("/")
async def root():
    return RedirectResponse(url="/dashboard")

# ================= API NHẬN DIỆN (GIỮ NGUYÊN) =================
@app.post("/precheck")
async def precheck_uid(request: Request):
    cleanup_sessions()
    try:
        payload = await request.json()
        uid = str(payload.get("uid", "")).strip()
    except Exception: return PlainTextResponse("no", status_code=400)
    if not uid: return PlainTextResponse("no", status_code=400)
    enc = load_uid_encoding(uid)
    if enc is None: return PlainTextResponse("no")
    active_sessions[uid] = {"status": "pending", "ts": now_ts()}
    return PlainTextResponse("yes")

@app.get("/result")
async def get_result(uid: str = Query(...)):
    cleanup_sessions()
    s = active_sessions.get(uid)
    if not s: return PlainTextResponse("no")
    s["ts"] = now_ts()
    return PlainTextResponse(s["status"])

@app.post("/recognize")
async def recognize_face(request: Request, x_uid: Optional[str] = Header(None), x_last_frame: Optional[str] = Header(None)):
    cleanup_sessions()
    image_bytes = await request.body()
    if not image_bytes: return PlainTextResponse("pending", status_code=400)
    
    uid = None
    if x_uid: uid = x_uid.strip()
    else:
        if active_sessions: uid = max(active_sessions.items(), key=lambda kv: kv[1]["ts"])[0]
    
    if not uid or uid not in active_sessions: return PlainTextResponse("pending", status_code=428)
    
    if active_sessions[uid]["status"] in ("yess", "noo"):
        active_sessions[uid]["ts"] = now_ts()
        return PlainTextResponse(active_sessions[uid]["status"])
    
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None: return PlainTextResponse("pending", status_code=400)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    raw_path = os.path.join(UPLOAD_FOLDER, f"{timestamp}_raw.jpg")
    cv2.imwrite(raw_path, frame) # Save ảnh để hiển thị ở Gallery
    
    enc_expected = load_uid_encoding(uid)
    if enc_expected is None:
        active_sessions[uid]["status"] = "noo"
        return PlainTextResponse("noo")
    
    is_last = (str(x_last_frame).strip() == "1")
    frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=5)
    
    try: faces = face_app.get(frame)
    except Exception:
        if is_last:
            active_sessions[uid]["status"] = "noo"
            return PlainTextResponse("noo")
        return PlainTextResponse("pending")
    
    matched = False
    best_similarity = 0.0
    if len(faces) > 0:
        for face in faces:
            embedding = face.embedding
            embedding = embedding / np.linalg.norm(embedding)
            similarity = cosine_similarity(embedding, enc_expected)
            if similarity > best_similarity: best_similarity = similarity
            if similarity >= THRESHOLD:
                matched = True
                break
    
    with open(os.path.join(LOG_FOLDER, "recognition_log.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "timestamp": datetime.now().isoformat(), "uid": uid,
            "image_path": raw_path, "similarity": round(float(best_similarity), 4),
            "matched": matched
        }, ensure_ascii=False) + "\n")
    
    if matched:
        active_sessions[uid]["status"] = "yess"
        return PlainTextResponse("yess")
    else:
        if is_last:
            active_sessions[uid]["status"] = "noo"
            return PlainTextResponse("noo")
        else:
            return PlainTextResponse("pending")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)