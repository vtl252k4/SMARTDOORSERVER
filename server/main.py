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
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi import FastAPI, Request, UploadFile, Form, File, HTTPException
app = FastAPI()

# Thư mục
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACE_FOLDER = os.path.join(BASE_DIR, "face_data")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
LOG_FOLDER = os.path.join(BASE_DIR, "logs")
os.makedirs(FACE_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(LOG_FOLDER, exist_ok=True)

app.mount("/uploads", StaticFiles(directory=UPLOAD_FOLDER), name="uploads")
app.mount("/face_data", StaticFiles(directory=FACE_FOLDER), name="face_data")

# Cache và session
uid_encoding_cache = {}  # { uid: embedding_vector }
active_sessions = {}     # { uid: {"status": "pending"/"yess"/"noo", "ts": epoch_seconds} }
SESSION_TTL_SEC = 45
THRESHOLD = 0.6  # Ngưỡng tương đồng

# Khởi tạo InsightFace
print("[InsightFace] Đang khởi tạo model...")
face_app = FaceAnalysis(
    name='buffalo_l',  # Model chính xác cao (dùng 'buffalo_s' nhanh hơn)
    providers=['CPUExecutionProvider']  # Tự động chọn GPU nếu có (thêm vào nếu có GPU'CUDAExecutionProvider')
)
face_app.prepare(ctx_id=0, det_size=(640, 640))  # det_size 
print("[InsightFace] Khởi tạo hoàn tất!")

# Danh sách nhận diện
known_face_embeddings = []  # List embedding vectors
known_face_names = []       # List UIDs tương ứng

# Password trang web
UPLOAD_PASSWORD = "123456"

def now_ts() -> int:
    return int(time.time())

def cleanup_sessions():
    """Xóa các session hết hạn"""
    expired = [uid for uid, s in active_sessions.items() if now_ts() - s["ts"] > SESSION_TTL_SEC]
    for uid in expired:
        del active_sessions[uid]

def find_uid_image_path(uid: str) -> Optional[str]:
    """Tìm file ảnh của UID"""
    targets = [f"{uid}.jpg", f"{uid}.jpeg", f"{uid}.png"]
    lower_targets = {t.lower() for t in targets}
    for fn in os.listdir(FACE_FOLDER):
        if fn.lower() in lower_targets:
            return os.path.join(FACE_FOLDER, fn)
    return None

def extract_embedding(image_path: str):
    """Trích xuất embedding từ ảnh sử dụng InsightFace"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Detect và lấy embedding
        faces = face_app.get(img)
        if len(faces) == 0:
            print(f"[Warning] Không phát hiện khuôn mặt trong {image_path}")
            return None
        
        # Lấy khuôn mặt đầu tiên (giả sử mỗi ảnh có 1 người)
        embedding = faces[0].embedding
        # Chuẩn hóa vector (L2 normalization)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    except Exception as e:
        print(f"[Error] {e}")
        return None

def load_uid_encoding(uid: str):
    """Đọc/đệm embedding cho UID"""
    if uid in uid_encoding_cache:
        return uid_encoding_cache[uid]
    
    path = find_uid_image_path(uid)
    if not path:
        return None
    
    embedding = extract_embedding(path)
    if embedding is not None:
        uid_encoding_cache[uid] = embedding
    return embedding

def load_known_faces():
    """Load tất cả khuôn mặt đã biết vào bộ nhớ"""
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

# Load faces khi khởi động
load_known_faces()

def load_uids():
    """Lấy danh sách UID"""
    return [os.path.splitext(f)[0] for f in os.listdir(FACE_FOLDER) 
            if f.lower().endswith((".jpg", ".jpeg", ".png"))]

def delete_uid_file(uid: str):
    """Xóa file ảnh và cache của UID"""
    deleted = False
    for ext in [".jpg", ".jpeg", ".png"]:
        path = os.path.join(FACE_FOLDER, f"{uid}{ext}")
        if os.path.exists(path):
            os.remove(path)
            deleted = True
    
    if deleted:
        # Xóa khỏi cache
        if uid in uid_encoding_cache:
            del uid_encoding_cache[uid]
        
        # Xóa khỏi danh sách known faces
        try:
            idx = known_face_names.index(uid)
            known_face_names.pop(idx)
            known_face_embeddings.pop(idx)
        except ValueError:
            pass
    
    return deleted

def cosine_similarity(emb1, emb2):
    """Tính độ tương đồng cosine (đã chuẩn hóa L2)"""
    return np.dot(emb1, emb2)

# ============= WiFi Config =============
# if not os.path.exists(WIFI_CONFIG_FILE):
#     with open(WIFI_CONFIG_FILE, "w", encoding="utf8") as f:
#         json.dump({"ssid": "", "password": ""}, f, ensure_ascii=False)

# def load_wifi():
#     with open(WIFI_CONFIG_FILE, "r", encoding="utf8") as f:
#         return json.load(f)

# def save_wifi(ssid, password):
#     with open(WIFI_CONFIG_FILE, "w", encoding="utf8") as f:
#         json.dump({"ssid": ssid, "password": password}, f, ensure_ascii=False)

# @app.get("/wifi_config")
# async def get_wifi_config():
#     """ESP32 lấy WiFi config"""
#     return load_wifi()

# @app.get("/wifi_panel", response_class=HTMLResponse)
# async def wifi_panel():
#     wifi = load_wifi()
#     return f"""
#     <html>
#     <head><title>WiFi Configuration</title></head>
#     <body style='font-family:Arial;padding:30px;'>
#         <h2>WiFi Configuration</h2>
#         <form method="POST" action="/wifi_panel">
#             <label>Admin Password:</label><br>
#             <input type="password" name="admin_pw" required><br><br>
#             <label>WiFi SSID:</label><br>
#             <input type="text" name="ssid" value="{wifi['ssid']}" required><br><br>
#             <label>WiFi Password:</label><br>
#             <input type="text" name="password" value="{wifi['password']}" required><br><br>
#             <button type="submit">Update WiFi</button>
#         </form>
#         <hr>
#         <p><b>Current Saved WiFi:</b><br>
#         SSID: {wifi['ssid']}<br>
#         Password: {wifi['password']}</p>
#     </body>
#     </html>
#     """

# @app.post("/wifi_panel", response_class=HTMLResponse)
# async def update_wifi(admin_pw: str = Form(...), ssid: str = Form(...), password: str = Form(...)):
#     if admin_pw != WIFI_PANEL_PASSWORD:
#         return HTMLResponse("Sai mật khẩu Admin<br><a href='/wifi_panel'>Quay lại</a>")
#     save_wifi(ssid, password)
#     return HTMLResponse(f"WiFi đã cập nhật!<br>SSID: {ssid}<br><a href='/wifi_panel'>Quay lại</a>")

# ============= Upload Panel =============
@app.get("/upload_panel", response_class=HTMLResponse)
async def upload_panel_get():
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
                    <td>{uid}</td>
                    <td style="text-align:center;">
                        <img src='{display_path}' width='80' height='80'
                            style="object-fit:cover;border-radius:8px;border:1px solid #ccc;">
                    </td>
                    <td>
                        <form method="POST" action="/upload_panel/delete" class="delete-form">
                            <input type="hidden" name="delete_uid" value="{uid}">
                            <input type="password" name="password" placeholder="Password" class="pw-input" required>
                            <button type="submit" class="delete-btn">Xóa</button>
                        </form>
                    </td>
                </tr>
                """
    else:
        uid_rows = "<tr><td colspan='3' style='text-align:center;'>Chưa có UID nào.</td></tr>"

    html = f"""
    <html>
    <head>
        <title>Upload Face Data - InsightFace</title>
        <style>
            body {{ font-family: Arial; background: #f7f7f7; padding: 30px; }}
            .container {{ max-width: 850px; margin: auto; background: white; padding: 25px; 
                         border-radius: 12px; box-shadow: 0 3px 10px rgba(0,0,0,0.15); }}
            h2 {{ color: #333; margin-bottom: 10px; }}
            input[type="text"], input[type="password"], input[type="file"] {{
                width: 100%; padding: 10px; border-radius: 8px; border: 1px solid #ccc;
                margin-top: 5px; margin-bottom: 15px; font-size: 15px;
            }}
            button {{ padding: 10px 18px; background: #0078ff; border: none; color: white;
                     font-size: 15px; border-radius: 8px; cursor: pointer; }}
            button:hover {{ background: #005fcc; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            th, td {{ padding: 12px; border-bottom: 1px solid #e5e5e5; text-align: left; }}
            th {{ background: #f0f0f0; }}
            .delete-btn {{ background: #ff4444; }}
            .delete-btn:hover {{ background: #cc0000; }}
            .delete-form {{ display: flex; gap: 8px; align-items: center; }}
            .pw-input {{ width: 150px; }}
            .badge {{ display: inline-block; padding: 4px 8px; background: #4CAF50; 
                     color: white; border-radius: 4px; font-size: 12px; margin-left: 10px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Upload Face Data <span class="badge">InsightFace</span></h2>
            <form method="POST" action="/upload_panel/upload" enctype="multipart/form-data">
                <label>Password:</label>
                <input type="password" name="password" required>
                <label>UID (Tên người):</label>
                <input type="text" name="uid" required>
                <label>Chọn ảnh (JPG/PNG):</label>
                <input type="file" name="file" accept=".jpg,.jpeg,.png" required>
                <button type="submit">Upload</button>
            </form>
            <hr style="margin: 30px 0;">
            <h3>Danh sách UID hiện có ({len(uids)})</h3>
            <table>
                <tr><th>UID</th><th>Ảnh</th><th>Hành động</th></tr>
                {uid_rows}
            </table>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(html)

@app.post("/upload_panel/upload", response_class=HTMLResponse)
async def upload_face(password: str = Form(...), uid: str = Form(...), file: UploadFile = File(...)):
    if password != UPLOAD_PASSWORD:
        raise HTTPException(status_code=403, detail="Sai mật khẩu")
    
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file .jpg, .jpeg, .png")
    
    # Lưu với extension gốc
    ext = os.path.splitext(file.filename)[1]
    save_path = os.path.join(FACE_FOLDER, f"{uid}{ext}")
    
    with open(save_path, "wb") as f:
        f.write(await file.read())
    
    # Trích xuất embedding và thêm vào database
    embedding = extract_embedding(save_path)
    if embedding is not None:
        # Xóa UID cũ nếu có
        if uid in known_face_names:
            idx = known_face_names.index(uid)
            known_face_names.pop(idx)
            known_face_embeddings.pop(idx)
        
        known_face_embeddings.append(embedding)
        known_face_names.append(uid)
        uid_encoding_cache[uid] = embedding
        msg = f"Upload thành công: {uid} ✓"
    else:
        msg = f"Upload file thành công nhưng không phát hiện khuôn mặt: {uid} ⚠️"
    
    return HTMLResponse(f"{msg}<br><a href='/upload_panel'>⬅ Quay lại</a>")

@app.post("/upload_panel/delete", response_class=HTMLResponse)
async def delete_face(password: str = Form(...), delete_uid: str = Form(...)):
    if password != UPLOAD_PASSWORD:
        raise HTTPException(status_code=403, detail="Sai mật khẩu")
    success = delete_uid_file(delete_uid)
    return HTMLResponse(f"{'Đã xóa UID: ' + delete_uid if success else 'Không tìm thấy UID'}<br><a href='/upload_panel'>⬅ Quay lại</a>")

# ============= Gallery =============
@app.get("/gallery", response_class=HTMLResponse)
async def gallery():
    files = sorted(os.listdir(UPLOAD_FOLDER), reverse=True)
    images_html = ""
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            url = f"/uploads/{f}"
            images_html += f"""
            <div style='display:inline-block;margin:10px;text-align:center;'>
                <img src='{url}' width='200' style='border-radius:10px;box-shadow:0 2px 6px rgba(0,0,0,0.3)'>
                <p>{f}</p>
            </div>
            """
    return HTMLResponse(f"""
    <html>
      <head><title>Gallery</title></head>
      <body style='font-family:Arial;text-align:center;padding:30px;'>
        <h2>Ảnh đã upload</h2>
        {images_html or '<p>Chưa có ảnh nào.</p>'}
        <br><a href="/upload_panel">⬅ Quay lại upload</a>
      </body>
    </html>
    """)

# ============= Recognition API =============
@app.post("/precheck")
async def precheck_uid(request: Request):
    """Kiểm tra UID có tồn tại ảnh không"""
    cleanup_sessions()
    try:
        payload = await request.json()
        uid = str(payload.get("uid", "")).strip()
    except Exception:
        return PlainTextResponse("no", status_code=400)
    
    if not uid:
        return PlainTextResponse("no", status_code=400)
    
    enc = load_uid_encoding(uid)
    if enc is None:
        return PlainTextResponse("no")
    
    # Tạo/refresh session
    active_sessions[uid] = {"status": "pending", "ts": now_ts()}
    return PlainTextResponse("yes")

@app.get("/result")
async def get_result(uid: str = Query(...)):
    """ESP32-DEV poll kết quả nhận diện"""
    cleanup_sessions()
    s = active_sessions.get(uid)
    if not s:
        return PlainTextResponse("no")
    s["ts"] = now_ts()
    return PlainTextResponse(s["status"])

@app.post("/recognize")
async def recognize_face(request: Request,
                         x_uid: Optional[str] = Header(default=None),
                         x_last_frame: Optional[str] = Header(default=None)):
    """Nhận diện khuôn mặt từ frame ESP32-CAM"""
    cleanup_sessions()
    
    image_bytes = await request.body()
    if not image_bytes:
        return PlainTextResponse("pending", status_code=400)
    
    # Xác định UID cho phiên này
    uid = None
    if x_uid:
        uid = x_uid.strip()
    else:
        if active_sessions:
            uid = max(active_sessions.items(), key=lambda kv: kv[1]["ts"])[0]
    
    if not uid or uid not in active_sessions:
        return PlainTextResponse("pending", status_code=428)
    
    # Early-exit nếu đã kết thúc
    if active_sessions[uid]["status"] in ("yess", "noo"):
        active_sessions[uid]["ts"] = now_ts()
        return PlainTextResponse(active_sessions[uid]["status"])
    
    # Decode ảnh
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return PlainTextResponse("pending", status_code=400)
    
    # Lưu ảnh debug
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    raw_path = os.path.join(UPLOAD_FOLDER, f"{timestamp}_raw.jpg")
    cv2.imwrite(raw_path, frame)
    
    # Load embedding cần so sánh
    enc_expected = load_uid_encoding(uid)
    if enc_expected is None:
        active_sessions[uid]["status"] = "noo"
        active_sessions[uid]["ts"] = now_ts()
        return PlainTextResponse("noo")
    
    is_last = (str(x_last_frame).strip() == "1")
    
    # Tiền xử lý ảnh (tăng sáng nhẹ)
    frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=5)
    
    # Detect faces và trích xuất embeddings
    try:
        faces = face_app.get(frame)
    except Exception as e:
        print(f"[Error] InsightFace detection error: {e}")
        if is_last:
            active_sessions[uid]["status"] = "noo"
            active_sessions[uid]["ts"] = now_ts()
            return PlainTextResponse("noo")
        return PlainTextResponse("pending")
    
    matched = False
    best_similarity = 0.0
    
    if len(faces) > 0:
        # So sánh từng khuôn mặt trong frame
        for face in faces:
            embedding = face.embedding
            # Chuẩn hóa
            embedding = embedding / np.linalg.norm(embedding)
            
            # Tính cosine similarity
            similarity = cosine_similarity(embedding, enc_expected)
            
            if similarity > best_similarity:
                best_similarity = similarity
            
            # Kiểm tra ngưỡng
            if similarity >= THRESHOLD:
                matched = True
                break
    
    # Log
    with open(os.path.join(LOG_FOLDER, "recognition_log.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "timestamp": datetime.now().isoformat(),
            "uid": uid,
            "image_path": raw_path,
            "face_count": len(faces),
            "best_similarity": round(float(best_similarity), 4),
            "threshold": THRESHOLD,
            "matched": matched
        }, ensure_ascii=False) + "\n")
    
    # Trả kết quả
    if matched:
        active_sessions[uid]["status"] = "yess"
        active_sessions[uid]["ts"] = now_ts()
        return PlainTextResponse("yess")
    else:
        if is_last:
            active_sessions[uid]["status"] = "noo"
            active_sessions[uid]["ts"] = now_ts()
            return PlainTextResponse("noo")
        else:
            active_sessions[uid]["status"] = "pending"
            active_sessions[uid]["ts"] = now_ts()
            return PlainTextResponse("pending")

# ============= Root =============
@app.get("/")
async def root():
    return {
        "status": "online",
        # "version": "2.5",
        "known_faces_count": len(known_face_names),
        "known_names": known_face_names,
        # "upload_panel": "/upload_panel",
        # "gallery": "/gallery",
        "endpoint_docs": "/docs"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)