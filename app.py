import streamlit as st
import face_recognition, sqlite3, numpy as np
from PIL import Image
import openai   # 可选

st.title("幼儿园小朋友识别 Agent")

# 加载人脸库
@st.experimental_singleton
def load_known():
    conn = sqlite3.connect("kids.db", check_same_thread=False)
    data = conn.execute("SELECT name, encoding FROM kids").fetchall()
    known = [(n, np.frombuffer(enc, dtype=np.float64)) for n,enc in data]
    return known
known_faces = load_known()

# 文件上传
uploaded = st.file_uploader("上传孩子合照或单人照", type=["jpg","png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="原图", use_column_width=True)
    img_arr = np.array(img)

    # 人脸检测与识别
    locs = face_recognition.face_locations(img_arr)
    encs = face_recognition.face_encodings(img_arr, locs)
    names = []
    for e in encs:
        dists = [np.linalg.norm(e - known_e) for _, known_e in known_faces]
        idx = int(np.argmin(dists))
        names.append(known_faces[idx][0] if dists[idx] < 0.6 else "未知")

    st.write("**识别结果：**", ", ".join(names))

    # 可选：成长记录生成
    if st.checkbox("生成成长鼓励语"):
        recs = []
        for name in names:
            prompt = f"请为幼儿园小朋友{name}写一句鼓励话语。"
            resp = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role":"user","content":prompt}],
                api_key=st.secrets["proj_uhrxk7onx31SfhyTe2QyP3gk"]
            )
            recs.append(resp.choices[0].message.content)
        for r in recs:
            st.info(r)