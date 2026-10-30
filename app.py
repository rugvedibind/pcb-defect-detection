import cv2, numpy as np, streamlit as st
from PIL import Image
from ultralytics import YOLO

CLASSES = ["open","short","mousebite","spur","copper","pinhole"]
COLORS = [(255,0,0),(0,160,0),(0,0,255),(255,140,0),(160,0,200),(0,170,170)]

@st.cache_resource
def load_models():
    return YOLO("models/baseline.pt"), YOLO("models/diff.pt")

def prep(im):
    g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    g = cv2.resize(g, (640, 640))
    _, g = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return g

def align_ecc(t, r):
    a = cv2.GaussianBlur(r, (5,5), 0).astype(np.float32)
    b = cv2.GaussianBlur(t, (5,5), 0).astype(np.float32)
    warp = np.eye(2, 3, dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
    _, warp = cv2.findTransformECC(a, b, warp, cv2.MOTION_AFFINE, crit, None, 5)
    return cv2.warpAffine(t, warp, (640, 640),
                          flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

def detect(model, img, conf):
    r = model.predict(img, conf=conf, imgsz=640, device="cpu", verbose=False)[0]
    out = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2RGB)
    n = len(r.boxes)
    for (x1,y1,x2,y2), c, p in zip(r.boxes.xyxy.numpy(),
                                   r.boxes.cls.numpy().astype(int),
                                   r.boxes.conf.numpy()):
        cv2.rectangle(out, (int(x1),int(y1)), (int(x2),int(y2)), COLORS[c], 2)
        cv2.putText(out, f"{CLASSES[c]} {p:.2f}", (int(x1), max(int(y1)-4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[c], 1)
    return out, n

st.set_page_config(page_title="PCB Defect Detection", layout="wide")
st.title("PCB Defect Detection (YOLOv8n)")
st.caption("Direct detector vs template-difference detector")

base_model, diff_model = load_models()

c1, c2 = st.columns(2)
test_file = c1.file_uploader("Test board image", type=["jpg","jpeg","png"])
ref_file = c2.file_uploader("Reference (defect-free) image, optional",
                            type=["jpg","jpeg","png"])
align = st.checkbox("ECC alignment", value=False)
conf = st.slider("Confidence", 0.1, 0.9, 0.25)

if test_file is not None:
    test_img = np.array(Image.open(test_file).convert("RGB"))
    t = prep(test_img)
    d_img, d_n = detect(base_model, cv2.merge([t]*3), conf)
    left, right = st.columns(2)
    left.subheader("Direct")
    left.image(d_img)
    left.write(f"**{'FAIL' if d_n else 'PASS'}** ({d_n} detections)")
    right.subheader("Template-difference")
    if ref_file is not None:
        r = prep(np.array(Image.open(ref_file).convert("RGB")))
        if align:
            t = align_ecc(t, r)
        diff = cv2.absdiff(t, r)
        t_img, t_n = detect(diff_model, cv2.merge([t, r, diff]), conf)
        right.image(t_img)
        right.write(f"**{'FAIL' if t_n else 'PASS'}** ({t_n} detections)")
    else:
        right.info("Upload a reference image to run this model.")
else:
    st.info("Upload a test board image to start.")
