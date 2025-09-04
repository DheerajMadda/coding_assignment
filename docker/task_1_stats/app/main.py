import os
import io
import cv2
from glob import glob
import streamlit as st

st.title("BDD100K Dataset Stats")
st.set_page_config(layout="wide")

train_data = {
    os.path.basename(path).replace(".png", ""): path
    for path in glob(os.path.join("plots", "train", "*.png"))
}

val_data = {
    os.path.basename(path).replace(".png", ""): path
    for path in glob(os.path.join("plots", "val", "*.png"))
}

data_dict = dict(
    Train=train_data,
    Val=val_data
)

plot_titles =  [
    "Distibution by Weather across Images",
    "Distibution by Scene across Images",
    "Distibution by Time Of Day across Images",
    "Distibution by Weather across Class",
    "Distibution by Scene across Class",
    "Distibution by Time Of Day across Class",
    "Distibution by Label across Labels",
    "Distibution by Occluded Labels across Labels",
    "Distibution by Truncated Labels across Labels",
    "Distibution by Small-Sized Labels across Labels",
    "Distibution by Medium-Sized Labels across Labels",
    "Distibution by Large-Sized Labels across Labels",
    "Distibution by Uncertain Labels across Labels"
]
choice = st.selectbox("Choose a plot:", plot_titles)

@st.cache_data
def get_image(split, choice):

    path = data_dict[split][choice]
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

col1, col2 = st.columns(2)

with col1:
    st.subheader("Train")

    img_train = get_image("Train", choice)
    st.image(img_train)
    
    _, train_buf_arr = cv2.imencode(".jpg", cv2.cvtColor(img_train, cv2.COLOR_RGB2BGR))
    train_buf = io.BytesIO(train_buf_arr)
    train_buf.seek(0)
    
    st.download_button(
        label="📥 Download Train Plot",
        data=train_buf,
        file_name=f"Train_{choice}.png",
        mime="image/png"
    )
    
with col2:
    st.subheader("Val")

    img_val = get_image("Val", choice)
    st.image(img_val)
    
    _, val_buf_arr = cv2.imencode(".jpg", cv2.cvtColor(img_val, cv2.COLOR_RGB2BGR))
    val_buf = io.BytesIO(val_buf_arr)
    val_buf.seek(0)

    st.download_button(
        label="📥 Download Val Plot",
        data=val_buf,
        file_name=f"Val_{choice}.png",
        mime="image/png"
    )
