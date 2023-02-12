import ast
import json
import os

import numpy as np
import pandas as pd
import requests
import streamlit as st


def init_page():
    st.set_page_config(layout="wide")


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


@st.cache_data
def mscoco_image_url(image_id):
    data = {"image_ids[]": image_id, "querytype": "getImages"}
    response = requests.post(
        "https://us-central1-open-images-dataset.cloudfunctions.net/coco-dataset-bigquery",
        data=data,
    )
    return response.json()[0]["coco_url"]


def parse_image_id(image_name):
    image_id = image_name.split("/")[-1].split("_")[-1]
    image_id = image_id.replace(".jpg", "")
    return image_id


def get_image_url(image_name):
    return image_id2image_filename[parse_image_id(image_name).lstrip('0')]


init_page()

st.header("Visualization of speaker adaptation")

filenames = sorted(
    [filename for filename in os.listdir("data/reports")
     if filename.endswith(".csv")]
)

file_path = st.selectbox("Select report", filenames)

with open("data/preprocess/mscoco_category2image_id.json") as fp:
    category2image_id = json.load(fp)

with open("data/preprocess/mscoco_image_id2image_filename.json") as fp:
    image_id2image_filename = json.load(fp)

supercat2cat = {
    "kitchen": "bowl",
    "furniture": "potted plant",
    "outdoor": "stop sign",
    "appliance": "microwave",
    "vehicle": "airplane",
    "food": "broccoli",
    "electronic": "laptop",
    "accessory": "umbrella",
    "indoor": "clock",
    "animal": "cow",
    "sports": "tennis racket"
}

with open("data/preprocess/mscoco_image_id2image_objects.json") as in_file:
    image_id2image_objects = json.load(in_file)

if file_path:
    file = open(os.path.join("data/reports", file_path), mode="rb")
    df = pd.read_csv(file)
    df["target_id"] = df["target"].apply(parse_image_id).apply(int)

    subsets = list(df.groupby(["context"]))
    index = st.selectbox(
        "Select context",
        range(len(subsets)),
        format_func=lambda x: f"Context {x + 1}"
    )

    removed_categories = set(supercat2cat.values())

    st.markdown("<div style='text-align: center; color: black;'>{}</div>".format(
        "<strong>Removed categories:</strong><br>" + ", ".join(removed_categories)), unsafe_allow_html=True)

    context, subset = subsets[index]
    context = ast.literal_eval(context)

    context_image_urls = [get_image_url(image_path) for image_path in context]

    context_image_ids = [parse_image_id(image_path).lstrip('0') for image_path in context]

    context_image_unknown_objects = []
    for image_id in context_image_ids:
        unknown_objects = image_id2image_objects[image_id]
        unknown_objects = set(unknown_objects).intersection(removed_categories)
        context_image_unknown_objects.append(unknown_objects)

    context_image_urls = [mscoco_image_url(idx) for idx in context_image_ids]
    context_other_image_urls = ["http://images.cocodataset.org/train2017/" +
                                fn for fn in [x.split("COCO_val2014_")[-1] for x in context]]

    cols = st.columns(4)
    for i, x in enumerate(context_image_ids):
        cols[i].image(image_id2image_filename[parse_image_id(
            x).lstrip('0')], use_column_width=True)

    st.markdown("**Image URLs:**")
    for url_index, url in enumerate(context_image_urls):
        st.write("Image {}: {}".format(url_index + 1, url))

    cols = st.columns(len(context_image_urls))
    subset = subset.sort_values(by="round_num").set_index("round_num")

    def normalize_scores(x):
        return [float(x) for x in x.replace("[", "").replace("]", "").split()]

    subset["prediction"] = subset["scores"]
    subset.loc[:, "prediction"] = subset.prediction.apply(
        lambda x: np.argmax(normalize_scores(x)) + 1)
    subset.loc[:, "scores"] = subset.scores.apply(
        lambda x: softmax(np.array(normalize_scores(x))).round(2))
    subset.loc[:, "target"] = subset.target.apply(
        lambda x: context.index(x) + 1)
    subset["target_memory"] = subset["target_memory"]
    subset["target_memory"] = subset.target_memory.apply(lambda x: eval(x))
    subset["target_memory"] = subset.target_memory.apply(
        lambda x: [i.split("/")[-1].replace("[", "").replace("]", "").split("_")[-1].replace(".jpg", "").lstrip("0").replace("'", "") for i in x])
    subset["target_memory"] = subset.target_memory.apply(
        lambda x: [context_image_ids.index(str(i)) + 1 for i in x])

    def color_correct(s):
        if s:
            return "background-color: green"
        else:
            return "background-color: red"

    final_subset = subset[
        ["target", "caption", "target_memory", "prediction", "scores", "correct"]
    ]

    st.dataframe(
        final_subset.reset_index(drop=True).style.applymap(
            color_correct, subset="correct").background_gradient(
                cmap="Greys",
                axis=0,
                subset="target"
        ),
        height=850
    )
