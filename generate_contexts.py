import csv
import json
import random
from collections import defaultdict

import numpy as np
from pycocotools.coco import COCO
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

data_type = "val2014"
annotation_path = "data/preprocess/annotations/instances_{}.json".format(data_type)
coco = COCO(annotation_path)

categories = coco.loadCats(coco.getCatIds())
category_list = [item["name"] for item in categories]

category_name2category_id = {category["name"]: category["id"] for category in categories}
category_id2category_name = {v: k for k, v in category_name2category_id.items()}

supercategory2category_id = {}
category_id2supercategory = {}
for category in categories:
    curr_supercategory = category["supercategory"]
    curr_id = category["id"]
    category_id2supercategory.update({curr_id: curr_supercategory})

    if curr_supercategory in supercategory2category_id.keys():
        supercategory2category_id[curr_supercategory].append(curr_id)
    else:
        supercategory2category_id.update({curr_supercategory: []})

category_id2images_ids = defaultdict(set)
filtered_categoryid2images_ids = defaultdict(set)

for category in categories:
    category_id2images_ids[category["id"]].update(
        coco.getImgIds(catIds=category["id"]))

image_filename2image_id = {}
for image in coco.loadImgs(coco.getImgIds()):
    image_filename2image_id[image["file_name"]] = image["id"]

image_id2filename = {v: k for k, v in image_filename2image_id.items()}

unknown_cats = ["bowl", "potted plant", "stop sign",
                "microwave",  "airplane", "broccoli",
                "laptop", "umbrella", "clock",
                "cow", "tennis racket"]
known_cats = [item for item in category_list if item not in unknown_cats]

for category_id, image_ids in category_id2images_ids.items():
    known_cat_flag = False
    category_name = category_id2category_name[category_id]
    if category_name in known_cats:
        known_cat_flag = True

    for image_id in image_ids:
        subject_total_area = 0
        unknown_cat_count = 0
        known_with_unknown = False

        temp = []
        annotations = coco.loadAnns(coco.getAnnIds(imgIds=image_id))
        for annotation in annotations:
            if annotation["category_id"] == category_id:
                subject_total_area += annotation["area"]

            category = category_id2category_name[annotation["category_id"]]

            if category in unknown_cats:
                if known_cat_flag:
                    known_with_unknown = True
                else:
                    unknown_cats.remove(category)
                    temp.append(category)
                    unknown_cat_count += 1

        unknown_cats = unknown_cats + temp
        if (unknown_cat_count <= 1) and (not known_with_unknown):
            image = coco.loadImgs(image_id)[0]
            image_area = image["width"] * image["height"]
            if subject_total_area / image_area > 0.15:
                filtered_categoryid2images_ids[category_id].add(image_id)

image_ids2category = {}
for category_id, image_ids in filtered_categoryid2images_ids.items():
    for id in image_ids:
        image_ids2category.update({id: category_id})


path = "data/preprocess/vgg_features"
filename = "FEATURES_vgg_FC6"

image_filename2feature_index = {}
feature_index2image_id = {}

with open("{}/METADATA.csv".format(path)) as in_file:
    reader = csv.reader(in_file)
    next(reader)
    for row in reader:
        feature_index, image_path = row
        feature_index = int(feature_index.strip())
        image_filename = image_path.strip().split("/")[-1]

        image_filename2feature_index[image_filename] = feature_index
        feature_index2image_id[feature_index] = image_filename2image_id[image_filename]

image_id2feature_index = {v: k for k, v in feature_index2image_id.items()}

feats = np.load("{}/{}.npy".format(path, filename))
feats = np.array(feats)

pca = PCA(n_components=512)
pca.fit(feats)
pca_features = pca.transform(feats)

no_of_unknown = 2

filtered_categoryid2images_ids = dict(filtered_categoryid2images_ids)
del filtered_categoryid2images_ids[1]


def get_similar_image(image_feature, image_feature_set, image_id_set):
    image_feature = image_feature.reshape(1, -1)
    similarity_matrix = (1 - cosine_similarity(image_feature, image_feature_set))[0]
    similar_image_feature_idx = np.argsort(similarity_matrix)[2]
    image_id = image_id_set[similar_image_feature_idx]

    return image_id


# image_set1 = n x feature
# image_set2 = m x feature
def get_similar_pairs(image_set1, image_set2, no_of_pairs):
    similarity_matrix = (1 - cosine_similarity(image_set1, image_set2))
    flattened_indices = np.argsort(similarity_matrix.ravel())[:no_of_pairs]
    og_indexes = np.unravel_index(
        flattened_indices, (image_set1.shape[0], image_set2.shape[0]))
    og_indexes = list(zip(og_indexes[0], og_indexes[1]))

    return og_indexes


dataset = {}
count = 0

for unknown_cat in unknown_cats:
    cat_id = category_name2category_id[unknown_cat]
    supercategory = category_id2supercategory[cat_id]

    unknown_cat_images = list(filtered_categoryid2images_ids[cat_id])
    unknown_cat_features = []
    for image_id in unknown_cat_images:
        feature_index = image_id2feature_index[image_id]
        unknown_cat_features.append(pca_features[feature_index])
    unknown_cat_features = np.array(unknown_cat_features)

    try:
        supercategory2category_id[supercategory].remove(cat_id)
    except (ValueError):
        pass

    category_list = supercategory2category_id[supercategory]
    known_cat_images = []
    for category in category_list:
        known_cat_images = known_cat_images + \
            list(filtered_categoryid2images_ids[category])

    known_cat_features = []
    for image_id in known_cat_images:
        feature_index = image_id2feature_index[image_id]
        known_cat_features.append(pca_features[feature_index])
    known_cat_features = np.array(known_cat_features)

    similar_image_pairs = get_similar_pairs(
        unknown_cat_features, known_cat_features, 40)

    for j, image_pair in enumerate(similar_image_pairs):
        index1 = image_pair[0]
        image_id1 = unknown_cat_images[index1]
        feature_index1 = image_id2feature_index[image_id1]

        index2 = image_pair[1]
        image_id2 = known_cat_images[index2]
        feature_index2 = image_id2feature_index[image_id2]

        context = [image_id1, image_id2]
        context_label = ["unknown_"+unknown_cat, "known_category"]

        known_category = image_ids2category[image_id2]
        image_list = list(filtered_categoryid2images_ids[known_category])

        feature_list = []
        for image_id in image_list:
            feature_index = image_id2feature_index[image_id]
            feature_list.append(pca_features[feature_index])
        feature_list = np.array(feature_list)

        image_id3 = get_similar_image(
            pca_features[feature_index1], unknown_cat_features, unknown_cat_images)
        context.append(image_id3)
        context_label.append("unknown_"+unknown_cat)

        image_id4 = get_similar_image(
            pca_features[feature_index2], known_cat_features, known_cat_images)
        context.append(image_id4)
        context_label.append("known_category")

        temp = list(zip(context, context_label))
        random.shuffle(temp)
        context, context_label = zip(*temp)
        context, context_label = list(context), list(context_label)

        count += 1
        dataset.update({count: (context, context_label)})

temp = list(dataset.values())
random.shuffle(temp)
dataset = dict(zip(dataset, temp))

with open("2_same_unknown_and_known_v2", "w") as out_file:
    json.dump(dataset, out_file)
