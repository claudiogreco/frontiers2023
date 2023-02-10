import json
from collections import defaultdict
import inflect
import spacy
from tqdm import tqdm

if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")
    inflect_engine = inflect.engine()

    with open("data/preprocess/2_unknown_2_known.json") as in_file:
        contexts = json.load(in_file)

    def get_unknowns(contexts):
        contexts_unknowns = set()
        for images_labels in [v[1] for k, v in contexts.items()]:
            for label in images_labels:
                if "unknown" in label:
                    contexts_unknowns.add(label.replace("unknown_", ""))
        return contexts_unknowns

    contexts_unknowns = get_unknowns(contexts)

    removed_categories_names = set()
    for contexts_unknowns in [contexts_unknowns]: # , contexts2_unknowns, contexts3_unknowns, contexts4_unknowns]:
        removed_categories_names = removed_categories_names.union(contexts_unknowns)

    with open("data/annotations/instances_train2014.json") as in_file:
        instances = json.load(in_file)

    image_id2image_objects = defaultdict(set)
    for annotation in instances["annotations"]:
        image_id2image_objects[annotation["image_id"]].add(annotation["category_id"])

    category_name2category_id = {x["name"]: x["id"] for x in instances["categories"]}
    category_id2category_name = {v: k for k, v in category_name2category_id.items()}

    with open("data/annotations/captions_train2014.json") as in_file:
        captions = json.load(in_file)

    image_id2captions = defaultdict(list)
    for annotation in captions["annotations"]:
        image_id2captions[annotation["image_id"]].append(annotation["caption"])

    removed_categories = {category_name2category_id[name] for name in removed_categories_names}

    removed_categories_names_ext = removed_categories_names.union(
        {inflect_engine.plural(x) for x in removed_categories_names})

    preserved_images_ids = set()
    removed_images_ids = set()
    for image_id in tqdm(image_id2image_objects, total=len(image_id2image_objects)):
        if image_id2image_objects[image_id].intersection(removed_categories):
            removed_images_ids.add(image_id)
        else:
            for caption in image_id2captions[image_id]:
                caption_tokens = {token.text for token in nlp(caption.lower())}
                if caption_tokens.intersection(removed_categories_names_ext):
                    print(caption)
                    removed_images_ids.add(image_id)
                    break
            if image_id not in removed_images_ids:
                preserved_images_ids.add(image_id)

    preserved_images = []
    for image in captions["images"]:
        if image["id"] in preserved_images_ids:
            preserved_images.append(image)

    preserved_captions = []
    removed_captions = []
    for annotation in captions["annotations"]:
        if annotation["image_id"] in preserved_images_ids:
            preserved_captions.append(annotation)
        else:
            removed_captions.append(annotation)

    with open("captions_train2014_wrt_2_unknown_2_known.json", mode="w") as out_file:
        json.dump(
            {
                "info": captions["info"],
                "images": preserved_images,
                "licenses": captions["licenses"],
                "annotations": preserved_captions
            },
            out_file
        )

    print("Number of images: {}".format(len(image_id2image_objects)))
    print("Number of captions: {}".format(len(captions["annotations"])))

    print("Number of preserved images: {}".format(len(preserved_images_ids)))
    print("Number of removed images: {}".format(len(removed_images_ids)))

    print("Number of preserved captions: {}".format(len(preserved_captions)))
    print("Number of removed captions: {}".format(len(removed_captions)))
