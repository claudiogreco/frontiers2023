from PIL import Image
import torchvision.transforms as transforms
from symspellpy.symspellpy import SymSpell, Verbosity
import csv
import time
from pycocotools.coco import COCO
import torch
import numpy as np
import nltk
import string
import spacy

data_dir = "data/preprocess/"
data_type = "val2014"
coco = COCO("{}/annotations/instances_{}.json".format(data_dir, data_type))
coco_caps = COCO("{}/annotations/captions_{}.json".format(data_dir, data_type))

device = torch.device("cuda:8" if torch.cuda.is_available() else "cpu")
print("device: ", device)
if torch.cuda.is_available() :
    torch.cuda.set_device(device)
nlp = spacy.load("en_core_web_sm")
sym_spell = SymSpell(2, 7)
sym_spell.load_dictionary("../data/preprocess/frequency_dictionary_en_82_765.txt",0,1)

class Writer():
    def __init__(self, args, save_file) :
        self.args = args
        self.save_file = save_file

    def init_output_csv(self, header=None):
        if not header:
            header = [["sample_num", "date", "image", "learning_rate",
                       "num_steps", "batch_size", "dataset_type",
                       "round", "num_words", "caption"]]
        with open(self.save_file, "w") as writeFile:
            writer = csv.writer(writeFile)
            writer.writerows(header)

    def writerow(self, row):
        """
        write a custom row
        """
        with open(self.save_file, "a") as writeFile:
            writer = csv.writer(writeFile)
            writer.writerows([row])

    def write(self, round_num, caption, sample_num = 0, image = None) :
        date = time.strftime("%Y-%b-%d-%H-%M-%S")
        len_caption = len(caption.split())-2

        # Fall back on image from initialization if it doesn"t change
        if image is None :
            image = self.args.raw_image

        with open(self.save_file, "a") as writeFile:
            writer = csv.writer(writeFile)
            row = [[sample_num, date, image,
                    self.args.learning_rate, self.args.num_steps, self.args.batch_size,
                    self.args.ds_type, round_num, len_caption, caption]]
            writer.writerows(row)

class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = len(self.word2idx)

    def word2idx(self, word):
        return self.word2idx[word]

    def add_word(self, word):
        assert not word in self.word2idx
        print("\nadding new word:", word)
        self.word2idx[word] = self.idx
        print("word2idx", self.word2idx[word])
        self.idx2word[self.idx] = word
        print("idx2word", self.idx2word[self.idx])
        self.idx += 1
        print("updated idx", self.idx)

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx["<unk>"]
        else :
            return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def sort_seqs(inputs, lengths):
    """sort_sequences
    Sort sequences according to lengths descendingly.
    """
    lengths_sorted, sorted_idx = lengths.sort(descending=True)
    _, unsorted_idx = sorted_idx.sort()
    return inputs[sorted_idx], lengths_sorted, unsorted_idx

def tokenize(caption) :
    if caption[:7] == "<start>":
        caption = caption[8:]
    if caption[-5:] == "<end>":
        caption = caption[:-5]

    # hard coding some corner cases
    caption = caption.replace("w/", "with")
    caption = caption.replace("there is ", "")
    caption = caption.replace("close up", "closeup")
    caption = caption.replace("/", " or ")
    caption = caption.replace("teh", "the")
    caption = caption.replace("wiht", "with")

    # strip punct
    caption = caption.translate(str.maketrans("", "", string.punctuation))
    return nltk.tokenize.word_tokenize(str(caption).lower())

def words_to_ids (caption, vocab) :
    """
    Converts words to ids
    * note that any oovs wanted to add to the vocab should"ve already happened
      here, the default is to transform to unk (as rehearsal requires)
    """
    if caption[:7] == "<start>" :
        caption = caption[8:-6]

    ids = []
    tokens = tokenize(caption)
    ids.append(vocab("<start>"))
    ids.extend([vocab(token) for token in tokens])
    ids.append(vocab("<end>"))
    return ids

def ids_to_words (ids, vocab) :
    out = []
    for word_id in ids :
        out.append(word_id)
        if word_id in [vocab.word2idx["<end>"], vocab.word2idx["<pad>"]]  :
            break

    # remove "<start>" and <"end"> from word representation
    if out[0] == vocab.word2idx["<start>"] :
        out = out[1:-1]

    return " ".join([vocab.idx2word[idx] for idx in out])

def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.

    Args:
    data: list of tuple (image, caption).
    - image: torch tensor of shape (3, 256, 256).
    - caption: torch tensor of shape (?); variable length.

    Returns:
    images: torch tensor of shape (batch_size, 3, 256, 256).
    targets: torch tensor of shape (batch_size, padded_length).
    lengths: list; valid length for each padded caption.
    """

    # Sort a data list by caption length (descending order).
#    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)
    lengths = [len(cap) for cap in captions]
    captions = torch.nn.utils.rnn.pad_sequence(captions, batch_first=True).long()
    return images, captions, lengths

def load_image(image_path, for_batching=False):
    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])
    #line below causes words to ids cap
    image = Image.open(image_path).convert("RGB")
    if transform is not None:
        image = transform(image)
    if not for_batching :
        image = image.unsqueeze(0)
    return image

def load_images(image_paths):
    context_images = []
    for path in image_paths:
        image = load_image(path)
        context_images.append(image)
    return torch.cat(context_images)

def get_img_path(img_name) :
    """
    Appends path to coco images
    """
    return "{}/{}/{}".format(data_dir, data_type, img_name)

def get_cat_names() :
    return [cat["name"] for cat in coco.loadCats(coco.getCatIds())]

def get_id_from_path(path):
    # extract img id from img_path
    start = path.rfind("_")
    end = path.rfind(".")
    return int(path[start+1:end])

def sample_img_from_tag(img_tag) :
    """
    Samples img (and meta-data) from provided coco category tag
    """
    cat_id = coco.getCatIds(catNms=img_tag)
    img_id = np.random.choice(coco.getImgIds(catIds=cat_id), 1)
    img_path = get_img_path(coco.loadImgs(int(img_id))[0]["file_name"])
    return load_image(img_path), img_path, img_tag

def _levenshtein(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros ((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    return (matrix[size_x - 1, size_y - 1])

def edit_distance(str1, str2, m, n):
    # Create a table to store results of subproblems
    dp = [[0 for x in range(n+1)] for x in range(m+1)]

    # Fill d[][] in bottom up manner
    for i in range(m+1):
        for j in range(n+1):

            # If first string is empty, only option is to
            # insert all characters of second string
            if i == 0:
                dp[i][j] = j    # Min. operations = j

            # If second string is empty, only option is to
            # remove all characters of second string
            elif j == 0:
                dp[i][j] = i    # Min. operations = i

            # If last characters are same, ignore last char
            # and recur for remaining string
            elif str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]

            # If last character are different, consider all
            # possibilities and find minimum
            else:
                dp[i][j] = 1 + min(dp[i][j-1],        # Insert
                                   dp[i-1][j],        # Remove
                                   dp[i-1][j-1])      # Replace

    return float(dp[m][n])/max(m,n)

def flatten(l) :
    return [item for sublist in l for item in sublist]

def find_invocab_word(word, discourse_history, word2idx):
    """
    word comes from tokenized/preprocessed utterance
    """
    #print("\nOOV word", word)
    #print("history", discourse_history)
    # we don"t want to be replacing with rare or other oov word...
    correction = sym_spell.lookup(word, Verbosity.ALL, 1)
    terms = [cor.term for cor in correction
                if cor.count > 100000 and
                cor.term in word2idx]
    discourse_words = list(set(flatten(map(lambda x: tokenize(x["cap"]),
                                           discourse_history))))
    inter = set(discourse_words).intersection(terms)
    print("possible typo corrections meeting criteria", terms)
    #print("inter", inter)
    # if *any* corrected option appears in prior discourse history
    # go with that one
    if len(inter) > 0 :
        invocab_word = list(inter)[0]
    # otherwise, take best option if there are any fitting criteria
    elif len(terms) >0 :
        invocab_word = terms[0]
    # if not, just leave it an unk
    else :
        invocab_word = word
    return invocab_word

def find_invocab_utterance(utterance, discourse_history, vocab):
    out = []
    for word in tokenize(utterance) :
        out.append(find_invocab_word(word, discourse_history, vocab.word2idx))
    return " ".join(out)

def get_human_caps(img_path, vocab):
    """
    Gets human annotated captions from img_path
    """
    # extract img id from img_path
    img_id = get_id_from_path(img_path)

    # sample an annotation from that image
    ann_ids = coco_caps.getAnnIds(imgIds=img_id)
    anns = coco_caps.loadAnns(ann_ids)
    img_cap = np.random.choice(anns, 1)[0]["caption"]
    img_cap_id = words_to_ids(img_cap, vocab)
    return torch.Tensor(img_cap_id)


def choose_diff_images(num_images, vocab, cats_to_avoid=[]):
    """
    Returns all image_dir, caption pairs from 1 category that is not in cats_to_avoid
    """

    # excluding cat names to avoid
    cat_names = get_cat_names()
    allowed_cat_names = list(set(cat_names) - set(cats_to_avoid))
    cat_ids = coco.getCatIds(catNms=allowed_cat_names)

    img_dir_caps = []

    ## TODO: since the API works in bulk, we don"t have to use this for loop...
    for i in range(num_images) :
        # sample a category
        cat_id = np.random.choice(cat_ids, 1)

        # sample an image from that category
        img_id = np.random.choice(coco.getImgIds(catIds = cat_id), 1)

        # sample an annotation from that image
        ann_ids = coco_caps.getAnnIds(imgIds=img_id)
        anns = coco_caps.loadAnns(ann_ids)
        img_cap = np.random.choice(anns, 1)[0]["caption"]
        img_cap_id = words_to_ids(img_cap, vocab)

        # return path to image with caption
        img_dir = get_img_path(coco.loadImgs(img_id)[0]["file_name"])

        img_dir_caps.append((img_dir, img_cap, img_cap_id))

    return img_dir_caps

def choose_similar_images(num_images, cat_name, vocab):
    """
    Chooses num_images images that are in category cat_name
    """
    cat_id = coco.getCatIds(catNms=[cat_name])
    img_ids = coco.getImgIds(catIds=cat_id)
    img_sample = np.random.choice(img_ids, num_images, replace=False)
    img_dir_caps = []
    ## TODO: since the API works in bulk, we don"t have to use this for loop...
    for img_id in img_sample:
        # sample an annotation from that image
        ann_ids = coco_caps.getAnnIds(imgIds=img_id)
        anns = coco_caps.loadAnns(ann_ids)
        img_cap = np.random.choice(anns, 1)[0]["caption"]
        img_cap_id = words_to_ids(img_cap, vocab)
        while not img_cap_id:
            ann_ids = coco_caps.getAnnIds(imgIds=img_id)
            anns = coco_caps.loadAnns(ann_ids)
            img_cap = np.random.choice(anns, 1)[0]["caption"]
            img_cap_id = words_to_ids(img_cap, vocab)
        # return path to image with caption
        img_dir = get_img_path(coco.loadImgs([img_id])[0]["file_name"])
        img_dir_caps.append((img_dir, img_cap, img_cap_id))

    return img_dir_caps
