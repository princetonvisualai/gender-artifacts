import pickle 
import re

from pycocotools.coco import COCO

directory = '/n/fs/visualai-scr/Data/Coco/2014data/'
SPLIT = 'val'
male_words = set(["male", "boy", "man", "gentleman", "boys", "men", "males", "gentlemen"])
fem_words = set(["female", "girl","woman", "lady", "girls", "women", "females", "ladies"])
num_cap = 2

COCO = COCO(directory + 'annotations/captions_{}2014.json'.format(SPLIT))

def is_overlap(sentence, words):
    return len(sentence.intersection(words)) > 0

def format(image): 
    filename = directory + '{0}2014/COCO_{0}2014_{1}.jpg'.format(SPLIT, str(image).zfill(12))
    return filename
images = COCO.getImgIds()
output = {}

for image in images:
    captions = COCO.loadAnns(ids=COCO.getAnnIds(imgIds=[image]))
    is_fem, is_mal = False, False
    fem_count, mal_count = 0, 0 
    for caption in captions: 
        cleaned_str = res = re.sub(r'[^\w\s]', '', caption['caption'].lower())
        sentence = set(cleaned_str.split(' '))
        fem_overlap = is_overlap(sentence, fem_words)
        male_overlap = is_overlap(sentence, male_words)
        if fem_overlap: 
            is_fem = True 
            fem_count += 1
        if male_overlap: 
            is_mal = True 
            mal_count += 1
    if is_fem and is_mal: continue 
    elif is_fem and fem_count >= num_cap: output[format(image)] = 1
    elif is_mal and mal_count >= num_cap: output[format(image)] = 0

pickle.dump(output, open('mals/{}.pkl'.format(SPLIT), 'wb'))
print(len(output))