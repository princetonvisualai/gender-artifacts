import pickle, argparse
import pandas as pd

#python person_only_labels_val.py --images_val2014_path images_val2014.csv --labels_filename gender_labels.pkl

parser = argparse.ArgumentParser()
parser.add_argument('--images_val2014_path', type=str, default=None)
parser.add_argument('--labels_filename', type=str, default=None)
arg = vars(parser.parse_args())
print('\n', arg, '\n')


labels = {} # create dictionary mapping image path to label
gender_labels = {'Female':1, 'Male':0}
images_val2014 = pd.read_csv(arg['images_val2014_path'])

def id_to_imgpath(id):
    # id is a 12 digit number so pad id with zeros
    zerofilled_id=str(id).zfill(12)
    return "/n/fs/visualai-scr/Data/Coco/2014data/val2014/COCO_val2014_"+zerofilled_id +".jpg"

# grab only the images containing female (1) or male (0) labels
for gender_label in list(gender_labels.keys()):
    print('Creating ' + gender_label + ' data...')
    data = images_val2014[list(images_val2014['bb_gender'] == gender_label)]
    for row in range(data.shape[0]):
        labels[id_to_imgpath(list(data['id'])[row])] = gender_labels[gender_label]

with open(arg['labels_filename'], 'wb') as handle:
    pickle.dump(labels, handle, protocol=4)
