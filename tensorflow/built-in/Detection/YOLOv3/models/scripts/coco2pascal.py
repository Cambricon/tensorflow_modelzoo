import baker
import json
import os
from cytoolz import merge, join, groupby
from cytoolz.compatibility import iteritems
from cytoolz.curried import update_in
from itertools import starmap
from collections import deque
from lxml import etree, objectify
from scipy.io import savemat
from scipy.ndimage import imread
from pathlib import Path
from tqdm import tqdm

def keyjoin(leftkey, leftseq, rightkey, rightseq):
    return starmap(merge, join(leftkey, leftseq, rightkey, rightseq))


def root(folder, filename, height, width):
    E = objectify.ElementMaker(annotate=False)
    return E.annotation(
            E.folder(folder),
            E.filename(filename),
            E.source(
                E.database('MS COCO 2014'),
                E.annotation('MS COCO 2014'),
                E.image('Flickr'),
                ),
            E.size(
                E.width(width),
                E.height(height),
                E.depth(3),
                ),
            E.segmented(0)
            )


def instance_to_xml(anno):
    E = objectify.ElementMaker(annotate=False)
    xmin, ymin, width, height = anno['bbox']
    return E.object(
            E.name(anno['category_id']),
            E.bndbox(
                E.xmin(xmin),
                E.ymin(ymin),
                E.xmax(xmin+width),
                E.ymax(ymin+height),
                ),
            )


@baker.command
def write_categories(coco_annotation, dst):
    with open(os.path.abspath(coco_annotation)) as file:
        content = json.load(file)
        categories = tuple( d['name'] for d in content['categories'])
        savemat(os.path.abspath(dst), {'categories': categories})


def get_instances(coco_annotation):
    coco_annotation = os.path.abspath(coco_annotation)
    with open(coco_annotation) as file:
        content = json.load(file)
        categories = {d['id']: d['name'] for d in content['categories']}
        return categories, tuple(keyjoin('id', content['images'], 'image_id', content['annotations']))

def rename(name, year=2014):
        out_name = os.path.splitext(name)[0]
        # out_name = out_name.split('_')[-1]
        # out_name = '{}_{}'.format(year, out_name)
        return out_name


@baker.command
def create_imageset(annotations, dst):
    annotations = os.path.abspath(annotations)
    dst = os.path.abspath(dst)
    val_txt = dst / 'val.txt'
    train_txt = dst / 'train.txt'

    for val in annotations.listdir('*val*'):
        val_txt.write_text('{}\n'.format(val.basename().stripext()), append=True)

    for train in annotations.listdir('*train*'):
        train_txt.write_text('{}\n'.format(train.basename().stripext()), append=True)
@baker.command
def create_annotations(dbPath, subset, dst='COCO_to_VOC2014/test/annotations'):
    """ converts annotations from coco to voc pascal. 
        parameters:

        dbPath: folder which contains the annotations subfolder which contains the annotations file in .json format. 
                Note: the corresponding images should be in the train2014 or val2014 subfolder.
        subset: which of the .json files should be opened e.g. train for the "instances_train2014.json" file
        dst: destination folder for the annotations. Will be created if it doesn't exist e.g. "annotations_voc"
     """
    if not os.path.exists(dst):
        os.makedirs(dst)
    annotations_path = os.path.join(os.path.abspath(dbPath),'annotations','instances_'+str(subset)+'2014.json')
    images_Path = os.path.join(os.path.abspath(dbPath),str(subset)+'2014')
    print("reading data...")
    categories , instances= get_instances(annotations_path)
    print("finished reading data")
    dst = os.path.abspath(dst)
    for i, instance in tqdm(enumerate(instances),desc="rewriting categories"):
        instances[i]['category_id'] = categories[instance['category_id']]

    for name, group in tqdm(iteritems(groupby('file_name', instances)), total=len(groupby('file_name', instances)), desc="processing annotations"):
        img = imread(os.path.abspath(os.path.join(images_Path,name)))
        if img.ndim == 3:
            out_name = rename(name)
            annotation = root('COCO_to_VOC2014', '{}.jpg'.format(out_name),  group[0]['height'], group[0]['width'])
            for instance in group:
                #print(instance);
                annotation.append(instance_to_xml(instance))
            etree.ElementTree(annotation).write(os.path.join(dst, '{}.xml'.format(out_name)))
            #print( out_name)
        #else:
            #print (instance['file_name'])
            
if __name__ == '__main__':
    baker.run()