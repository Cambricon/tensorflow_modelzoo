import json

label = json.load(open("./path_to/datasets/COCO17/annotations/instances_val2017.json"))
cats = label['categories']
f = open("coco17.names", "w")
for cat in cats:
    f.write("'{}', ".format(cat['name']))
f.close()
