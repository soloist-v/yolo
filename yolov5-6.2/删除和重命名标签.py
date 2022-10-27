import os
from voc_xml_parse import VocXML, get_pair_sample_from_dir
from pathlib import Path
from shutil import copy
from inference import calc_iou

if __name__ == '__main__':
    img_dir = r"D:\dataset\project\QZDL\panpa_copy"
    xml_dir = r"D:\dataset\project\QZDL\panpa_copy"
    remove_names = []
    rename_names = {"panpa": "ren_panpa"}

    for name, img_path, xml_path in get_pair_sample_from_dir(img_dir, xml_dir):
        if not Path(xml_path).exists():
            continue
        voc = VocXML(xml_path)
        rm_ls = []
        for i, obj in enumerate(voc.annotation.objects):
            name = obj.name
            if name == "ren":
                for o in voc.annotation.objects:
                    if o.name not in ("panpa", "ren_panpa"):
                        continue
                    iou = calc_iou(obj.bndbox.bbox, o.bndbox.bbox)
                    print(iou)
                    if iou > 0.90:
                        rm_ls.append(i)
                        break
        for i in reversed(rm_ls):
            voc.annotation.objects.pop(i)
        for obj in voc.annotation.objects:
            name = obj.name
            if name in rename_names:
                obj.name = rename_names[name]
        voc.save(xml_path)
