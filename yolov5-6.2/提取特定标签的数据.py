import os

from voc_xml_parse import VocXML, get_pair_sample_from_dir
from pathlib import Path
from shutil import copy

if __name__ == '__main__':
    img_dir = r"D:\dataset\project\衢州电力\all"
    xml_dir = r"D:\dataset\project\衢州电力\all"
    sava_dir = r"D:\dataset\project\衢州电力\tacai_copy"
    targets_names = {"tacai", ""}
    for name, img_path, xml_path in get_pair_sample_from_dir(img_dir, xml_dir):
        if not Path(xml_path).exists():
            continue
        voc = VocXML(xml_path)
        for obj in voc.annotation.objects:
            if obj.name not in targets_names:
                continue
            break
        else:
            continue
        base_name, ext = os.path.splitext(name)
        t_img = os.path.join(sava_dir, name)
        t_xml = os.path.join(sava_dir, f'{base_name}.xml')
        copy(img_path, t_img)
        copy(xml_path, t_xml)
