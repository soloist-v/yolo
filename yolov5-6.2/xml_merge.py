import os
import shutil

from voc_xml_parse import VocXML, img_formats, walk_dir


def merge_xml(img_dir, xml_dir, save_dir, clear_old=False):
    xml_map = {}
    img_save_dir = os.path.join(save_dir, "")
    xml_save_dir = os.path.join(save_dir, "")
    if clear_old:
        shutil.rmtree(img_save_dir, ignore_errors=True)
        shutil.rmtree(xml_save_dir, ignore_errors=True)
    os.makedirs(img_save_dir, exist_ok=True)
    os.makedirs(xml_save_dir, exist_ok=True)
    for name in os.listdir(xml_dir):
        cur_label = name
        label_dir = os.path.join(xml_dir, name)
        print("walk xml dir:", label_dir)
        for name, filepath in walk_dir(label_dir):
            base, ext = os.path.splitext(name)
            if ext.lower() != ".xml":
                continue
            if name not in xml_map:
                xml_map[name] = VocXML.create()
            for obj in VocXML(filepath).annotation.objects:
                if obj.name != cur_label:
                    continue
                xml_map[name].annotation.objects.append(obj)
    for name, xml in xml_map.items():
        save_path = os.path.join(xml_save_dir, name)
        print("merge xml:", save_path)
        xml.save(save_path)
    for name, img_path in walk_dir(img_dir):
        base, ext = os.path.splitext(name)
        if ext.lower() not in img_formats:
            continue
        print("copy img:", img_path)
        shutil.copy(img_path, os.path.join(img_save_dir, name))


if __name__ == '__main__':
    is_clear = True
    img_dir = r"/mnt/mydrive/public/衢州电力/0-数据集ALL/一二期合并/img/"
    xml_dir = r"/mnt/mydrive/public/衢州电力/0-数据集ALL/一二期合并/xml/"
    save_dir = r"/mnt/mydrive/public/衢州电力/0-数据集ALL/一二期合并/train/true"
    merge_xml(img_dir, xml_dir, save_dir, is_clear)
