import os
import shutil

from voc_xml_parse import VocXML, Object, Bndbox, get_pair_sample_from_dir


def split_xml(src_dir, xml_save_dir, img_save_dir=None):
    os.makedirs(xml_save_dir, exist_ok=True)
    if img_save_dir is not None:
        os.makedirs(img_save_dir, exist_ok=True)
    os.makedirs(os.path.join(xml_save_dir, "background"), exist_ok=True)
    for img_name, img_path, label_path in get_pair_sample_from_dir(src_dir, src_dir):
        if not os.path.exists(label_path):
            continue
        print(img_path)
        base_name, ext = os.path.splitext(img_name)
        if img_save_dir is not None:
            img_save_path = os.path.join(img_save_dir, img_name)
            shutil.copy(img_path, img_save_path)
        else:
            img_save_path = img_path
        xml = VocXML(label_path)
        xml_name = f"{base_name}.xml"
        objects = xml.annotation.objects
        if len(objects) == 0:
            xml_save_path = os.path.join(xml_save_dir, "background", xml_name)
            VocXML.create(img_save_path).save(xml_save_path)
        obj_map = {}
        for obj in objects:
            name = obj.name
            if name not in obj_map:
                obj_map[name] = VocXML.create()
            single_xml = obj_map[name]
            single_xml.annotation.objects.append(obj)
        for name, single_xml in obj_map.items():
            os.makedirs(os.path.join(xml_save_dir, name), exist_ok=True)
            save_path = os.path.join(xml_save_dir, name, xml_name)
            single_xml.save(save_path)


if __name__ == '__main__':
    src_dir = r"D:\Videos\dataset\dataset_single\temp"
    xml_save_dir = r"D:\Videos\dataset\tizi_ciping_img"
    img_save_dir = r"D:\Videos\dataset\tizi_ciping_xml"
    split_xml(src_dir, xml_save_dir, img_save_dir)
