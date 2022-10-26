from toolset.image_tools import imread, imwrite
from voc_xml_parse import VocXML, Objects, Object, Bndbox, get_pair_sample_from_dir
from pathlib import Path

if __name__ == '__main__':
    img_dir = r"D:\dataset\Crossfire\zombie_all"
    """
    {
    'lvjuren', 'diyu', 'person', 'weizhuang', 
    'gangtie', 'xukong', 'zhongjie', 'miwu', 'fkbb', 'shikong', 
    'xiaohong', 'baojun', 'yaoji', 'renzhe', 'nvhuang'
    }
    """
    names = set()
    for name, img_path, label_path in get_pair_sample_from_dir(img_dir, img_dir):
        img = imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        if not Path(label_path).exists():
            xml = VocXML.create(img_path, w, h)
            xml.save(label_path)
        else:
            xml = VocXML(label_path)
        for obj in xml.annotation.objects:
            names.add(obj.name)

    print(names)
