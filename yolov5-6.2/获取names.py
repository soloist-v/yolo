from voc_xml_parse import VocXML, get_pair_sample_from_dir
from pathlib import Path
from collections import defaultdict

if __name__ == '__main__':
    img_dir = r"D:\dataset\project\all"
    label_dir = r"D:\dataset\project\all"
    names = set()
    target_names = ["gongxie", "yuxie", "tacai"]
    counter = defaultdict(lambda: 0)
    # counter.default_factory = 0

    for name, image, label in get_pair_sample_from_dir(img_dir, label_dir):
        # print(name)
        if not Path(label).exists():
            continue
        xml = VocXML(label)
        for obj in xml.annotation.objects:
            names.add(obj.name)
            counter[obj.name] += 1
    for k, v in counter.items():
        if k not in target_names:
            continue
        print(k, v)
