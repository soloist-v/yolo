import os
import cv2
from voc_xml_parse import VocXML, Object, Bndbox
from pathlib import Path
from inference import Predictor, plot
from toolset.image_tools import imread, auto_resize


def auto_annotation(detectors, img, save_name: str, names):
    reference = []
    for detector in detectors:
        labels, boxes, scores = detector.predict_original(img, 416, 416, 0.4)
        reference.extend(list(zip(labels, boxes)))
        plot(img, labels, boxes, scores, img.shape)
    cv2.imshow("src", auto_resize(img, 1280, 640)[0])
    if cv2.waitKey(1) == 27:
        exit()
    if Path(save_name).exists():
        xml = VocXML(save_name)
    else:
        xml = VocXML.create()
    exists_names = set(map(lambda x: x.name, xml.annotation.objects))
    objects = xml.annotation.objects
    for label, xyxy in reference:
        if label not in names:
            continue
        if label in exists_names:
            continue
        label = fix_name_map.get(label, label)
        objects.append(Object.create(label, Bndbox.create(*xyxy)))
    xml.save(save_name)


def main(model_names, names, source_dir, exist_only=False, one=False, device="cuda:0"):
    detectors = []
    all_names = []
    for model_name, size in model_names:
        if not model_name:
            continue
        model = Predictor(model_name, device, size, classes=names)
        detectors.append(model)
        all_names.extend(model.names)
    if names is None:
        names = all_names
    if one:
        files = [source_dir]
    else:
        files = []
        for name in os.listdir(source_dir):
            if name[name.rfind(".") + 1:] not in img_suffix:
                continue
            file = os.path.join(source_dir, name)
            files.append(file)
    for index, file in enumerate(files):
        img = imread(file)
        if img is None:
            continue
        if exist_only:
            if not os.path.isfile(os.path.splitext(file)[0] + ".xml"):
                continue
        xml_path = os.path.splitext(file)[0] + ".xml"
        print(index, xml_path)
        auto_annotation(detectors, img, xml_path, names)


file_suffix = ("xml", "png", "jpg", 'bmp')
img_suffix = {"png", "jpg", 'bmp'}
fix_name_map = {"person": 'ren', "helmet": "anquanmao"}
conf_thres = .3
if __name__ == '__main__':
    target_dir = r'D:\dataset\project\衢州电力\高空抛物_安全帽_图片'  # 要预测的图片路径
    names = ['anquanmao']
    # names = ['person']
    # 设置模型
    models = [
        ["weights/qzdl_small.pt", 640],
    ]
    main(models, None, target_dir)
