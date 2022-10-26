import argparse
import os
from rknn.api import RKNN

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', type=str, required=True, help='weights path')
    parser.add_argument('--rknn', type=str, default='', help='保存路径')
    parser.add_argument('--dataset', type=str, default="./dataset.txt", help='dataset txt')
    parser.add_argument('--precompile', action="store_true", help='是否是预编译模型')
    parser.add_argument("--div_255", action="store_true", help="input / 255")
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--platform', type=str, default="rk3399pro", help='target platform')
    opt = parser.parse_args()
    print("options:\n\t", opt)
    ONNX_MODEL = opt.onnx
    if opt.rknn:
        RKNN_MODEL = opt.rknn
    else:
        RKNN_MODEL = "%s.rknn" % os.path.splitext(ONNX_MODEL)[0]
    rknn = RKNN()
    print('--> config model')
    if opt.div_255:
        mean_val = 255
    else:
        mean_val = 1
    print("mean value:>>", mean_val)
    # rknn.config(channel_mean_value='0 0 0 %s' % mean_val, reorder_channel='2 1 0', batch_size=opt.batch_size,
    #             target_platform=opt.platform, quantize_input_node=True)
    # (input[..., :C] - mean_values) / std_values
    rknn.config(mean_values=[[0, 0, 0]],
                std_values=[[255, 255, 255]],
                reorder_channel='2 1 0',
                batch_size=opt.batch_size,
                target_platform=opt.platform,
                quantize_input_node=True)
    # Load model
    print('--> Loading model')
    print('onnx model path:', ONNX_MODEL)
    ret = rknn.load_onnx(model=ONNX_MODEL)
    assert ret == 0, "Load onnx failed!"
    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=True, dataset=opt.dataset)
    assert ret == 0, "Build onnx failed!"
    # Export model
    print('--> Export RKNN model')
    # ret = rknn.export_rknn(RKNN_MODEL)
    ret = rknn.export_rknn_precompile_model(RKNN_MODEL)
    assert ret == 0, "Export %s.rknn failed!" % opt.rknn
    print("rknn export success, saved as %s" % RKNN_MODEL)
    print('done')
