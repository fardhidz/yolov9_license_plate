from abc import ABC

import detect
import train
import train_dual
from segment import train as train_seg
from segment import predict as predict_seg
from panoptic import train as train_pan
from panoptic import predict as predict_pan


class Yolov9(ABC):

    def __init__(self, mode):
        self.mode = mode

        if self.mode in ['object-detection']:
            self.model = Yolov9ObjectDetection(None)
        elif self.mode in ['instance-segmentation']:
            self.model = Yolov9InstanceSegmentation(None)
        elif self.mode in ['panoptic-segmentation']:
            self.model = Yolov9PanopticSegmentation(None)

    def train(self, **kwargs):
        return self.model.train(**kwargs)

    def predict(self, **kwargs):
        return self.model.predict(**kwargs)


class Yolov9ObjectDetection(Yolov9):

    def __init__(self, mode):
        super().__init__(mode)

    def train(self, **kwargs):

        mode = kwargs.get('mode', 'dual')
        data = kwargs.get('data', None)

        assert data is not None
        assert mode in ['dual', 'gelan']

        opt = train_dual.parse_opt() if mode in ['dual'] else train.parse_opt()

        opt.data = data
        opt.epochs = kwargs.get('epochs', 1)

        opt.workers = kwargs.get('workers', 8)
        opt.device = kwargs.get('device', 0)
        opt.batch_size = kwargs.get('batch_size', 8)
        opt.imgsz = kwargs.get('imgsz', 640)
        opt.weights = kwargs.get('weights', '')
        opt.hyp = kwargs.get('hyp', 'hyp.scratch-high.yaml')
        opt.min_items = kwargs.get('min_items', 0)
        opt.close_mosaic = kwargs.get('close_mosaic', 15)

        if mode in ['dual']:

            opt.cfg = kwargs.get('cfg', 'models/detect/yolov9-c.yaml')
            opt.name = kwargs.get('name', 'yolov9-c')

            train_dual.main(opt)

        elif mode in ['gelan']:

            opt.cfg = 'models/detect/gelan-c.yaml'
            opt.name = 'gelan-c'

            train.main(opt)

    def predict(self, **kwargs):
        source = kwargs.get('source', None)
        weights = kwargs.get('weights', None)

        assert source is not None
        assert weights is not None

        opt = detect.parse_opt()

        opt.source = source
        opt.weights = weights

        opt.imgsz = kwargs.get('imgsz', 640)
        opt.device = kwargs.get('device', 0)
        opt.device = kwargs.get('name', 'yolov9-c')

        detect.main(opt)


class Yolov9InstanceSegmentation(Yolov9):

    def __init__(self, mode):
        super().__init__(mode)

    def train(self, **kwargs):

        data = kwargs.get('data', None)

        assert data is not None

        opt = train_seg.parse_opt()

        opt.data = data
        opt.epochs = kwargs.get('epochs', 1)

        opt.workers = kwargs.get('workers', 8)
        opt.device = kwargs.get('device', 0)
        opt.batch_size = kwargs.get('batch_size', 8)
        opt.imgsz = kwargs.get('imgsz', 640)
        opt.weights = kwargs.get('weights', '')
        opt.hyp = kwargs.get('hyp', 'hyp.scratch-high.yaml')
        opt.close_mosaic = kwargs.get('close_mosaic', 10)
        opt.no_overlap = kwargs.get('no_overlap', True)

        opt.cfg = kwargs.get('cfg', 'models/segment/gelan-c-seg.yaml')
        opt.name = kwargs.get('name', 'gelan-c-seg')

        train_seg.main(opt)

    def predict(self, **kwargs):
        source = kwargs.get('source', None)
        weights = kwargs.get('weights', None)

        assert source is not None
        assert weights is not None

        opt = predict_seg.parse_opt()

        opt.source = source
        opt.weights = weights

        opt.imgsz = kwargs.get('imgsz', 640)
        opt.device = kwargs.get('device', 0)
        opt.device = kwargs.get('name', 'yolov9-seg')

        predict_seg.main(opt)


class Yolov9PanopticSegmentation(Yolov9):

    def __init__(self, mode):
        super().__init__(mode)

    def train(self, **kwargs):

        data = kwargs.get('data', None)

        assert data is not None

        opt = train_pan.parse_opt()

        opt.data = data
        opt.epochs = kwargs.get('epochs', 1)

        opt.workers = kwargs.get('workers', 8)
        opt.device = kwargs.get('device', 0)
        opt.batch_size = kwargs.get('batch_size', 8)
        opt.imgsz = kwargs.get('imgsz', 640)
        opt.weights = kwargs.get('weights', '')
        opt.hyp = kwargs.get('hyp', 'hyp.scratch-high.yaml')
        opt.close_mosaic = kwargs.get('close_mosaic', 10)
        opt.no_overlap = kwargs.get('no_overlap', True)

        opt.cfg = kwargs.get('cfg', 'models/panoptic/gelan-c-pan.yaml')
        opt.name = kwargs.get('name', 'gelan-c-pan')

        train_pan.main(opt)

    def predict(self, **kwargs):
        source = kwargs.get('source', None)
        weights = kwargs.get('weights', None)

        assert source is not None
        assert weights is not None

        opt = predict_pan.parse_opt()

        opt.source = source
        opt.weights = weights

        opt.imgsz = kwargs.get('imgsz', 640)
        opt.device = kwargs.get('device', 0)
        opt.device = kwargs.get('name', 'yolov9-seg')

        predict_pan.main(opt)

