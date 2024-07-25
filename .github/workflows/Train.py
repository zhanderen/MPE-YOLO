import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'C:\Users\qinyi\Desktop\ultralytics-main-vs\runs\train\exp4\yolov8s.yaml')
    #model.load('yolov8n.pt') # loading pretrain weights
    model.train(data=r'C:\Users\qinyi\Desktop\ultralytics-main-vs\ultralytics\cfg\datasets\coco.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=4,
                close_mosaic=0,
                workers=4,
                device='0',
                cos_lr=True,
                optimizer='SGD', # using SGD
               # resume='True', # last.pt path
                # amp=False # close amp
                # fraction=0.2,
                project='runs/train',
                name='exp',
                )