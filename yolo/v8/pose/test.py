from tbknet import YOLO

# Load a model
#model = YOLO('yolov8n-pose.pt')  # load an official model
model = YOLO('D:/phycharmcsl/ultralytics_pose/ultralytics\yolo/v8\pose/runs\pose/4090train9\weights/best.pt')  # load a custom model


# Predict with the model  D:\phycharmcsl\ultralytics_pose\testpic\215.png
#results = model('../testpic/215.png',show=True,save=True, )  # predict on an images



# Predict with the model  D:\CSL\tbknet-pose(new\tbknet\yolo\v8\pose\datasets\Tea_keypoint\images\test
results = model(source='D:\phycharmcsl/ultralytics_pose/testpic/215.png',imgsz=640,project='615(val',name ='4090train9',save=True)  # predict on an images
# for result in results:
#     boxes = result.boxes  # Boxes object for bbox outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     probs = result.probs  # Class probabilities for classification outputs

#yolo pose predict model='D:/phycharmcsl/ultralytics_pose/tbknet\yolo/v8\pose/runs\pose/train\weights/best.pt'  source='D:\phycharmcsl/ultralytics_pose/testpic/215.png' visualize=True

#yolo pose predict model=tbknet/yolo/v8/pose/runs/pose/train/weights/best.pt source='./tbknet/yolo/v8/testpic/Leaves572.png' show=True save=True visualize=True

#results = model.predict(source='D:\phycharmcsl/ultralytics_pose/testpic/215.png',visualize=True,name='test1')
# results = model( batch=1, conf=0.001, iou=0.5, name='Test',save_json= True,visualize=True)  # 参数和训练用到的一样
# results = model(visualize=True)  # 参数和训练用到的一样
