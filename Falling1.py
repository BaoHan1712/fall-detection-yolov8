from ultralytics import YOLO
model = YOLO('fall_det_1.pt')
results = model(source = 0, show = True, conf = 0.4, imgsz = 320,
max_det=200 ,iou= 0.4 )
#if u want to save video and img , u can :
#results = model(source = "your video ", show = True, conf = 0.4, imgsz = 320,max_det=200 ,iou= 0.4 , save = True)
