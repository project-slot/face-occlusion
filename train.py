from ultralytics import YOLO


# Load a model
model = YOLO("yolov8s.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="data.yaml", epochs=2, device='mps', classes=[0,1,3,4,6])  # train the model
# metrics = model.val()  # evaluate model performance on the validation set
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
path = model.export()  # export the model to ONNX format
