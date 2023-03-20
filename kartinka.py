from imageai.Detection import ObjectDetection
import os   #встр.библ в py-, позволяет работать с виндой?

exec_path = os.getcwd()  #функция позволяет указать путь к проекту

detector = ObjectDetection() #объект детектор на основе класса ObjDetect
detector.setModelTypeAsRetinaNet() #используется ретинанет модель для обнаружения объекта
detector.setModelPath(os.path.join(
    exec_path, "resnet50_coco_best_v2.0.1.h5")
)
detector.loadModel()

list = detector.detectObjectsFromImage(
    input_image=os.path.join(exec_path, "obj.jpg"),
     output_image_path=os.path.join(exec_path, "new_obj.jpg")
)
