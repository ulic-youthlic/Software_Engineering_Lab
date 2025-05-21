from ultralytics import YOLO
import cv2

#script start
import pyautogui
from pynput import mouse
from pynput.mouse import Button, Controller
import keyboard
import time
import pynput

ctr = pynput.mouse.Controller()

# 鼠标点击函数
def move_and_click(x, y):
    '''
    screen_width, screen_height = pyautogui.size()
    center_x = screen_width // 2
    center_y = screen_height // 2

    ctr.move((x - center_x), (center_y - y))
    ctr.click(Button.left, 1)
    '''

# 压枪函数
#def spray():
#    for _ in range(10):
#        pyautogui.move(0, 10, duration=0)


# 主函数
def aim_and_fire(enemy):

    x1, y1, x2, y2 = enemy[0]
    enemy_center_x = (x1 + x2) // 2
    enemy_center_y = (y1 + y2) // 2

    # 按住鼠标并持续射击
    #move_and_click(enemy_center_x, enemy_center_y)

    # 压枪
    #spray()

    # 释放鼠标
    print("开火")

#script end

model=YOLO('best.pt')

cap=cv2.VideoCapture(1)#0是默认内置摄像头，1是vtb那个摄像头,2是obs
while True:
    ret,frame=cap.read()
    if not ret:#failed to open camera
        break
    #运行模型预测
    results=model(frame)
    #获取结果并打印
    for r in results:
        boxes=r.boxes#包含所有检测框的信息    xyxy/xywh/xyxyn/xywhn框位置    conf置信度 cls类别   id标号(我是predict所以没有)
        masks=r.masks#包含检测掩码的mask对象
        probs=r.probs#包含分类中每个类别概率的Probs对象
        xyxy=boxes.xyxy
        xyxy_np=xyxy.numpy()
        cls=boxes.cls
        cls_np=cls.numpy()
        conf=boxes.conf
        conf_np=conf.numpy()
        print(xyxy)
        print(cls)#enemies0，1，3对应的cls值分别是3,4,5
        print(conf)
        #遍历cls,找到数值为3，4，5的点对应的索引，按3>4>5,同级取置信度高的的优先级取xyxy中优先级最高的那一个，作为enemy
        index3 = -1
        conf3=0
        index4 = -1
        conf4=0
        index5 = -1
        conf5=0
        for i, cls_value in enumerate(cls_np):
            if cls_value == 3.:
                if conf_np[i] > conf3:
                    index3 = i
            if cls_value == 4.:
                if conf_np[i] > conf4:
                    index4 = i
            if cls_value == 5.:
                if conf_np[i] > conf5:
                    index5 = i
        if(index3>0):
            enemy=xyxy_np[index3:]
            print("enemy")
            print(enemy)
            aim_and_fire(enemy)
        elif index4>0:
            enemy=xyxy_np[index4:]
            print("enemy")
            print(enemy)
            aim_and_fire(enemy)
        elif index5>0:
            enemy=xyxy_np[index5:]
            print("enemy")
            print(enemy)
            aim_and_fire(enemy)
    '''if results:
        print(results)
        for det in results[0]:
            x1,y1,x2,y2,conf,class_id=det
            print(f"Detected class: {class_id}, Confidence: {conf:.2f}, Box: [{x1}, {y1}, {x2}, {y2}]")
    '''
    #显示带标注的图像
    annotated_frame=results[0].plot()
    cv2.imshow('YOLOv8',annotated_frame)
    if cv2.waitKey(1)==ord('i'):#按i退出
        break
cap.release()
cv2.destroyAllWindows()