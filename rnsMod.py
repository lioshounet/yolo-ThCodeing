
def modCheack(Rs,ModArr):
    if(ModArr==0):
        ModArr=['person','bicycle','car','motorcycle','airplane','bus','train',
                'truck','boat', 'traffic light','fire hydrant', 'stop sign',
                'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
                'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
                'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
                'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
                'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
                'hair drier', 'toothbrush']
    Ws = [ModArr,[]]
    Msg = []

    # 空0填补
    for i in Ws[0]:
        Ws[1].append(0)

    # 计数数组
    for r in Rs:
        Ws[1][r] = Ws[1][r] + 1

    # 字符串转化
    for m in range(0,len(Ws[0])):
        if(Ws[1][m]):
            Msg.append(str(Ws[0][m])+":"+str(Ws[1][m]))
    print("信息统计完成")
    return Msg

if __name__ == "__main__":
    print(modCheack([44,55,44],0))