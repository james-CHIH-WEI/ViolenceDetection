import urllib.request as urllib
import re
import cv2


def getTable():
    url = "https://en.wikipedia.org/wiki/List_of_common_resolutions"
    ret = urllib.urlopen(url)
    html = ret.read().decode("utf-8")
    match = re.search(
        '<table class="wikitable sortable">(.|\n)*?</table>', html)
    table = match.group()
    return table


def getAllResolution(table):
    resolution = []
    result = re.findall("<tr>((.|\n)*?)</tr>", table)
    for p in result:
        match_width = re.search(
            '<td style="text-align:right;border-right:none">((.|\n)*?)</td>', str(
                p)
        )
        match_height = re.search(
            '<td style="text-align:left;border-left:none">((.|\n)*?)</td>', str(
                p)
        )
        if match_width is None or match_height is None:
            continue

        width = int(match_width.group(1).replace("\\n", ""))
        height = int(match_height.group(1).replace("\\n", ""))

        resolution.append({"width": width, "height": height})
    return resolution


def resolutionTest(width, height):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    rw = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    rh = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    if rw == width and rh == height:
        print("{} x {}: OK".format(width, height))


cap = cv2.VideoCapture(2)
_, frame = cap.read()
h, w, c = frame.shape
print("預設解析度為 {} x {}".format(w, h))
print("準備從維基百科取得解析度資料")
table = getTable()
resolutionList = getAllResolution(table)
print("已解析出所有解析度")
print("開始測試攝影機支援的解析度")
for p in resolutionList:
    resolutionTest(p["width"], p["height"])
