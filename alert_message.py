# -*- coding: UTF-8 -*-
from linebot import LineBotApi
from linebot.models import *
import os


def copy_video(alert_video_folder_image, alert_video_folder_video, alert_video_name):
    path = "/var/www/html/violence/"
    image_path = path + alert_video_folder_image[15:]
    video_path = path + alert_video_folder_video[15:]
    if not os.path.isdir(image_path):
        os.makedirs(image_path)
    if not os.path.isdir(video_path):
        os.makedirs(video_path)

    os.system("cp " + alert_video_folder_image + "/" + alert_video_name +
              ".png " + image_path + "/" + alert_video_name + ".png")
    os.system("cp " + alert_video_folder_video + "/" + alert_video_name +
              ".mp4 " + video_path + "/" + alert_video_name + ".mp4")


def push(alert_video_folder_image, alert_video_folder_video, alert_video_name):
    copy_video(alert_video_folder_image,
               alert_video_folder_video, alert_video_name)
    url = "https://jamesproject.ddns.net/violence/"
    img = url + alert_video_folder_image[15:] + \
        "/" + alert_video_name + ".png"
    video = url + alert_video_folder_video[15:] + \
        "/" + alert_video_name + ".mp4"

    to = "U36f04328f40023e68dfc079aa1a537e3"

    line_bot_api = LineBotApi(
        "qgUrLjc3fkr3y7lIw1AndkWI2M5eX16Ah7VJxrkGDU2HEkhCz39I+aQDhxk5z9U1dGj3nOG6IBscjldbe3ELIxv0H0iNkCS8lFJNMrPLRpqMzrP7kuMqDddoRrRVjf0gA9OUByJ6FDnQyf9haHvLWAdB04t89/1O/w1cDnyilFU="
    )

    message = TextSendMessage(text="發生疑似虐待行為，請進行確認~")

    video_message = VideoSendMessage(
        original_content_url=video,
        preview_image_url=img
    )

    line_bot_api.push_message(to, [message, video_message])
