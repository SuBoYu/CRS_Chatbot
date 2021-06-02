from __future__ import unicode_literals
import os
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import *
import configparser
import datetime

import chat_model

import pickle
import json

import random

app = Flask(__name__)

# LINE 聊天機器人的基本資料
# 每個python file都要放下面幾行
config = configparser.ConfigParser()
config.read('config.ini')

line_bot_api = LineBotApi(config.get('line-bot', 'channel_access_token'))
handler = WebhookHandler(config.get('line-bot', 'channel_secret'))

chatmodel_dict = dict()

dir = "FM-train-data"

with open('../../data/{}/FM_test_list.pickle'.format(dir), 'rb') as f:
    test_list = pickle.load(f)

with open('../../data/{}/item_dict.json'.format(dir), 'r') as f:
    item_dict = json.load(f)

with open('dict5.json') as f:
    item_image_url_map = json.load(f)

with open('../../data/{}/tag_question_map.json'.format(dir), 'r') as f:
     tq_map = json.load(f)

with open('../../data/{}/question_id.json'.format(dir), 'r') as f:
    qid_map = json.load(f)

print(len(item_dict))
print(len(item_image_url_map))

# 接收 LINE 的資訊
@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']

    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    print(body)

    try:
        handler.handle(body, signature)

    except InvalidSignatureError:
        abort(400)

    return 'OK'


# 對話
@handler.add(MessageEvent, message=TextMessage)
def reply(event):
    if event.source.user_id != "Udeadbeefdeadbeefdeadbeefdeadbeef":
        user_res = event.message.text
        print(f"user response : {user_res}")

        if user_res == "S" or user_res == "s" or user_res == "Change":
            # Model construct and First Conversation
            # key: userID, value: chatmodel
            '''
            (21767, 1100)
            (10711, 4558)
            (16437, 5639)
            (4861, 998)
            (30811, 581)
            (17940, 4451)
            (12018, 1899)
            (26631, 1515)
            (25914, 1936)
            (29816, 5127)
            '''
            random.seed(datetime.datetime.now())
            ran_int = random.randint(0, len(test_list))
            user_item_pair = test_list[ran_int]

            user = str(user_item_pair[0])
            item = str(user_item_pair[1])
            chatmodel_dict[event.source.user_id] = chat_model.chat_model(user, item)

            if len(item_image_url_map[item]["name"]) <= 40:
                real_name = item_image_url_map[item]["name"]
            else:
                real_name = item_image_url_map[item]["name"]
                real_name = real_name.split(" ")
                if type(real_name) == list:
                    real_name = real_name[-5] + " " + real_name[-4] + " " + real_name[-3] + " " + real_name[-2] + " " + real_name[-1]


            info_message = "User_id: " + user + "\nItem_name: " + real_name +"\nItem_entity:\n"

            true_entity_list = item_dict[item]["categories"]

            for i,j in enumerate(true_entity_list):
                info_message += str(i+1)+". "+j+"\n"

            Confirm_template = TemplateSendMessage(
                alt_text='confirm template請使用手機版line',
                template=ConfirmTemplate(
                    title='ConfirmTemplate',
                    text=info_message,
                    actions=[
                        MessageTemplateAction(
                            label='OK',
                            text='OK',
                        ),
                        MessageTemplateAction(
                            label='Change',
                            text='Change'
                        )
                    ]
                )
            )

            line_bot_api.reply_message(event.reply_token, Confirm_template)

        if user_res == "OK":

            print("chatmodel_dict", chatmodel_dict)
            chatmodel = chatmodel_dict[event.source.user_id]
            print("first conversation")
            k = chatmodel.first_conversation(user_res)

            if k[0] == "D": # ask entity
                # "Do you like" + str(self.agent_utterance.data['facet']) + "?"
                k = k[1:]
                question_id = tq_map[k]
                question = qid_map[str(question_id)]
                k = question + " \"" + k + "\"?"
                print(k)
                Confirm_template = TemplateSendMessage(
                    alt_text='confirm template請使用手機版line',
                    template=ConfirmTemplate(
                        title='ConfirmTemplate',
                        text=k,
                        actions=[
                            MessageTemplateAction(
                                label='Yes',
                                text='Y',
                            ),
                            MessageTemplateAction(
                                label='No',
                                text='N'
                            )
                        ]
                    )
                )
            elif k[0] == "r":
                print(k)
                # "recommendation list: "
                Confirm_template = TemplateSendMessage(
                    alt_text='confirm template請使用手機版line',
                    template=ConfirmTemplate(
                        title='ConfirmTemplate',
                        text=k,
                        actions=[
                            MessageTemplateAction(
                                label='Hit',
                                text='H',
                            ),
                            MessageTemplateAction(
                                label='Reject',
                                text='R'
                            )
                        ]
                    )
                )

            line_bot_api.reply_message(event.reply_token, Confirm_template)

        elif user_res == "Q": # Quit
            print("remove userid_chatmodel pair")
            del chatmodel_dict[event.source.user_id]

        elif user_res in ["Y", "N", "H", "R"]: # Conversation
            chatmodel = chatmodel_dict[event.source.user_id]
            print("conversation")
            k = chatmodel.conversation(user_res)

            if k[0] == "D": # ask entity
                # "Do you like" + str(self.agent_utterance.data['facet']) + "?"
                k = k[1:]
                question_id = tq_map[k]
                question = qid_map[str(question_id)]
                k = question + " \"" + k + "\"?"
                print(k)
                Confirm_template = TemplateSendMessage(
                    alt_text='confirm template請使用手機版line',
                    template=ConfirmTemplate(
                        title='ConfirmTemplate',
                        text=k,
                        actions=[
                            MessageTemplateAction(
                                label='Yes',
                                text='Y',
                            ),
                            MessageTemplateAction(
                                label='No',
                                text='N'
                            )
                        ]
                    )
                )
                line_bot_api.reply_message(event.reply_token, Confirm_template)
            elif k[0] == "r":
                print(k)
                rec_item_id_list = k[1]
                item_name_list = []
                item_image_list = []
                item_des_list = []
                item_url_list = []
                for i in rec_item_id_list:
                    print("get-item-info's item id: ", i)
                    if type(item_image_url_map[i]["name"]) == list:
                        name = "".join(item_image_url_map[i]["name"])
                    else:
                        name = item_image_url_map[i]["name"]
                    if len(name) <= 40:
                        item_name_list.append(name)
                    else:
                        name = name.split(" ")
                        if type(name) == list:
                            name = name[-5]+" "+name[-4]+" "+name[-3]+" "+name[-2]+" "+name[-1]
                            if len(name) > 40:
                                name = name.split(" ")
                                name = name[-4] + " " + name[-3] + " " + name[-2] + " " + name[-1]
                        item_name_list.append(name)
                    #item_des_list.append(item_image_url_map[i]["short_description"])
                    if item_image_url_map[i]['images'] != None:
                        diction = json.loads(item_image_url_map[i]['images'])
                        image_list = []
                        for k, v in diction.items():
                            image_list.append([k, v])
                        item_image_list.append(image_list)
                    else:
                        image_list = [["https://2.bp.blogspot.com/-Ado6ei4W5YU/WY-RnHsRzzI/AAAAAAABoXA/p0AEw7GIMaUgK_-pyrwH4pwBbwGyKRaowCEwYBhgL/s1600/70533052.jpg", None]]
                        item_image_list.append(image_list)

                    if item_image_url_map[i]['url'] != None:
                        item_url_list.append(item_image_url_map[i]['url'])
                    else:
                        item_url_list.append("https://2.bp.blogspot.com/-Ado6ei4W5YU/WY-RnHsRzzI/AAAAAAABoXA/p0AEw7GIMaUgK_-pyrwH4pwBbwGyKRaowCEwYBhgL/s1600/70533052.jpg")

                print("item_name_list: ", item_name_list)
                # print("item_image_list: ", item_image_list)
                # print("item_des_list: ", item_des_list)

                # "recommendation list: "
                # Confirm_template = TemplateSendMessage(
                #     alt_text='confirm template請使用手機版line',
                #     template=ConfirmTemplate(
                #         title='ConfirmTemplate',
                #         text=k,
                #         actions=[
                #             MessageTemplateAction(
                #                 label='Hit',
                #                 text='H',
                #             ),
                #             MessageTemplateAction(
                #                 label='Reject',
                #                 text='R'
                #             )
                #         ]
                #     )
                # )
                col = []
                for i in range(len(item_name_list)):
                    col.append(CarouselColumn(
                                thumbnail_image_url=item_image_list[i][0][0],
                                title=item_name_list[i],
                                text=str(i+1),
                                actions=[
                                    MessageTemplateAction(
                                        label='Hit',
                                        text='H',
                                    ),
                                    MessageTemplateAction(
                                        label='Reject',
                                        text='R',
                                    ),
                                    URIAction(
                                        label='go to the website',
                                        uri=item_url_list[i]
                                    )
                                ]
                            ))
                Carousel_template = TemplateSendMessage(
                    alt_text='Carousel template請使用手機版line',
                    template=CarouselTemplate(
                        columns=col
                    )
                )
                line_bot_api.reply_message(event.reply_token, Carousel_template)

            elif k[0] == "R" or k[0] == "A":
                Confirm_template = TemplateSendMessage(
                    alt_text='confirm template請使用手機版line',
                    template=ConfirmTemplate(
                        title='ConfirmTemplate',
                        text=k+'\n'+"Do you want to restart the session?",
                        actions=[
                            MessageTemplateAction(
                                label='Sure',
                                text='S',
                            ),
                            MessageTemplateAction(
                                label='Quit',
                                text='Q'
                            )
                        ]
                    )
                )
                line_bot_api.reply_message(event.reply_token, Confirm_template)

if __name__ == "__main__":
    app.run()