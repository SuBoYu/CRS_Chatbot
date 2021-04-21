from __future__ import unicode_literals
import os
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import *
import configparser

import chat_model

import random

app = Flask(__name__)

# LINE 聊天機器人的基本資料
# 每個python file都要放下面幾行
config = configparser.ConfigParser()
config.read('config.ini')

line_bot_api = LineBotApi(config.get('line-bot', 'channel_access_token'))
handler = WebhookHandler(config.get('line-bot', 'channel_secret'))

chatmodel_dict = dict()

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
        if user_res == "S": # Model construct and First Conversation

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
            chatmodel_dict[event.source.user_id] = chat_model.chat_model('29816', '5127')
            print("chatmodel_dict", chatmodel_dict)

            chatmodel = chatmodel_dict[event.source.user_id]
            print("first conversation")
            k = chatmodel.first_conversation(user_res)
            if k[0] == "D": # ask entity
                # "Do you like" + str(self.agent_utterance.data['facet']) + "?"
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