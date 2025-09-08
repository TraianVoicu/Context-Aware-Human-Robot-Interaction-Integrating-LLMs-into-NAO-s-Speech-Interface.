# -*- coding: utf-8 -*-
from naoqi import ALProxy
import time

ROBOT_IP = "192.168.1.104"  
PORT = 9559

tts = ALProxy("ALTextToSpeech", ROBOT_IP, PORT)

asr = ALProxy("ALSpeechRecognition", ROBOT_IP, PORT)
memory = ALProxy("ALMemory", ROBOT_IP, PORT)

vocab = ["hello", "goodbye", "yes", "no", "NAO"]
asr.setVocabulary(vocab, False)  

def on_word_recognized(value):
    if value and len(value) > 0:
        word = value[0]
        print("Heard:", word)

        with open("/home/nao/recognized_words.txt", "a") as f:
            f.write(word + "\n")

        tts.say("You said " + word)

subscriber = memory.subscriber("WordRecognized")
subscriber.signal.connect(on_word_recognized)

asr.subscribe("SimpleSTT")

print("Listening for words: ", vocab)
print("Press Ctrl+C to stop...")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Stopping...")
    asr.unsubscribe("SimpleSTT")
    print("Stopped.")
