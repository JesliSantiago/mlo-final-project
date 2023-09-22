from model2 import *
import gradio as gr

_model = poem_classifier_model()

epochs = 50
lr = 0.001

_model.load_data()
_model.preprocess()
_model.train(epochs=epochs, lr=lr)

def extract_genre(verse):
    return _model.classify_poem(verse)

# Gradio interface
interface = gr.Interface(fn=extract_genre, 
                         inputs="text", 
                         outputs="text",
                         live=True)
interface.launch()