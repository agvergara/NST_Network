# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 17:48:04 2019

@author: Antonio Gomez Vergara
"""

# I know it looks awful, but it was for testing
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from flask import Flask, request, redirect, render_template
from NST_Network_class import NSTNetwork
import tensorflow.compat.v1 as tf
import time
import os

# I was using TensorFlow 1
tf.disable_v2_behavior()

app = Flask(__name__)

app.config["IMAGE_UPLOADS"] = "./imgs"
app.config["IMAGE_OUTPUT"] = "./static/output/"
app.config["MODEL_PATH"] = os.path.join("./model", "imagenet-vgg-verydeep-19.mat")

if not os.path.isdir(app.config["IMAGE_UPLOADS"]):
    os.mkdir(app.config["IMAGE_UPLOADS"], 777)

if not os.path.isdir(app.config["IMAGE_OUTPUT"]):
    os.mkdir(app.config["IMAGE_OUTPUT"], 777)


def make_output(model_path, content_img, style_img):
    nst_net = NSTNetwork(app.config["IMAGE_OUTPUT"])

    with tf.Graph().as_default() as g:
        sess = tf.InteractiveSession(graph=g)
        nst_net.load_model(model_path)
        content_img = nst_net.load_normalize_content_img(content_img)
        style_img = nst_net.load_normalize_content_img(style_img)
        input_img = nst_net.generate_noisy_img(content_img)
        optimizer = nst_net.choose_optimizer("Adam")
        generated_img, path_generated_img = nst_net.model_nst(input_img, optimizer, content_img, style_img, "output_img", g, sess)
        sess.close()
    return path_generated_img, generated_img


@app.route('/', methods=["GET", "POST"])
def upload_img():
    path_out_img = None

    if request.method == "POST":
        if request.files:
            image_content = request.files["image_content"]
            image_style = request.files["image_style"]

            path_content = os.path.join(app.config["IMAGE_UPLOADS"], image_content.filename)
            path_style = os.path.join(app.config["IMAGE_UPLOADS"], image_style.filename)

            image_content.save(path_content)
            image_style.save(path_style)

            start_time = time.time()
            path_out_img, out_img = make_output(app.config["MODEL_PATH"], path_content, path_style)
            print("Image generated in {} secs".format(time.time() - start_time))
    return render_template("upload_img.html", output_path=path_out_img)


@app.route('/delete')
def delete_input_imgs():
    for filename in os.listdir(app.config["IMAGE_UPLOADS"]):
        os.remove(os.path.join(app.config["IMAGE_UPLOADS"], filename))
    return redirect('/')


if __name__ == "__main__":
    warnings.simplefilter(action="ignore", category=FutureWarning)
    app.run()
