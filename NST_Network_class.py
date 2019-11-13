# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 10:18:59 2019

@author: Antonio Gomez Vergara
"""


from matplotlib.pyplot import imread
from nst_utils import load_vgg_model, generate_noise_image, reshape_and_normalize_image, save_image
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


class NSTNetwork:

    def __init__(self, output_dir, learning_rate=2.0, num_iterations=200):
        self.style_layers = [
            ('conv1_1', 0.2),
            ('conv2_1', 0.2),
            ('conv3_1', 0.2),
            ('conv4_1', 0.2),
            ('conv5_1', 0.2)]
        self.model = None
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.output_dir = output_dir
        
    @staticmethod   
    def load_normalize_content_img(content_img_path):
        
        content_img = imread(content_img_path)
        content_img = reshape_and_normalize_image(content_img)
        return content_img
        
    @staticmethod   
    def load_normalize_style_img(style_img_path):
        
        style_img = imread(style_img_path)
        style_img = reshape_and_normalize_image(style_img)
        return style_img
        
    @staticmethod   
    def generate_noisy_img(content_img):
        
        generated_noisy_img = generate_noise_image(content_img)
        return generated_noisy_img

    def load_model(self, model_path):
        
        self.model = load_vgg_model(model_path)

    def choose_optimizer(self, optimizer_select="Adam"):
        
        if optimizer_select == "Adam":
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
        else:
            optimizer = "Optimizer not recognised"
        return optimizer

    # Functions to compute Style/Content cost, Total cost and the network.
    @staticmethod   
    def compute_content_cost(a_c, a_g):
        
        m, nh, nw, nc = a_g.get_shape().as_list()
        a_c = tf.transpose(tf.reshape(a_c, [nh * nw, nc]))
        a_g = tf.transpose(tf.reshape(a_g, [nh * nw, nc]))
        j_content = (1/(4 * nh * nw * nc)) * tf.reduce_sum(tf.square(tf.subtract(a_c, a_g)))
        return j_content
    
    @staticmethod   
    def gram_matrix(matrix):
        
        ga = tf.matmul(matrix, tf.transpose(matrix))
        return ga
    
    def compute_layer_style_cost(self, a_s, a_g):
        
        m, nh, nw, nc = a_g.get_shape().as_list()
        a_s = tf.transpose(tf.reshape(a_s, [nh * nw, nc]))
        a_g = tf.transpose(tf.reshape(a_g, [nh * nw, nc]))
        gs = self.gram_matrix(a_s)
        gg = self.gram_matrix(a_g)
        normalize_term = 1 / (4 * nc**2 * (nh * nw)**2)
        j_style_layer = normalize_term * (tf.reduce_sum(tf.square(tf.subtract(gs, gg))))
        return j_style_layer

    def compute_style_cost(self, model, style_layers, sess):
        
        j_style = 0
        for layer_name, coeff in style_layers:
            out = model[layer_name]
            a_s = sess.run(out)
            a_g = out
            j_style_layer = self.compute_layer_style_cost(a_s, a_g)
            j_style += coeff * j_style_layer
        return j_style
    
    @staticmethod   
    def total_cost(j_content, j_style, alpha=10, beta=40):
        
        j = alpha * j_content + beta * j_style
        return j
    
    def model_nst(self, input_image, optimizer, content_img, style_img, output_name, graph, sess, print_output=False):
        
        sess.run(self.model['input'].assign(content_img))
        out = self.model['conv4_2']
        a_c = sess.run(out)
        a_g = out
        j_content = self.compute_content_cost(a_c, a_g)
          
        sess.run(self.model['input'].assign(style_img))
        j_style = self.compute_style_cost(self.model, self.style_layers, sess)
         
        j = self.total_cost(j_content, j_style)
          
        train_step = optimizer.minimize(j)
        sess.run(tf.global_variables_initializer())
        sess.run(self.model['input'].assign(input_image))
            
        for i in range(self.num_iterations):
            sess.run(train_step)
            generated_image = sess.run(self.model['input'])
            if i % 20 == 0 and print_output:
                jt, jc, js = sess.run([j, j_content, j_style])
                print("Iteration " + str(i) + " :")
                print("Total cost = " + str(jt))
                print("Content cost = " + str(jc))
                print("Style cost = " + str(js))
                save_image(self.output_dir + output_name + '_' + str(i) + ".png", generated_image)   
        sess.close()
        path_generated_img = self.output_dir + output_name + '.png'
        save_image(path_generated_img, generated_image)      
        return generated_image, path_generated_img
