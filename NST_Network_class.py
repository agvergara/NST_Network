# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 10:18:59 2019

@author: Antonio Gomez Vergara
"""


import scipy.io
import scipy.misc
from .nst_utils import load_vgg_model, generate_noise_image, reshape_and_normalize_image, save_image
import tensorflow as tf

class NST_Network():
    
    def __init__(self, learning_rate=2.0, num_iterations=200):
        self.style_layers = [
            ('conv1_1', 0.2),
            ('conv2_1', 0.2),
            ('conv3_1', 0.2),
            ('conv4_1', 0.2),
            ('conv5_1', 0.2)]
        self.model = None
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.output_dir = "output/"
        
    @staticmethod   
    def load_normalize_content_img (content_img_path):
        
        content_img = scipy.misc.imread(content_img_path)
        content_img = reshape_and_normalize_image(content_img)
        return content_img
        
    @staticmethod   
    def load_normalize_style_img (style_img_path):
        
        style_img = scipy.misc.imread(style_img_path)
        style_img = reshape_and_normalize_image(style_img)
        return style_img
        
    @staticmethod   
    def generate_noisy_img (content_img):
        
        generated_noisy_img = generate_noise_image(content_img)
        return generated_noisy_img
        
    
    def load_model (self, model_path):
        
        self.model = load_vgg_model(model_path)
        
        
    def choose_optimizer (self, optimizer_select="Adam"):
        
        optimizer = None
        if (optimizer_select == "Adam"):
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
        else:
            optimizer = "Optimizer not recognised"
        return optimizer
    
    
    # Functions to compute Style/Content cost, Total cost and the network.
    @staticmethod   
    def compute_content_cost(aC, aG):
        
        m, nH, nW, nC = aG.get_shape().as_list() 
        aC = tf.transpose(tf.reshape(aC, [nH * nW, nC]))
        aG = tf.transpose(tf.reshape(aG, [nH * nW, nC]))
        J_content = (1/(4 * nH * nW * nC)) * tf.reduce_sum(tf.square(tf.subtract(aC, aG)))
        return J_content
    
    @staticmethod   
    def gram_matrix(A):
        
        GA = tf.matmul(A,tf.transpose(A))
        return GA
    
    def compute_layer_style_cost(self, aS, aG):
        
        m, nH, nW, nC = aG.get_shape().as_list()
        aS = tf.transpose(tf.reshape(aS, [nH * nW, nC]))
        aG = tf.transpose(tf.reshape(aG, [nH * nW, nC]))
        GS = self.gram_matrix(aS)
        GG = self.gram_matrix(aG)     
        normalize_term = 1 / (4 * nC**2 * (nH * nW)**2)
        J_style_layer = normalize_term * (tf.reduce_sum(tf.square(tf.subtract(GS, GG))))
        return J_style_layer
    
    
    def compute_style_cost(self, model, style_layers, sess):
        
        J_style = 0
        for layer_name, coeff in style_layers:
            out = model[layer_name]
            aS = sess.run(out)
            aG = out
            J_style_layer = self.compute_layer_style_cost(aS, aG)
            J_style += coeff * J_style_layer
        return J_style
    
    @staticmethod   
    def total_cost(J_content, J_style, alpha=10, beta=40): 
        
        J = alpha * J_content + beta * J_style
        return J 
    
    def model_nst(self, input_image, optimizer, content_img, style_img, output_name, graph, sess, print_output=False):
        
        sess.run(self.model['input'].assign(content_img))
        out = self.model['conv4_2']
        aC = sess.run(out)
        aG = out
        J_content = self.compute_content_cost(aC, aG)
          
        sess.run(self.model['input'].assign(style_img))
        J_style = self.compute_style_cost(self.model, self.style_layers, sess)
         
        J = self.total_cost(J_content, J_style)
          
        train_step = optimizer.minimize(J)
        sess.run(tf.global_variables_initializer())
        sess.run(self.model['input'].assign(input_image))
            
        for i in range(self.num_iterations):
            sess.run(train_step)
            generated_image = sess.run(self.model['input'])
            if ((i%20 == 0) and (print_output)):
                Jt, Jc, Js = sess.run([J, J_content, J_style])
                print("Iteration " + str(i) + " :")
                print("Total cost = " + str(Jt))
                print("Content cost = " + str(Jc))
                print("Style cost = " + str(Js))  
                save_image(self.output_dir + output_name + '_' + str(i) + ".png", generated_image)   
        sess.close()
        path_generated_img = self.output_dir + output_name + '.png'
        save_image(path_generated_img, generated_image)      
        return generated_image, path_generated_img
