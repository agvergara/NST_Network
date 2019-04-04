# Neural Style Transfer Network

Created during the Week 4 of the course: "Convolutional Neural Networks" (Coursera)

* nst_utils.py: Provided in the course, only a litle bit of modifications were done. 

* NST_Network_class.py: Full class of the network, it allows the user to load a pretrained model and change the hyperparameters easily.

The model used is the VGG-19,  you can find it in __[Kaggle](https://www.kaggle.com/teksab/imagenetvggverydeep19mat/version/1)__ or in __[MatConvNet](http://www.vlfeat.org/matconvnet/pretrained/)__

Why Neural Style Transfer? I think this is a fun application of convolutional neural networks and why not?

Example of use:
```python
content_img = "[HERE YOUR PATH]" + content_img_name
style_img = "[HERE YOUR PATH]" + content_img_name
model_path = "[PATH TO MODEL]"

nst_net = NST_Network()
with tf.Graph().as_default() as g:
	sess = tf.InteractiveSession(graph=g)
	nst_net.load_model(model_path)
	content_img = nst_net.load_normalize_content_img(content_img)
	style_img = nst_net.load_normalize_content_img(style_img)
	input_img = nst_net.generate_noisy_img(content_img)
	optimizer = nst_net.choos_optimizer("Adam")
	generated_img, path_generated_img = nst_net.model_nst(input_img, optimizer, content_img, "output_img", g, sess)
	sess.close()
```

The network was done using CPU (my GPU is worse)

## Network hyperparameters:

The parameters/hyperparameters are fully customizable for tunnint purposes. 

*__Learning rate__ : 2.0
*__Iterations__ : 200
*__Activation layer to evaluate__: conv4_1
