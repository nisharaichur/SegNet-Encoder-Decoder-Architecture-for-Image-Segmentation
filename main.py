if __name__ == '__main__':
    main()

def main():
	with open("/content/drive/My Drive/SegNet/config.json") as f:
	      config = json.load(f)
	#load the weights of VGG16(transfer learning)
	vgg_weights = np.load('/content/drive/My Drive/SegNet/vgg16.npy', encoding='latin1', allow_pickle=True)
	print("The VGG16 weights are loading for the transfer learning.....")
	vgg_weights = vgg_weights.item()
	use_vgg = True
	train_file = '/content/drive/My Drive/SegNet/'+config["TRAIN_FILE"]
	test_file = '/content/drive/My Drive/SegNet/'+config["TEST_FILE"]
	val_file = '/content/drive/My Drive/SegNet/'+config["VAL_FILE"]
	train_loss, train_accuracy = [], []
	val_loss, val_acc = [], []
	num_classes = config["NUM_CLASSES"]
	image_filename, label_filename = get_filename_list(train_file, config)     
	val_image_filename, val_label_filename = get_filename_list(val_file, config)
	print("The length of the training images")
	print(len(image_filename))
	print("The length of the validation images")
	print(len(val_image_filename))
	tr_images, tr_labels = generate_data(image_filename, label_filename)
	vl_images, vl_labels = generate_data(val_image_filename, val_label_filename)
	
	#convert the images to tensors
	tr_images =  tf.cast(tr_images, dtype=tf.float32)
	tr_labels =  tf.cast(tr_labels[:,:,:,0:1], dtype=tf.float32)
	vl_images =  tf.cast(vl_images, dtype=tf.float32)
	vl_labels =  tf.cast(vl_labels[:,:,:,0:1], dtype=tf.float32)
	
	#create an iterator for both training and validation data, which returns the data in batches
	train_dataset = tf.data.Dataset.from_tensor_slices((tr_images, tr_labels))
	train_dataset = train_dataset.shuffle(buffer_size=50).batch(3)
	iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
	train_init_op = iterator.make_initializer(train_dataset)
	validation_dataset = tf.data.Dataset.from_tensor_slices((vl_images, vl_labels))
	validation_dataset = validation_dataset.shuffle(buffer_size=50).batch(3)
	validation_init_op = iterator.make_initializer(validation_dataset)
	[features, labels] = iterator.get_next()
	
	#model architecture
	norm1 = tf.nn.lrn(features, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75, name='norm1')
	conv1_1 = conv_layer(norm1, "conv1_1", [3, 3, 3, 64], vgg_weights, use_vgg )
	conv1_2 = conv_layer(conv1_1, "conv1_2", [3, 3, 64, 64], vgg_weights, use_vgg)
	pool1, pool1_index, shape_1 = max_pool(conv1_2, "pool1")

	conv2_1 = conv_layer(pool1, "conv2_1", [3, 3, 64, 128], vgg_weights, use_vgg)
	conv2_2 = conv_layer(conv2_1,"conv2_2", [3, 3, 128, 128], vgg_weights, use_vgg)
	pool2, pool2_index, shape_2 = max_pool(conv2_2, "pool2")


	conv3_1 = conv_layer(pool2, "conv3_1", [3, 3, 128, 256], vgg_weights, use_vgg)
	conv3_2 = conv_layer(conv3_1, "conv3_2", [3, 3, 256, 256], vgg_weights, use_vgg)
	conv3_3 = conv_layer(conv3_2, "conv3_3", [3, 3, 256, 256], vgg_weights, use_vgg)
	pool3, pool3_index, shape_3 = max_pool(conv3_3, "pool3")

	dropout1 = tf.layers.dropout(pool3, rate=(1 - 0.5), training=True, name="dropout1")

	conv4_1 = conv_layer(dropout1, "conv4_1", [3, 3, 256, 512], vgg_weights, use_vgg)
	conv4_2 = conv_layer(conv4_1, "conv4_2", [3, 3, 512, 512], vgg_weights, use_vgg)
	conv4_3 = conv_layer(conv4_2, "conv4_3", [3, 3, 512, 512], vgg_weights, use_vgg)
	pool4, pool4_index, shape_4 = max_pool(conv4_3, "pool4")


	dropout2 = tf.layers.dropout(pool4, rate=(1 - 0.5), training=True, name="dropout2")

	conv5_1 = conv_layer(dropout2, "conv5_1", [3, 3, 512, 512], vgg_weights, use_vgg)
	conv5_2 = conv_layer(conv5_1, "conv5_2", [3, 3, 512, 512], vgg_weights, use_vgg)
	conv5_3 = conv_layer(conv5_2, "conv5_3", [3, 3, 512, 512], vgg_weights, use_vgg)
	pool5, pool5_index, shape_5 = max_pool(conv5_3, "pool5")

	dropout3 = tf.layers.dropout(pool5, rate=(1 - 0.5), training=True, name="dropout3")

	deconv5_1 = up_sampling(dropout3, pool5_index, shape_5, 3, "unpool_5")
	deconv5_2 = conv_layer(deconv5_1, "deconv5_2", [3, 3, 512, 512])
	deconv5_3 = conv_layer(deconv5_2, "deconv5_3", [3, 3, 512, 512])
	deconv5_4 = conv_layer(deconv5_3, "deconv5_4", [3, 3, 512, 512])

	dropout4 = tf.layers.dropout(deconv5_4, rate=(1 - 0.5), training=True, name="dropout4")

	deconv4_1 = up_sampling(dropout4, pool4_index, shape_4, 3, "unpool_4")
	deconv4_2 = conv_layer(deconv4_1, "deconv4_2", [3, 3, 512, 512])
	deconv4_3 = conv_layer(deconv4_2, "deconv4_3", [3, 3, 512, 512])
	deconv4_4 = conv_layer(deconv4_3, "deconv4_4", [3, 3, 512, 256])

	dropout5 = tf.layers.dropout(deconv4_4, rate=(1 - 0.5), training=True, name="dropout5")

	deconv3_1 = up_sampling(dropout5, pool3_index, shape_3, 3, "unpool_3")
	deconv3_2 = conv_layer(deconv3_1, "deconv3_2", [3, 3, 256, 256])
	deconv3_3 = conv_layer(deconv3_2, "deconv3_3", [3, 3, 256, 256])
	deconv3_4 = conv_layer(deconv3_3, "deconv3_4", [3, 3, 256, 128])


	dropout6 = tf.layers.dropout(deconv3_4, rate=(1 - 0.5), training=True, name="dropout6")

	deconv2_1 = up_sampling(dropout6, pool2_index, shape_2, 3, "unpool_2")
	deconv2_2 = conv_layer(deconv2_1, "deconv2_2", [3, 3, 128, 128])
	deconv2_3 = conv_layer(deconv2_2, "deconv2_3", [3, 3, 128, 64])

	#deconvolve block 1
	deconv1_1 = up_sampling(deconv2_3, pool1_index, shape_1, 3, "unpool_1")
	deconv1_2 = conv_layer(deconv1_1, "deconv1_2", [3, 3, 64, 64])
	deconv1_3 = conv_layer(deconv1_2, "deconv1_3", [3, 3, 64, 64])


	with tf.variable_scope('conv_classifier') as scope: # size of the kernel is the hight width inchannel and outchannel
	  kernel = tf.get_variable('weights', initializer=initialization(1, 64), shape=[1, 1, 64, num_classes])
	  conv = tf.nn.conv2d(deconv1_3, kernel, [1, 1, 1, 1], padding='SAME')
	  biases = tf.get_variable('biases', initializer=tf.constant_initializer(0.0), shape=[num_classes])
	  logits = tf.nn.bias_add(conv, biases, name=scope.name)


	loss, accuracy, prediction = cal_loss(logits=logits, labels=labels, num_class=num_classes)
	train, global_step, grads = train_op(total_loss=loss)
	#Training the model
	with tf.Session() as sess:
	  sess.run(tf.local_variables_initializer())
	  sess.run(tf.global_variables_initializer())
	  for step in range(9000):
	      sess.run(train_init_op)
	      _, tr_loss, tr_accuracy = sess.run([train, loss, accuracy])
	      train_loss.append(tr_loss)
	      train_accuracy.append(tr_accuracy)
	      print("Iteration {}: Train Loss{:6.3f}, Train Accu {:6.3f}".format(step, train_loss[-1], train_accuracy[-1]))
	      if step%100 == 0:
		print("....validating....")
		_val_loss = []
		_val_acc = []
		#initializing the iterator with validation data
		sess.run(validation_init_op)
		hist = np.zeros((num_classes, num_classes))
		#validation data length is 101,  50 images in each iterations
		for test_step in range(int(19)):
		  _loss, _accuracy, val_pred, val_lab = sess.run([loss, accuracy, logits, labels])
		  _val_loss.append(_loss)
		  _val_acc.append(_accuracy)
		  hist += get_hist(val_pred, val_lab)
		print_hist_summary(hist)
		val_loss.append(np.mean(_val_loss))
		val_acc.append(np.mean(_val_acc))
		print("Iteration {}: Train Loss {:6.3f}, Train Acc {:6.3f}, Val Loss {:6.3f}, Val Acc {:6.3f}".format(step, train_loss[-1], train_accuracy[-1], val_loss[-1], val_acc[-1]))






