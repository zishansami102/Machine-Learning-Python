import tensorflow as tf
import time
import gzip
import numpy as np


NUM_FILT1 = 8
NUM_FILT2 = 8
FILT_SIZE = 5

IMG_DEPTH = 1
IMG_WIDTH = 28
LEARNING_RATE = 0.01

N_CLASSES = 10
BATCH_SIZE = 20
N_EPOCHS = 7


x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')


##############################################################################################################################
################################################## ------ START HERE --------  ###############################################
########################################### ----------  NEURAL NETWORK ---------------  ######################################
######## ----ARCHITECTURE PROPOSED : [INPUT - HIDDEN1 - RELU - HIDDEN2 - RELU - HIDDEN3 - RELU - OUTPUT]---- #################
##############################################################################################################################


def ConvNet(x):

	weights = {'conv1':tf.Variable(tf.random_normal([FILT_SIZE,FILT_SIZE, IMG_DEPTH,NUM_FILT1])),
			   'conv2':tf.Variable(tf.random_normal([FILT_SIZE,FILT_SIZE, NUM_FILT1,NUM_FILT2])),
			   'fc':tf.Variable(tf.random_normal([IMG_WIDTH/2*IMG_WIDTH/2*NUM_FILT2, N_CLASSES]))}
			   
	biases = {'conv1':tf.Variable(tf.random_normal([NUM_FILT1])),
			  'conv2':tf.Variable(tf.random_normal([NUM_FILT2])),
			  'fc':tf.Variable(tf.random_normal([N_CLASSES]))}

	x = tf.reshape(x, shape=[-1, IMG_WIDTH, IMG_WIDTH, IMG_DEPTH])
	conv1 = tf.nn.conv2d(x, weights['conv1'], strides=[1,1,1,1], padding='SAME')+biases['conv1']
	conv1 = tf.nn.relu(conv1)

	conv2 = tf.nn.conv2d(conv1, weights['conv2'], strides=[1,1,1,1], padding='SAME')+biases['conv2']
	conv2 = tf.nn.relu(conv2)

	pooled_layer = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

	fc = tf.reshape(pooled_layer, shape=[-1, IMG_WIDTH/2*IMG_WIDTH/2*NUM_FILT2])

	output = tf.matmul(fc, weights['fc'])+biases['fc']
	return output

def train_nn(x):
	print(x.shape)
	prediction = ConvNet(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
	batch_loop = int(NUM_IMAGES/BATCH_SIZE)
 	#default learning rate = 0.001
	optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(N_EPOCHS):
			epoch_loss = 0
			for i in range(batch_loop):
				stime = time.time()
				epoch_x, epoch_y = train_data['X'][i*BATCH_SIZE:(i+1)*BATCH_SIZE,:], train_data['y'][i*BATCH_SIZE:(i+1)*BATCH_SIZE,:]
				_, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
				epoch_loss += c
				if (i+1)%50==0:
					per = float(i+1)/int(NUM_IMAGES/BATCH_SIZE)*100
					print("Epoch:"+str(round(per,2))+"% Of "+str(epoch+1)+"/"+str(N_EPOCHS)+", Batch loss:"+str(c))

					ftime = time.time()
					deltime = ftime-stime
					remtime = (batch_loop-i-1)*deltime+deltime*batch_loop*(N_EPOCHS-epoch-1)
					hrs = int(remtime)/3600
					mins = int((remtime/60-hrs*60))
					secs = int(remtime-mins*60-hrs*3600)
					print(str(int(deltime))+"secs/batch : ########  "+str(hrs)+"Hrs "+str(mins)+"Mins "+str(secs)+"Secs remaining  ########")

			print('Epoch', (epoch+1), 'completed out of', N_EPOCHS,'loss:',epoch_loss)
		pred_classes = tf.argmax(prediction,1)
		true_classes = tf.argmax(y,1)

		correct = tf.equal(pred_classes, true_classes)

		accuray = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy:',accuray.eval({x:test_data['X'], y:test_data['y']}))



def extract_data(filename, num_images, IMAGE_WIDTH):
	"""Extract the images into a 4D tensor [image index, y, x, channels].
	Values are rescaled from [0, 255] down to [-0.5, 0.5].
	"""
	print('Extracting', filename)
	with gzip.open(filename) as bytestream:
		bytestream.read(16)
		buf = bytestream.read(IMAGE_WIDTH * IMAGE_WIDTH * num_images)
		data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
		data = data.reshape(num_images, IMAGE_WIDTH*IMAGE_WIDTH)
		return data

def extract_labels(filename, num_images):
	"""Extract the labels into a vector of int64 label IDs."""
	print('Extracting', filename)
	with gzip.open(filename) as bytestream:
		bytestream.read(8)
		buf = bytestream.read(1 * num_images)
		labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
	return labels


## Data extracting
m = 10000
X = extract_data('t10k-images-idx3-ubyte.gz', m, IMG_WIDTH)
y_dash = extract_labels('t10k-labels-idx1-ubyte.gz', m).reshape(m,1)
X-= int(np.mean(X))
X/= int(np.std(X))
test_data_dash = np.hstack((X,y_dash))
np.random.shuffle(test_data_dash)

y_dash = np.zeros((m,N_CLASSES))
for i in range(0,m):
	y_dash[i,int(test_data_dash[i,-1])]=1

test_data = {'X':test_data_dash[:,:-1], 'y':y_dash }


m =50000
X = extract_data('train-images-idx3-ubyte.gz', m, IMG_WIDTH)
y_dash = extract_labels('train-labels-idx1-ubyte.gz', m).reshape(m,1)
X-= int(np.mean(X))
X/= int(np.std(X))
train_data_dash = np.hstack((X,y_dash))
np.random.shuffle(train_data_dash)

NUM_IMAGES = train_data_dash.shape[0]

y_dash = np.zeros((NUM_IMAGES,N_CLASSES))
for i in range(0,NUM_IMAGES):
	y_dash[i,int(train_data_dash[i,-1])]=1

train_data = {'X':train_data_dash[:,:-1], 'y':y_dash }

print("Data preprocessing completed.")
print("Training starts here....")

train_nn(x)
