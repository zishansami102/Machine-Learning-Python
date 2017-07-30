import tensorflow as tf
import time
import numpy as np


NUM_FILT1 = 16
NUM_FILT2 = 16
FILT_SIZE = 5

IMG_DEPTH = 3
IMG_WIDTH = 32
LEARNING_RATE = 0.01

N_CLASSES = 10
BATCH_SIZE = 8
N_EPOCHS = 7


x = tf.placeholder('float', [None, IMG_WIDTH*IMG_WIDTH*IMG_DEPTH])
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

			print("Epoch"+ str(epoch+1)+"completed out of "+str(N_EPOCHS)+" E.loss:"+str(epoch_loss))
		pred_classes = tf.argmax(prediction,1)
		true_classes = tf.argmax(y,1)

		correct = tf.equal(pred_classes, true_classes)

		accuray = tf.reduce_mean(tf.cast(correct, 'float'))
		bacc=0
		print("Computing Accuracy...")
		for xx in range(int(N_TSAMPLE/BATCH_SIZE)):
			bacc += accuray.eval({x:test_data['X'][xx*BATCH_SIZE:(xx+1)*BATCH_SIZE,:], y:test_data['y'][xx*BATCH_SIZE:(xx+1)*BATCH_SIZE,:]})
			if (xx+1)%50==0:
				per = float(xx+1)/int(N_TSAMPLE/BATCH_SIZE)*100
				print("Accuracy Computation running..  "+str(per)+"% Completed")
		acc = bacc/batch_loop
		print("Accuracy:"+str(acc*100)+"%")



## Get the data from the file
def unpickle(file):
	import cPickle
	with open(file, 'rb') as fo:
		dict = cPickle.load(fo)
	return dict



## Data extracting
m = 10000
data_dash = unpickle("test_batch")
X= data_dash['data']	# m * n
y_dash = np.array(data_dash['labels']).reshape((m,1))	# m * 1
X-= int(np.mean(X))
X/= int(np.std(X))
test_data_dash = np.hstack((X,y_dash))
np.random.shuffle(test_data_dash)


N_TSAMPLE = 10000

y_dash = np.zeros((N_TSAMPLE,N_CLASSES))
for i in range(0,N_TSAMPLE):
	y_dash[i,int(test_data_dash[i,-1])]=1

test_data = {'X':test_data_dash[:N_TSAMPLE,:-1], 'y':y_dash }


m =10000
data_dash = unpickle("data_batch_1")
X= data_dash['data']	# m * n
y_dash = np.array(data_dash['labels']).reshape((m,1))	# m * 1
X-= int(np.mean(X))
X/= int(np.std(X))
train_data_dash = np.hstack((X,y_dash))

for xx in range(2,6):
	data_dash = unpickle("data_batch_"+str(xx))
	X= data_dash['data']	# m * n
	y_dash = np.array(data_dash['labels']).reshape((m,1))	# m * 1
	X-= int(np.mean(X))
	X/= int(np.std(X))
	train_data_dash = np.vstack((train_data_dash, np.hstack((X,y_dash))))
	
np.random.shuffle(train_data_dash)

NUM_IMAGES = train_data_dash.shape[0]

y_dash = np.zeros((NUM_IMAGES,N_CLASSES))
for i in range(0,NUM_IMAGES):
	y_dash[i,int(train_data_dash[i,-1])]=1

train_data = {'X':train_data_dash[:,:-1], 'y':y_dash }

print("Data preprocessing completed.")
print("Training starts here....")

train_nn(x)
