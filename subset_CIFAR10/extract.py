#def unpickle(file):
import cPickle
fo = open("small_data_batch_5.mat", 'rb')
dict = cPickle.load(fo)
fo.close()