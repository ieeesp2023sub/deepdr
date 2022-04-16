import sys
import train
import test

logpath = sys.argv[1]
window = sys.argv[2]
dimension = sys.argv[3]

print 'training...'
train.training(logpath,window,dimension)
print 'testing...'
test.training(logpath,window,dimension)


