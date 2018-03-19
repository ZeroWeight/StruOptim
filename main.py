#import sklearn.model_selection
import cntk
from core import *
'''
arr = []
with open('../MNIST/Train-28x28_cntk_text.txt') as f:
    for line in f:
        arr.append(line)
train,valid = sklearn.model_selection.train_test_split(arr,test_size=1.0/6.0)

with open('../MNIST/Train_Split-28x28_cntk_text.txt','w') as f:
    for line in train:
        f.write(line)
with open('../MNIST/Valid_Split-28x28_cntk_text.txt','w') as f:
    for line in valid:
        f.write(line)
'''

def create_reader(path):
    label_stream = cntk.io.StreamDef(field='labels', shape=10, is_sparse=False)
    feature_stream = cntk.io.StreamDef(field='features', shape=28 * 28, is_sparse=False)
    deserializer = cntk.io.CTFDeserializer(path,cntk.io.StreamDefs(
        labels=label_stream, features=feature_stream))
    return cntk.io.MinibatchSource(deserializer, randomize=True, max_sweeps=cntk.io.INFINITELY_REPEAT)

train_reader = create_reader('../MNIST/Train_Split-28x28_cntk_text.txt')
valid_reader = create_reader('../MNIST/Valid_Split-28x28_cntk_text.txt')
test_reader = create_reader('../MNIST/Test-28x28_cntk_text.txt')

network_input = cntk.input_variable((1, 28, 28))
network_label = cntk.input_variable(10)

def mapping(reader):
    return {
        network_label : reader.streams.labels,
        network_input : reader.streams.features
    }


def create_network(para, verbose=False):
    with cntk.layers.default_options(init=cntk.glorot_uniform(), activation=cntk.ops.relu):
        # In order to accelerate the debugging step, we choose a simple structure with only 2 parameters
        h = cntk.layers.Convolution2D(filter_shape=(3, 3), num_filters=para[0],
                                      strides=(1, 1), pad=True, name='C1')(network_input / 255.0)
        h = cntk.layers.layers.MaxPooling(filter_shape=(3, 3), strides=(2, 2), )(h)

        h = cntk.layers.Convolution2D(filter_shape=(3, 3), num_filters=para[1],
                                      strides=(1, 1), pad=True, name='C2')(h)
        h = cntk.layers.layers.MaxPooling(filter_shape=(3, 3), strides=(2, 2))(h)

        h = cntk.layers.Convolution2D(filter_shape=(3, 3), num_filters=para[2],
                                      strides=(1, 1), pad=True, name='C3')(h)
        h = cntk.layers.layers.MaxPooling(filter_shape=(3, 3), strides=(2, 2))(h)

        z = cntk.layers.Dense(10, activation=None, name='R')(h)
        loss = cntk.cross_entropy_with_softmax(z, network_label)
    label_error = cntk.classification_error(z, network_label)
    lr_schedule = cntk.learning_rate_schedule(0.1, cntk.UnitType.minibatch)
    learner = cntk.momentum_sgd(z.parameters, lr_schedule, cntk.momentum_schedule(0.9))
    trainer = cntk.Trainer(z, (loss, label_error), [learner])
    if verbose: log = cntk.logging.ProgressPrinter(100)
    for _ in xrange(500):
        data = train_reader.next_minibatch(100, input_map=mapping(train_reader))
        trainer.train_minibatch(data)
        if verbose: log.update_with_trainer(trainer)
    return trainer


if __name__ == '__main__':
    optimizer = StruOptim(create_network,valid_reader,mapping,100,100,network_input,[2,2,2],[130,130,130])
    optimizer.start_optim(init_samples = 100)