from core import *
image_height = 32
image_width = 32
num_channels = 3
num_classes = 10
data_path = '../CIFAR-10'
def create_reader(map_file, mean_file):

    transforms = [cntk.io.transforms.crop(crop_type='randomside', side_ratio=0.8)]
    transforms += [
        cntk.io.transforms.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear'),
        cntk.io.transforms.mean(mean_file)
    ]
    return cntk.io.MinibatchSource(cntk.io.ImageDeserializer(map_file, cntk.io.StreamDefs(
        features=cntk.io.StreamDef(field='image', transforms=transforms),
        # first column in map file is referred to as 'image'
        labels=cntk.io.StreamDef(field='label', shape=num_classes)  # and second as 'label'
    )))
reader_train = create_reader(os.path.join(data_path, 'train_map.txt'),
                             os.path.join(data_path, 'CIFAR-10_mean.xml'))
reader_test  = create_reader(os.path.join(data_path, 'test_map.txt'),
                             os.path.join(data_path, 'CIFAR-10_mean.xml'))

network_input = cntk.input_variable((num_channels, image_height, image_width))
network_label = cntk.input_variable(num_classes)
def mapping(reader):
    return  {
        network_input: reader.streams.features,
        network_label: reader.streams.labels
    }

def create_network(para):
    with cntk.layers.default_options(activation=cntk.relu, init=cntk.glorot_uniform()):
        model = cntk.layers.Sequential([
            cntk.layers.For(range(3), lambda i: [
                cntk.layers.Convolution((3, 3), para[i], pad=True),
                cntk.layers.BatchNormalization(map_rank=1),
                cntk.layers.Convolution((3, 3), para[i], pad=True),
                cntk.layers.BatchNormalization(map_rank=1),
                cntk.layers.MaxPooling((3, 3), strides=(2, 2)),
                cntk.layers.Dropout(0.5),
            ]),
            cntk.layers.For(range(2), lambda: [
                cntk.layers.Dense(1024),
                cntk.layers.Dropout(0.5),
            ]),
            cntk.layers.Dense(10, activation=None)
        ])
    z = model(network_input / 256.0)

    ce = cntk.cross_entropy_with_softmax(z, network_label)
    pe = cntk.classification_error(z, network_label)
    # training config
    epoch_size = 50000
    minibatch_size = 1024
    # Set training parameters
    lr_schedule = cntk.learning_rate_schedule(1, cntk.UnitType.minibatch)
    momentums = cntk.momentum_schedule(0.9, minibatch_size=minibatch_size)
    #l2_reg_weight = 0.001
    # trainer object
    learner = cntk.adam(z.parameters,lr=lr_schedule, momentum=momentums)
    #learner = cntk.momentum_sgd(z.parameters,lr=lr_schedule,momentum=momentums,l2_regularization_weight=l2_reg_weight)
    progress_printer = cntk.logging.ProgressPrinter(tag='Training', num_epochs=5000)
    trainer = cntk.Trainer(z, (ce, pe), [learner],[progress_printer])
    #log = cntk.logging.ProgressPrinter(7)
    for epoch in range(5000):
        sample_count = 0
        while sample_count < epoch_size:
            data = reader_train.next_minibatch(minibatch_size,input_map=mapping(reader_train))
            trainer.train_minibatch(data)
            sample_count += data[network_label].num_samples
            #log.update_with_trainer(trainer)
        trainer.summarize_training_progress()
        data = reader_test.next_minibatch(1000, input_map=mapping(reader_test))
        trainer.test_minibatch(data)
        trainer.summarize_test_progress()
        #data = reader_train.next_minibatch(5000, input_map=mapping(reader_train))
        #print (' Train accuracy = {}'.format((1 - trainer.test_minibatch(data)) * 100))
    return trainer
if __name__ == '__main__':
    #train_and_evaluate([64, 96, 128])
    print Node([32,64,96],create_network,reader_test,mapping,100,100,network_input)


