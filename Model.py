class Model():
    def __init__(self):
        super(Model,self).__init__()
        self.Input_Size = (299,299,1)
        self.optimize = tf.keras.optimizers.Adam()
        self.earlystop = EarlyStopping(monitor='val_loss',patience=10)
        self.savebest = ModelCheckpoint('./model/Inception_small.h5',monitor='val_loss',save_best_only=True)

    def conv2d_bn(self,x,filter,kernel,stride=(1,1),padding='same'):
        x = layers.Conv2D(filters=filter,kernel_size=kernel,strides=stride,padding=padding)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('elu')(x)
        return x

    def Stem(self,x):
        x = self.conv2d_bn(x,32,3,2,padding='valid')
        x = self.conv2d_bn(x,32,3,padding='valid')
        x = self.conv2d_bn(x,64,3)

        branch1 = layers.MaxPooling2D(pool_size=(3,3),strides=2,padding='valid')(x)
        branch2 = self.conv2d_bn(x,96,3,2,padding='valid')
        x = layers.concatenate([branch1,branch2],axis=3)

        branch1 = self.conv2d_bn(x,64,1)
        branch1 = self.conv2d_bn(branch1,96,3,padding='valid')
        branch2 = self.conv2d_bn(x,64,1)
        branch2 = self.conv2d_bn(branch2,64,(7,1))
        branch2 = self.conv2d_bn(branch2,64,(1,7))
        branch2 = self.conv2d_bn(branch2,96,3,padding='valid')
        x = layers.concatenate([branch1,branch2],axis=3)

        branch1 = self.conv2d_bn(x,192,3,2,padding='valid')
        branch2 = layers.MaxPooling2D(padding='valid')(x)
        x = layers.concatenate([branch1,branch2],axis=3)

        return x


    def Inception_a(self,x):
        branch1 = layers.AveragePooling2D(pool_size=(3,3),strides=(1,1),padding='same')(x)
        branch1 = self.conv2d_bn(branch1,96,1)

        branch2 = self.conv2d_bn(x,96,1)

        branch3 = self.conv2d_bn(x,64,1)
        branch3 = self.conv2d_bn(branch3,96,3)

        branch4 = self.conv2d_bn(x,64,1)
        branch4 = self.conv2d_bn(branch4,96,3)
        branch4 = self.conv2d_bn(branch4,96,3)

        x = layers.concatenate([branch1,branch2,branch3,branch4],axis=3)

        return x

    def reduction_a(self,x):
        branch1 = self.conv2d_bn(x,384,3,2,padding='valid')

        branch2 = self.conv2d_bn(x,192,1)
        branch2 = self.conv2d_bn(branch2,224,3)
        branch2 = self.conv2d_bn(branch2,256,3,2,padding='valid')

        branch3 = layers.MaxPool2D(pool_size=(3,3),strides=(2,2),padding='valid')(x)

        x = layers.concatenate([branch1,branch2,branch3],axis=3)
        return x

    def Inception_b(self,x):
        branch1 = layers.AveragePooling2D(pool_size=(3,3),strides=(1,1),padding='same')(x)
        branch1 = self.conv2d_bn(branch1,128,(1,1))

        branch2 = self.conv2d_bn(x,384,(1,1))

        branch3 = self.conv2d_bn(x,192,(1,1))
        branch3 = self.conv2d_bn(branch3,224,(7,1))
        branch3 = self.conv2d_bn(branch3,256,(1,7))

        branch4 = self.conv2d_bn(x,192,(1,1))
        branch4 = self.conv2d_bn(branch4,192,(1,7))
        branch4 = self.conv2d_bn(branch4,224,(7,1))
        branch4 = self.conv2d_bn(branch4,224,(1,7))
        branch4 = self.conv2d_bn(branch4,256,(7,1))

        x = layers.concatenate([branch1,branch2,branch3,branch4],axis=3)
        return x

    def reduction_b(self,x):
        branch1 = self.conv2d_bn(x,192,1)
        branch1 = self.conv2d_bn(branch1,192,3,2,padding='valid')

        branch2 = self.conv2d_bn(x,256,1)
        branch2 = self.conv2d_bn(branch2,256,(1,7))
        branch2 = self.conv2d_bn(branch2,320,(7,1))
        branch2 = self.conv2d_bn(branch2,320,3,2,padding='valid')

        x = layers.concatenate([branch1,branch2],axis=3)
        return x

    def Inception_c(self,x):
        branch1 = layers.AveragePooling2D(pool_size=(3,3),strides=1,padding='same')(x)

        branch2 = self.conv2d_bn(x,256,(1,1))

        branch3 = self.conv2d_bn(x,384,(1,1))
        branch31 = self.conv2d_bn(branch3,256,(1,3))
        branch32 = self.conv2d_bn(branch3,256,(3,1))
        branch3 = layers.concatenate([branch31,branch32],axis=3)

        branch4 = self.conv2d_bn(x,384,1)
        branch4 = self.conv2d_bn(branch4,448,(1,3))
        branch4 = self.conv2d_bn(branch4,512,(3,1))
        branch41 = self.conv2d_bn(branch4,256,(3,1))
        branch42 = self.conv2d_bn(branch4,256,(1,3))
        branch4 = layers.concatenate([branch41,branch42],axis=3)

        x = layers.concatenate([branch1,branch2,branch3,branch4],axis=3)
        return x

    def Net(self):
        img = layers.Input(self.Input_Size)

        x = self.Stem(img)

        x = self.Inception_a(x)

        x = self.reduction_a(x)

        x = self.Inception_b(x)

        x = self.reduction_b(x)

        x = self.Inception_c(x)

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.8)(x)
        x = layers.Dense(3,activation='softmax')(x)

        model = tf.keras.models.Model(img,x)
        model.summary()
        return model

    def Train(self,train_gen,test_gen,class_weights):

        model = self.Net()
        model.compile(optimizer=self.optimize,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit_generator(train_gen, steps_per_epoch=500, epochs=50,
                            validation_data=test_gen, validation_steps=50,
                            callbacks=[TensorBoard(log_dir='./log'),self.savebest,self.earlystop],
                            class_weight=class_weights)
        return model
