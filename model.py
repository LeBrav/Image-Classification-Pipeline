from models_code.shufflenetv2 import *
from distiller import *
from learning_rate_scheduler import *
from tensorflow.python.client import device_lib
from keras.models import load_model
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.layers import Dense, Input, Lambda
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten, Dense, Reshape, Conv2D, Dropout, GlobalAveragePooling2D,Input
from tensorflow.keras.layers import Dropout
import tensorflow.keras.optimizers as optimizers
from wandb.keras import WandbCallback
from keras.models import Model
from keras.callbacks import LearningRateScheduler
from models_code.resnet18 import ResNet18
from keras.callbacks import EarlyStopping


class TotalParamsCallback(WandbCallback):
    def __init__(self, total_params, *args, **kwargs):
        self.total_params = total_params
        self.params_logged = False
        super().__init__(*args, **kwargs)

    def on_epoch_end(self, epoch, logs={}):
        if not self.params_logged:
            logs["total_params"] = float("{:.2f}".format(self.total_params / 1000000))
            self.params_logged = True
        logs["val_accuracy/total_params"] = logs["val_accuracy"] * 100 / (self.total_params/1000000)

        super().on_epoch_end(epoch, logs)


def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]


def get_model(cfg, distilation):
    optimizer_cfg = cfg["optimizer"]
    loss_name = cfg["loss"]

    # create model
    model, total_params = build_model(cfg)
    # optimizer
    optimizer = getattr(optimizers, optimizer_cfg["name"])(**optimizer_cfg["settings"])
    if not distilation:
        # compile with loss function
        model.compile(loss=loss_name, optimizer=optimizer, metrics=['accuracy'])
    else:
        model.compile(optimizer=optimizer,
                      loss=lambda y_true, y_pred: distillation_loss(y_true, y_pred,
                      temperature=cfg["teacher_model"]["temperature"],
                      alpha=cfg["teacher_model"]["alpha"]),
                      metrics=['accuracy'])

    print(model.summary())

    return model, total_params


def build_model(cfg):
    if cfg["model"]["name"] == "shufflenet":
        model = ShuffleNetV2(include_top=True,
                             input_tensor=None,
                             scale_factor=cfg["model"]["scale_factor"],
                             pooling='avg',
                             input_shape=(cfg["dataset"]["input_shape"], cfg["dataset"]["input_shape"], 3),
                             load_model=False,
                             num_shuffle_units=[cfg["model"]["shuffle_unit1"],cfg["model"]["shuffle_unit2"],cfg["model"]["shuffle_unit3"]],
                             bottleneck_ratio=cfg["model"]["bottleneck_ratio"],
                             dropout=cfg["model"]["dropout"],
                             classes=8)

    elif cfg["model"]["name"] == "resnet":
        model = ResNet18(num_classes=8, n_blocks=cfg["model"]["n_blocks"], channels=cfg["model"]["channels"],
                         dropout_rate=cfg["model"]["dropout"])
        model.build(input_shape=(None, cfg["dataset"]["input_shape"], cfg["dataset"]["input_shape"], 3))

    elif cfg["model"]["name"] == "resnet50":
        chosen_output_str = cfg["model"]["last_layer"]
        model = ResNet50(weights=cfg["model"]["weights"])
        # # Iterate over all layers in the model
        # for layer in model.layers:
        #     # Check if the layer is a convolutional layer
        #     if 'conv' in layer.name:
        #         # Add dropout after the layer
        #         x = Dropout(cfg["model"]["dropout"])(layer.output)
        #         # Update the layer with the dropout

        x = model.get_layer(chosen_output_str).output  # destroy stages
        x = GlobalAveragePooling2D()(x)
        x = Dense(cfg["model"]["n_classes"], activation='softmax', name='predictions')(x)
        model = Model(inputs=model.input, outputs=x)

    total_params = model.count_params()

    return model, total_params


def train(cfg, model, total_params, train_generator, validation_generator, project_name, distilation):
    if distilation:
        teacher_model = load_model(cfg["teacher_model"]["model_path"])
        teacher_model.load_weights(cfg["teacher_model"]["weights_path"])
        print(teacher_model.summary())
        # Freeze the layers of the teacher model
        for layer in teacher_model.layers:
            layer.trainable = False

    n_epochs = cfg["n_epochs"]
    if "early_stopping" in cfg:
        early_stopping = EarlyStopping(monitor=cfg["early_stopping"]["monitor"],
                                       patience=cfg["early_stopping"]["patience"],
                                       min_delta=cfg["early_stopping"]["min_delta"],
                                       mode=cfg["early_stopping"]["mode"],
                                       restore_best_weights=cfg["early_stopping"]["restore_best_weights"])

    if "warmup_steps" in cfg["optimizer"]:
        lr_scheduler = LearningRateScheduler(lambda epoch: warmup_cosine_decay(epoch,
                                                                           cfg["optimizer"]["settings"]["learning_rate"],
                                                                           cfg,
                                                                           len(train_generator)))

    print(get_available_devices())

    wandb_callback = TotalParamsCallback(total_params, save_model=False, save_weights_only=False)
    callbacks = ([wandb_callback] if project_name is not None else [])+\
                ([lr_scheduler] if "warmup_steps" in cfg["optimizer"] and lr_scheduler is not None else [])+\
                ([early_stopping] if "early_stopping" in cfg and early_stopping is not None else [])

    history = model.fit(train_generator,
                        steps_per_epoch=(len(train_generator)),
                        epochs=n_epochs,
                        validation_data=validation_generator,
                        validation_steps=(len(validation_generator)),
                        callbacks=callbacks)

    if "model_path" in cfg.keys():
        model_path = cfg["model_path"]
        print('Saving the model into ' + model_path + ' \n')
        try:
            model.save(model_path)
        except Exception as e:
            print(f"Failed to save the model: {e}. Saving the weights only...")
            model_path = model_path.replace(".h5", "_onlyweights.h5")
            model.save_weights(model_path)

        print('Done!\n')


def freeze_layers(model):
    # Freeze the feature extractor
    for layer in model.layers[:-1]:
        layer.trainable = False
    model.layers[-1].trainable = True
    return model
