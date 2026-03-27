# Keras Support Coverage

Tracking exhaustive coverage of the `keras` python package API.


## Detailed API

| Object Name | Type | Signature |
|---|---|---|
| `DTypePolicy` | Class | `(name)` |
| `FloatDTypePolicy` | Class | `(...)` |
| `Function` | Class | `(inputs, outputs, name)` |
| `Initializer` | Class | `(...)` |
| `Input` | Function | `(shape, batch_size, dtype, sparse, ragged, batch_shape, name, tensor, optional)` |
| `InputSpec` | Class | `(dtype, shape, ndim, max_ndim, min_ndim, axes, allow_last_axis_squeeze, name, optional)` |
| `KerasTensor` | Class | `(shape, dtype, sparse, ragged, record_history, name, kwargs)` |
| `Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `Loss` | Class | `(name, reduction, dtype)` |
| `Metric` | Class | `(dtype, name)` |
| `Model` | Class | `(args, kwargs)` |
| `Operation` | Class | `(name)` |
| `Optimizer` | Class | `(...)` |
| `Quantizer` | Class | `(output_dtype)` |
| `Regularizer` | Class | `(...)` |
| `RematScope` | Class | `(mode, output_size_threshold, layer_names)` |
| `Sequential` | Class | `(layers, trainable, name)` |
| `StatelessScope` | Class | `(state_mapping, collect_losses, initialize_variables)` |
| `SymbolicScope` | Class | `(...)` |
| `Variable` | Class | `(...)` |
| `activations.celu` | Function | `(x, alpha)` |
| `activations.deserialize` | Function | `(config, custom_objects)` |
| `activations.elu` | Function | `(x, alpha)` |
| `activations.exponential` | Function | `(x)` |
| `activations.gelu` | Function | `(x, approximate)` |
| `activations.get` | Function | `(identifier)` |
| `activations.glu` | Function | `(x, axis)` |
| `activations.hard_shrink` | Function | `(x, threshold)` |
| `activations.hard_sigmoid` | Function | `(x)` |
| `activations.hard_silu` | Function | `(x)` |
| `activations.hard_swish` | Function | `(x)` |
| `activations.hard_tanh` | Function | `(x)` |
| `activations.leaky_relu` | Function | `(x, negative_slope)` |
| `activations.linear` | Function | `(x)` |
| `activations.log_sigmoid` | Function | `(x)` |
| `activations.log_softmax` | Function | `(x, axis)` |
| `activations.mish` | Function | `(x)` |
| `activations.relu` | Function | `(x, negative_slope, max_value, threshold)` |
| `activations.relu6` | Function | `(x)` |
| `activations.selu` | Function | `(x)` |
| `activations.serialize` | Function | `(activation)` |
| `activations.sigmoid` | Function | `(x)` |
| `activations.silu` | Function | `(x)` |
| `activations.soft_shrink` | Function | `(x, threshold)` |
| `activations.softmax` | Function | `(x, axis)` |
| `activations.softplus` | Function | `(x)` |
| `activations.softsign` | Function | `(x)` |
| `activations.sparse_plus` | Function | `(x)` |
| `activations.sparse_sigmoid` | Function | `(x)` |
| `activations.sparsemax` | Function | `(x, axis)` |
| `activations.squareplus` | Function | `(x, b)` |
| `activations.swish` | Function | `(x)` |
| `activations.tanh` | Function | `(x)` |
| `activations.tanh_shrink` | Function | `(x)` |
| `activations.threshold` | Function | `(x, threshold, default_value)` |
| `applications.ConvNeXtBase` | Function | `(include_top, include_preprocessing, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `applications.ConvNeXtLarge` | Function | `(include_top, include_preprocessing, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `applications.ConvNeXtSmall` | Function | `(include_top, include_preprocessing, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `applications.ConvNeXtTiny` | Function | `(include_top, include_preprocessing, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `applications.ConvNeXtXLarge` | Function | `(include_top, include_preprocessing, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `applications.DenseNet121` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `applications.DenseNet169` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `applications.DenseNet201` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `applications.EfficientNetB0` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `applications.EfficientNetB1` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `applications.EfficientNetB2` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `applications.EfficientNetB3` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `applications.EfficientNetB4` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `applications.EfficientNetB5` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `applications.EfficientNetB6` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `applications.EfficientNetB7` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `applications.EfficientNetV2B0` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, include_preprocessing, name)` |
| `applications.EfficientNetV2B1` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, include_preprocessing, name)` |
| `applications.EfficientNetV2B2` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, include_preprocessing, name)` |
| `applications.EfficientNetV2B3` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, include_preprocessing, name)` |
| `applications.EfficientNetV2L` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, include_preprocessing, name)` |
| `applications.EfficientNetV2M` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, include_preprocessing, name)` |
| `applications.EfficientNetV2S` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, include_preprocessing, name)` |
| `applications.InceptionResNetV2` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `applications.InceptionV3` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `applications.MobileNet` | Function | `(input_shape, alpha, depth_multiplier, dropout, include_top, weights, input_tensor, pooling, classes, classifier_activation, name)` |
| `applications.MobileNetV2` | Function | `(input_shape, alpha, include_top, weights, input_tensor, pooling, classes, classifier_activation, name)` |
| `applications.MobileNetV3Large` | Function | `(input_shape, alpha, minimalistic, include_top, weights, input_tensor, classes, pooling, dropout_rate, classifier_activation, include_preprocessing, name)` |
| `applications.MobileNetV3Small` | Function | `(input_shape, alpha, minimalistic, include_top, weights, input_tensor, classes, pooling, dropout_rate, classifier_activation, include_preprocessing, name)` |
| `applications.NASNetLarge` | Function | `(input_shape, include_top, weights, input_tensor, pooling, classes, classifier_activation, name)` |
| `applications.NASNetMobile` | Function | `(input_shape, include_top, weights, input_tensor, pooling, classes, classifier_activation, name)` |
| `applications.ResNet101` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `applications.ResNet101V2` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `applications.ResNet152` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `applications.ResNet152V2` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `applications.ResNet50` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `applications.ResNet50V2` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `applications.VGG16` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `applications.VGG19` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `applications.Xception` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `applications.convnext.ConvNeXtBase` | Function | `(include_top, include_preprocessing, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `applications.convnext.ConvNeXtLarge` | Function | `(include_top, include_preprocessing, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `applications.convnext.ConvNeXtSmall` | Function | `(include_top, include_preprocessing, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `applications.convnext.ConvNeXtTiny` | Function | `(include_top, include_preprocessing, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `applications.convnext.ConvNeXtXLarge` | Function | `(include_top, include_preprocessing, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `applications.convnext.decode_predictions` | Function | `(preds, top)` |
| `applications.convnext.preprocess_input` | Function | `(x, data_format)` |
| `applications.densenet.DenseNet121` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `applications.densenet.DenseNet169` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `applications.densenet.DenseNet201` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `applications.densenet.decode_predictions` | Function | `(preds, top)` |
| `applications.densenet.preprocess_input` | Function | `(x, data_format)` |
| `applications.efficientnet.EfficientNetB0` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `applications.efficientnet.EfficientNetB1` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `applications.efficientnet.EfficientNetB2` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `applications.efficientnet.EfficientNetB3` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `applications.efficientnet.EfficientNetB4` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `applications.efficientnet.EfficientNetB5` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `applications.efficientnet.EfficientNetB6` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `applications.efficientnet.EfficientNetB7` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `applications.efficientnet.decode_predictions` | Function | `(preds, top)` |
| `applications.efficientnet.preprocess_input` | Function | `(x, data_format)` |
| `applications.efficientnet_v2.EfficientNetV2B0` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, include_preprocessing, name)` |
| `applications.efficientnet_v2.EfficientNetV2B1` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, include_preprocessing, name)` |
| `applications.efficientnet_v2.EfficientNetV2B2` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, include_preprocessing, name)` |
| `applications.efficientnet_v2.EfficientNetV2B3` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, include_preprocessing, name)` |
| `applications.efficientnet_v2.EfficientNetV2L` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, include_preprocessing, name)` |
| `applications.efficientnet_v2.EfficientNetV2M` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, include_preprocessing, name)` |
| `applications.efficientnet_v2.EfficientNetV2S` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, include_preprocessing, name)` |
| `applications.efficientnet_v2.decode_predictions` | Function | `(preds, top)` |
| `applications.efficientnet_v2.preprocess_input` | Function | `(x, data_format)` |
| `applications.imagenet_utils.decode_predictions` | Function | `(preds, top)` |
| `applications.imagenet_utils.preprocess_input` | Function | `(x, data_format, mode)` |
| `applications.inception_resnet_v2.InceptionResNetV2` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `applications.inception_resnet_v2.decode_predictions` | Function | `(preds, top)` |
| `applications.inception_resnet_v2.preprocess_input` | Function | `(x, data_format)` |
| `applications.inception_v3.InceptionV3` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `applications.inception_v3.decode_predictions` | Function | `(preds, top)` |
| `applications.inception_v3.preprocess_input` | Function | `(x, data_format)` |
| `applications.mobilenet.MobileNet` | Function | `(input_shape, alpha, depth_multiplier, dropout, include_top, weights, input_tensor, pooling, classes, classifier_activation, name)` |
| `applications.mobilenet.decode_predictions` | Function | `(preds, top)` |
| `applications.mobilenet.preprocess_input` | Function | `(x, data_format)` |
| `applications.mobilenet_v2.MobileNetV2` | Function | `(input_shape, alpha, include_top, weights, input_tensor, pooling, classes, classifier_activation, name)` |
| `applications.mobilenet_v2.decode_predictions` | Function | `(preds, top)` |
| `applications.mobilenet_v2.preprocess_input` | Function | `(x, data_format)` |
| `applications.mobilenet_v3.decode_predictions` | Function | `(preds, top)` |
| `applications.mobilenet_v3.preprocess_input` | Function | `(x, data_format)` |
| `applications.nasnet.NASNetLarge` | Function | `(input_shape, include_top, weights, input_tensor, pooling, classes, classifier_activation, name)` |
| `applications.nasnet.NASNetMobile` | Function | `(input_shape, include_top, weights, input_tensor, pooling, classes, classifier_activation, name)` |
| `applications.nasnet.decode_predictions` | Function | `(preds, top)` |
| `applications.nasnet.preprocess_input` | Function | `(x, data_format)` |
| `applications.resnet.ResNet101` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `applications.resnet.ResNet152` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `applications.resnet.ResNet50` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `applications.resnet.decode_predictions` | Function | `(preds, top)` |
| `applications.resnet.preprocess_input` | Function | `(x, data_format)` |
| `applications.resnet50.ResNet50` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `applications.resnet50.decode_predictions` | Function | `(preds, top)` |
| `applications.resnet50.preprocess_input` | Function | `(x, data_format)` |
| `applications.resnet_v2.ResNet101V2` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `applications.resnet_v2.ResNet152V2` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `applications.resnet_v2.ResNet50V2` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `applications.resnet_v2.decode_predictions` | Function | `(preds, top)` |
| `applications.resnet_v2.preprocess_input` | Function | `(x, data_format)` |
| `applications.vgg16.VGG16` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `applications.vgg16.decode_predictions` | Function | `(preds, top)` |
| `applications.vgg16.preprocess_input` | Function | `(x, data_format)` |
| `applications.vgg19.VGG19` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `applications.vgg19.decode_predictions` | Function | `(preds, top)` |
| `applications.vgg19.preprocess_input` | Function | `(x, data_format)` |
| `applications.xception.Xception` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `applications.xception.decode_predictions` | Function | `(preds, top)` |
| `applications.xception.preprocess_input` | Function | `(x, data_format)` |
| `backend.backend` | Function | `()` |
| `backend.clear_session` | Function | `(free_memory)` |
| `backend.epsilon` | Function | `()` |
| `backend.floatx` | Function | `()` |
| `backend.get_uid` | Function | `(prefix)` |
| `backend.image_data_format` | Function | `()` |
| `backend.is_float_dtype` | Function | `(dtype)` |
| `backend.is_int_dtype` | Function | `(dtype)` |
| `backend.is_keras_tensor` | Function | `(x)` |
| `backend.result_type` | Function | `(dtypes)` |
| `backend.set_epsilon` | Function | `(value)` |
| `backend.set_floatx` | Function | `(value)` |
| `backend.set_image_data_format` | Function | `(data_format)` |
| `backend.standardize_dtype` | Function | `(dtype)` |
| `callbacks.BackupAndRestore` | Class | `(backup_dir, save_freq, double_checkpoint, delete_checkpoint)` |
| `callbacks.CSVLogger` | Class | `(filename, separator, append)` |
| `callbacks.Callback` | Class | `()` |
| `callbacks.CallbackList` | Class | `(callbacks, add_history, add_progbar, model, params)` |
| `callbacks.EarlyStopping` | Class | `(monitor, min_delta, patience, verbose, mode, baseline, restore_best_weights, start_from_epoch)` |
| `callbacks.History` | Class | `()` |
| `callbacks.LambdaCallback` | Class | `(on_epoch_begin, on_epoch_end, on_train_begin, on_train_end, on_train_batch_begin, on_train_batch_end, kwargs)` |
| `callbacks.LearningRateScheduler` | Class | `(schedule, verbose)` |
| `callbacks.ModelCheckpoint` | Class | `(filepath, monitor, verbose, save_best_only, save_weights_only, mode, save_freq, initial_value_threshold)` |
| `callbacks.ProgbarLogger` | Class | `()` |
| `callbacks.ReduceLROnPlateau` | Class | `(monitor, factor, patience, verbose, mode, min_delta, cooldown, min_lr, kwargs)` |
| `callbacks.RemoteMonitor` | Class | `(root, path, field, headers, send_as_json)` |
| `callbacks.SwapEMAWeights` | Class | `(swap_on_epoch)` |
| `callbacks.TensorBoard` | Class | `(log_dir, histogram_freq, write_graph, write_images, write_steps_per_second, update_freq, profile_batch, embeddings_freq, embeddings_metadata)` |
| `callbacks.TerminateOnNaN` | Class | `(raise_error: bool)` |
| `config.backend` | Function | `()` |
| `config.disable_flash_attention` | Function | `()` |
| `config.disable_interactive_logging` | Function | `()` |
| `config.disable_traceback_filtering` | Function | `()` |
| `config.dtype_policy` | Function | `()` |
| `config.enable_flash_attention` | Function | `()` |
| `config.enable_interactive_logging` | Function | `()` |
| `config.enable_traceback_filtering` | Function | `()` |
| `config.enable_unsafe_deserialization` | Function | `()` |
| `config.epsilon` | Function | `()` |
| `config.floatx` | Function | `()` |
| `config.image_data_format` | Function | `()` |
| `config.is_flash_attention_enabled` | Function | `()` |
| `config.is_interactive_logging_enabled` | Function | `()` |
| `config.is_nnx_enabled` | Function | `()` |
| `config.is_traceback_filtering_enabled` | Function | `()` |
| `config.max_epochs` | Function | `()` |
| `config.max_steps_per_epoch` | Function | `()` |
| `config.set_backend` | Function | `(backend)` |
| `config.set_dtype_policy` | Function | `(policy)` |
| `config.set_epsilon` | Function | `(value)` |
| `config.set_floatx` | Function | `(value)` |
| `config.set_image_data_format` | Function | `(data_format)` |
| `config.set_max_epochs` | Function | `(max_epochs)` |
| `config.set_max_steps_per_epoch` | Function | `(max_steps_per_epoch)` |
| `constraints.Constraint` | Class | `(...)` |
| `constraints.MaxNorm` | Class | `(max_value, axis)` |
| `constraints.MinMaxNorm` | Class | `(min_value, max_value, rate, axis)` |
| `constraints.NonNeg` | Class | `(...)` |
| `constraints.UnitNorm` | Class | `(axis)` |
| `constraints.deserialize` | Function | `(config, custom_objects)` |
| `constraints.get` | Function | `(identifier)` |
| `constraints.max_norm` | Class | `(max_value, axis)` |
| `constraints.min_max_norm` | Class | `(min_value, max_value, rate, axis)` |
| `constraints.non_neg` | Class | `(...)` |
| `constraints.serialize` | Function | `(constraint)` |
| `constraints.unit_norm` | Class | `(axis)` |
| `datasets.boston_housing.load_data` | Function | `(path, test_split, seed)` |
| `datasets.california_housing.load_data` | Function | `(version, path, test_split, seed)` |
| `datasets.cifar10.load_data` | Function | `()` |
| `datasets.cifar100.load_data` | Function | `(label_mode)` |
| `datasets.fashion_mnist.load_data` | Function | `()` |
| `datasets.imdb.get_word_index` | Function | `(path)` |
| `datasets.imdb.load_data` | Function | `(path, num_words, skip_top, maxlen, seed, start_char, oov_char, index_from, kwargs)` |
| `datasets.mnist.load_data` | Function | `(path)` |
| `datasets.reuters.get_label_names` | Function | `()` |
| `datasets.reuters.get_word_index` | Function | `(path)` |
| `datasets.reuters.load_data` | Function | `(path, num_words, skip_top, maxlen, test_split, seed, start_char, oov_char, index_from)` |
| `device` | Function | `(device_name)` |
| `distillation.DistillationLoss` | Class | `(...)` |
| `distillation.Distiller` | Class | `(teacher, student, distillation_losses, distillation_loss_weights, student_loss_weight, name, kwargs)` |
| `distillation.FeatureDistillation` | Class | `(loss, teacher_layer_name, student_layer_name)` |
| `distillation.LogitsDistillation` | Class | `(temperature, loss)` |
| `distribution.DataParallel` | Class | `(device_mesh, devices, auto_shard_dataset)` |
| `distribution.DeviceMesh` | Class | `(shape, axis_names, devices)` |
| `distribution.LayoutMap` | Class | `(device_mesh)` |
| `distribution.ModelParallel` | Class | `(layout_map, batch_dim_name, auto_shard_dataset, kwargs)` |
| `distribution.TensorLayout` | Class | `(axes, device_mesh)` |
| `distribution.distribute_tensor` | Function | `(tensor, layout)` |
| `distribution.distribution` | Function | `()` |
| `distribution.get_device_count` | Function | `(device_type)` |
| `distribution.initialize` | Function | `(job_addresses, num_processes, process_id)` |
| `distribution.list_devices` | Function | `(device_type)` |
| `distribution.set_distribution` | Function | `(value)` |
| `dtype_policies.DTypePolicy` | Class | `(name)` |
| `dtype_policies.DTypePolicyMap` | Class | `(default_policy, policy_map)` |
| `dtype_policies.FloatDTypePolicy` | Class | `(...)` |
| `dtype_policies.GPTQDTypePolicy` | Class | `(mode, source_name)` |
| `dtype_policies.QuantizedDTypePolicy` | Class | `(mode, source_name)` |
| `dtype_policies.QuantizedFloat8DTypePolicy` | Class | `(mode, source_name, amax_history_length)` |
| `dtype_policies.deserialize` | Function | `(config, custom_objects)` |
| `dtype_policies.get` | Function | `(identifier)` |
| `dtype_policies.serialize` | Function | `(dtype_policy)` |
| `export.ExportArchive` | Class | `()` |
| `initializers.Constant` | Class | `(value)` |
| `initializers.GlorotNormal` | Class | `(seed)` |
| `initializers.GlorotUniform` | Class | `(seed)` |
| `initializers.HeNormal` | Class | `(seed)` |
| `initializers.HeUniform` | Class | `(seed)` |
| `initializers.Identity` | Class | `(gain)` |
| `initializers.IdentityInitializer` | Class | `(gain)` |
| `initializers.Initializer` | Class | `(...)` |
| `initializers.LecunNormal` | Class | `(seed)` |
| `initializers.LecunUniform` | Class | `(seed)` |
| `initializers.Ones` | Class | `(...)` |
| `initializers.Orthogonal` | Class | `(gain, seed)` |
| `initializers.OrthogonalInitializer` | Class | `(gain, seed)` |
| `initializers.RandomNormal` | Class | `(mean, stddev, seed)` |
| `initializers.RandomUniform` | Class | `(minval, maxval, seed)` |
| `initializers.STFT` | Class | `(side, window, scaling, periodic)` |
| `initializers.STFTInitializer` | Class | `(side, window, scaling, periodic)` |
| `initializers.TruncatedNormal` | Class | `(mean, stddev, seed)` |
| `initializers.VarianceScaling` | Class | `(scale, mode, distribution, seed)` |
| `initializers.Zeros` | Class | `(...)` |
| `initializers.constant` | Class | `(value)` |
| `initializers.deserialize` | Function | `(config, custom_objects)` |
| `initializers.get` | Function | `(identifier)` |
| `initializers.glorot_normal` | Class | `(seed)` |
| `initializers.glorot_uniform` | Class | `(seed)` |
| `initializers.he_normal` | Class | `(seed)` |
| `initializers.he_uniform` | Class | `(seed)` |
| `initializers.identity` | Class | `(gain)` |
| `initializers.lecun_normal` | Class | `(seed)` |
| `initializers.lecun_uniform` | Class | `(seed)` |
| `initializers.ones` | Class | `(...)` |
| `initializers.orthogonal` | Class | `(gain, seed)` |
| `initializers.random_normal` | Class | `(mean, stddev, seed)` |
| `initializers.random_uniform` | Class | `(minval, maxval, seed)` |
| `initializers.serialize` | Function | `(initializer)` |
| `initializers.stft` | Class | `(side, window, scaling, periodic)` |
| `initializers.truncated_normal` | Class | `(mean, stddev, seed)` |
| `initializers.variance_scaling` | Class | `(scale, mode, distribution, seed)` |
| `initializers.zeros` | Class | `(...)` |
| `layers.Activation` | Class | `(activation, kwargs)` |
| `layers.ActivityRegularization` | Class | `(l1, l2, kwargs)` |
| `layers.AdaptiveAveragePooling1D` | Class | `(output_size, data_format, kwargs)` |
| `layers.AdaptiveAveragePooling2D` | Class | `(output_size, data_format, kwargs)` |
| `layers.AdaptiveAveragePooling3D` | Class | `(output_size, data_format, kwargs)` |
| `layers.AdaptiveMaxPooling1D` | Class | `(output_size, data_format, kwargs)` |
| `layers.AdaptiveMaxPooling2D` | Class | `(output_size, data_format, kwargs)` |
| `layers.AdaptiveMaxPooling3D` | Class | `(output_size, data_format, kwargs)` |
| `layers.Add` | Class | `(...)` |
| `layers.AdditiveAttention` | Class | `(use_scale, dropout, kwargs)` |
| `layers.AlphaDropout` | Class | `(rate, noise_shape, seed, kwargs)` |
| `layers.Attention` | Class | `(use_scale, score_mode, dropout, seed, kwargs)` |
| `layers.AugMix` | Class | `(value_range, num_chains, chain_depth, factor, alpha, all_ops, interpolation, seed, data_format, kwargs)` |
| `layers.AutoContrast` | Class | `(value_range, kwargs)` |
| `layers.Average` | Class | `(...)` |
| `layers.AveragePooling1D` | Class | `(pool_size, strides, padding, data_format, name, kwargs)` |
| `layers.AveragePooling2D` | Class | `(pool_size, strides, padding, data_format, name, kwargs)` |
| `layers.AveragePooling3D` | Class | `(pool_size, strides, padding, data_format, name, kwargs)` |
| `layers.AvgPool1D` | Class | `(pool_size, strides, padding, data_format, name, kwargs)` |
| `layers.AvgPool2D` | Class | `(pool_size, strides, padding, data_format, name, kwargs)` |
| `layers.AvgPool3D` | Class | `(pool_size, strides, padding, data_format, name, kwargs)` |
| `layers.BatchNormalization` | Class | `(axis, momentum, epsilon, center, scale, beta_initializer, gamma_initializer, moving_mean_initializer, moving_variance_initializer, beta_regularizer, gamma_regularizer, beta_constraint, gamma_constraint, synchronized, kwargs)` |
| `layers.Bidirectional` | Class | `(layer, merge_mode, weights, backward_layer, kwargs)` |
| `layers.CategoryEncoding` | Class | `(num_tokens, output_mode, sparse, kwargs)` |
| `layers.CenterCrop` | Class | `(height, width, data_format, kwargs)` |
| `layers.Concatenate` | Class | `(axis, kwargs)` |
| `layers.Conv1D` | Class | `(filters, kernel_size, strides, padding, data_format, dilation_rate, groups, activation, use_bias, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint, kwargs)` |
| `layers.Conv1DTranspose` | Class | `(filters, kernel_size, strides, padding, output_padding, data_format, dilation_rate, activation, use_bias, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint, kwargs)` |
| `layers.Conv2D` | Class | `(filters, kernel_size, strides, padding, data_format, dilation_rate, groups, activation, use_bias, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint, kwargs)` |
| `layers.Conv2DTranspose` | Class | `(filters, kernel_size, strides, padding, output_padding, data_format, dilation_rate, activation, use_bias, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint, kwargs)` |
| `layers.Conv3D` | Class | `(filters, kernel_size, strides, padding, data_format, dilation_rate, groups, activation, use_bias, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint, kwargs)` |
| `layers.Conv3DTranspose` | Class | `(filters, kernel_size, strides, padding, data_format, output_padding, dilation_rate, activation, use_bias, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint, kwargs)` |
| `layers.ConvLSTM1D` | Class | `(filters, kernel_size, strides, padding, data_format, dilation_rate, activation, recurrent_activation, use_bias, kernel_initializer, recurrent_initializer, bias_initializer, unit_forget_bias, kernel_regularizer, recurrent_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, recurrent_constraint, bias_constraint, dropout, recurrent_dropout, seed, return_sequences, return_state, go_backwards, stateful, kwargs)` |
| `layers.ConvLSTM2D` | Class | `(filters, kernel_size, strides, padding, data_format, dilation_rate, activation, recurrent_activation, use_bias, kernel_initializer, recurrent_initializer, bias_initializer, unit_forget_bias, kernel_regularizer, recurrent_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, recurrent_constraint, bias_constraint, dropout, recurrent_dropout, seed, return_sequences, return_state, go_backwards, stateful, kwargs)` |
| `layers.ConvLSTM3D` | Class | `(filters, kernel_size, strides, padding, data_format, dilation_rate, activation, recurrent_activation, use_bias, kernel_initializer, recurrent_initializer, bias_initializer, unit_forget_bias, kernel_regularizer, recurrent_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, recurrent_constraint, bias_constraint, dropout, recurrent_dropout, seed, return_sequences, return_state, go_backwards, stateful, kwargs)` |
| `layers.Convolution1D` | Class | `(filters, kernel_size, strides, padding, data_format, dilation_rate, groups, activation, use_bias, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint, kwargs)` |
| `layers.Convolution1DTranspose` | Class | `(filters, kernel_size, strides, padding, output_padding, data_format, dilation_rate, activation, use_bias, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint, kwargs)` |
| `layers.Convolution2D` | Class | `(filters, kernel_size, strides, padding, data_format, dilation_rate, groups, activation, use_bias, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint, kwargs)` |
| `layers.Convolution2DTranspose` | Class | `(filters, kernel_size, strides, padding, output_padding, data_format, dilation_rate, activation, use_bias, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint, kwargs)` |
| `layers.Convolution3D` | Class | `(filters, kernel_size, strides, padding, data_format, dilation_rate, groups, activation, use_bias, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint, kwargs)` |
| `layers.Convolution3DTranspose` | Class | `(filters, kernel_size, strides, padding, data_format, output_padding, dilation_rate, activation, use_bias, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint, kwargs)` |
| `layers.Cropping1D` | Class | `(cropping, kwargs)` |
| `layers.Cropping2D` | Class | `(cropping, data_format, kwargs)` |
| `layers.Cropping3D` | Class | `(cropping, data_format, kwargs)` |
| `layers.CutMix` | Class | `(factor, seed, data_format, kwargs)` |
| `layers.Dense` | Class | `(units, activation, use_bias, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint, lora_rank, lora_alpha, quantization_config, kwargs)` |
| `layers.DepthwiseConv1D` | Class | `(kernel_size, strides, padding, depth_multiplier, data_format, dilation_rate, activation, use_bias, depthwise_initializer, bias_initializer, depthwise_regularizer, bias_regularizer, activity_regularizer, depthwise_constraint, bias_constraint, kwargs)` |
| `layers.DepthwiseConv2D` | Class | `(kernel_size, strides, padding, depth_multiplier, data_format, dilation_rate, activation, use_bias, depthwise_initializer, bias_initializer, depthwise_regularizer, bias_regularizer, activity_regularizer, depthwise_constraint, bias_constraint, kwargs)` |
| `layers.Discretization` | Class | `(bin_boundaries, num_bins, epsilon, output_mode, sparse, dtype, name)` |
| `layers.Dot` | Class | `(axes, normalize, kwargs)` |
| `layers.Dropout` | Class | `(rate, noise_shape, seed, kwargs)` |
| `layers.ELU` | Class | `(alpha, kwargs)` |
| `layers.EinsumDense` | Class | `(equation, output_shape, activation, bias_axes, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, kernel_constraint, bias_constraint, lora_rank, lora_alpha, gptq_unpacked_column_size, quantization_config, kwargs)` |
| `layers.Embedding` | Class | `(input_dim, output_dim, embeddings_initializer, embeddings_regularizer, embeddings_constraint, mask_zero, weights, lora_rank, lora_alpha, quantization_config, kwargs)` |
| `layers.Equalization` | Class | `(value_range, bins, data_format, kwargs)` |
| `layers.Flatten` | Class | `(data_format, kwargs)` |
| `layers.FlaxLayer` | Class | `(module, method, variables, kwargs)` |
| `layers.GRU` | Class | `(units, activation, recurrent_activation, use_bias, kernel_initializer, recurrent_initializer, bias_initializer, kernel_regularizer, recurrent_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, recurrent_constraint, bias_constraint, dropout, recurrent_dropout, seed, return_sequences, return_state, go_backwards, stateful, unroll, reset_after, use_cudnn, kwargs)` |
| `layers.GRUCell` | Class | `(units, activation, recurrent_activation, use_bias, kernel_initializer, recurrent_initializer, bias_initializer, kernel_regularizer, recurrent_regularizer, bias_regularizer, kernel_constraint, recurrent_constraint, bias_constraint, dropout, recurrent_dropout, reset_after, seed, kwargs)` |
| `layers.GaussianDropout` | Class | `(rate, seed, kwargs)` |
| `layers.GaussianNoise` | Class | `(stddev, seed, kwargs)` |
| `layers.GlobalAveragePooling1D` | Class | `(data_format, keepdims, kwargs)` |
| `layers.GlobalAveragePooling2D` | Class | `(data_format, keepdims, kwargs)` |
| `layers.GlobalAveragePooling3D` | Class | `(data_format, keepdims, kwargs)` |
| `layers.GlobalAvgPool1D` | Class | `(data_format, keepdims, kwargs)` |
| `layers.GlobalAvgPool2D` | Class | `(data_format, keepdims, kwargs)` |
| `layers.GlobalAvgPool3D` | Class | `(data_format, keepdims, kwargs)` |
| `layers.GlobalMaxPool1D` | Class | `(data_format, keepdims, kwargs)` |
| `layers.GlobalMaxPool2D` | Class | `(data_format, keepdims, kwargs)` |
| `layers.GlobalMaxPool3D` | Class | `(data_format, keepdims, kwargs)` |
| `layers.GlobalMaxPooling1D` | Class | `(data_format, keepdims, kwargs)` |
| `layers.GlobalMaxPooling2D` | Class | `(data_format, keepdims, kwargs)` |
| `layers.GlobalMaxPooling3D` | Class | `(data_format, keepdims, kwargs)` |
| `layers.GroupNormalization` | Class | `(groups, axis, epsilon, center, scale, beta_initializer, gamma_initializer, beta_regularizer, gamma_regularizer, beta_constraint, gamma_constraint, kwargs)` |
| `layers.GroupQueryAttention` | Class | `(head_dim, num_query_heads, num_key_value_heads, dropout, use_bias, flash_attention, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint, seed, kwargs)` |
| `layers.HashedCrossing` | Class | `(num_bins, output_mode, sparse, name, dtype, kwargs)` |
| `layers.Hashing` | Class | `(num_bins, mask_value, salt, output_mode, sparse, kwargs)` |
| `layers.Identity` | Class | `(kwargs)` |
| `layers.Input` | Function | `(shape, batch_size, dtype, sparse, ragged, batch_shape, name, tensor, optional)` |
| `layers.InputLayer` | Class | `(shape, batch_size, dtype, sparse, ragged, batch_shape, input_tensor, optional, name, kwargs)` |
| `layers.InputSpec` | Class | `(dtype, shape, ndim, max_ndim, min_ndim, axes, allow_last_axis_squeeze, name, optional)` |
| `layers.IntegerLookup` | Class | `(max_tokens, num_oov_indices, mask_token, oov_token, vocabulary, vocabulary_dtype, idf_weights, invert, output_mode, sparse, pad_to_max_tokens, name, kwargs)` |
| `layers.JaxLayer` | Class | `(call_fn, init_fn, params, state, seed, kwargs)` |
| `layers.LSTM` | Class | `(units, activation, recurrent_activation, use_bias, kernel_initializer, recurrent_initializer, bias_initializer, unit_forget_bias, kernel_regularizer, recurrent_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, recurrent_constraint, bias_constraint, dropout, recurrent_dropout, seed, return_sequences, return_state, go_backwards, stateful, unroll, use_cudnn, kwargs)` |
| `layers.LSTMCell` | Class | `(units, activation, recurrent_activation, use_bias, kernel_initializer, recurrent_initializer, bias_initializer, unit_forget_bias, kernel_regularizer, recurrent_regularizer, bias_regularizer, kernel_constraint, recurrent_constraint, bias_constraint, dropout, recurrent_dropout, seed, kwargs)` |
| `layers.Lambda` | Class | `(function, output_shape, mask, arguments, kwargs)` |
| `layers.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `layers.LayerNormalization` | Class | `(axis, epsilon, center, scale, beta_initializer, gamma_initializer, beta_regularizer, gamma_regularizer, beta_constraint, gamma_constraint, kwargs)` |
| `layers.LeakyReLU` | Class | `(negative_slope, kwargs)` |
| `layers.Masking` | Class | `(mask_value, kwargs)` |
| `layers.MaxNumBoundingBoxes` | Class | `(max_number, fill_value, kwargs)` |
| `layers.MaxPool1D` | Class | `(pool_size, strides, padding, data_format, name, kwargs)` |
| `layers.MaxPool2D` | Class | `(pool_size, strides, padding, data_format, name, kwargs)` |
| `layers.MaxPool3D` | Class | `(pool_size, strides, padding, data_format, name, kwargs)` |
| `layers.MaxPooling1D` | Class | `(pool_size, strides, padding, data_format, name, kwargs)` |
| `layers.MaxPooling2D` | Class | `(pool_size, strides, padding, data_format, name, kwargs)` |
| `layers.MaxPooling3D` | Class | `(pool_size, strides, padding, data_format, name, kwargs)` |
| `layers.Maximum` | Class | `(...)` |
| `layers.MelSpectrogram` | Class | `(fft_length, sequence_stride, sequence_length, window, sampling_rate, num_mel_bins, min_freq, max_freq, power_to_db, top_db, mag_exp, min_power, ref_power, kwargs)` |
| `layers.Minimum` | Class | `(...)` |
| `layers.MixUp` | Class | `(alpha, data_format, seed, kwargs)` |
| `layers.MultiHeadAttention` | Class | `(num_heads, key_dim, value_dim, dropout, use_bias, output_shape, attention_axes, flash_attention, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint, seed, kwargs)` |
| `layers.Multiply` | Class | `(...)` |
| `layers.Normalization` | Class | `(axis, mean, variance, invert, kwargs)` |
| `layers.PReLU` | Class | `(alpha_initializer, alpha_regularizer, alpha_constraint, shared_axes, kwargs)` |
| `layers.Permute` | Class | `(dims, kwargs)` |
| `layers.Pipeline` | Class | `(layers, name)` |
| `layers.RMSNormalization` | Class | `(axis, epsilon, kwargs)` |
| `layers.RNN` | Class | `(cell, return_sequences, return_state, go_backwards, stateful, unroll, zero_output_for_mask, kwargs)` |
| `layers.RandAugment` | Class | `(value_range, num_ops, factor, interpolation, seed, data_format, kwargs)` |
| `layers.RandomBrightness` | Class | `(factor, value_range, seed, kwargs)` |
| `layers.RandomColorDegeneration` | Class | `(factor, value_range, data_format, seed, kwargs)` |
| `layers.RandomColorJitter` | Class | `(value_range, brightness_factor, contrast_factor, saturation_factor, hue_factor, seed, data_format, kwargs)` |
| `layers.RandomContrast` | Class | `(factor, value_range, seed, kwargs)` |
| `layers.RandomCrop` | Class | `(height, width, seed, data_format, name, kwargs)` |
| `layers.RandomElasticTransform` | Class | `(factor, scale, interpolation, fill_mode, fill_value, value_range, seed, data_format, kwargs)` |
| `layers.RandomErasing` | Class | `(factor, scale, fill_value, value_range, seed, data_format, kwargs)` |
| `layers.RandomFlip` | Class | `(mode, seed, data_format, kwargs)` |
| `layers.RandomGaussianBlur` | Class | `(factor, kernel_size, sigma, value_range, data_format, seed, kwargs)` |
| `layers.RandomGrayscale` | Class | `(factor, data_format, seed, kwargs)` |
| `layers.RandomHue` | Class | `(factor, value_range, data_format, seed, kwargs)` |
| `layers.RandomInvert` | Class | `(factor, value_range, seed, data_format, kwargs)` |
| `layers.RandomPerspective` | Class | `(factor, scale, interpolation, fill_value, seed, data_format, kwargs)` |
| `layers.RandomPosterization` | Class | `(factor, value_range, data_format, seed, kwargs)` |
| `layers.RandomRotation` | Class | `(factor, fill_mode, interpolation, seed, fill_value, data_format, kwargs)` |
| `layers.RandomSaturation` | Class | `(factor, value_range, data_format, seed, kwargs)` |
| `layers.RandomSharpness` | Class | `(factor, value_range, data_format, seed, kwargs)` |
| `layers.RandomShear` | Class | `(x_factor, y_factor, interpolation, fill_mode, fill_value, data_format, seed, kwargs)` |
| `layers.RandomTranslation` | Class | `(height_factor, width_factor, fill_mode, interpolation, seed, fill_value, data_format, kwargs)` |
| `layers.RandomZoom` | Class | `(height_factor, width_factor, fill_mode, interpolation, seed, fill_value, data_format, kwargs)` |
| `layers.ReLU` | Class | `(max_value, negative_slope, threshold, kwargs)` |
| `layers.RepeatVector` | Class | `(n, kwargs)` |
| `layers.Rescaling` | Class | `(scale, offset, kwargs)` |
| `layers.Reshape` | Class | `(target_shape, kwargs)` |
| `layers.Resizing` | Class | `(height, width, interpolation, crop_to_aspect_ratio, pad_to_aspect_ratio, fill_mode, fill_value, antialias, data_format, kwargs)` |
| `layers.ReversibleEmbedding` | Class | `(input_dim, output_dim, tie_weights, embeddings_initializer, embeddings_regularizer, embeddings_constraint, mask_zero, reverse_dtype, logit_soft_cap, kwargs)` |
| `layers.STFTSpectrogram` | Class | `(mode, frame_length, frame_step, fft_length, window, periodic, scaling, padding, expand_dims, data_format, kwargs)` |
| `layers.SeparableConv1D` | Class | `(filters, kernel_size, strides, padding, data_format, dilation_rate, depth_multiplier, activation, use_bias, depthwise_initializer, pointwise_initializer, bias_initializer, depthwise_regularizer, pointwise_regularizer, bias_regularizer, activity_regularizer, depthwise_constraint, pointwise_constraint, bias_constraint, kwargs)` |
| `layers.SeparableConv2D` | Class | `(filters, kernel_size, strides, padding, data_format, dilation_rate, depth_multiplier, activation, use_bias, depthwise_initializer, pointwise_initializer, bias_initializer, depthwise_regularizer, pointwise_regularizer, bias_regularizer, activity_regularizer, depthwise_constraint, pointwise_constraint, bias_constraint, kwargs)` |
| `layers.SeparableConvolution1D` | Class | `(filters, kernel_size, strides, padding, data_format, dilation_rate, depth_multiplier, activation, use_bias, depthwise_initializer, pointwise_initializer, bias_initializer, depthwise_regularizer, pointwise_regularizer, bias_regularizer, activity_regularizer, depthwise_constraint, pointwise_constraint, bias_constraint, kwargs)` |
| `layers.SeparableConvolution2D` | Class | `(filters, kernel_size, strides, padding, data_format, dilation_rate, depth_multiplier, activation, use_bias, depthwise_initializer, pointwise_initializer, bias_initializer, depthwise_regularizer, pointwise_regularizer, bias_regularizer, activity_regularizer, depthwise_constraint, pointwise_constraint, bias_constraint, kwargs)` |
| `layers.SimpleRNN` | Class | `(units, activation, use_bias, kernel_initializer, recurrent_initializer, bias_initializer, kernel_regularizer, recurrent_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, recurrent_constraint, bias_constraint, dropout, recurrent_dropout, return_sequences, return_state, go_backwards, stateful, unroll, seed, kwargs)` |
| `layers.SimpleRNNCell` | Class | `(units, activation, use_bias, kernel_initializer, recurrent_initializer, bias_initializer, kernel_regularizer, recurrent_regularizer, bias_regularizer, kernel_constraint, recurrent_constraint, bias_constraint, dropout, recurrent_dropout, seed, kwargs)` |
| `layers.Softmax` | Class | `(axis, kwargs)` |
| `layers.Solarization` | Class | `(addition_factor, threshold_factor, value_range, seed, kwargs)` |
| `layers.SpatialDropout1D` | Class | `(rate, seed, name, dtype)` |
| `layers.SpatialDropout2D` | Class | `(rate, data_format, seed, name, dtype)` |
| `layers.SpatialDropout3D` | Class | `(rate, data_format, seed, name, dtype)` |
| `layers.SpectralNormalization` | Class | `(layer, power_iterations, kwargs)` |
| `layers.StackedRNNCells` | Class | `(cells, kwargs)` |
| `layers.StringLookup` | Class | `(max_tokens, num_oov_indices, mask_token, oov_token, vocabulary, idf_weights, invert, output_mode, pad_to_max_tokens, sparse, encoding, name, kwargs)` |
| `layers.Subtract` | Class | `(...)` |
| `layers.TFSMLayer` | Class | `(filepath, call_endpoint, call_training_endpoint, trainable, name, dtype)` |
| `layers.TextVectorization` | Class | `(max_tokens, standardize, split, ngrams, output_mode, output_sequence_length, pad_to_max_tokens, vocabulary, idf_weights, sparse, ragged, encoding, name, kwargs)` |
| `layers.TimeDistributed` | Class | `(layer, kwargs)` |
| `layers.TorchModuleWrapper` | Class | `(module, name, output_shape, kwargs)` |
| `layers.UnitNormalization` | Class | `(axis, kwargs)` |
| `layers.UpSampling1D` | Class | `(size, kwargs)` |
| `layers.UpSampling2D` | Class | `(size, data_format, interpolation, kwargs)` |
| `layers.UpSampling3D` | Class | `(size, data_format, kwargs)` |
| `layers.Wrapper` | Class | `(layer, kwargs)` |
| `layers.ZeroPadding1D` | Class | `(padding, data_format, kwargs)` |
| `layers.ZeroPadding2D` | Class | `(padding, data_format, kwargs)` |
| `layers.ZeroPadding3D` | Class | `(padding, data_format, kwargs)` |
| `layers.add` | Function | `(inputs, kwargs)` |
| `layers.average` | Function | `(inputs, kwargs)` |
| `layers.concatenate` | Function | `(inputs, axis, kwargs)` |
| `layers.deserialize` | Function | `(config, custom_objects)` |
| `layers.dot` | Function | `(inputs, axes, kwargs)` |
| `layers.maximum` | Function | `(inputs, kwargs)` |
| `layers.minimum` | Function | `(inputs, kwargs)` |
| `layers.multiply` | Function | `(inputs, kwargs)` |
| `layers.serialize` | Function | `(layer)` |
| `layers.subtract` | Function | `(inputs, kwargs)` |
| `legacy.saving.deserialize_keras_object` | Function | `(identifier, module_objects, custom_objects, printable_module_name)` |
| `legacy.saving.serialize_keras_object` | Function | `(instance)` |
| `losses.BinaryCrossentropy` | Class | `(from_logits, label_smoothing, axis, reduction, name, dtype)` |
| `losses.BinaryFocalCrossentropy` | Class | `(apply_class_balancing, alpha, gamma, from_logits, label_smoothing, axis, reduction, name, dtype)` |
| `losses.CTC` | Class | `(reduction, name, dtype)` |
| `losses.CategoricalCrossentropy` | Class | `(from_logits, label_smoothing, axis, reduction, name, dtype)` |
| `losses.CategoricalFocalCrossentropy` | Class | `(alpha, gamma, from_logits, label_smoothing, axis, reduction, name, dtype)` |
| `losses.CategoricalGeneralizedCrossEntropy` | Class | `(q, reduction, name, dtype)` |
| `losses.CategoricalHinge` | Class | `(reduction, name, dtype)` |
| `losses.Circle` | Class | `(gamma, margin, remove_diagonal, reduction, name, dtype)` |
| `losses.CosineSimilarity` | Class | `(axis, reduction, name, dtype)` |
| `losses.Dice` | Class | `(reduction, name, axis, dtype)` |
| `losses.Hinge` | Class | `(reduction, name, dtype)` |
| `losses.Huber` | Class | `(delta, reduction, name, dtype)` |
| `losses.KLDivergence` | Class | `(reduction, name, dtype)` |
| `losses.LogCosh` | Class | `(reduction, name, dtype)` |
| `losses.Loss` | Class | `(name, reduction, dtype)` |
| `losses.MeanAbsoluteError` | Class | `(reduction, name, dtype)` |
| `losses.MeanAbsolutePercentageError` | Class | `(reduction, name, dtype)` |
| `losses.MeanSquaredError` | Class | `(reduction, name, dtype)` |
| `losses.MeanSquaredLogarithmicError` | Class | `(reduction, name, dtype)` |
| `losses.Poisson` | Class | `(reduction, name, dtype)` |
| `losses.SparseCategoricalCrossentropy` | Class | `(from_logits, ignore_class, reduction, axis, name, dtype)` |
| `losses.SquaredHinge` | Class | `(reduction, name, dtype)` |
| `losses.Tversky` | Class | `(alpha, beta, reduction, name, axis, dtype)` |
| `losses.binary_crossentropy` | Function | `(y_true, y_pred, from_logits, label_smoothing, axis)` |
| `losses.binary_focal_crossentropy` | Function | `(y_true, y_pred, apply_class_balancing, alpha, gamma, from_logits, label_smoothing, axis)` |
| `losses.categorical_crossentropy` | Function | `(y_true, y_pred, from_logits, label_smoothing, axis)` |
| `losses.categorical_focal_crossentropy` | Function | `(y_true, y_pred, alpha, gamma, from_logits, label_smoothing, axis)` |
| `losses.categorical_generalized_cross_entropy` | Function | `(y_true, y_pred, q)` |
| `losses.categorical_hinge` | Function | `(y_true, y_pred)` |
| `losses.circle` | Function | `(y_true, y_pred, ref_labels, ref_embeddings, remove_diagonal, gamma, margin)` |
| `losses.cosine_similarity` | Function | `(y_true, y_pred, axis)` |
| `losses.ctc` | Function | `(y_true, y_pred)` |
| `losses.deserialize` | Function | `(name, custom_objects)` |
| `losses.dice` | Function | `(y_true, y_pred, axis)` |
| `losses.get` | Function | `(identifier)` |
| `losses.hinge` | Function | `(y_true, y_pred)` |
| `losses.huber` | Function | `(y_true, y_pred, delta)` |
| `losses.kl_divergence` | Function | `(y_true, y_pred)` |
| `losses.log_cosh` | Function | `(y_true, y_pred)` |
| `losses.mean_absolute_error` | Function | `(y_true, y_pred)` |
| `losses.mean_absolute_percentage_error` | Function | `(y_true, y_pred)` |
| `losses.mean_squared_error` | Function | `(y_true, y_pred)` |
| `losses.mean_squared_logarithmic_error` | Function | `(y_true, y_pred)` |
| `losses.poisson` | Function | `(y_true, y_pred)` |
| `losses.serialize` | Function | `(loss)` |
| `losses.sparse_categorical_crossentropy` | Function | `(y_true, y_pred, from_logits, ignore_class, axis)` |
| `losses.squared_hinge` | Function | `(y_true, y_pred)` |
| `losses.tversky` | Function | `(y_true, y_pred, alpha, beta, axis)` |
| `metrics.AUC` | Class | `(num_thresholds, curve, summation_method, name, dtype, thresholds, multi_label, num_labels, label_weights, from_logits)` |
| `metrics.Accuracy` | Class | `(name, dtype)` |
| `metrics.BinaryAccuracy` | Class | `(name, dtype, threshold)` |
| `metrics.BinaryCrossentropy` | Class | `(name, dtype, from_logits, label_smoothing)` |
| `metrics.BinaryIoU` | Class | `(target_class_ids, threshold, name, dtype)` |
| `metrics.CategoricalAccuracy` | Class | `(name, dtype)` |
| `metrics.CategoricalCrossentropy` | Class | `(name, dtype, from_logits, label_smoothing, axis)` |
| `metrics.CategoricalHinge` | Class | `(name, dtype)` |
| `metrics.ConcordanceCorrelation` | Class | `(name, dtype, axis)` |
| `metrics.CosineSimilarity` | Class | `(name, dtype, axis)` |
| `metrics.F1Score` | Class | `(average, threshold, name, dtype)` |
| `metrics.FBetaScore` | Class | `(average, beta, threshold, name, dtype)` |
| `metrics.FalseNegatives` | Class | `(thresholds, name, dtype)` |
| `metrics.FalsePositives` | Class | `(thresholds, name, dtype)` |
| `metrics.Hinge` | Class | `(name, dtype)` |
| `metrics.IoU` | Class | `(num_classes, target_class_ids, name, dtype, ignore_class, sparse_y_true, sparse_y_pred, axis)` |
| `metrics.KLDivergence` | Class | `(name, dtype)` |
| `metrics.LogCoshError` | Class | `(name, dtype)` |
| `metrics.Mean` | Class | `(name, dtype)` |
| `metrics.MeanAbsoluteError` | Class | `(name, dtype)` |
| `metrics.MeanAbsolutePercentageError` | Class | `(name, dtype)` |
| `metrics.MeanIoU` | Class | `(num_classes, name, dtype, ignore_class, sparse_y_true, sparse_y_pred, axis)` |
| `metrics.MeanMetricWrapper` | Class | `(fn, name, dtype, kwargs)` |
| `metrics.MeanSquaredError` | Class | `(name, dtype)` |
| `metrics.MeanSquaredLogarithmicError` | Class | `(name, dtype)` |
| `metrics.Metric` | Class | `(dtype, name)` |
| `metrics.OneHotIoU` | Class | `(num_classes, target_class_ids, name, dtype, ignore_class, sparse_y_pred, axis)` |
| `metrics.OneHotMeanIoU` | Class | `(num_classes, name, dtype, ignore_class, sparse_y_pred, axis)` |
| `metrics.PearsonCorrelation` | Class | `(name, dtype, axis)` |
| `metrics.Poisson` | Class | `(name, dtype)` |
| `metrics.Precision` | Class | `(thresholds, top_k, class_id, name, dtype)` |
| `metrics.PrecisionAtRecall` | Class | `(recall, num_thresholds, class_id, name, dtype)` |
| `metrics.R2Score` | Class | `(class_aggregation, num_regressors, name, dtype)` |
| `metrics.Recall` | Class | `(thresholds, top_k, class_id, name, dtype)` |
| `metrics.RecallAtPrecision` | Class | `(precision, num_thresholds, class_id, name, dtype)` |
| `metrics.RootMeanSquaredError` | Class | `(name, dtype)` |
| `metrics.SensitivityAtSpecificity` | Class | `(specificity, num_thresholds, class_id, name, dtype)` |
| `metrics.SparseCategoricalAccuracy` | Class | `(name, dtype)` |
| `metrics.SparseCategoricalCrossentropy` | Class | `(name, dtype, from_logits, axis)` |
| `metrics.SparseTopKCategoricalAccuracy` | Class | `(k, name, dtype, from_sorted_ids)` |
| `metrics.SpecificityAtSensitivity` | Class | `(sensitivity, num_thresholds, class_id, name, dtype)` |
| `metrics.SquaredHinge` | Class | `(name, dtype)` |
| `metrics.Sum` | Class | `(name, dtype)` |
| `metrics.TopKCategoricalAccuracy` | Class | `(k, name, dtype)` |
| `metrics.TrueNegatives` | Class | `(thresholds, name, dtype)` |
| `metrics.TruePositives` | Class | `(thresholds, name, dtype)` |
| `metrics.binary_accuracy` | Function | `(y_true, y_pred, threshold)` |
| `metrics.binary_crossentropy` | Function | `(y_true, y_pred, from_logits, label_smoothing, axis)` |
| `metrics.binary_focal_crossentropy` | Function | `(y_true, y_pred, apply_class_balancing, alpha, gamma, from_logits, label_smoothing, axis)` |
| `metrics.categorical_accuracy` | Function | `(y_true, y_pred)` |
| `metrics.categorical_crossentropy` | Function | `(y_true, y_pred, from_logits, label_smoothing, axis)` |
| `metrics.categorical_focal_crossentropy` | Function | `(y_true, y_pred, alpha, gamma, from_logits, label_smoothing, axis)` |
| `metrics.categorical_hinge` | Function | `(y_true, y_pred)` |
| `metrics.concordance_correlation` | Function | `(y_true, y_pred, axis)` |
| `metrics.deserialize` | Function | `(config, custom_objects)` |
| `metrics.get` | Function | `(identifier)` |
| `metrics.hinge` | Function | `(y_true, y_pred)` |
| `metrics.huber` | Function | `(y_true, y_pred, delta)` |
| `metrics.kl_divergence` | Function | `(y_true, y_pred)` |
| `metrics.log_cosh` | Function | `(y_true, y_pred)` |
| `metrics.mean_absolute_error` | Function | `(y_true, y_pred)` |
| `metrics.mean_absolute_percentage_error` | Function | `(y_true, y_pred)` |
| `metrics.mean_squared_error` | Function | `(y_true, y_pred)` |
| `metrics.mean_squared_logarithmic_error` | Function | `(y_true, y_pred)` |
| `metrics.pearson_correlation` | Function | `(y_true, y_pred, axis)` |
| `metrics.poisson` | Function | `(y_true, y_pred)` |
| `metrics.serialize` | Function | `(metric)` |
| `metrics.sparse_categorical_accuracy` | Function | `(y_true, y_pred)` |
| `metrics.sparse_categorical_crossentropy` | Function | `(y_true, y_pred, from_logits, ignore_class, axis)` |
| `metrics.sparse_top_k_categorical_accuracy` | Function | `(y_true, y_pred, k, from_sorted_ids)` |
| `metrics.squared_hinge` | Function | `(y_true, y_pred)` |
| `metrics.top_k_categorical_accuracy` | Function | `(y_true, y_pred, k)` |
| `mixed_precision.DTypePolicy` | Class | `(name)` |
| `mixed_precision.LossScaleOptimizer` | Class | `(inner_optimizer, initial_scale, dynamic_growth_steps, name, kwargs)` |
| `mixed_precision.Policy` | Class | `(name)` |
| `mixed_precision.dtype_policy` | Function | `()` |
| `mixed_precision.global_policy` | Function | `()` |
| `mixed_precision.set_dtype_policy` | Function | `(policy)` |
| `mixed_precision.set_global_policy` | Function | `(policy)` |
| `models.Model` | Class | `(args, kwargs)` |
| `models.Sequential` | Class | `(layers, trainable, name)` |
| `models.clone_model` | Function | `(model, input_tensors, clone_function, call_function, recursive, kwargs)` |
| `models.load_model` | Function | `(filepath, custom_objects, compile, safe_mode)` |
| `models.model_from_json` | Function | `(json_string, custom_objects)` |
| `models.save_model` | Function | `(model, filepath, overwrite, zipped, kwargs)` |
| `name_scope` | Class | `(...)` |
| `ops.abs` | Function | `(x)` |
| `ops.absolute` | Function | `(x)` |
| `ops.adaptive_average_pool` | Function | `(inputs, output_size, data_format)` |
| `ops.adaptive_max_pool` | Function | `(inputs, output_size, data_format)` |
| `ops.add` | Function | `(x1, x2)` |
| `ops.all` | Function | `(x, axis, keepdims)` |
| `ops.amax` | Function | `(x, axis, keepdims)` |
| `ops.amin` | Function | `(x, axis, keepdims)` |
| `ops.angle` | Function | `(x)` |
| `ops.any` | Function | `(x, axis, keepdims)` |
| `ops.append` | Function | `(x1, x2, axis)` |
| `ops.arange` | Function | `(start, stop, step, dtype)` |
| `ops.arccos` | Function | `(x)` |
| `ops.arccosh` | Function | `(x)` |
| `ops.arcsin` | Function | `(x)` |
| `ops.arcsinh` | Function | `(x)` |
| `ops.arctan` | Function | `(x)` |
| `ops.arctan2` | Function | `(x1, x2)` |
| `ops.arctanh` | Function | `(x)` |
| `ops.argmax` | Function | `(x, axis, keepdims)` |
| `ops.argmin` | Function | `(x, axis, keepdims)` |
| `ops.argpartition` | Function | `(x, kth, axis)` |
| `ops.argsort` | Function | `(x, axis)` |
| `ops.array` | Function | `(x, dtype)` |
| `ops.array_split` | Function | `(x, indices_or_sections, axis)` |
| `ops.associative_scan` | Function | `(f, elems, reverse, axis)` |
| `ops.average` | Function | `(x, axis, weights)` |
| `ops.average_pool` | Function | `(inputs, pool_size, strides, padding, data_format)` |
| `ops.bartlett` | Function | `(x)` |
| `ops.batch_normalization` | Function | `(x, mean, variance, axis, offset, scale, epsilon)` |
| `ops.binary_crossentropy` | Function | `(target, output, from_logits)` |
| `ops.bincount` | Function | `(x, weights, minlength, sparse)` |
| `ops.bitwise_and` | Function | `(x, y)` |
| `ops.bitwise_invert` | Function | `(x)` |
| `ops.bitwise_left_shift` | Function | `(x, y)` |
| `ops.bitwise_not` | Function | `(x)` |
| `ops.bitwise_or` | Function | `(x, y)` |
| `ops.bitwise_right_shift` | Function | `(x, y)` |
| `ops.bitwise_xor` | Function | `(x, y)` |
| `ops.blackman` | Function | `(x)` |
| `ops.broadcast_to` | Function | `(x, shape)` |
| `ops.cast` | Function | `(x, dtype)` |
| `ops.categorical_crossentropy` | Function | `(target, output, from_logits, axis)` |
| `ops.cbrt` | Function | `(x)` |
| `ops.ceil` | Function | `(x)` |
| `ops.celu` | Function | `(x, alpha)` |
| `ops.cholesky` | Function | `(x, upper)` |
| `ops.cholesky_inverse` | Function | `(x, upper)` |
| `ops.clip` | Function | `(x, x_min, x_max)` |
| `ops.concatenate` | Function | `(xs, axis)` |
| `ops.cond` | Function | `(pred, true_fn, false_fn)` |
| `ops.conj` | Function | `(x)` |
| `ops.conjugate` | Function | `(x)` |
| `ops.conv` | Function | `(inputs, kernel, strides, padding, data_format, dilation_rate)` |
| `ops.conv_transpose` | Function | `(inputs, kernel, strides, padding, output_padding, data_format, dilation_rate)` |
| `ops.convert_to_numpy` | Function | `(x)` |
| `ops.convert_to_tensor` | Function | `(x, dtype, sparse, ragged)` |
| `ops.copy` | Function | `(x)` |
| `ops.corrcoef` | Function | `(x)` |
| `ops.correlate` | Function | `(x1, x2, mode)` |
| `ops.cos` | Function | `(x)` |
| `ops.cosh` | Function | `(x)` |
| `ops.count_nonzero` | Function | `(x, axis)` |
| `ops.cross` | Function | `(x1, x2, axisa, axisb, axisc, axis)` |
| `ops.ctc_decode` | Function | `(inputs, sequence_lengths, strategy, beam_width, top_paths, merge_repeated, mask_index)` |
| `ops.ctc_loss` | Function | `(target, output, target_length, output_length, mask_index)` |
| `ops.cumprod` | Function | `(x, axis, dtype)` |
| `ops.cumsum` | Function | `(x, axis, dtype)` |
| `ops.custom_gradient` | Function | `(f)` |
| `ops.deg2rad` | Function | `(x)` |
| `ops.depthwise_conv` | Function | `(inputs, kernel, strides, padding, data_format, dilation_rate)` |
| `ops.det` | Function | `(x)` |
| `ops.diag` | Function | `(x, k)` |
| `ops.diagflat` | Function | `(x, k)` |
| `ops.diagonal` | Function | `(x, offset, axis1, axis2)` |
| `ops.diff` | Function | `(a, n, axis)` |
| `ops.digitize` | Function | `(x, bins)` |
| `ops.divide` | Function | `(x1, x2)` |
| `ops.divide_no_nan` | Function | `(x1, x2)` |
| `ops.dot` | Function | `(x1, x2)` |
| `ops.dot_product_attention` | Function | `(query, key, value, bias, mask, scale, is_causal, flash_attention, attn_logits_soft_cap)` |
| `ops.dtype` | Function | `(x)` |
| `ops.eig` | Function | `(x)` |
| `ops.eigh` | Function | `(x)` |
| `ops.einsum` | Function | `(subscripts, operands, kwargs)` |
| `ops.elu` | Function | `(x, alpha)` |
| `ops.empty` | Function | `(shape, dtype)` |
| `ops.empty_like` | Function | `(x, dtype)` |
| `ops.equal` | Function | `(x1, x2)` |
| `ops.erf` | Function | `(x)` |
| `ops.erfinv` | Function | `(x)` |
| `ops.exp` | Function | `(x)` |
| `ops.exp2` | Function | `(x)` |
| `ops.expand_dims` | Function | `(x, axis)` |
| `ops.expm1` | Function | `(x)` |
| `ops.extract_sequences` | Function | `(x, sequence_length, sequence_stride)` |
| `ops.eye` | Function | `(N, M, k, dtype)` |
| `ops.fft` | Function | `(x)` |
| `ops.fft2` | Function | `(x)` |
| `ops.flip` | Function | `(x, axis)` |
| `ops.floor` | Function | `(x)` |
| `ops.floor_divide` | Function | `(x1, x2)` |
| `ops.fori_loop` | Function | `(lower, upper, body_fun, init_val)` |
| `ops.full` | Function | `(shape, fill_value, dtype)` |
| `ops.full_like` | Function | `(x, fill_value, dtype)` |
| `ops.gcd` | Function | `(x1, x2)` |
| `ops.gelu` | Function | `(x, approximate)` |
| `ops.get_item` | Function | `(x, key)` |
| `ops.glu` | Function | `(x, axis)` |
| `ops.greater` | Function | `(x1, x2)` |
| `ops.greater_equal` | Function | `(x1, x2)` |
| `ops.hamming` | Function | `(x)` |
| `ops.hanning` | Function | `(x)` |
| `ops.hard_shrink` | Function | `(x, threshold)` |
| `ops.hard_sigmoid` | Function | `(x)` |
| `ops.hard_silu` | Function | `(x)` |
| `ops.hard_swish` | Function | `(x)` |
| `ops.hard_tanh` | Function | `(x)` |
| `ops.heaviside` | Function | `(x1, x2)` |
| `ops.histogram` | Function | `(x, bins, range)` |
| `ops.hstack` | Function | `(xs)` |
| `ops.hypot` | Function | `(x1, x2)` |
| `ops.identity` | Function | `(n, dtype)` |
| `ops.ifft2` | Function | `(x)` |
| `ops.imag` | Function | `(x)` |
| `ops.image.affine_transform` | Function | `(images, transform, interpolation, fill_mode, fill_value, data_format)` |
| `ops.image.crop_images` | Function | `(images, top_cropping, left_cropping, bottom_cropping, right_cropping, target_height, target_width, data_format)` |
| `ops.image.elastic_transform` | Function | `(images, alpha, sigma, interpolation, fill_mode, fill_value, seed, data_format)` |
| `ops.image.extract_patches` | Function | `(images, size, strides, dilation_rate, padding, data_format)` |
| `ops.image.extract_patches_3d` | Function | `(volumes, size, strides, dilation_rate, padding, data_format)` |
| `ops.image.gaussian_blur` | Function | `(images, kernel_size, sigma, data_format)` |
| `ops.image.hsv_to_rgb` | Function | `(images, data_format)` |
| `ops.image.map_coordinates` | Function | `(inputs, coordinates, order, fill_mode, fill_value)` |
| `ops.image.pad_images` | Function | `(images, top_padding, left_padding, bottom_padding, right_padding, target_height, target_width, data_format)` |
| `ops.image.perspective_transform` | Function | `(images, start_points, end_points, interpolation, fill_value, data_format)` |
| `ops.image.resize` | Function | `(images, size, interpolation, antialias, crop_to_aspect_ratio, pad_to_aspect_ratio, fill_mode, fill_value, data_format)` |
| `ops.image.rgb_to_grayscale` | Function | `(images, data_format)` |
| `ops.image.rgb_to_hsv` | Function | `(images, data_format)` |
| `ops.image.scale_and_translate` | Function | `(images, output_shape, scale, translation, spatial_dims, method, antialias)` |
| `ops.in_top_k` | Function | `(targets, predictions, k)` |
| `ops.inner` | Function | `(x1, x2)` |
| `ops.inv` | Function | `(x)` |
| `ops.irfft` | Function | `(x, fft_length)` |
| `ops.is_tensor` | Function | `(x)` |
| `ops.isclose` | Function | `(x1, x2, rtol, atol, equal_nan)` |
| `ops.isfinite` | Function | `(x)` |
| `ops.isin` | Function | `(x1, x2, assume_unique, invert)` |
| `ops.isinf` | Function | `(x)` |
| `ops.isnan` | Function | `(x)` |
| `ops.isneginf` | Function | `(x)` |
| `ops.isposinf` | Function | `(x)` |
| `ops.isreal` | Function | `(x)` |
| `ops.istft` | Function | `(x, sequence_length, sequence_stride, fft_length, length, window, center)` |
| `ops.jvp` | Function | `(fun, primals, tangents, has_aux)` |
| `ops.kaiser` | Function | `(x, beta)` |
| `ops.kron` | Function | `(x1, x2)` |
| `ops.layer_normalization` | Function | `(x, gamma, beta, axis, epsilon, kwargs)` |
| `ops.lcm` | Function | `(x1, x2)` |
| `ops.ldexp` | Function | `(x1, x2)` |
| `ops.leaky_relu` | Function | `(x, negative_slope)` |
| `ops.left_shift` | Function | `(x, y)` |
| `ops.less` | Function | `(x1, x2)` |
| `ops.less_equal` | Function | `(x1, x2)` |
| `ops.linalg.cholesky` | Function | `(x, upper)` |
| `ops.linalg.cholesky_inverse` | Function | `(x, upper)` |
| `ops.linalg.det` | Function | `(x)` |
| `ops.linalg.eig` | Function | `(x)` |
| `ops.linalg.eigh` | Function | `(x)` |
| `ops.linalg.inv` | Function | `(x)` |
| `ops.linalg.jvp` | Function | `(fun, primals, tangents, has_aux)` |
| `ops.linalg.lstsq` | Function | `(a, b, rcond)` |
| `ops.linalg.lu_factor` | Function | `(x)` |
| `ops.linalg.norm` | Function | `(x, ord, axis, keepdims)` |
| `ops.linalg.qr` | Function | `(x, mode)` |
| `ops.linalg.solve` | Function | `(a, b)` |
| `ops.linalg.solve_triangular` | Function | `(a, b, lower)` |
| `ops.linalg.svd` | Function | `(x, full_matrices, compute_uv)` |
| `ops.linspace` | Function | `(start, stop, num, endpoint, retstep, dtype, axis)` |
| `ops.log` | Function | `(x)` |
| `ops.log10` | Function | `(x)` |
| `ops.log1p` | Function | `(x)` |
| `ops.log2` | Function | `(x)` |
| `ops.log_sigmoid` | Function | `(x)` |
| `ops.log_softmax` | Function | `(x, axis)` |
| `ops.logaddexp` | Function | `(x1, x2)` |
| `ops.logaddexp2` | Function | `(x1, x2)` |
| `ops.logdet` | Function | `(x)` |
| `ops.logical_and` | Function | `(x1, x2)` |
| `ops.logical_not` | Function | `(x)` |
| `ops.logical_or` | Function | `(x1, x2)` |
| `ops.logical_xor` | Function | `(x1, x2)` |
| `ops.logspace` | Function | `(start, stop, num, endpoint, base, dtype, axis)` |
| `ops.logsumexp` | Function | `(x, axis, keepdims)` |
| `ops.lstsq` | Function | `(a, b, rcond)` |
| `ops.lu_factor` | Function | `(x)` |
| `ops.map` | Function | `(f, xs)` |
| `ops.matmul` | Function | `(x1, x2)` |
| `ops.max` | Function | `(x, axis, keepdims, initial)` |
| `ops.max_pool` | Function | `(inputs, pool_size, strides, padding, data_format)` |
| `ops.maximum` | Function | `(x1, x2)` |
| `ops.mean` | Function | `(x, axis, keepdims)` |
| `ops.median` | Function | `(x, axis, keepdims)` |
| `ops.meshgrid` | Function | `(x, indexing)` |
| `ops.min` | Function | `(x, axis, keepdims, initial)` |
| `ops.minimum` | Function | `(x1, x2)` |
| `ops.mod` | Function | `(x1, x2)` |
| `ops.moments` | Function | `(x, axes, keepdims, synchronized)` |
| `ops.moveaxis` | Function | `(x, source, destination)` |
| `ops.multi_hot` | Function | `(inputs, num_classes, axis, dtype, sparse, kwargs)` |
| `ops.multiply` | Function | `(x1, x2)` |
| `ops.nan_to_num` | Function | `(x, nan, posinf, neginf)` |
| `ops.ndim` | Function | `(x)` |
| `ops.negative` | Function | `(x)` |
| `ops.nn.adaptive_average_pool` | Function | `(inputs, output_size, data_format)` |
| `ops.nn.adaptive_max_pool` | Function | `(inputs, output_size, data_format)` |
| `ops.nn.average_pool` | Function | `(inputs, pool_size, strides, padding, data_format)` |
| `ops.nn.batch_normalization` | Function | `(x, mean, variance, axis, offset, scale, epsilon)` |
| `ops.nn.binary_crossentropy` | Function | `(target, output, from_logits)` |
| `ops.nn.categorical_crossentropy` | Function | `(target, output, from_logits, axis)` |
| `ops.nn.celu` | Function | `(x, alpha)` |
| `ops.nn.conv` | Function | `(inputs, kernel, strides, padding, data_format, dilation_rate)` |
| `ops.nn.conv_transpose` | Function | `(inputs, kernel, strides, padding, output_padding, data_format, dilation_rate)` |
| `ops.nn.ctc_decode` | Function | `(inputs, sequence_lengths, strategy, beam_width, top_paths, merge_repeated, mask_index)` |
| `ops.nn.ctc_loss` | Function | `(target, output, target_length, output_length, mask_index)` |
| `ops.nn.depthwise_conv` | Function | `(inputs, kernel, strides, padding, data_format, dilation_rate)` |
| `ops.nn.dot_product_attention` | Function | `(query, key, value, bias, mask, scale, is_causal, flash_attention, attn_logits_soft_cap)` |
| `ops.nn.elu` | Function | `(x, alpha)` |
| `ops.nn.gelu` | Function | `(x, approximate)` |
| `ops.nn.glu` | Function | `(x, axis)` |
| `ops.nn.hard_shrink` | Function | `(x, threshold)` |
| `ops.nn.hard_sigmoid` | Function | `(x)` |
| `ops.nn.hard_silu` | Function | `(x)` |
| `ops.nn.hard_swish` | Function | `(x)` |
| `ops.nn.hard_tanh` | Function | `(x)` |
| `ops.nn.layer_normalization` | Function | `(x, gamma, beta, axis, epsilon, kwargs)` |
| `ops.nn.leaky_relu` | Function | `(x, negative_slope)` |
| `ops.nn.log_sigmoid` | Function | `(x)` |
| `ops.nn.log_softmax` | Function | `(x, axis)` |
| `ops.nn.max_pool` | Function | `(inputs, pool_size, strides, padding, data_format)` |
| `ops.nn.moments` | Function | `(x, axes, keepdims, synchronized)` |
| `ops.nn.multi_hot` | Function | `(inputs, num_classes, axis, dtype, sparse, kwargs)` |
| `ops.nn.normalize` | Function | `(x, axis, order, epsilon)` |
| `ops.nn.one_hot` | Function | `(x, num_classes, axis, dtype, sparse)` |
| `ops.nn.polar` | Function | `(abs_, angle)` |
| `ops.nn.psnr` | Function | `(x1, x2, max_val)` |
| `ops.nn.relu` | Function | `(x)` |
| `ops.nn.relu6` | Function | `(x)` |
| `ops.nn.rms_normalization` | Function | `(x, scale, axis, epsilon)` |
| `ops.nn.selu` | Function | `(x)` |
| `ops.nn.separable_conv` | Function | `(inputs, depthwise_kernel, pointwise_kernel, strides, padding, data_format, dilation_rate)` |
| `ops.nn.sigmoid` | Function | `(x)` |
| `ops.nn.silu` | Function | `(x)` |
| `ops.nn.soft_shrink` | Function | `(x, threshold)` |
| `ops.nn.softmax` | Function | `(x, axis)` |
| `ops.nn.softplus` | Function | `(x)` |
| `ops.nn.softsign` | Function | `(x)` |
| `ops.nn.sparse_categorical_crossentropy` | Function | `(target, output, from_logits, axis)` |
| `ops.nn.sparse_plus` | Function | `(x)` |
| `ops.nn.sparse_sigmoid` | Function | `(x)` |
| `ops.nn.sparsemax` | Function | `(x, axis)` |
| `ops.nn.squareplus` | Function | `(x, b)` |
| `ops.nn.swish` | Function | `(x)` |
| `ops.nn.tanh_shrink` | Function | `(x)` |
| `ops.nn.threshold` | Function | `(x, threshold, default_value)` |
| `ops.nn.unfold` | Function | `(x, kernel_size, dilation, padding, stride)` |
| `ops.nonzero` | Function | `(x)` |
| `ops.norm` | Function | `(x, ord, axis, keepdims)` |
| `ops.normalize` | Function | `(x, axis, order, epsilon)` |
| `ops.not_equal` | Function | `(x1, x2)` |
| `ops.numpy.abs` | Function | `(x)` |
| `ops.numpy.absolute` | Function | `(x)` |
| `ops.numpy.add` | Function | `(x1, x2)` |
| `ops.numpy.all` | Function | `(x, axis, keepdims)` |
| `ops.numpy.amax` | Function | `(x, axis, keepdims)` |
| `ops.numpy.amin` | Function | `(x, axis, keepdims)` |
| `ops.numpy.angle` | Function | `(x)` |
| `ops.numpy.any` | Function | `(x, axis, keepdims)` |
| `ops.numpy.append` | Function | `(x1, x2, axis)` |
| `ops.numpy.arange` | Function | `(start, stop, step, dtype)` |
| `ops.numpy.arccos` | Function | `(x)` |
| `ops.numpy.arccosh` | Function | `(x)` |
| `ops.numpy.arcsin` | Function | `(x)` |
| `ops.numpy.arcsinh` | Function | `(x)` |
| `ops.numpy.arctan` | Function | `(x)` |
| `ops.numpy.arctan2` | Function | `(x1, x2)` |
| `ops.numpy.arctanh` | Function | `(x)` |
| `ops.numpy.argmax` | Function | `(x, axis, keepdims)` |
| `ops.numpy.argmin` | Function | `(x, axis, keepdims)` |
| `ops.numpy.argpartition` | Function | `(x, kth, axis)` |
| `ops.numpy.argsort` | Function | `(x, axis)` |
| `ops.numpy.array` | Function | `(x, dtype)` |
| `ops.numpy.array_split` | Function | `(x, indices_or_sections, axis)` |
| `ops.numpy.average` | Function | `(x, axis, weights)` |
| `ops.numpy.bartlett` | Function | `(x)` |
| `ops.numpy.bincount` | Function | `(x, weights, minlength, sparse)` |
| `ops.numpy.bitwise_and` | Function | `(x, y)` |
| `ops.numpy.bitwise_invert` | Function | `(x)` |
| `ops.numpy.bitwise_left_shift` | Function | `(x, y)` |
| `ops.numpy.bitwise_not` | Function | `(x)` |
| `ops.numpy.bitwise_or` | Function | `(x, y)` |
| `ops.numpy.bitwise_right_shift` | Function | `(x, y)` |
| `ops.numpy.bitwise_xor` | Function | `(x, y)` |
| `ops.numpy.blackman` | Function | `(x)` |
| `ops.numpy.broadcast_to` | Function | `(x, shape)` |
| `ops.numpy.cbrt` | Function | `(x)` |
| `ops.numpy.ceil` | Function | `(x)` |
| `ops.numpy.clip` | Function | `(x, x_min, x_max)` |
| `ops.numpy.concatenate` | Function | `(xs, axis)` |
| `ops.numpy.conj` | Function | `(x)` |
| `ops.numpy.conjugate` | Function | `(x)` |
| `ops.numpy.copy` | Function | `(x)` |
| `ops.numpy.corrcoef` | Function | `(x)` |
| `ops.numpy.correlate` | Function | `(x1, x2, mode)` |
| `ops.numpy.cos` | Function | `(x)` |
| `ops.numpy.cosh` | Function | `(x)` |
| `ops.numpy.count_nonzero` | Function | `(x, axis)` |
| `ops.numpy.cross` | Function | `(x1, x2, axisa, axisb, axisc, axis)` |
| `ops.numpy.cumprod` | Function | `(x, axis, dtype)` |
| `ops.numpy.cumsum` | Function | `(x, axis, dtype)` |
| `ops.numpy.deg2rad` | Function | `(x)` |
| `ops.numpy.diag` | Function | `(x, k)` |
| `ops.numpy.diagflat` | Function | `(x, k)` |
| `ops.numpy.diagonal` | Function | `(x, offset, axis1, axis2)` |
| `ops.numpy.diff` | Function | `(a, n, axis)` |
| `ops.numpy.digitize` | Function | `(x, bins)` |
| `ops.numpy.divide` | Function | `(x1, x2)` |
| `ops.numpy.divide_no_nan` | Function | `(x1, x2)` |
| `ops.numpy.dot` | Function | `(x1, x2)` |
| `ops.numpy.einsum` | Function | `(subscripts, operands, kwargs)` |
| `ops.numpy.empty` | Function | `(shape, dtype)` |
| `ops.numpy.empty_like` | Function | `(x, dtype)` |
| `ops.numpy.equal` | Function | `(x1, x2)` |
| `ops.numpy.exp` | Function | `(x)` |
| `ops.numpy.exp2` | Function | `(x)` |
| `ops.numpy.expand_dims` | Function | `(x, axis)` |
| `ops.numpy.expm1` | Function | `(x)` |
| `ops.numpy.eye` | Function | `(N, M, k, dtype)` |
| `ops.numpy.flip` | Function | `(x, axis)` |
| `ops.numpy.floor` | Function | `(x)` |
| `ops.numpy.floor_divide` | Function | `(x1, x2)` |
| `ops.numpy.full` | Function | `(shape, fill_value, dtype)` |
| `ops.numpy.full_like` | Function | `(x, fill_value, dtype)` |
| `ops.numpy.gcd` | Function | `(x1, x2)` |
| `ops.numpy.get_item` | Function | `(x, key)` |
| `ops.numpy.greater` | Function | `(x1, x2)` |
| `ops.numpy.greater_equal` | Function | `(x1, x2)` |
| `ops.numpy.hamming` | Function | `(x)` |
| `ops.numpy.hanning` | Function | `(x)` |
| `ops.numpy.heaviside` | Function | `(x1, x2)` |
| `ops.numpy.histogram` | Function | `(x, bins, range)` |
| `ops.numpy.hstack` | Function | `(xs)` |
| `ops.numpy.hypot` | Function | `(x1, x2)` |
| `ops.numpy.identity` | Function | `(n, dtype)` |
| `ops.numpy.imag` | Function | `(x)` |
| `ops.numpy.inner` | Function | `(x1, x2)` |
| `ops.numpy.isclose` | Function | `(x1, x2, rtol, atol, equal_nan)` |
| `ops.numpy.isfinite` | Function | `(x)` |
| `ops.numpy.isin` | Function | `(x1, x2, assume_unique, invert)` |
| `ops.numpy.isinf` | Function | `(x)` |
| `ops.numpy.isnan` | Function | `(x)` |
| `ops.numpy.isneginf` | Function | `(x)` |
| `ops.numpy.isposinf` | Function | `(x)` |
| `ops.numpy.isreal` | Function | `(x)` |
| `ops.numpy.kaiser` | Function | `(x, beta)` |
| `ops.numpy.kron` | Function | `(x1, x2)` |
| `ops.numpy.lcm` | Function | `(x1, x2)` |
| `ops.numpy.ldexp` | Function | `(x1, x2)` |
| `ops.numpy.left_shift` | Function | `(x, y)` |
| `ops.numpy.less` | Function | `(x1, x2)` |
| `ops.numpy.less_equal` | Function | `(x1, x2)` |
| `ops.numpy.linspace` | Function | `(start, stop, num, endpoint, retstep, dtype, axis)` |
| `ops.numpy.log` | Function | `(x)` |
| `ops.numpy.log10` | Function | `(x)` |
| `ops.numpy.log1p` | Function | `(x)` |
| `ops.numpy.log2` | Function | `(x)` |
| `ops.numpy.logaddexp` | Function | `(x1, x2)` |
| `ops.numpy.logaddexp2` | Function | `(x1, x2)` |
| `ops.numpy.logical_and` | Function | `(x1, x2)` |
| `ops.numpy.logical_not` | Function | `(x)` |
| `ops.numpy.logical_or` | Function | `(x1, x2)` |
| `ops.numpy.logical_xor` | Function | `(x1, x2)` |
| `ops.numpy.logspace` | Function | `(start, stop, num, endpoint, base, dtype, axis)` |
| `ops.numpy.matmul` | Function | `(x1, x2)` |
| `ops.numpy.max` | Function | `(x, axis, keepdims, initial)` |
| `ops.numpy.maximum` | Function | `(x1, x2)` |
| `ops.numpy.mean` | Function | `(x, axis, keepdims)` |
| `ops.numpy.median` | Function | `(x, axis, keepdims)` |
| `ops.numpy.meshgrid` | Function | `(x, indexing)` |
| `ops.numpy.min` | Function | `(x, axis, keepdims, initial)` |
| `ops.numpy.minimum` | Function | `(x1, x2)` |
| `ops.numpy.mod` | Function | `(x1, x2)` |
| `ops.numpy.moveaxis` | Function | `(x, source, destination)` |
| `ops.numpy.multiply` | Function | `(x1, x2)` |
| `ops.numpy.nan_to_num` | Function | `(x, nan, posinf, neginf)` |
| `ops.numpy.ndim` | Function | `(x)` |
| `ops.numpy.negative` | Function | `(x)` |
| `ops.numpy.nonzero` | Function | `(x)` |
| `ops.numpy.not_equal` | Function | `(x1, x2)` |
| `ops.numpy.ones` | Function | `(shape, dtype)` |
| `ops.numpy.ones_like` | Function | `(x, dtype)` |
| `ops.numpy.outer` | Function | `(x1, x2)` |
| `ops.numpy.pad` | Function | `(x, pad_width, mode, constant_values)` |
| `ops.numpy.power` | Function | `(x1, x2)` |
| `ops.numpy.prod` | Function | `(x, axis, keepdims, dtype)` |
| `ops.numpy.quantile` | Function | `(x, q, axis, method, keepdims)` |
| `ops.numpy.ravel` | Function | `(x)` |
| `ops.numpy.real` | Function | `(x)` |
| `ops.numpy.reciprocal` | Function | `(x)` |
| `ops.numpy.repeat` | Function | `(x, repeats, axis)` |
| `ops.numpy.reshape` | Function | `(x, newshape)` |
| `ops.numpy.right_shift` | Function | `(x, y)` |
| `ops.numpy.roll` | Function | `(x, shift, axis)` |
| `ops.numpy.rot90` | Function | `(array, k, axes)` |
| `ops.numpy.round` | Function | `(x, decimals)` |
| `ops.numpy.searchsorted` | Function | `(sorted_sequence, values, side)` |
| `ops.numpy.select` | Function | `(condlist, choicelist, default)` |
| `ops.numpy.sign` | Function | `(x)` |
| `ops.numpy.signbit` | Function | `(x)` |
| `ops.numpy.sin` | Function | `(x)` |
| `ops.numpy.sinh` | Function | `(x)` |
| `ops.numpy.size` | Function | `(x)` |
| `ops.numpy.slogdet` | Function | `(x)` |
| `ops.numpy.sort` | Function | `(x, axis)` |
| `ops.numpy.split` | Function | `(x, indices_or_sections, axis)` |
| `ops.numpy.sqrt` | Function | `(x)` |
| `ops.numpy.square` | Function | `(x)` |
| `ops.numpy.squeeze` | Function | `(x, axis)` |
| `ops.numpy.stack` | Function | `(x, axis)` |
| `ops.numpy.std` | Function | `(x, axis, keepdims)` |
| `ops.numpy.subtract` | Function | `(x1, x2)` |
| `ops.numpy.sum` | Function | `(x, axis, keepdims)` |
| `ops.numpy.swapaxes` | Function | `(x, axis1, axis2)` |
| `ops.numpy.take` | Function | `(x, indices, axis)` |
| `ops.numpy.take_along_axis` | Function | `(x, indices, axis)` |
| `ops.numpy.tan` | Function | `(x)` |
| `ops.numpy.tanh` | Function | `(x)` |
| `ops.numpy.tensordot` | Function | `(x1, x2, axes)` |
| `ops.numpy.tile` | Function | `(x, repeats)` |
| `ops.numpy.trace` | Function | `(x, offset, axis1, axis2)` |
| `ops.numpy.transpose` | Function | `(x, axes)` |
| `ops.numpy.trapezoid` | Function | `(y, x, dx, axis)` |
| `ops.numpy.tri` | Function | `(N, M, k, dtype)` |
| `ops.numpy.tril` | Function | `(x, k)` |
| `ops.numpy.triu` | Function | `(x, k)` |
| `ops.numpy.true_divide` | Function | `(x1, x2)` |
| `ops.numpy.trunc` | Function | `(x)` |
| `ops.numpy.unravel_index` | Function | `(indices, shape)` |
| `ops.numpy.vander` | Function | `(x, N, increasing)` |
| `ops.numpy.var` | Function | `(x, axis, keepdims)` |
| `ops.numpy.vdot` | Function | `(x1, x2)` |
| `ops.numpy.vectorize` | Function | `(pyfunc, excluded, signature)` |
| `ops.numpy.view` | Function | `(x, dtype)` |
| `ops.numpy.vstack` | Function | `(xs)` |
| `ops.numpy.where` | Function | `(condition, x1, x2)` |
| `ops.numpy.zeros` | Function | `(shape, dtype)` |
| `ops.numpy.zeros_like` | Function | `(x, dtype)` |
| `ops.one_hot` | Function | `(x, num_classes, axis, dtype, sparse)` |
| `ops.ones` | Function | `(shape, dtype)` |
| `ops.ones_like` | Function | `(x, dtype)` |
| `ops.outer` | Function | `(x1, x2)` |
| `ops.pad` | Function | `(x, pad_width, mode, constant_values)` |
| `ops.polar` | Function | `(abs_, angle)` |
| `ops.power` | Function | `(x1, x2)` |
| `ops.prod` | Function | `(x, axis, keepdims, dtype)` |
| `ops.psnr` | Function | `(x1, x2, max_val)` |
| `ops.qr` | Function | `(x, mode)` |
| `ops.quantile` | Function | `(x, q, axis, method, keepdims)` |
| `ops.ravel` | Function | `(x)` |
| `ops.real` | Function | `(x)` |
| `ops.rearrange` | Function | `(tensor, pattern, axes_lengths)` |
| `ops.reciprocal` | Function | `(x)` |
| `ops.relu` | Function | `(x)` |
| `ops.relu6` | Function | `(x)` |
| `ops.repeat` | Function | `(x, repeats, axis)` |
| `ops.reshape` | Function | `(x, newshape)` |
| `ops.rfft` | Function | `(x, fft_length)` |
| `ops.right_shift` | Function | `(x, y)` |
| `ops.rms_normalization` | Function | `(x, scale, axis, epsilon)` |
| `ops.roll` | Function | `(x, shift, axis)` |
| `ops.rot90` | Function | `(array, k, axes)` |
| `ops.round` | Function | `(x, decimals)` |
| `ops.rsqrt` | Function | `(x)` |
| `ops.saturate_cast` | Function | `(x, dtype)` |
| `ops.scan` | Function | `(f, init, xs, length, reverse, unroll)` |
| `ops.scatter` | Function | `(indices, values, shape)` |
| `ops.scatter_update` | Function | `(inputs, indices, updates)` |
| `ops.searchsorted` | Function | `(sorted_sequence, values, side)` |
| `ops.segment_max` | Function | `(data, segment_ids, num_segments, sorted)` |
| `ops.segment_sum` | Function | `(data, segment_ids, num_segments, sorted)` |
| `ops.select` | Function | `(condlist, choicelist, default)` |
| `ops.selu` | Function | `(x)` |
| `ops.separable_conv` | Function | `(inputs, depthwise_kernel, pointwise_kernel, strides, padding, data_format, dilation_rate)` |
| `ops.shape` | Function | `(x)` |
| `ops.sigmoid` | Function | `(x)` |
| `ops.sign` | Function | `(x)` |
| `ops.signbit` | Function | `(x)` |
| `ops.silu` | Function | `(x)` |
| `ops.sin` | Function | `(x)` |
| `ops.sinh` | Function | `(x)` |
| `ops.size` | Function | `(x)` |
| `ops.slice` | Function | `(inputs, start_indices, shape)` |
| `ops.slice_update` | Function | `(inputs, start_indices, updates)` |
| `ops.slogdet` | Function | `(x)` |
| `ops.soft_shrink` | Function | `(x, threshold)` |
| `ops.softmax` | Function | `(x, axis)` |
| `ops.softplus` | Function | `(x)` |
| `ops.softsign` | Function | `(x)` |
| `ops.solve` | Function | `(a, b)` |
| `ops.solve_triangular` | Function | `(a, b, lower)` |
| `ops.sort` | Function | `(x, axis)` |
| `ops.sparse_categorical_crossentropy` | Function | `(target, output, from_logits, axis)` |
| `ops.sparse_plus` | Function | `(x)` |
| `ops.sparse_sigmoid` | Function | `(x)` |
| `ops.sparsemax` | Function | `(x, axis)` |
| `ops.split` | Function | `(x, indices_or_sections, axis)` |
| `ops.sqrt` | Function | `(x)` |
| `ops.square` | Function | `(x)` |
| `ops.squareplus` | Function | `(x, b)` |
| `ops.squeeze` | Function | `(x, axis)` |
| `ops.stack` | Function | `(x, axis)` |
| `ops.std` | Function | `(x, axis, keepdims)` |
| `ops.stft` | Function | `(x, sequence_length, sequence_stride, fft_length, window, center)` |
| `ops.stop_gradient` | Function | `(variable)` |
| `ops.subtract` | Function | `(x1, x2)` |
| `ops.sum` | Function | `(x, axis, keepdims)` |
| `ops.svd` | Function | `(x, full_matrices, compute_uv)` |
| `ops.swapaxes` | Function | `(x, axis1, axis2)` |
| `ops.swish` | Function | `(x)` |
| `ops.switch` | Function | `(index, branches, operands)` |
| `ops.take` | Function | `(x, indices, axis)` |
| `ops.take_along_axis` | Function | `(x, indices, axis)` |
| `ops.tan` | Function | `(x)` |
| `ops.tanh` | Function | `(x)` |
| `ops.tanh_shrink` | Function | `(x)` |
| `ops.tensordot` | Function | `(x1, x2, axes)` |
| `ops.threshold` | Function | `(x, threshold, default_value)` |
| `ops.tile` | Function | `(x, repeats)` |
| `ops.top_k` | Function | `(x, k, sorted)` |
| `ops.trace` | Function | `(x, offset, axis1, axis2)` |
| `ops.transpose` | Function | `(x, axes)` |
| `ops.trapezoid` | Function | `(y, x, dx, axis)` |
| `ops.tri` | Function | `(N, M, k, dtype)` |
| `ops.tril` | Function | `(x, k)` |
| `ops.triu` | Function | `(x, k)` |
| `ops.true_divide` | Function | `(x1, x2)` |
| `ops.trunc` | Function | `(x)` |
| `ops.unfold` | Function | `(x, kernel_size, dilation, padding, stride)` |
| `ops.unravel_index` | Function | `(indices, shape)` |
| `ops.unstack` | Function | `(x, num, axis)` |
| `ops.vander` | Function | `(x, N, increasing)` |
| `ops.var` | Function | `(x, axis, keepdims)` |
| `ops.vdot` | Function | `(x1, x2)` |
| `ops.vectorize` | Function | `(pyfunc, excluded, signature)` |
| `ops.vectorized_map` | Function | `(function, elements)` |
| `ops.view` | Function | `(x, dtype)` |
| `ops.view_as_complex` | Function | `(x)` |
| `ops.view_as_real` | Function | `(x)` |
| `ops.vstack` | Function | `(xs)` |
| `ops.where` | Function | `(condition, x1, x2)` |
| `ops.while_loop` | Function | `(cond, body, loop_vars, maximum_iterations)` |
| `ops.zeros` | Function | `(shape, dtype)` |
| `ops.zeros_like` | Function | `(x, dtype)` |
| `optimizers.Adadelta` | Class | `(learning_rate, rho, epsilon, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps, name, kwargs)` |
| `optimizers.Adafactor` | Class | `(learning_rate, beta_2_decay, epsilon_1, epsilon_2, clip_threshold, relative_step, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps, name, kwargs)` |
| `optimizers.Adagrad` | Class | `(learning_rate, initial_accumulator_value, epsilon, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps, name, kwargs)` |
| `optimizers.Adam` | Class | `(learning_rate, beta_1, beta_2, epsilon, amsgrad, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps, name, kwargs)` |
| `optimizers.AdamW` | Class | `(learning_rate, weight_decay, beta_1, beta_2, epsilon, amsgrad, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps, name, kwargs)` |
| `optimizers.Adamax` | Class | `(learning_rate, beta_1, beta_2, epsilon, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps, name, kwargs)` |
| `optimizers.Ftrl` | Class | `(learning_rate, learning_rate_power, initial_accumulator_value, l1_regularization_strength, l2_regularization_strength, l2_shrinkage_regularization_strength, beta, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps, name, kwargs)` |
| `optimizers.Lamb` | Class | `(learning_rate, beta_1, beta_2, epsilon, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps, name, kwargs)` |
| `optimizers.Lion` | Class | `(learning_rate, beta_1, beta_2, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps, name, kwargs)` |
| `optimizers.LossScaleOptimizer` | Class | `(inner_optimizer, initial_scale, dynamic_growth_steps, name, kwargs)` |
| `optimizers.Muon` | Class | `(learning_rate, adam_beta_1, adam_beta_2, adam_weight_decay, epsilon, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps, name, exclude_layers, exclude_embeddings, muon_a, muon_b, muon_c, adam_lr_ratio, momentum, ns_steps, nesterov, rms_rate, kwargs)` |
| `optimizers.Nadam` | Class | `(learning_rate, beta_1, beta_2, epsilon, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps, name, kwargs)` |
| `optimizers.Optimizer` | Class | `(...)` |
| `optimizers.RMSprop` | Class | `(learning_rate, rho, momentum, epsilon, centered, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps, name, kwargs)` |
| `optimizers.SGD` | Class | `(learning_rate, momentum, nesterov, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps, name, kwargs)` |
| `optimizers.deserialize` | Function | `(config, custom_objects)` |
| `optimizers.get` | Function | `(identifier)` |
| `optimizers.legacy.Adagrad` | Class | `(args, kwargs)` |
| `optimizers.legacy.Adam` | Class | `(args, kwargs)` |
| `optimizers.legacy.Ftrl` | Class | `(args, kwargs)` |
| `optimizers.legacy.Optimizer` | Class | `(args, kwargs)` |
| `optimizers.legacy.RMSprop` | Class | `(args, kwargs)` |
| `optimizers.legacy.SGD` | Class | `(args, kwargs)` |
| `optimizers.schedules.CosineDecay` | Class | `(initial_learning_rate, decay_steps, alpha, name, warmup_target, warmup_steps)` |
| `optimizers.schedules.CosineDecayRestarts` | Class | `(initial_learning_rate, first_decay_steps, t_mul, m_mul, alpha, name)` |
| `optimizers.schedules.ExponentialDecay` | Class | `(initial_learning_rate, decay_steps, decay_rate, staircase, name)` |
| `optimizers.schedules.InverseTimeDecay` | Class | `(initial_learning_rate, decay_steps, decay_rate, staircase, name)` |
| `optimizers.schedules.LearningRateSchedule` | Class | `(...)` |
| `optimizers.schedules.PiecewiseConstantDecay` | Class | `(boundaries, values, name)` |
| `optimizers.schedules.PolynomialDecay` | Class | `(initial_learning_rate, decay_steps, end_learning_rate, power, cycle, name)` |
| `optimizers.schedules.deserialize` | Function | `(config, custom_objects)` |
| `optimizers.schedules.serialize` | Function | `(learning_rate_schedule)` |
| `optimizers.serialize` | Function | `(optimizer)` |
| `preprocessing.image.array_to_img` | Function | `(x, data_format, scale, dtype)` |
| `preprocessing.image.img_to_array` | Function | `(img, data_format, dtype)` |
| `preprocessing.image.load_img` | Function | `(path, color_mode, target_size, interpolation, keep_aspect_ratio)` |
| `preprocessing.image.save_img` | Function | `(path, x, data_format, file_format, scale, kwargs)` |
| `preprocessing.image.smart_resize` | Function | `(x, size, interpolation, data_format, backend_module)` |
| `preprocessing.image_dataset_from_directory` | Function | `(directory, labels, label_mode, class_names, color_mode, batch_size, image_size, shuffle, seed, validation_split, subset, interpolation, follow_links, crop_to_aspect_ratio, pad_to_aspect_ratio, data_format, format, verbose)` |
| `preprocessing.sequence.pad_sequences` | Function | `(sequences, maxlen, dtype, padding, truncating, value)` |
| `preprocessing.text_dataset_from_directory` | Function | `(directory, labels, label_mode, class_names, batch_size, max_length, shuffle, seed, validation_split, subset, follow_links, format, verbose)` |
| `preprocessing.timeseries_dataset_from_array` | Function | `(data, targets, sequence_length, sequence_stride, sampling_rate, batch_size, shuffle, seed, start_index, end_index)` |
| `quantizers.AbsMaxQuantizer` | Class | `(axis, value_range, epsilon, output_dtype)` |
| `quantizers.Float8QuantizationConfig` | Class | `()` |
| `quantizers.GPTQConfig` | Class | `(dataset, tokenizer, weight_bits: int, num_samples: int, per_channel: bool, sequence_length: int, hessian_damping: float, group_size: int, symmetric: bool, activation_order: bool, quantization_layer_structure: dict)` |
| `quantizers.Int4QuantizationConfig` | Class | `(weight_quantizer, activation_quantizer)` |
| `quantizers.Int8QuantizationConfig` | Class | `(weight_quantizer, activation_quantizer)` |
| `quantizers.QuantizationConfig` | Class | `(weight_quantizer, activation_quantizer)` |
| `quantizers.Quantizer` | Class | `(output_dtype)` |
| `quantizers.abs_max_quantize` | Function | `(inputs, axis, value_range, dtype, epsilon, to_numpy)` |
| `quantizers.compute_float8_amax_history` | Function | `(x, amax_history)` |
| `quantizers.compute_float8_scale` | Function | `(amax, scale, dtype_max, margin)` |
| `quantizers.deserialize` | Function | `(config, custom_objects)` |
| `quantizers.fake_quant_with_min_max_vars` | Function | `(inputs, min_vals, max_vals, num_bits, narrow_range, axis)` |
| `quantizers.get` | Function | `(identifier, kwargs)` |
| `quantizers.pack_int4` | Function | `(arr, axis, dtype)` |
| `quantizers.quantize_and_dequantize` | Function | `(inputs, scale, quantized_dtype, compute_dtype)` |
| `quantizers.serialize` | Function | `(initializer)` |
| `quantizers.unpack_int4` | Function | `(packed, orig_len, axis, dtype)` |
| `random.SeedGenerator` | Class | `(seed, name, kwargs)` |
| `random.beta` | Function | `(shape, alpha, beta, dtype, seed)` |
| `random.binomial` | Function | `(shape, counts, probabilities, dtype, seed)` |
| `random.categorical` | Function | `(logits, num_samples, dtype, seed)` |
| `random.dropout` | Function | `(inputs, rate, noise_shape, seed)` |
| `random.gamma` | Function | `(shape, alpha, dtype, seed)` |
| `random.normal` | Function | `(shape, mean, stddev, dtype, seed)` |
| `random.randint` | Function | `(shape, minval, maxval, dtype, seed)` |
| `random.shuffle` | Function | `(x, axis, seed)` |
| `random.truncated_normal` | Function | `(shape, mean, stddev, dtype, seed)` |
| `random.uniform` | Function | `(shape, minval, maxval, dtype, seed)` |
| `regularizers.L1` | Class | `(l1)` |
| `regularizers.L1L2` | Class | `(l1, l2)` |
| `regularizers.L2` | Class | `(l2)` |
| `regularizers.OrthogonalRegularizer` | Class | `(factor, mode)` |
| `regularizers.Regularizer` | Class | `(...)` |
| `regularizers.deserialize` | Function | `(config, custom_objects)` |
| `regularizers.get` | Function | `(identifier)` |
| `regularizers.l1` | Class | `(l1)` |
| `regularizers.l1_l2` | Class | `(l1, l2)` |
| `regularizers.l2` | Class | `(l2)` |
| `regularizers.orthogonal_regularizer` | Class | `(factor, mode)` |
| `regularizers.serialize` | Function | `(regularizer)` |
| `remat` | Function | `(f)` |
| `saving.CustomObjectScope` | Class | `(custom_objects)` |
| `saving.KerasFileEditor` | Class | `(filepath)` |
| `saving.custom_object_scope` | Class | `(custom_objects)` |
| `saving.deserialize_keras_object` | Function | `(config, custom_objects, safe_mode, kwargs)` |
| `saving.get_custom_objects` | Function | `()` |
| `saving.get_registered_name` | Function | `(obj)` |
| `saving.get_registered_object` | Function | `(name, custom_objects, module_objects)` |
| `saving.load_model` | Function | `(filepath, custom_objects, compile, safe_mode)` |
| `saving.load_weights` | Function | `(model, filepath, skip_mismatch, kwargs)` |
| `saving.register_keras_serializable` | Function | `(package, name)` |
| `saving.save_model` | Function | `(model, filepath, overwrite, zipped, kwargs)` |
| `saving.save_weights` | Function | `(model, filepath, overwrite, max_shard_size, kwargs)` |
| `saving.serialize_keras_object` | Function | `(obj)` |
| `src.Functional` | Class | `(inputs, outputs, name, kwargs)` |
| `src.Input` | Function | `(shape, batch_size, dtype, sparse, ragged, batch_shape, name, tensor, optional)` |
| `src.KerasTensor` | Class | `(shape, dtype, sparse, ragged, record_history, name, kwargs)` |
| `src.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.Model` | Class | `(args, kwargs)` |
| `src.Sequential` | Class | `(layers, trainable, name)` |
| `src.activations.ALL_OBJECTS` | Object | `` |
| `src.activations.ALL_OBJECTS_DICT` | Object | `` |
| `src.activations.activations.Mish` | Class | `(...)` |
| `src.activations.activations.ReLU` | Class | `(negative_slope, max_value, threshold, name)` |
| `src.activations.activations.backend` | Object | `` |
| `src.activations.activations.celu` | Function | `(x, alpha)` |
| `src.activations.activations.elu` | Function | `(x, alpha)` |
| `src.activations.activations.exponential` | Function | `(x)` |
| `src.activations.activations.gelu` | Function | `(x, approximate)` |
| `src.activations.activations.glu` | Function | `(x, axis)` |
| `src.activations.activations.hard_shrink` | Function | `(x, threshold)` |
| `src.activations.activations.hard_sigmoid` | Function | `(x)` |
| `src.activations.activations.hard_silu` | Function | `(x)` |
| `src.activations.activations.hard_tanh` | Function | `(x)` |
| `src.activations.activations.keras_export` | Class | `(path)` |
| `src.activations.activations.leaky_relu` | Function | `(x, negative_slope)` |
| `src.activations.activations.linear` | Function | `(x)` |
| `src.activations.activations.log_sigmoid` | Function | `(x)` |
| `src.activations.activations.log_softmax` | Function | `(x, axis)` |
| `src.activations.activations.mish` | Function | `(x)` |
| `src.activations.activations.ops` | Object | `` |
| `src.activations.activations.relu` | Function | `(x, negative_slope, max_value, threshold)` |
| `src.activations.activations.relu6` | Function | `(x)` |
| `src.activations.activations.selu` | Function | `(x)` |
| `src.activations.activations.sigmoid` | Function | `(x)` |
| `src.activations.activations.silu` | Function | `(x)` |
| `src.activations.activations.soft_shrink` | Function | `(x, threshold)` |
| `src.activations.activations.softmax` | Function | `(x, axis)` |
| `src.activations.activations.softplus` | Function | `(x)` |
| `src.activations.activations.softsign` | Function | `(x)` |
| `src.activations.activations.sparse_plus` | Function | `(x)` |
| `src.activations.activations.sparse_sigmoid` | Function | `(x)` |
| `src.activations.activations.sparsemax` | Function | `(x, axis)` |
| `src.activations.activations.squareplus` | Function | `(x, b)` |
| `src.activations.activations.tanh` | Function | `(x)` |
| `src.activations.activations.tanh_shrink` | Function | `(x)` |
| `src.activations.activations.threshold` | Function | `(x, threshold, default_value)` |
| `src.activations.celu` | Function | `(x, alpha)` |
| `src.activations.deserialize` | Function | `(config, custom_objects)` |
| `src.activations.elu` | Function | `(x, alpha)` |
| `src.activations.exponential` | Function | `(x)` |
| `src.activations.gelu` | Function | `(x, approximate)` |
| `src.activations.get` | Function | `(identifier)` |
| `src.activations.glu` | Function | `(x, axis)` |
| `src.activations.hard_shrink` | Function | `(x, threshold)` |
| `src.activations.hard_sigmoid` | Function | `(x)` |
| `src.activations.hard_silu` | Function | `(x)` |
| `src.activations.hard_tanh` | Function | `(x)` |
| `src.activations.keras_export` | Class | `(path)` |
| `src.activations.leaky_relu` | Function | `(x, negative_slope)` |
| `src.activations.linear` | Function | `(x)` |
| `src.activations.log_sigmoid` | Function | `(x)` |
| `src.activations.log_softmax` | Function | `(x, axis)` |
| `src.activations.mish` | Function | `(x)` |
| `src.activations.object_registration` | Object | `` |
| `src.activations.relu` | Function | `(x, negative_slope, max_value, threshold)` |
| `src.activations.relu6` | Function | `(x)` |
| `src.activations.selu` | Function | `(x)` |
| `src.activations.serialization_lib` | Object | `` |
| `src.activations.serialize` | Function | `(activation)` |
| `src.activations.sigmoid` | Function | `(x)` |
| `src.activations.silu` | Function | `(x)` |
| `src.activations.soft_shrink` | Function | `(x, threshold)` |
| `src.activations.softmax` | Function | `(x, axis)` |
| `src.activations.softplus` | Function | `(x)` |
| `src.activations.softsign` | Function | `(x)` |
| `src.activations.sparse_plus` | Function | `(x)` |
| `src.activations.sparse_sigmoid` | Function | `(x)` |
| `src.activations.sparsemax` | Function | `(x, axis)` |
| `src.activations.squareplus` | Function | `(x, b)` |
| `src.activations.tanh` | Function | `(x)` |
| `src.activations.tanh_shrink` | Function | `(x)` |
| `src.activations.threshold` | Function | `(x, threshold, default_value)` |
| `src.api_export.REGISTERED_NAMES_TO_OBJS` | Object | `` |
| `src.api_export.REGISTERED_OBJS_TO_NAMES` | Object | `` |
| `src.api_export.get_name_from_symbol` | Function | `(symbol)` |
| `src.api_export.get_symbol_from_name` | Function | `(name)` |
| `src.api_export.keras_export` | Class | `(path)` |
| `src.api_export.register_internal_serializable` | Function | `(path, symbol)` |
| `src.applications.convnext.BASE_DOCSTRING` | Object | `` |
| `src.applications.convnext.BASE_WEIGHTS_PATH` | Object | `` |
| `src.applications.convnext.ConvNeXt` | Function | `(depths, projection_dims, drop_path_rate, layer_scale_init_value, default_size, name, include_preprocessing, include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, weights_name)` |
| `src.applications.convnext.ConvNeXtBase` | Function | `(include_top, include_preprocessing, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `src.applications.convnext.ConvNeXtBlock` | Function | `(projection_dim, drop_path_rate, layer_scale_init_value, name)` |
| `src.applications.convnext.ConvNeXtLarge` | Function | `(include_top, include_preprocessing, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `src.applications.convnext.ConvNeXtSmall` | Function | `(include_top, include_preprocessing, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `src.applications.convnext.ConvNeXtTiny` | Function | `(include_top, include_preprocessing, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `src.applications.convnext.ConvNeXtXLarge` | Function | `(include_top, include_preprocessing, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `src.applications.convnext.Functional` | Class | `(inputs, outputs, name, kwargs)` |
| `src.applications.convnext.Head` | Function | `(num_classes, classifier_activation, name)` |
| `src.applications.convnext.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.applications.convnext.LayerScale` | Class | `(init_values, projection_dim, kwargs)` |
| `src.applications.convnext.MODEL_CONFIGS` | Object | `` |
| `src.applications.convnext.PreStem` | Function | `(name)` |
| `src.applications.convnext.Sequential` | Class | `(layers, trainable, name)` |
| `src.applications.convnext.StochasticDepth` | Class | `(drop_path_rate, kwargs)` |
| `src.applications.convnext.WEIGHTS_HASHES` | Object | `` |
| `src.applications.convnext.backend` | Object | `` |
| `src.applications.convnext.decode_predictions` | Function | `(preds, top)` |
| `src.applications.convnext.file_utils` | Object | `` |
| `src.applications.convnext.imagenet_utils` | Object | `` |
| `src.applications.convnext.initializers` | Object | `` |
| `src.applications.convnext.keras_export` | Class | `(path)` |
| `src.applications.convnext.layers` | Object | `` |
| `src.applications.convnext.operation_utils` | Object | `` |
| `src.applications.convnext.ops` | Object | `` |
| `src.applications.convnext.preprocess_input` | Function | `(x, data_format)` |
| `src.applications.convnext.random` | Object | `` |
| `src.applications.densenet.BASE_WEIGHTS_PATH` | Object | `` |
| `src.applications.densenet.DENSENET121_WEIGHT_PATH` | Object | `` |
| `src.applications.densenet.DENSENET121_WEIGHT_PATH_NO_TOP` | Object | `` |
| `src.applications.densenet.DENSENET169_WEIGHT_PATH` | Object | `` |
| `src.applications.densenet.DENSENET169_WEIGHT_PATH_NO_TOP` | Object | `` |
| `src.applications.densenet.DENSENET201_WEIGHT_PATH` | Object | `` |
| `src.applications.densenet.DENSENET201_WEIGHT_PATH_NO_TOP` | Object | `` |
| `src.applications.densenet.DOC` | Object | `` |
| `src.applications.densenet.DenseNet` | Function | `(blocks, include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `src.applications.densenet.DenseNet121` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `src.applications.densenet.DenseNet169` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `src.applications.densenet.DenseNet201` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `src.applications.densenet.Functional` | Class | `(inputs, outputs, name, kwargs)` |
| `src.applications.densenet.backend` | Object | `` |
| `src.applications.densenet.conv_block` | Function | `(x, growth_rate, name)` |
| `src.applications.densenet.decode_predictions` | Function | `(preds, top)` |
| `src.applications.densenet.dense_block` | Function | `(x, blocks, name)` |
| `src.applications.densenet.file_utils` | Object | `` |
| `src.applications.densenet.imagenet_utils` | Object | `` |
| `src.applications.densenet.keras_export` | Class | `(path)` |
| `src.applications.densenet.layers` | Object | `` |
| `src.applications.densenet.operation_utils` | Object | `` |
| `src.applications.densenet.preprocess_input` | Function | `(x, data_format)` |
| `src.applications.densenet.transition_block` | Function | `(x, reduction, name)` |
| `src.applications.efficientnet.BASE_DOCSTRING` | Object | `` |
| `src.applications.efficientnet.BASE_WEIGHTS_PATH` | Object | `` |
| `src.applications.efficientnet.CONV_KERNEL_INITIALIZER` | Object | `` |
| `src.applications.efficientnet.DEFAULT_BLOCKS_ARGS` | Object | `` |
| `src.applications.efficientnet.DENSE_KERNEL_INITIALIZER` | Object | `` |
| `src.applications.efficientnet.EfficientNet` | Function | `(width_coefficient, depth_coefficient, default_size, dropout_rate, drop_connect_rate, depth_divisor, activation, blocks_args, name, include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, weights_name)` |
| `src.applications.efficientnet.EfficientNetB0` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `src.applications.efficientnet.EfficientNetB1` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `src.applications.efficientnet.EfficientNetB2` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `src.applications.efficientnet.EfficientNetB3` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `src.applications.efficientnet.EfficientNetB4` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `src.applications.efficientnet.EfficientNetB5` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `src.applications.efficientnet.EfficientNetB6` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `src.applications.efficientnet.EfficientNetB7` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `src.applications.efficientnet.Functional` | Class | `(inputs, outputs, name, kwargs)` |
| `src.applications.efficientnet.IMAGENET_STDDEV_RGB` | Object | `` |
| `src.applications.efficientnet.WEIGHTS_HASHES` | Object | `` |
| `src.applications.efficientnet.backend` | Object | `` |
| `src.applications.efficientnet.block` | Function | `(inputs, activation, drop_rate, name, filters_in, filters_out, kernel_size, strides, expand_ratio, se_ratio, id_skip)` |
| `src.applications.efficientnet.decode_predictions` | Function | `(preds, top)` |
| `src.applications.efficientnet.file_utils` | Object | `` |
| `src.applications.efficientnet.imagenet_utils` | Object | `` |
| `src.applications.efficientnet.keras_export` | Class | `(path)` |
| `src.applications.efficientnet.layers` | Object | `` |
| `src.applications.efficientnet.operation_utils` | Object | `` |
| `src.applications.efficientnet.preprocess_input` | Function | `(x, data_format)` |
| `src.applications.efficientnet_v2.BASE_DOCSTRING` | Object | `` |
| `src.applications.efficientnet_v2.BASE_WEIGHTS_PATH` | Object | `` |
| `src.applications.efficientnet_v2.CONV_KERNEL_INITIALIZER` | Object | `` |
| `src.applications.efficientnet_v2.DEFAULT_BLOCKS_ARGS` | Object | `` |
| `src.applications.efficientnet_v2.DENSE_KERNEL_INITIALIZER` | Object | `` |
| `src.applications.efficientnet_v2.EfficientNetV2` | Function | `(width_coefficient, depth_coefficient, default_size, dropout_rate, drop_connect_rate, depth_divisor, min_depth, bn_momentum, activation, blocks_args, name, include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, include_preprocessing, weights_name)` |
| `src.applications.efficientnet_v2.EfficientNetV2B0` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, include_preprocessing, name)` |
| `src.applications.efficientnet_v2.EfficientNetV2B1` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, include_preprocessing, name)` |
| `src.applications.efficientnet_v2.EfficientNetV2B2` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, include_preprocessing, name)` |
| `src.applications.efficientnet_v2.EfficientNetV2B3` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, include_preprocessing, name)` |
| `src.applications.efficientnet_v2.EfficientNetV2L` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, include_preprocessing, name)` |
| `src.applications.efficientnet_v2.EfficientNetV2M` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, include_preprocessing, name)` |
| `src.applications.efficientnet_v2.EfficientNetV2S` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, include_preprocessing, name)` |
| `src.applications.efficientnet_v2.Functional` | Class | `(inputs, outputs, name, kwargs)` |
| `src.applications.efficientnet_v2.FusedMBConvBlock` | Function | `(input_filters, output_filters, expand_ratio, kernel_size, strides, se_ratio, bn_momentum, activation, survival_probability, name)` |
| `src.applications.efficientnet_v2.MBConvBlock` | Function | `(input_filters, output_filters, expand_ratio, kernel_size, strides, se_ratio, bn_momentum, activation, survival_probability, name)` |
| `src.applications.efficientnet_v2.WEIGHTS_HASHES` | Object | `` |
| `src.applications.efficientnet_v2.backend` | Object | `` |
| `src.applications.efficientnet_v2.decode_predictions` | Function | `(preds, top)` |
| `src.applications.efficientnet_v2.file_utils` | Object | `` |
| `src.applications.efficientnet_v2.imagenet_utils` | Object | `` |
| `src.applications.efficientnet_v2.initializers` | Object | `` |
| `src.applications.efficientnet_v2.keras_export` | Class | `(path)` |
| `src.applications.efficientnet_v2.layers` | Object | `` |
| `src.applications.efficientnet_v2.operation_utils` | Object | `` |
| `src.applications.efficientnet_v2.preprocess_input` | Function | `(x, data_format)` |
| `src.applications.efficientnet_v2.round_filters` | Function | `(filters, width_coefficient, min_depth, depth_divisor)` |
| `src.applications.efficientnet_v2.round_repeats` | Function | `(repeats, depth_coefficient)` |
| `src.applications.imagenet_utils.CLASS_INDEX` | Object | `` |
| `src.applications.imagenet_utils.CLASS_INDEX_PATH` | Object | `` |
| `src.applications.imagenet_utils.PREPROCESS_INPUT_DEFAULT_ERROR_DOC` | Object | `` |
| `src.applications.imagenet_utils.PREPROCESS_INPUT_DOC` | Object | `` |
| `src.applications.imagenet_utils.PREPROCESS_INPUT_ERROR_DOC` | Object | `` |
| `src.applications.imagenet_utils.PREPROCESS_INPUT_MODE_DOC` | Object | `` |
| `src.applications.imagenet_utils.PREPROCESS_INPUT_RET_DOC_CAFFE` | Object | `` |
| `src.applications.imagenet_utils.PREPROCESS_INPUT_RET_DOC_TF` | Object | `` |
| `src.applications.imagenet_utils.PREPROCESS_INPUT_RET_DOC_TORCH` | Object | `` |
| `src.applications.imagenet_utils.activations` | Object | `` |
| `src.applications.imagenet_utils.backend` | Object | `` |
| `src.applications.imagenet_utils.correct_pad` | Function | `(inputs, kernel_size)` |
| `src.applications.imagenet_utils.decode_predictions` | Function | `(preds, top)` |
| `src.applications.imagenet_utils.file_utils` | Object | `` |
| `src.applications.imagenet_utils.keras_export` | Class | `(path)` |
| `src.applications.imagenet_utils.obtain_input_shape` | Function | `(input_shape, default_size, min_size, data_format, require_flatten, weights)` |
| `src.applications.imagenet_utils.ops` | Object | `` |
| `src.applications.imagenet_utils.preprocess_input` | Function | `(x, data_format, mode)` |
| `src.applications.imagenet_utils.validate_activation` | Function | `(classifier_activation, weights)` |
| `src.applications.inception_resnet_v2.BASE_WEIGHT_URL` | Object | `` |
| `src.applications.inception_resnet_v2.CustomScaleLayer` | Class | `(scale, kwargs)` |
| `src.applications.inception_resnet_v2.Functional` | Class | `(inputs, outputs, name, kwargs)` |
| `src.applications.inception_resnet_v2.InceptionResNetV2` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `src.applications.inception_resnet_v2.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.applications.inception_resnet_v2.backend` | Object | `` |
| `src.applications.inception_resnet_v2.conv2d_bn` | Function | `(x, filters, kernel_size, strides, padding, activation, use_bias, name)` |
| `src.applications.inception_resnet_v2.decode_predictions` | Function | `(preds, top)` |
| `src.applications.inception_resnet_v2.file_utils` | Object | `` |
| `src.applications.inception_resnet_v2.imagenet_utils` | Object | `` |
| `src.applications.inception_resnet_v2.inception_resnet_block` | Function | `(x, scale, block_type, block_idx, activation)` |
| `src.applications.inception_resnet_v2.keras_export` | Class | `(path)` |
| `src.applications.inception_resnet_v2.layers` | Object | `` |
| `src.applications.inception_resnet_v2.operation_utils` | Object | `` |
| `src.applications.inception_resnet_v2.preprocess_input` | Function | `(x, data_format)` |
| `src.applications.inception_v3.Functional` | Class | `(inputs, outputs, name, kwargs)` |
| `src.applications.inception_v3.InceptionV3` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `src.applications.inception_v3.WEIGHTS_PATH` | Object | `` |
| `src.applications.inception_v3.WEIGHTS_PATH_NO_TOP` | Object | `` |
| `src.applications.inception_v3.backend` | Object | `` |
| `src.applications.inception_v3.conv2d_bn` | Function | `(x, filters, num_row, num_col, padding, strides, name)` |
| `src.applications.inception_v3.decode_predictions` | Function | `(preds, top)` |
| `src.applications.inception_v3.file_utils` | Object | `` |
| `src.applications.inception_v3.imagenet_utils` | Object | `` |
| `src.applications.inception_v3.keras_export` | Class | `(path)` |
| `src.applications.inception_v3.layers` | Object | `` |
| `src.applications.inception_v3.operation_utils` | Object | `` |
| `src.applications.inception_v3.preprocess_input` | Function | `(x, data_format)` |
| `src.applications.mobilenet.BASE_WEIGHT_PATH` | Object | `` |
| `src.applications.mobilenet.Functional` | Class | `(inputs, outputs, name, kwargs)` |
| `src.applications.mobilenet.MobileNet` | Function | `(input_shape, alpha, depth_multiplier, dropout, include_top, weights, input_tensor, pooling, classes, classifier_activation, name)` |
| `src.applications.mobilenet.backend` | Object | `` |
| `src.applications.mobilenet.decode_predictions` | Function | `(preds, top)` |
| `src.applications.mobilenet.file_utils` | Object | `` |
| `src.applications.mobilenet.imagenet_utils` | Object | `` |
| `src.applications.mobilenet.keras_export` | Class | `(path)` |
| `src.applications.mobilenet.layers` | Object | `` |
| `src.applications.mobilenet.operation_utils` | Object | `` |
| `src.applications.mobilenet.preprocess_input` | Function | `(x, data_format)` |
| `src.applications.mobilenet_v2.BASE_WEIGHT_PATH` | Object | `` |
| `src.applications.mobilenet_v2.Functional` | Class | `(inputs, outputs, name, kwargs)` |
| `src.applications.mobilenet_v2.MobileNetV2` | Function | `(input_shape, alpha, include_top, weights, input_tensor, pooling, classes, classifier_activation, name)` |
| `src.applications.mobilenet_v2.backend` | Object | `` |
| `src.applications.mobilenet_v2.decode_predictions` | Function | `(preds, top)` |
| `src.applications.mobilenet_v2.file_utils` | Object | `` |
| `src.applications.mobilenet_v2.imagenet_utils` | Object | `` |
| `src.applications.mobilenet_v2.keras_export` | Class | `(path)` |
| `src.applications.mobilenet_v2.layers` | Object | `` |
| `src.applications.mobilenet_v2.operation_utils` | Object | `` |
| `src.applications.mobilenet_v2.preprocess_input` | Function | `(x, data_format)` |
| `src.applications.mobilenet_v3.BASE_DOCSTRING` | Object | `` |
| `src.applications.mobilenet_v3.BASE_WEIGHT_PATH` | Object | `` |
| `src.applications.mobilenet_v3.Functional` | Class | `(inputs, outputs, name, kwargs)` |
| `src.applications.mobilenet_v3.MobileNetV3` | Function | `(stack_fn, last_point_ch, input_shape, alpha, model_type, minimalistic, include_top, weights, input_tensor, classes, pooling, dropout_rate, classifier_activation, include_preprocessing, name)` |
| `src.applications.mobilenet_v3.MobileNetV3Large` | Function | `(input_shape, alpha, minimalistic, include_top, weights, input_tensor, classes, pooling, dropout_rate, classifier_activation, include_preprocessing, name)` |
| `src.applications.mobilenet_v3.MobileNetV3Small` | Function | `(input_shape, alpha, minimalistic, include_top, weights, input_tensor, classes, pooling, dropout_rate, classifier_activation, include_preprocessing, name)` |
| `src.applications.mobilenet_v3.WEIGHTS_HASHES` | Object | `` |
| `src.applications.mobilenet_v3.backend` | Object | `` |
| `src.applications.mobilenet_v3.decode_predictions` | Function | `(preds, top)` |
| `src.applications.mobilenet_v3.file_utils` | Object | `` |
| `src.applications.mobilenet_v3.hard_sigmoid` | Function | `(x)` |
| `src.applications.mobilenet_v3.hard_swish` | Function | `(x)` |
| `src.applications.mobilenet_v3.imagenet_utils` | Object | `` |
| `src.applications.mobilenet_v3.keras_export` | Class | `(path)` |
| `src.applications.mobilenet_v3.layers` | Object | `` |
| `src.applications.mobilenet_v3.operation_utils` | Object | `` |
| `src.applications.mobilenet_v3.preprocess_input` | Function | `(x, data_format)` |
| `src.applications.mobilenet_v3.relu` | Function | `(x)` |
| `src.applications.nasnet.BASE_WEIGHTS_PATH` | Object | `` |
| `src.applications.nasnet.Functional` | Class | `(inputs, outputs, name, kwargs)` |
| `src.applications.nasnet.NASNET_LARGE_WEIGHT_PATH` | Object | `` |
| `src.applications.nasnet.NASNET_LARGE_WEIGHT_PATH_NO_TOP` | Object | `` |
| `src.applications.nasnet.NASNET_MOBILE_WEIGHT_PATH` | Object | `` |
| `src.applications.nasnet.NASNET_MOBILE_WEIGHT_PATH_NO_TOP` | Object | `` |
| `src.applications.nasnet.NASNet` | Function | `(input_shape, penultimate_filters, num_blocks, stem_block_filters, skip_reduction, filter_multiplier, include_top, weights, input_tensor, pooling, classes, default_size, classifier_activation, name)` |
| `src.applications.nasnet.NASNetLarge` | Function | `(input_shape, include_top, weights, input_tensor, pooling, classes, classifier_activation, name)` |
| `src.applications.nasnet.NASNetMobile` | Function | `(input_shape, include_top, weights, input_tensor, pooling, classes, classifier_activation, name)` |
| `src.applications.nasnet.backend` | Object | `` |
| `src.applications.nasnet.decode_predictions` | Function | `(preds, top)` |
| `src.applications.nasnet.file_utils` | Object | `` |
| `src.applications.nasnet.imagenet_utils` | Object | `` |
| `src.applications.nasnet.keras_export` | Class | `(path)` |
| `src.applications.nasnet.layers` | Object | `` |
| `src.applications.nasnet.operation_utils` | Object | `` |
| `src.applications.nasnet.preprocess_input` | Function | `(x, data_format)` |
| `src.applications.resnet.BASE_WEIGHTS_PATH` | Object | `` |
| `src.applications.resnet.DOC` | Object | `` |
| `src.applications.resnet.Functional` | Class | `(inputs, outputs, name, kwargs)` |
| `src.applications.resnet.ResNet` | Function | `(stack_fn, preact, use_bias, include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name, weights_name)` |
| `src.applications.resnet.ResNet101` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `src.applications.resnet.ResNet152` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `src.applications.resnet.ResNet50` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `src.applications.resnet.WEIGHTS_HASHES` | Object | `` |
| `src.applications.resnet.backend` | Object | `` |
| `src.applications.resnet.decode_predictions` | Function | `(preds, top)` |
| `src.applications.resnet.file_utils` | Object | `` |
| `src.applications.resnet.imagenet_utils` | Object | `` |
| `src.applications.resnet.keras_export` | Class | `(path)` |
| `src.applications.resnet.layers` | Object | `` |
| `src.applications.resnet.operation_utils` | Object | `` |
| `src.applications.resnet.preprocess_input` | Function | `(x, data_format)` |
| `src.applications.resnet.residual_block_v1` | Function | `(x, filters, kernel_size, stride, conv_shortcut, name)` |
| `src.applications.resnet.residual_block_v2` | Function | `(x, filters, kernel_size, stride, conv_shortcut, name)` |
| `src.applications.resnet.stack_residual_blocks_v1` | Function | `(x, filters, blocks, stride1, name)` |
| `src.applications.resnet.stack_residual_blocks_v2` | Function | `(x, filters, blocks, stride1, name)` |
| `src.applications.resnet_v2.DOC` | Object | `` |
| `src.applications.resnet_v2.ResNet101V2` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `src.applications.resnet_v2.ResNet152V2` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `src.applications.resnet_v2.ResNet50V2` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `src.applications.resnet_v2.decode_predictions` | Function | `(preds, top)` |
| `src.applications.resnet_v2.imagenet_utils` | Object | `` |
| `src.applications.resnet_v2.keras_export` | Class | `(path)` |
| `src.applications.resnet_v2.preprocess_input` | Function | `(x, data_format)` |
| `src.applications.resnet_v2.resnet` | Object | `` |
| `src.applications.vgg16.Functional` | Class | `(inputs, outputs, name, kwargs)` |
| `src.applications.vgg16.VGG16` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `src.applications.vgg16.WEIGHTS_PATH` | Object | `` |
| `src.applications.vgg16.WEIGHTS_PATH_NO_TOP` | Object | `` |
| `src.applications.vgg16.backend` | Object | `` |
| `src.applications.vgg16.decode_predictions` | Function | `(preds, top)` |
| `src.applications.vgg16.file_utils` | Object | `` |
| `src.applications.vgg16.imagenet_utils` | Object | `` |
| `src.applications.vgg16.keras_export` | Class | `(path)` |
| `src.applications.vgg16.layers` | Object | `` |
| `src.applications.vgg16.operation_utils` | Object | `` |
| `src.applications.vgg16.preprocess_input` | Function | `(x, data_format)` |
| `src.applications.vgg19.Functional` | Class | `(inputs, outputs, name, kwargs)` |
| `src.applications.vgg19.VGG19` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `src.applications.vgg19.WEIGHTS_PATH` | Object | `` |
| `src.applications.vgg19.WEIGHTS_PATH_NO_TOP` | Object | `` |
| `src.applications.vgg19.backend` | Object | `` |
| `src.applications.vgg19.decode_predictions` | Function | `(preds, top)` |
| `src.applications.vgg19.file_utils` | Object | `` |
| `src.applications.vgg19.imagenet_utils` | Object | `` |
| `src.applications.vgg19.keras_export` | Class | `(path)` |
| `src.applications.vgg19.layers` | Object | `` |
| `src.applications.vgg19.operation_utils` | Object | `` |
| `src.applications.vgg19.preprocess_input` | Function | `(x, data_format)` |
| `src.applications.xception.Functional` | Class | `(inputs, outputs, name, kwargs)` |
| `src.applications.xception.WEIGHTS_PATH` | Object | `` |
| `src.applications.xception.WEIGHTS_PATH_NO_TOP` | Object | `` |
| `src.applications.xception.Xception` | Function | `(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)` |
| `src.applications.xception.backend` | Object | `` |
| `src.applications.xception.decode_predictions` | Function | `(preds, top)` |
| `src.applications.xception.file_utils` | Object | `` |
| `src.applications.xception.imagenet_utils` | Object | `` |
| `src.applications.xception.keras_export` | Class | `(path)` |
| `src.applications.xception.layers` | Object | `` |
| `src.applications.xception.operation_utils` | Object | `` |
| `src.applications.xception.preprocess_input` | Function | `(x, data_format)` |
| `src.backend.AutocastScope` | Class | `(dtype)` |
| `src.backend.BackendVariable` | Class | `(...)` |
| `src.backend.IS_THREAD_SAFE` | Object | `` |
| `src.backend.KerasTensor` | Class | `(shape, dtype, sparse, ragged, record_history, name, kwargs)` |
| `src.backend.SUPPORTS_RAGGED_TENSORS` | Object | `` |
| `src.backend.SUPPORTS_SPARSE_TENSORS` | Object | `` |
| `src.backend.StatelessScope` | Class | `(state_mapping, collect_losses, initialize_variables)` |
| `src.backend.SymbolicScope` | Class | `(...)` |
| `src.backend.Variable` | Class | `(...)` |
| `src.backend.any_symbolic_tensors` | Function | `(args, kwargs)` |
| `src.backend.backend` | Function | `()` |
| `src.backend.backend_name_scope` | Object | `` |
| `src.backend.cast` | Function | `(x, dtype)` |
| `src.backend.common.AutocastScope` | Class | `(dtype)` |
| `src.backend.common.KerasVariable` | Class | `(initializer, shape, dtype, trainable, autocast, aggregation, synchronization, name, kwargs)` |
| `src.backend.common.backend_utils.canonicalize_axis` | Function | `(axis, num_dims)` |
| `src.backend.common.backend_utils.compute_adaptive_pooling_window_sizes` | Function | `(input_dim, output_dim)` |
| `src.backend.common.backend_utils.compute_conv_transpose_output_shape` | Function | `(input_shape, kernel_size, filters, strides, padding, output_padding, data_format, dilation_rate)` |
| `src.backend.common.backend_utils.compute_conv_transpose_padding_args_for_jax` | Function | `(input_shape, kernel_shape, strides, padding, output_padding, dilation_rate)` |
| `src.backend.common.backend_utils.compute_conv_transpose_padding_args_for_torch` | Function | `(input_shape, kernel_shape, strides, padding, output_padding, dilation_rate)` |
| `src.backend.common.backend_utils.slice_along_axis` | Function | `(x, start, stop, step, axis)` |
| `src.backend.common.backend_utils.standardize_axis_for_numpy` | Function | `(axis)` |
| `src.backend.common.backend_utils.to_tuple_or_list` | Function | `(value)` |
| `src.backend.common.backend_utils.vectorize_impl` | Function | `(pyfunc, vmap_fn, excluded, signature)` |
| `src.backend.common.dtypes.ALLOWED_DTYPES` | Object | `` |
| `src.backend.common.dtypes.BIT64_TO_BIT32_DTYPE` | Object | `` |
| `src.backend.common.dtypes.BOOL_TYPES` | Object | `` |
| `src.backend.common.dtypes.COMPLEX_TYPES` | Object | `` |
| `src.backend.common.dtypes.FLOAT8_TYPES` | Object | `` |
| `src.backend.common.dtypes.FLOAT_TYPES` | Object | `` |
| `src.backend.common.dtypes.INT_TYPES` | Object | `` |
| `src.backend.common.dtypes.LATTICE_UPPER_BOUNDS` | Object | `` |
| `src.backend.common.dtypes.PYTHON_DTYPES_MAP` | Object | `` |
| `src.backend.common.dtypes.WEAK_TYPES` | Object | `` |
| `src.backend.common.dtypes.config` | Object | `` |
| `src.backend.common.dtypes.keras_export` | Class | `(path)` |
| `src.backend.common.dtypes.result_type` | Function | `(dtypes)` |
| `src.backend.common.dtypes.standardize_dtype` | Function | `(dtype)` |
| `src.backend.common.get_autocast_scope` | Function | `()` |
| `src.backend.common.global_state.GLOBAL_SETTINGS_TRACKER` | Object | `` |
| `src.backend.common.global_state.GLOBAL_STATE_TRACKER` | Object | `` |
| `src.backend.common.global_state.backend` | Object | `` |
| `src.backend.common.global_state.clear_session` | Function | `(free_memory)` |
| `src.backend.common.global_state.get_global_attribute` | Function | `(name, default, set_to_default)` |
| `src.backend.common.global_state.keras_export` | Class | `(path)` |
| `src.backend.common.global_state.set_global_attribute` | Function | `(name, value)` |
| `src.backend.common.is_float_dtype` | Function | `(dtype)` |
| `src.backend.common.is_int_dtype` | Function | `(dtype)` |
| `src.backend.common.keras_tensor.KerasTensor` | Class | `(shape, dtype, sparse, ragged, record_history, name, kwargs)` |
| `src.backend.common.keras_tensor.any_symbolic_tensors` | Function | `(args, kwargs)` |
| `src.backend.common.keras_tensor.auto_name` | Function | `(prefix)` |
| `src.backend.common.keras_tensor.is_keras_tensor` | Function | `(x)` |
| `src.backend.common.keras_tensor.keras_export` | Class | `(path)` |
| `src.backend.common.keras_tensor.tree` | Object | `` |
| `src.backend.common.masking.get_keras_mask` | Function | `(x)` |
| `src.backend.common.masking.get_tensor_attr` | Function | `(tensor, attr)` |
| `src.backend.common.masking.set_keras_mask` | Function | `(x, mask)` |
| `src.backend.common.masking.set_tensor_attr` | Function | `(tensor, attr, value)` |
| `src.backend.common.name_scope.current_path` | Function | `()` |
| `src.backend.common.name_scope.global_state` | Object | `` |
| `src.backend.common.name_scope.name_scope` | Class | `(name, caller, deduplicate, override_parent)` |
| `src.backend.common.random` | Object | `` |
| `src.backend.common.remat.RematMode` | Object | `` |
| `src.backend.common.remat.RematScope` | Class | `(mode, output_size_threshold, layer_names)` |
| `src.backend.common.remat.backend` | Object | `` |
| `src.backend.common.remat.get_current_remat_mode` | Function | `()` |
| `src.backend.common.remat.global_state` | Object | `` |
| `src.backend.common.remat.keras_export` | Class | `(path)` |
| `src.backend.common.remat.remat` | Function | `(f)` |
| `src.backend.common.result_type` | Function | `(dtypes)` |
| `src.backend.common.standardize_dtype` | Function | `(dtype)` |
| `src.backend.common.standardize_shape` | Function | `(shape)` |
| `src.backend.common.stateless_scope.StatelessScope` | Class | `(state_mapping, collect_losses, initialize_variables)` |
| `src.backend.common.stateless_scope.get_stateless_scope` | Function | `()` |
| `src.backend.common.stateless_scope.global_state` | Object | `` |
| `src.backend.common.stateless_scope.in_stateless_scope` | Function | `()` |
| `src.backend.common.stateless_scope.keras_export` | Class | `(path)` |
| `src.backend.common.symbolic_scope.SymbolicScope` | Class | `(...)` |
| `src.backend.common.symbolic_scope.get_symbolic_scope` | Function | `()` |
| `src.backend.common.symbolic_scope.global_state` | Object | `` |
| `src.backend.common.symbolic_scope.in_symbolic_scope` | Function | `()` |
| `src.backend.common.symbolic_scope.keras_export` | Class | `(path)` |
| `src.backend.common.tensor_attributes.get_tensor_attr` | Function | `(tensor, attr)` |
| `src.backend.common.tensor_attributes.global_state` | Object | `` |
| `src.backend.common.tensor_attributes.set_tensor_attr` | Function | `(tensor, attr, value)` |
| `src.backend.common.variables.AutocastScope` | Class | `(dtype)` |
| `src.backend.common.variables.Variable` | Class | `(initializer, shape, dtype, trainable, autocast, aggregation, synchronization, name, kwargs)` |
| `src.backend.common.variables.auto_name` | Function | `(prefix)` |
| `src.backend.common.variables.backend` | Object | `` |
| `src.backend.common.variables.config` | Object | `` |
| `src.backend.common.variables.current_path` | Function | `()` |
| `src.backend.common.variables.dtypes` | Object | `` |
| `src.backend.common.variables.get_autocast_scope` | Function | `()` |
| `src.backend.common.variables.get_stateless_scope` | Function | `()` |
| `src.backend.common.variables.global_state` | Object | `` |
| `src.backend.common.variables.in_stateless_scope` | Function | `()` |
| `src.backend.common.variables.initialize_all_variables` | Function | `()` |
| `src.backend.common.variables.is_float_dtype` | Function | `(dtype)` |
| `src.backend.common.variables.is_int_dtype` | Function | `(dtype)` |
| `src.backend.common.variables.keras_export` | Class | `(path)` |
| `src.backend.common.variables.register_uninitialized_variable` | Function | `(variable)` |
| `src.backend.common.variables.shape_equal` | Function | `(a_shape, b_shape)` |
| `src.backend.common.variables.standardize_dtype` | Function | `(dtype)` |
| `src.backend.common.variables.standardize_shape` | Function | `(shape)` |
| `src.backend.common.variables.tf` | Object | `` |
| `src.backend.compute_output_spec` | Function | `(fn, args, kwargs)` |
| `src.backend.cond` | Function | `(pred, true_fn, false_fn)` |
| `src.backend.config.backend` | Function | `()` |
| `src.backend.config.disable_flash_attention` | Function | `()` |
| `src.backend.config.enable_flash_attention` | Function | `()` |
| `src.backend.config.env_val` | Object | `` |
| `src.backend.config.epsilon` | Function | `()` |
| `src.backend.config.floatx` | Function | `()` |
| `src.backend.config.image_data_format` | Function | `()` |
| `src.backend.config.is_flash_attention_enabled` | Function | `()` |
| `src.backend.config.is_nnx_enabled` | Function | `()` |
| `src.backend.config.keras_export` | Class | `(path)` |
| `src.backend.config.keras_home` | Function | `()` |
| `src.backend.config.max_epochs` | Function | `()` |
| `src.backend.config.max_steps_per_epoch` | Function | `()` |
| `src.backend.config.set_epsilon` | Function | `(value)` |
| `src.backend.config.set_floatx` | Function | `(value)` |
| `src.backend.config.set_image_data_format` | Function | `(data_format)` |
| `src.backend.config.set_max_epochs` | Function | `(max_epochs)` |
| `src.backend.config.set_max_steps_per_epoch` | Function | `(max_steps_per_epoch)` |
| `src.backend.config.set_nnx_enabled` | Function | `(value)` |
| `src.backend.config.standardize_data_format` | Function | `(data_format)` |
| `src.backend.convert_to_numpy` | Function | `(x)` |
| `src.backend.convert_to_tensor` | Function | `(x, dtype, sparse, ragged)` |
| `src.backend.core` | Object | `` |
| `src.backend.cudnn_ok` | Function | `(args, kwargs)` |
| `src.backend.device` | Function | `(device_name)` |
| `src.backend.device_scope` | Function | `(device_name)` |
| `src.backend.distribution_lib` | Object | `` |
| `src.backend.epsilon` | Function | `()` |
| `src.backend.floatx` | Function | `()` |
| `src.backend.get_autocast_scope` | Function | `()` |
| `src.backend.get_keras_mask` | Function | `(x)` |
| `src.backend.get_stateless_scope` | Function | `()` |
| `src.backend.gru` | Function | `(args, kwargs)` |
| `src.backend.image` | Object | `` |
| `src.backend.image_data_format` | Function | `()` |
| `src.backend.in_stateless_scope` | Function | `()` |
| `src.backend.in_symbolic_scope` | Function | `()` |
| `src.backend.is_float_dtype` | Function | `(dtype)` |
| `src.backend.is_int_dtype` | Function | `(dtype)` |
| `src.backend.is_keras_tensor` | Function | `(x)` |
| `src.backend.is_nnx_enabled` | Function | `()` |
| `src.backend.is_tensor` | Function | `(x)` |
| `src.backend.jax.IS_THREAD_SAFE` | Object | `` |
| `src.backend.jax.SUPPORTS_RAGGED_TENSORS` | Object | `` |
| `src.backend.jax.SUPPORTS_SPARSE_TENSORS` | Object | `` |
| `src.backend.jax.Variable` | Object | `` |
| `src.backend.jax.cast` | Function | `(x, dtype)` |
| `src.backend.jax.compute_output_spec` | Function | `(fn, args, kwargs)` |
| `src.backend.jax.cond` | Function | `(pred, true_fn, false_fn)` |
| `src.backend.jax.convert_to_numpy` | Function | `(x)` |
| `src.backend.jax.convert_to_tensor` | Function | `(x, dtype, sparse, ragged)` |
| `src.backend.jax.core.IS_THREAD_SAFE` | Object | `` |
| `src.backend.jax.core.JaxVariable` | Class | `(args, layout, kwargs)` |
| `src.backend.jax.core.KerasTensor` | Class | `(shape, dtype, sparse, ragged, record_history, name, kwargs)` |
| `src.backend.jax.core.KerasVariable` | Class | `(initializer, shape, dtype, trainable, autocast, aggregation, synchronization, name, kwargs)` |
| `src.backend.jax.core.NnxVariable` | Class | `(initializer, shape, dtype, trainable, autocast, aggregation, synchronization, name, layout, mutable, nnx_metadata)` |
| `src.backend.jax.core.SUPPORTS_RAGGED_TENSORS` | Object | `` |
| `src.backend.jax.core.SUPPORTS_SPARSE_TENSORS` | Object | `` |
| `src.backend.jax.core.StatelessScope` | Class | `(state_mapping, collect_losses, initialize_variables)` |
| `src.backend.jax.core.SymbolicScope` | Class | `(...)` |
| `src.backend.jax.core.Variable` | Object | `` |
| `src.backend.jax.core.associative_scan` | Function | `(f, elems, reverse, axis)` |
| `src.backend.jax.core.base_name_scope` | Class | `(name, caller, deduplicate, override_parent)` |
| `src.backend.jax.core.cast` | Function | `(x, dtype)` |
| `src.backend.jax.core.compute_output_spec` | Function | `(fn, args, kwargs)` |
| `src.backend.jax.core.cond` | Function | `(pred, true_fn, false_fn)` |
| `src.backend.jax.core.config` | Object | `` |
| `src.backend.jax.core.convert_to_numpy` | Function | `(x)` |
| `src.backend.jax.core.convert_to_tensor` | Function | `(x, dtype, sparse, ragged)` |
| `src.backend.jax.core.custom_gradient` | Function | `(fun)` |
| `src.backend.jax.core.device_scope` | Function | `(device_name)` |
| `src.backend.jax.core.distribution_lib` | Object | `` |
| `src.backend.jax.core.fori_loop` | Function | `(lower, upper, body_fun, init_val)` |
| `src.backend.jax.core.get_stateless_scope` | Function | `()` |
| `src.backend.jax.core.global_state` | Object | `` |
| `src.backend.jax.core.in_stateless_scope` | Function | `()` |
| `src.backend.jax.core.is_tensor` | Function | `(x)` |
| `src.backend.jax.core.map` | Function | `(f, xs)` |
| `src.backend.jax.core.name_scope` | Class | `(name, kwargs)` |
| `src.backend.jax.core.random_seed_dtype` | Function | `()` |
| `src.backend.jax.core.remat` | Function | `(f)` |
| `src.backend.jax.core.scan` | Function | `(f, init, xs, length, reverse, unroll)` |
| `src.backend.jax.core.scatter` | Function | `(indices, values, shape)` |
| `src.backend.jax.core.scatter_update` | Function | `(inputs, indices, updates)` |
| `src.backend.jax.core.shape` | Function | `(x)` |
| `src.backend.jax.core.should_shard_at_init` | Function | `(init_layout, shape)` |
| `src.backend.jax.core.slice` | Function | `(inputs, start_indices, shape)` |
| `src.backend.jax.core.slice_update` | Function | `(inputs, start_indices, updates)` |
| `src.backend.jax.core.standardize_dtype` | Function | `(dtype)` |
| `src.backend.jax.core.stop_gradient` | Function | `(variable)` |
| `src.backend.jax.core.switch` | Function | `(index, branches, operands)` |
| `src.backend.jax.core.tree` | Object | `` |
| `src.backend.jax.core.unstack` | Function | `(x, num, axis)` |
| `src.backend.jax.core.vectorized_map` | Function | `(function, elements)` |
| `src.backend.jax.core.while_loop` | Function | `(cond, body, loop_vars, maximum_iterations)` |
| `src.backend.jax.cudnn_ok` | Function | `(args, kwargs)` |
| `src.backend.jax.device_scope` | Function | `(device_name)` |
| `src.backend.jax.distribution_lib.distribute_data_input` | Function | `(per_process_batch, layout, batch_dim_name)` |
| `src.backend.jax.distribution_lib.distribute_tensor` | Function | `(tensor, layout)` |
| `src.backend.jax.distribution_lib.distribute_variable` | Function | `(value, layout)` |
| `src.backend.jax.distribution_lib.get_device_count` | Function | `(device_type)` |
| `src.backend.jax.distribution_lib.global_state` | Object | `` |
| `src.backend.jax.distribution_lib.initialize` | Function | `(job_addresses, num_processes, process_id)` |
| `src.backend.jax.distribution_lib.initialize_rng` | Function | `()` |
| `src.backend.jax.distribution_lib.jax_utils` | Object | `` |
| `src.backend.jax.distribution_lib.list_devices` | Function | `(device_type)` |
| `src.backend.jax.distribution_lib.num_processes` | Function | `()` |
| `src.backend.jax.distribution_lib.process_id` | Function | `()` |
| `src.backend.jax.distribution_lib.rng_utils` | Object | `` |
| `src.backend.jax.distribution_lib.seed_generator` | Object | `` |
| `src.backend.jax.export.JaxExportArchive` | Class | `()` |
| `src.backend.jax.export.StatelessScope` | Class | `(state_mapping, collect_losses, initialize_variables)` |
| `src.backend.jax.export.tf` | Object | `` |
| `src.backend.jax.export.tree` | Object | `` |
| `src.backend.jax.gru` | Function | `(args, kwargs)` |
| `src.backend.jax.image.AFFINE_TRANSFORM_FILL_MODES` | Object | `` |
| `src.backend.jax.image.AFFINE_TRANSFORM_INTERPOLATIONS` | Object | `` |
| `src.backend.jax.image.MAP_COORDINATES_FILL_MODES` | Object | `` |
| `src.backend.jax.image.RESIZE_INTERPOLATIONS` | Object | `` |
| `src.backend.jax.image.SCALE_AND_TRANSLATE_METHODS` | Object | `` |
| `src.backend.jax.image.affine_transform` | Function | `(images, transform, interpolation, fill_mode, fill_value, data_format)` |
| `src.backend.jax.image.backend` | Object | `` |
| `src.backend.jax.image.compute_homography_matrix` | Function | `(start_points, end_points)` |
| `src.backend.jax.image.convert_to_tensor` | Function | `(x, dtype, sparse, ragged)` |
| `src.backend.jax.image.draw_seed` | Function | `(seed)` |
| `src.backend.jax.image.elastic_transform` | Function | `(images, alpha, sigma, interpolation, fill_mode, fill_value, seed, data_format)` |
| `src.backend.jax.image.gaussian_blur` | Function | `(images, kernel_size, sigma, data_format)` |
| `src.backend.jax.image.hsv_to_rgb` | Function | `(images, data_format)` |
| `src.backend.jax.image.map_coordinates` | Function | `(inputs, coordinates, order, fill_mode, fill_value)` |
| `src.backend.jax.image.perspective_transform` | Function | `(images, start_points, end_points, interpolation, fill_value, data_format)` |
| `src.backend.jax.image.resize` | Function | `(images, size, interpolation, antialias, crop_to_aspect_ratio, pad_to_aspect_ratio, fill_mode, fill_value, data_format)` |
| `src.backend.jax.image.rgb_to_grayscale` | Function | `(images, data_format)` |
| `src.backend.jax.image.rgb_to_hsv` | Function | `(images, data_format)` |
| `src.backend.jax.image.scale_and_translate` | Function | `(images, output_shape, scale, translation, spatial_dims, method, antialias)` |
| `src.backend.jax.is_nnx_enabled` | Function | `()` |
| `src.backend.jax.is_tensor` | Function | `(x)` |
| `src.backend.jax.layer.BaseLayer` | Class | `(...)` |
| `src.backend.jax.layer.JaxLayer` | Class | `(...)` |
| `src.backend.jax.layer.is_nnx_enabled` | Function | `()` |
| `src.backend.jax.linalg.cast` | Function | `(x, dtype)` |
| `src.backend.jax.linalg.cholesky` | Function | `(a, upper)` |
| `src.backend.jax.linalg.cholesky_inverse` | Function | `(a, upper)` |
| `src.backend.jax.linalg.config` | Object | `` |
| `src.backend.jax.linalg.convert_to_tensor` | Function | `(x, dtype, sparse, ragged)` |
| `src.backend.jax.linalg.det` | Function | `(a)` |
| `src.backend.jax.linalg.dtypes` | Object | `` |
| `src.backend.jax.linalg.eig` | Function | `(x)` |
| `src.backend.jax.linalg.eigh` | Function | `(x)` |
| `src.backend.jax.linalg.inv` | Function | `(a)` |
| `src.backend.jax.linalg.jvp` | Function | `(fun, primals, tangents, has_aux)` |
| `src.backend.jax.linalg.lstsq` | Function | `(a, b, rcond)` |
| `src.backend.jax.linalg.lu_factor` | Function | `(x)` |
| `src.backend.jax.linalg.norm` | Function | `(x, ord, axis, keepdims)` |
| `src.backend.jax.linalg.qr` | Function | `(x, mode)` |
| `src.backend.jax.linalg.solve` | Function | `(a, b)` |
| `src.backend.jax.linalg.solve_triangular` | Function | `(a, b, lower)` |
| `src.backend.jax.linalg.standardize_dtype` | Function | `(dtype)` |
| `src.backend.jax.linalg.svd` | Function | `(x, full_matrices, compute_uv)` |
| `src.backend.jax.lstm` | Function | `(args, kwargs)` |
| `src.backend.jax.math.cast` | Function | `(x, dtype)` |
| `src.backend.jax.math.config` | Object | `` |
| `src.backend.jax.math.convert_to_tensor` | Function | `(x, dtype, sparse, ragged)` |
| `src.backend.jax.math.dtypes` | Object | `` |
| `src.backend.jax.math.erf` | Function | `(x)` |
| `src.backend.jax.math.erfinv` | Function | `(x)` |
| `src.backend.jax.math.extract_sequences` | Function | `(x, sequence_length, sequence_stride)` |
| `src.backend.jax.math.fft` | Function | `(x)` |
| `src.backend.jax.math.fft2` | Function | `(x)` |
| `src.backend.jax.math.ifft2` | Function | `(x)` |
| `src.backend.jax.math.in_top_k` | Function | `(targets, predictions, k)` |
| `src.backend.jax.math.irfft` | Function | `(x, fft_length)` |
| `src.backend.jax.math.istft` | Function | `(x, sequence_length, sequence_stride, fft_length, length, window, center)` |
| `src.backend.jax.math.logdet` | Function | `(x)` |
| `src.backend.jax.math.logsumexp` | Function | `(x, axis, keepdims)` |
| `src.backend.jax.math.norm` | Function | `(x, ord, axis, keepdims)` |
| `src.backend.jax.math.qr` | Function | `(x, mode)` |
| `src.backend.jax.math.rfft` | Function | `(x, fft_length)` |
| `src.backend.jax.math.rsqrt` | Function | `(x)` |
| `src.backend.jax.math.scipy` | Object | `` |
| `src.backend.jax.math.segment_max` | Function | `(data, segment_ids, num_segments, sorted)` |
| `src.backend.jax.math.segment_sum` | Function | `(data, segment_ids, num_segments, sorted)` |
| `src.backend.jax.math.solve` | Function | `(a, b)` |
| `src.backend.jax.math.standardize_dtype` | Function | `(dtype)` |
| `src.backend.jax.math.stft` | Function | `(x, sequence_length, sequence_stride, fft_length, window, center)` |
| `src.backend.jax.math.top_k` | Function | `(x, k, sorted)` |
| `src.backend.jax.name_scope` | Class | `(name, kwargs)` |
| `src.backend.jax.nn.adaptive_average_pool` | Function | `(inputs, output_size, data_format)` |
| `src.backend.jax.nn.adaptive_max_pool` | Function | `(inputs, output_size, data_format)` |
| `src.backend.jax.nn.average_pool` | Function | `(inputs, pool_size, strides, padding, data_format)` |
| `src.backend.jax.nn.backend` | Object | `` |
| `src.backend.jax.nn.batch_normalization` | Function | `(x, mean, variance, axis, offset, scale, epsilon)` |
| `src.backend.jax.nn.binary_crossentropy` | Function | `(target, output, from_logits)` |
| `src.backend.jax.nn.cast` | Function | `(x, dtype)` |
| `src.backend.jax.nn.categorical_crossentropy` | Function | `(target, output, from_logits, axis)` |
| `src.backend.jax.nn.celu` | Function | `(x, alpha)` |
| `src.backend.jax.nn.compute_adaptive_pooling_window_sizes` | Function | `(input_dim, output_dim)` |
| `src.backend.jax.nn.compute_conv_transpose_padding_args_for_jax` | Function | `(input_shape, kernel_shape, strides, padding, output_padding, dilation_rate)` |
| `src.backend.jax.nn.conv` | Function | `(inputs, kernel, strides, padding, data_format, dilation_rate)` |
| `src.backend.jax.nn.conv_transpose` | Function | `(inputs, kernel, strides, padding, output_padding, data_format, dilation_rate)` |
| `src.backend.jax.nn.convert_to_tensor` | Function | `(x, dtype, sparse, ragged)` |
| `src.backend.jax.nn.ctc_decode` | Function | `(inputs, sequence_lengths, strategy, beam_width, top_paths, merge_repeated, mask_index)` |
| `src.backend.jax.nn.ctc_loss` | Function | `(target, output, target_length, output_length, mask_index)` |
| `src.backend.jax.nn.depthwise_conv` | Function | `(inputs, kernel, strides, padding, data_format, dilation_rate)` |
| `src.backend.jax.nn.dot_product_attention` | Function | `(query, key, value, bias, mask, scale, is_causal, flash_attention, attn_logits_soft_cap)` |
| `src.backend.jax.nn.elu` | Function | `(x, alpha)` |
| `src.backend.jax.nn.gelu` | Function | `(x, approximate)` |
| `src.backend.jax.nn.glu` | Function | `(x, axis)` |
| `src.backend.jax.nn.hard_shrink` | Function | `(x, threshold)` |
| `src.backend.jax.nn.hard_sigmoid` | Function | `(x)` |
| `src.backend.jax.nn.hard_silu` | Function | `(x)` |
| `src.backend.jax.nn.hard_tanh` | Function | `(x)` |
| `src.backend.jax.nn.leaky_relu` | Function | `(x, negative_slope)` |
| `src.backend.jax.nn.log_sigmoid` | Function | `(x)` |
| `src.backend.jax.nn.log_softmax` | Function | `(x, axis)` |
| `src.backend.jax.nn.max_pool` | Function | `(inputs, pool_size, strides, padding, data_format)` |
| `src.backend.jax.nn.moments` | Function | `(x, axes, keepdims, synchronized)` |
| `src.backend.jax.nn.multi_hot` | Function | `(x, num_classes, axis, dtype, sparse)` |
| `src.backend.jax.nn.one_hot` | Function | `(x, num_classes, axis, dtype, sparse)` |
| `src.backend.jax.nn.psnr` | Function | `(x1, x2, max_val)` |
| `src.backend.jax.nn.relu` | Function | `(x)` |
| `src.backend.jax.nn.relu6` | Function | `(x)` |
| `src.backend.jax.nn.selu` | Function | `(x)` |
| `src.backend.jax.nn.separable_conv` | Function | `(inputs, depthwise_kernel, pointwise_kernel, strides, padding, data_format, dilation_rate)` |
| `src.backend.jax.nn.sigmoid` | Function | `(x)` |
| `src.backend.jax.nn.silu` | Function | `(x)` |
| `src.backend.jax.nn.soft_shrink` | Function | `(x, threshold)` |
| `src.backend.jax.nn.softmax` | Function | `(x, axis)` |
| `src.backend.jax.nn.softplus` | Function | `(x)` |
| `src.backend.jax.nn.softsign` | Function | `(x)` |
| `src.backend.jax.nn.sparse_categorical_crossentropy` | Function | `(target, output, from_logits, axis)` |
| `src.backend.jax.nn.sparse_plus` | Function | `(x)` |
| `src.backend.jax.nn.sparse_sigmoid` | Function | `(x)` |
| `src.backend.jax.nn.sparsemax` | Function | `(x, axis)` |
| `src.backend.jax.nn.squareplus` | Function | `(x, b)` |
| `src.backend.jax.nn.tanh` | Function | `(x)` |
| `src.backend.jax.nn.tanh_shrink` | Function | `(x)` |
| `src.backend.jax.nn.threshold` | Function | `(x, threshold, default_value)` |
| `src.backend.jax.nn.unfold` | Function | `(input, kernel_size, dilation, padding, stride)` |
| `src.backend.jax.nn.wrap_flash_attention` | Function | `(query, key, value, decoder_segment_ids, custom_mask, attn_logits_soft_cap, head_shards, q_seq_shards)` |
| `src.backend.jax.numpy.abs` | Function | `(x)` |
| `src.backend.jax.numpy.absolute` | Function | `(x)` |
| `src.backend.jax.numpy.add` | Function | `(x1, x2)` |
| `src.backend.jax.numpy.all` | Function | `(x, axis, keepdims)` |
| `src.backend.jax.numpy.amax` | Function | `(x, axis, keepdims)` |
| `src.backend.jax.numpy.amin` | Function | `(x, axis, keepdims)` |
| `src.backend.jax.numpy.angle` | Function | `(x)` |
| `src.backend.jax.numpy.any` | Function | `(x, axis, keepdims)` |
| `src.backend.jax.numpy.append` | Function | `(x1, x2, axis)` |
| `src.backend.jax.numpy.arange` | Function | `(start, stop, step, dtype)` |
| `src.backend.jax.numpy.arccos` | Function | `(x)` |
| `src.backend.jax.numpy.arccosh` | Function | `(x)` |
| `src.backend.jax.numpy.arcsin` | Function | `(x)` |
| `src.backend.jax.numpy.arcsinh` | Function | `(x)` |
| `src.backend.jax.numpy.arctan` | Function | `(x)` |
| `src.backend.jax.numpy.arctan2` | Function | `(x1, x2)` |
| `src.backend.jax.numpy.arctanh` | Function | `(x)` |
| `src.backend.jax.numpy.argmax` | Function | `(x, axis, keepdims)` |
| `src.backend.jax.numpy.argmin` | Function | `(x, axis, keepdims)` |
| `src.backend.jax.numpy.argpartition` | Function | `(x, kth, axis)` |
| `src.backend.jax.numpy.argsort` | Function | `(x, axis)` |
| `src.backend.jax.numpy.array` | Function | `(x, dtype)` |
| `src.backend.jax.numpy.array_split` | Function | `(x, indices_or_sections, axis)` |
| `src.backend.jax.numpy.average` | Function | `(x, axis, weights)` |
| `src.backend.jax.numpy.bartlett` | Function | `(x)` |
| `src.backend.jax.numpy.bincount` | Function | `(x, weights, minlength, sparse)` |
| `src.backend.jax.numpy.bitwise_and` | Function | `(x, y)` |
| `src.backend.jax.numpy.bitwise_invert` | Function | `(x)` |
| `src.backend.jax.numpy.bitwise_left_shift` | Function | `(x, y)` |
| `src.backend.jax.numpy.bitwise_not` | Function | `(x)` |
| `src.backend.jax.numpy.bitwise_or` | Function | `(x, y)` |
| `src.backend.jax.numpy.bitwise_right_shift` | Function | `(x, y)` |
| `src.backend.jax.numpy.bitwise_xor` | Function | `(x, y)` |
| `src.backend.jax.numpy.blackman` | Function | `(x)` |
| `src.backend.jax.numpy.broadcast_to` | Function | `(x, shape)` |
| `src.backend.jax.numpy.canonicalize_axis` | Function | `(axis, num_dims)` |
| `src.backend.jax.numpy.cast` | Function | `(x, dtype)` |
| `src.backend.jax.numpy.cbrt` | Function | `(x)` |
| `src.backend.jax.numpy.ceil` | Function | `(x)` |
| `src.backend.jax.numpy.clip` | Function | `(x, x_min, x_max)` |
| `src.backend.jax.numpy.concatenate` | Function | `(xs, axis)` |
| `src.backend.jax.numpy.config` | Object | `` |
| `src.backend.jax.numpy.conj` | Function | `(x)` |
| `src.backend.jax.numpy.conjugate` | Function | `(x)` |
| `src.backend.jax.numpy.convert_to_tensor` | Function | `(x, dtype, sparse, ragged)` |
| `src.backend.jax.numpy.copy` | Function | `(x)` |
| `src.backend.jax.numpy.corrcoef` | Function | `(x)` |
| `src.backend.jax.numpy.correlate` | Function | `(x1, x2, mode)` |
| `src.backend.jax.numpy.cos` | Function | `(x)` |
| `src.backend.jax.numpy.cosh` | Function | `(x)` |
| `src.backend.jax.numpy.count_nonzero` | Function | `(x, axis)` |
| `src.backend.jax.numpy.cross` | Function | `(x1, x2, axisa, axisb, axisc, axis)` |
| `src.backend.jax.numpy.cumprod` | Function | `(x, axis, dtype)` |
| `src.backend.jax.numpy.cumsum` | Function | `(x, axis, dtype)` |
| `src.backend.jax.numpy.deg2rad` | Function | `(x)` |
| `src.backend.jax.numpy.diag` | Function | `(x, k)` |
| `src.backend.jax.numpy.diagflat` | Function | `(x, k)` |
| `src.backend.jax.numpy.diagonal` | Function | `(x, offset, axis1, axis2)` |
| `src.backend.jax.numpy.diff` | Function | `(a, n, axis)` |
| `src.backend.jax.numpy.digitize` | Function | `(x, bins)` |
| `src.backend.jax.numpy.divide` | Function | `(x1, x2)` |
| `src.backend.jax.numpy.divide_no_nan` | Function | `(x1, x2)` |
| `src.backend.jax.numpy.dot` | Function | `(x1, x2)` |
| `src.backend.jax.numpy.dtypes` | Object | `` |
| `src.backend.jax.numpy.einsum` | Function | `(subscripts, operands, kwargs)` |
| `src.backend.jax.numpy.empty` | Function | `(shape, dtype)` |
| `src.backend.jax.numpy.empty_like` | Function | `(x, dtype)` |
| `src.backend.jax.numpy.equal` | Function | `(x1, x2)` |
| `src.backend.jax.numpy.exp` | Function | `(x)` |
| `src.backend.jax.numpy.exp2` | Function | `(x)` |
| `src.backend.jax.numpy.expand_dims` | Function | `(x, axis)` |
| `src.backend.jax.numpy.expm1` | Function | `(x)` |
| `src.backend.jax.numpy.eye` | Function | `(N, M, k, dtype)` |
| `src.backend.jax.numpy.flip` | Function | `(x, axis)` |
| `src.backend.jax.numpy.floor` | Function | `(x)` |
| `src.backend.jax.numpy.floor_divide` | Function | `(x1, x2)` |
| `src.backend.jax.numpy.full` | Function | `(shape, fill_value, dtype)` |
| `src.backend.jax.numpy.full_like` | Function | `(x, fill_value, dtype)` |
| `src.backend.jax.numpy.gcd` | Function | `(x1, x2)` |
| `src.backend.jax.numpy.greater` | Function | `(x1, x2)` |
| `src.backend.jax.numpy.greater_equal` | Function | `(x1, x2)` |
| `src.backend.jax.numpy.hamming` | Function | `(x)` |
| `src.backend.jax.numpy.hanning` | Function | `(x)` |
| `src.backend.jax.numpy.heaviside` | Function | `(x1, x2)` |
| `src.backend.jax.numpy.histogram` | Function | `(x, bins, range)` |
| `src.backend.jax.numpy.hstack` | Function | `(xs)` |
| `src.backend.jax.numpy.hypot` | Function | `(x1, x2)` |
| `src.backend.jax.numpy.identity` | Function | `(n, dtype)` |
| `src.backend.jax.numpy.imag` | Function | `(x)` |
| `src.backend.jax.numpy.inner` | Function | `(x1, x2)` |
| `src.backend.jax.numpy.isclose` | Function | `(x1, x2, rtol, atol, equal_nan)` |
| `src.backend.jax.numpy.isfinite` | Function | `(x)` |
| `src.backend.jax.numpy.isin` | Function | `(x1, x2, assume_unique, invert)` |
| `src.backend.jax.numpy.isinf` | Function | `(x)` |
| `src.backend.jax.numpy.isnan` | Function | `(x)` |
| `src.backend.jax.numpy.isneginf` | Function | `(x)` |
| `src.backend.jax.numpy.isposinf` | Function | `(x)` |
| `src.backend.jax.numpy.isreal` | Function | `(x)` |
| `src.backend.jax.numpy.kaiser` | Function | `(x, beta)` |
| `src.backend.jax.numpy.kron` | Function | `(x1, x2)` |
| `src.backend.jax.numpy.lcm` | Function | `(x1, x2)` |
| `src.backend.jax.numpy.ldexp` | Function | `(x1, x2)` |
| `src.backend.jax.numpy.left_shift` | Function | `(x, y)` |
| `src.backend.jax.numpy.less` | Function | `(x1, x2)` |
| `src.backend.jax.numpy.less_equal` | Function | `(x1, x2)` |
| `src.backend.jax.numpy.linspace` | Function | `(start, stop, num, endpoint, retstep, dtype, axis)` |
| `src.backend.jax.numpy.log` | Function | `(x)` |
| `src.backend.jax.numpy.log10` | Function | `(x)` |
| `src.backend.jax.numpy.log1p` | Function | `(x)` |
| `src.backend.jax.numpy.log2` | Function | `(x)` |
| `src.backend.jax.numpy.logaddexp` | Function | `(x1, x2)` |
| `src.backend.jax.numpy.logaddexp2` | Function | `(x1, x2)` |
| `src.backend.jax.numpy.logical_and` | Function | `(x1, x2)` |
| `src.backend.jax.numpy.logical_not` | Function | `(x)` |
| `src.backend.jax.numpy.logical_or` | Function | `(x1, x2)` |
| `src.backend.jax.numpy.logical_xor` | Function | `(x1, x2)` |
| `src.backend.jax.numpy.logspace` | Function | `(start, stop, num, endpoint, base, dtype, axis)` |
| `src.backend.jax.numpy.matmul` | Function | `(x1, x2)` |
| `src.backend.jax.numpy.max` | Function | `(x, axis, keepdims, initial)` |
| `src.backend.jax.numpy.maximum` | Function | `(x1, x2)` |
| `src.backend.jax.numpy.mean` | Function | `(x, axis, keepdims)` |
| `src.backend.jax.numpy.median` | Function | `(x, axis, keepdims)` |
| `src.backend.jax.numpy.meshgrid` | Function | `(x, indexing)` |
| `src.backend.jax.numpy.min` | Function | `(x, axis, keepdims, initial)` |
| `src.backend.jax.numpy.minimum` | Function | `(x1, x2)` |
| `src.backend.jax.numpy.mod` | Function | `(x1, x2)` |
| `src.backend.jax.numpy.moveaxis` | Function | `(x, source, destination)` |
| `src.backend.jax.numpy.multiply` | Function | `(x1, x2)` |
| `src.backend.jax.numpy.nan_to_num` | Function | `(x, nan, posinf, neginf)` |
| `src.backend.jax.numpy.ndim` | Function | `(x)` |
| `src.backend.jax.numpy.negative` | Function | `(x)` |
| `src.backend.jax.numpy.nn` | Object | `` |
| `src.backend.jax.numpy.nonzero` | Function | `(x)` |
| `src.backend.jax.numpy.not_equal` | Function | `(x1, x2)` |
| `src.backend.jax.numpy.ones` | Function | `(shape, dtype)` |
| `src.backend.jax.numpy.ones_like` | Function | `(x, dtype)` |
| `src.backend.jax.numpy.outer` | Function | `(x1, x2)` |
| `src.backend.jax.numpy.pad` | Function | `(x, pad_width, mode, constant_values)` |
| `src.backend.jax.numpy.power` | Function | `(x1, x2)` |
| `src.backend.jax.numpy.prod` | Function | `(x, axis, keepdims, dtype)` |
| `src.backend.jax.numpy.quantile` | Function | `(x, q, axis, method, keepdims)` |
| `src.backend.jax.numpy.ravel` | Function | `(x)` |
| `src.backend.jax.numpy.real` | Function | `(x)` |
| `src.backend.jax.numpy.reciprocal` | Function | `(x)` |
| `src.backend.jax.numpy.repeat` | Function | `(x, repeats, axis)` |
| `src.backend.jax.numpy.reshape` | Function | `(x, newshape)` |
| `src.backend.jax.numpy.right_shift` | Function | `(x, y)` |
| `src.backend.jax.numpy.roll` | Function | `(x, shift, axis)` |
| `src.backend.jax.numpy.rot90` | Function | `(array, k, axes)` |
| `src.backend.jax.numpy.round` | Function | `(x, decimals)` |
| `src.backend.jax.numpy.searchsorted` | Function | `(sorted_sequence, values, side)` |
| `src.backend.jax.numpy.select` | Function | `(condlist, choicelist, default)` |
| `src.backend.jax.numpy.sign` | Function | `(x)` |
| `src.backend.jax.numpy.signbit` | Function | `(x)` |
| `src.backend.jax.numpy.sin` | Function | `(x)` |
| `src.backend.jax.numpy.sinh` | Function | `(x)` |
| `src.backend.jax.numpy.size` | Function | `(x)` |
| `src.backend.jax.numpy.slogdet` | Function | `(x)` |
| `src.backend.jax.numpy.sort` | Function | `(x, axis)` |
| `src.backend.jax.numpy.sparse` | Object | `` |
| `src.backend.jax.numpy.split` | Function | `(x, indices_or_sections, axis)` |
| `src.backend.jax.numpy.sqrt` | Function | `(x)` |
| `src.backend.jax.numpy.square` | Function | `(x)` |
| `src.backend.jax.numpy.squeeze` | Function | `(x, axis)` |
| `src.backend.jax.numpy.stack` | Function | `(x, axis)` |
| `src.backend.jax.numpy.standardize_dtype` | Function | `(dtype)` |
| `src.backend.jax.numpy.std` | Function | `(x, axis, keepdims)` |
| `src.backend.jax.numpy.subtract` | Function | `(x1, x2)` |
| `src.backend.jax.numpy.sum` | Function | `(x, axis, keepdims)` |
| `src.backend.jax.numpy.swapaxes` | Function | `(x, axis1, axis2)` |
| `src.backend.jax.numpy.take` | Function | `(x, indices, axis)` |
| `src.backend.jax.numpy.take_along_axis` | Function | `(x, indices, axis)` |
| `src.backend.jax.numpy.tan` | Function | `(x)` |
| `src.backend.jax.numpy.tanh` | Function | `(x)` |
| `src.backend.jax.numpy.tensordot` | Function | `(x1, x2, axes)` |
| `src.backend.jax.numpy.tile` | Function | `(x, repeats)` |
| `src.backend.jax.numpy.to_tuple_or_list` | Function | `(value)` |
| `src.backend.jax.numpy.trace` | Function | `(x, offset, axis1, axis2)` |
| `src.backend.jax.numpy.transpose` | Function | `(x, axes)` |
| `src.backend.jax.numpy.trapezoid` | Function | `(y, x, dx, axis)` |
| `src.backend.jax.numpy.tri` | Function | `(N, M, k, dtype)` |
| `src.backend.jax.numpy.tril` | Function | `(x, k)` |
| `src.backend.jax.numpy.triu` | Function | `(x, k)` |
| `src.backend.jax.numpy.true_divide` | Function | `(x1, x2)` |
| `src.backend.jax.numpy.trunc` | Function | `(x)` |
| `src.backend.jax.numpy.unravel_index` | Function | `(indices, shape)` |
| `src.backend.jax.numpy.vander` | Function | `(x, N, increasing)` |
| `src.backend.jax.numpy.var` | Function | `(x, axis, keepdims)` |
| `src.backend.jax.numpy.vdot` | Function | `(x1, x2)` |
| `src.backend.jax.numpy.vectorize` | Function | `(pyfunc, excluded, signature)` |
| `src.backend.jax.numpy.view` | Function | `(x, dtype)` |
| `src.backend.jax.numpy.vstack` | Function | `(xs)` |
| `src.backend.jax.numpy.where` | Function | `(condition, x1, x2)` |
| `src.backend.jax.numpy.zeros` | Function | `(shape, dtype)` |
| `src.backend.jax.numpy.zeros_like` | Function | `(x, dtype)` |
| `src.backend.jax.optimizer.JaxOptimizer` | Class | `(...)` |
| `src.backend.jax.optimizer.base_optimizer` | Object | `` |
| `src.backend.jax.random.SeedGenerator` | Class | `(seed, name, kwargs)` |
| `src.backend.jax.random.beta` | Function | `(shape, alpha, beta, dtype, seed)` |
| `src.backend.jax.random.binomial` | Function | `(shape, counts, probabilities, dtype, seed)` |
| `src.backend.jax.random.categorical` | Function | `(logits, num_samples, dtype, seed)` |
| `src.backend.jax.random.draw_seed` | Function | `(seed)` |
| `src.backend.jax.random.dropout` | Function | `(inputs, rate, noise_shape, seed)` |
| `src.backend.jax.random.floatx` | Function | `()` |
| `src.backend.jax.random.gamma` | Function | `(shape, alpha, dtype, seed)` |
| `src.backend.jax.random.jax_draw_seed` | Function | `(seed)` |
| `src.backend.jax.random.make_default_seed` | Function | `()` |
| `src.backend.jax.random.normal` | Function | `(shape, mean, stddev, dtype, seed)` |
| `src.backend.jax.random.randint` | Function | `(shape, minval, maxval, dtype, seed)` |
| `src.backend.jax.random.shuffle` | Function | `(x, axis, seed)` |
| `src.backend.jax.random.truncated_normal` | Function | `(shape, mean, stddev, dtype, seed)` |
| `src.backend.jax.random.uniform` | Function | `(shape, minval, maxval, dtype, seed)` |
| `src.backend.jax.random_seed_dtype` | Function | `()` |
| `src.backend.jax.rnn.cudnn_ok` | Function | `(args, kwargs)` |
| `src.backend.jax.rnn.gru` | Function | `(args, kwargs)` |
| `src.backend.jax.rnn.lstm` | Function | `(args, kwargs)` |
| `src.backend.jax.rnn.rnn` | Function | `(step_function, inputs, initial_states, go_backwards, mask, constants, unroll, input_length, time_major, zero_output_for_mask, return_all_outputs)` |
| `src.backend.jax.rnn.stateless_scope` | Object | `` |
| `src.backend.jax.rnn.tree` | Object | `` |
| `src.backend.jax.rnn.unstack` | Function | `(x, axis)` |
| `src.backend.jax.scatter` | Function | `(indices, values, shape)` |
| `src.backend.jax.shape` | Function | `(x)` |
| `src.backend.jax.sparse.axis_shape_dims_for_broadcast_in_dim` | Function | `(axis, input_shape, insert_dims)` |
| `src.backend.jax.sparse.bcoo_add_indices` | Function | `(x1, x2, sum_duplicates)` |
| `src.backend.jax.sparse.densifying_unary` | Function | `(func)` |
| `src.backend.jax.sparse.elementwise_binary_union` | Function | `(linear, use_sparsify)` |
| `src.backend.jax.sparse.elementwise_division` | Function | `(func)` |
| `src.backend.jax.sparse.elementwise_unary` | Function | `(linear)` |
| `src.backend.jax.sparse.jax_utils` | Object | `` |
| `src.backend.jax.stop_gradient` | Function | `(variable)` |
| `src.backend.jax.tensorboard.jax` | Object | `` |
| `src.backend.jax.tensorboard.start_batch_trace` | Function | `(batch)` |
| `src.backend.jax.tensorboard.start_trace` | Function | `(logdir)` |
| `src.backend.jax.tensorboard.stop_batch_trace` | Function | `(batch_trace_context)` |
| `src.backend.jax.tensorboard.stop_trace` | Function | `(save)` |
| `src.backend.jax.trainer.EpochIterator` | Class | `(x, y, sample_weight, batch_size, steps_per_epoch, shuffle, class_weight, steps_per_execution)` |
| `src.backend.jax.trainer.JAXEpochIterator` | Class | `(...)` |
| `src.backend.jax.trainer.JAXTrainer` | Class | `()` |
| `src.backend.jax.trainer.array_slicing` | Object | `` |
| `src.backend.jax.trainer.backend` | Object | `` |
| `src.backend.jax.trainer.base_trainer` | Object | `` |
| `src.backend.jax.trainer.callbacks_module` | Object | `` |
| `src.backend.jax.trainer.config` | Object | `` |
| `src.backend.jax.trainer.data_adapter_utils` | Object | `` |
| `src.backend.jax.trainer.distribution_lib` | Object | `` |
| `src.backend.jax.trainer.is_nnx_enabled` | Function | `()` |
| `src.backend.jax.trainer.jax_distribution_lib` | Object | `` |
| `src.backend.jax.trainer.jit` | Object | `` |
| `src.backend.jax.trainer.optimizers_module` | Object | `` |
| `src.backend.jax.trainer.traceback_utils` | Object | `` |
| `src.backend.jax.trainer.tree` | Object | `` |
| `src.backend.jax.vectorized_map` | Function | `(function, elements)` |
| `src.backend.keras_export` | Class | `(path)` |
| `src.backend.linalg` | Object | `` |
| `src.backend.lstm` | Function | `(args, kwargs)` |
| `src.backend.math` | Object | `` |
| `src.backend.name_scope` | Class | `(...)` |
| `src.backend.nn` | Object | `` |
| `src.backend.numpy.IS_THREAD_SAFE` | Object | `` |
| `src.backend.numpy.SUPPORTS_RAGGED_TENSORS` | Object | `` |
| `src.backend.numpy.SUPPORTS_SPARSE_TENSORS` | Object | `` |
| `src.backend.numpy.Variable` | Class | `(...)` |
| `src.backend.numpy.cast` | Function | `(x, dtype)` |
| `src.backend.numpy.compute_output_spec` | Function | `(fn, args, kwargs)` |
| `src.backend.numpy.cond` | Function | `(pred, true_fn, false_fn)` |
| `src.backend.numpy.convert_to_numpy` | Function | `(x)` |
| `src.backend.numpy.convert_to_tensor` | Function | `(x, dtype, sparse, ragged)` |
| `src.backend.numpy.core.IS_THREAD_SAFE` | Object | `` |
| `src.backend.numpy.core.KerasTensor` | Class | `(shape, dtype, sparse, ragged, record_history, name, kwargs)` |
| `src.backend.numpy.core.KerasVariable` | Class | `(initializer, shape, dtype, trainable, autocast, aggregation, synchronization, name, kwargs)` |
| `src.backend.numpy.core.SUPPORTS_RAGGED_TENSORS` | Object | `` |
| `src.backend.numpy.core.SUPPORTS_SPARSE_TENSORS` | Object | `` |
| `src.backend.numpy.core.StatelessScope` | Class | `(state_mapping, collect_losses, initialize_variables)` |
| `src.backend.numpy.core.SymbolicScope` | Class | `(...)` |
| `src.backend.numpy.core.Variable` | Class | `(...)` |
| `src.backend.numpy.core.associative_scan` | Function | `(f, elems, reverse, axis)` |
| `src.backend.numpy.core.cast` | Function | `(x, dtype)` |
| `src.backend.numpy.core.compute_output_spec` | Function | `(fn, args, kwargs)` |
| `src.backend.numpy.core.cond` | Function | `(pred, true_fn, false_fn)` |
| `src.backend.numpy.core.convert_to_numpy` | Function | `(x)` |
| `src.backend.numpy.core.convert_to_tensor` | Function | `(x, dtype, sparse, ragged)` |
| `src.backend.numpy.core.custom_gradient` | Class | `(fun)` |
| `src.backend.numpy.core.device_scope` | Function | `(device_name)` |
| `src.backend.numpy.core.fori_loop` | Function | `(lower, upper, body_fun, init_val)` |
| `src.backend.numpy.core.is_tensor` | Function | `(x)` |
| `src.backend.numpy.core.map` | Function | `(f, xs)` |
| `src.backend.numpy.core.random_seed_dtype` | Function | `()` |
| `src.backend.numpy.core.remat` | Function | `(f)` |
| `src.backend.numpy.core.result_type` | Function | `(dtypes)` |
| `src.backend.numpy.core.scan` | Function | `(f, init, xs, length, reverse, unroll)` |
| `src.backend.numpy.core.scatter` | Function | `(indices, values, shape)` |
| `src.backend.numpy.core.scatter_update` | Function | `(inputs, indices, updates)` |
| `src.backend.numpy.core.shape` | Function | `(x)` |
| `src.backend.numpy.core.slice` | Function | `(inputs, start_indices, shape)` |
| `src.backend.numpy.core.slice_along_axis` | Function | `(x, start, stop, step, axis)` |
| `src.backend.numpy.core.slice_update` | Function | `(inputs, start_indices, updates)` |
| `src.backend.numpy.core.standardize_dtype` | Function | `(dtype)` |
| `src.backend.numpy.core.stop_gradient` | Function | `(variable)` |
| `src.backend.numpy.core.switch` | Function | `(index, branches, operands)` |
| `src.backend.numpy.core.tree` | Object | `` |
| `src.backend.numpy.core.unstack` | Function | `(x, num, axis)` |
| `src.backend.numpy.core.vectorized_map` | Function | `(function, elements)` |
| `src.backend.numpy.core.while_loop` | Function | `(cond, body, loop_vars, maximum_iterations)` |
| `src.backend.numpy.cudnn_ok` | Function | `(args, kwargs)` |
| `src.backend.numpy.device_scope` | Function | `(device_name)` |
| `src.backend.numpy.export.NumpyExportArchive` | Class | `(...)` |
| `src.backend.numpy.gru` | Function | `(args, kwargs)` |
| `src.backend.numpy.image.AFFINE_TRANSFORM_FILL_MODES` | Object | `` |
| `src.backend.numpy.image.AFFINE_TRANSFORM_INTERPOLATIONS` | Object | `` |
| `src.backend.numpy.image.MAP_COORDINATES_FILL_MODES` | Object | `` |
| `src.backend.numpy.image.RESIZE_INTERPOLATIONS` | Object | `` |
| `src.backend.numpy.image.SCALE_AND_TRANSLATE_METHODS` | Object | `` |
| `src.backend.numpy.image.affine_transform` | Function | `(images, transform, interpolation, fill_mode, fill_value, data_format)` |
| `src.backend.numpy.image.backend` | Object | `` |
| `src.backend.numpy.image.compute_homography_matrix` | Function | `(start_points, end_points)` |
| `src.backend.numpy.image.convert_to_tensor` | Function | `(x, dtype, sparse, ragged)` |
| `src.backend.numpy.image.draw_seed` | Function | `(seed)` |
| `src.backend.numpy.image.elastic_transform` | Function | `(images, alpha, sigma, interpolation, fill_mode, fill_value, seed, data_format)` |
| `src.backend.numpy.image.gaussian_blur` | Function | `(images, kernel_size, sigma, data_format)` |
| `src.backend.numpy.image.hsv_to_rgb` | Function | `(images, data_format)` |
| `src.backend.numpy.image.map_coordinates` | Function | `(inputs, coordinates, order, fill_mode, fill_value)` |
| `src.backend.numpy.image.perspective_transform` | Function | `(images, start_points, end_points, interpolation, fill_value, data_format)` |
| `src.backend.numpy.image.resize` | Function | `(images, size, interpolation, antialias, crop_to_aspect_ratio, pad_to_aspect_ratio, fill_mode, fill_value, data_format)` |
| `src.backend.numpy.image.rgb_to_grayscale` | Function | `(images, data_format)` |
| `src.backend.numpy.image.rgb_to_hsv` | Function | `(images, data_format)` |
| `src.backend.numpy.image.scale_and_translate` | Function | `(images, output_shape, scale, translation, spatial_dims, method, antialias)` |
| `src.backend.numpy.image.scipy` | Object | `` |
| `src.backend.numpy.is_tensor` | Function | `(x)` |
| `src.backend.numpy.layer.NumpyLayer` | Class | `(...)` |
| `src.backend.numpy.linalg.cholesky` | Function | `(a, upper)` |
| `src.backend.numpy.linalg.cholesky_inverse` | Function | `(a, upper)` |
| `src.backend.numpy.linalg.convert_to_tensor` | Function | `(x, dtype, sparse, ragged)` |
| `src.backend.numpy.linalg.det` | Function | `(a)` |
| `src.backend.numpy.linalg.dtypes` | Object | `` |
| `src.backend.numpy.linalg.eig` | Function | `(a)` |
| `src.backend.numpy.linalg.eigh` | Function | `(a)` |
| `src.backend.numpy.linalg.inv` | Function | `(a)` |
| `src.backend.numpy.linalg.jvp` | Function | `(fun, primals, tangents, has_aux)` |
| `src.backend.numpy.linalg.lstsq` | Function | `(a, b, rcond)` |
| `src.backend.numpy.linalg.lu_factor` | Function | `(a)` |
| `src.backend.numpy.linalg.norm` | Function | `(x, ord, axis, keepdims)` |
| `src.backend.numpy.linalg.qr` | Function | `(x, mode)` |
| `src.backend.numpy.linalg.solve` | Function | `(a, b)` |
| `src.backend.numpy.linalg.solve_triangular` | Function | `(a, b, lower)` |
| `src.backend.numpy.linalg.standardize_dtype` | Function | `(dtype)` |
| `src.backend.numpy.linalg.svd` | Function | `(x, full_matrices, compute_uv)` |
| `src.backend.numpy.lstm` | Function | `(args, kwargs)` |
| `src.backend.numpy.math.convert_to_tensor` | Function | `(x, dtype, sparse, ragged)` |
| `src.backend.numpy.math.dtypes` | Object | `` |
| `src.backend.numpy.math.erf` | Function | `(x)` |
| `src.backend.numpy.math.erfinv` | Function | `(x)` |
| `src.backend.numpy.math.extract_sequences` | Function | `(x, sequence_length, sequence_stride)` |
| `src.backend.numpy.math.fft` | Function | `(x)` |
| `src.backend.numpy.math.fft2` | Function | `(x)` |
| `src.backend.numpy.math.ifft2` | Function | `(x)` |
| `src.backend.numpy.math.in_top_k` | Function | `(targets, predictions, k)` |
| `src.backend.numpy.math.irfft` | Function | `(x, fft_length)` |
| `src.backend.numpy.math.istft` | Function | `(x, sequence_length, sequence_stride, fft_length, length, window, center)` |
| `src.backend.numpy.math.jax_fft` | Function | `(x)` |
| `src.backend.numpy.math.jax_fft2` | Function | `(x)` |
| `src.backend.numpy.math.logdet` | Function | `(x)` |
| `src.backend.numpy.math.logsumexp` | Function | `(x, axis, keepdims)` |
| `src.backend.numpy.math.norm` | Function | `(x, ord, axis, keepdims)` |
| `src.backend.numpy.math.qr` | Function | `(x, mode)` |
| `src.backend.numpy.math.rfft` | Function | `(x, fft_length)` |
| `src.backend.numpy.math.rsqrt` | Function | `(x)` |
| `src.backend.numpy.math.scipy` | Object | `` |
| `src.backend.numpy.math.segment_max` | Function | `(data, segment_ids, num_segments, sorted)` |
| `src.backend.numpy.math.segment_sum` | Function | `(data, segment_ids, num_segments, sorted)` |
| `src.backend.numpy.math.solve` | Function | `(a, b)` |
| `src.backend.numpy.math.standardize_dtype` | Function | `(dtype)` |
| `src.backend.numpy.math.stft` | Function | `(x, sequence_length, sequence_stride, fft_length, window, center)` |
| `src.backend.numpy.math.top_k` | Function | `(x, k, sorted)` |
| `src.backend.numpy.name_scope` | Class | `(name, caller, deduplicate, override_parent)` |
| `src.backend.numpy.nn.adaptive_average_pool` | Function | `(inputs, output_size, data_format)` |
| `src.backend.numpy.nn.adaptive_max_pool` | Function | `(inputs, output_size, data_format)` |
| `src.backend.numpy.nn.average_pool` | Function | `(inputs, pool_size, strides, padding, data_format)` |
| `src.backend.numpy.nn.backend` | Object | `` |
| `src.backend.numpy.nn.batch_normalization` | Function | `(x, mean, variance, axis, offset, scale, epsilon)` |
| `src.backend.numpy.nn.binary_crossentropy` | Function | `(target, output, from_logits)` |
| `src.backend.numpy.nn.cast` | Function | `(x, dtype)` |
| `src.backend.numpy.nn.categorical_crossentropy` | Function | `(target, output, from_logits, axis)` |
| `src.backend.numpy.nn.celu` | Function | `(x, alpha)` |
| `src.backend.numpy.nn.compute_adaptive_pooling_window_sizes` | Function | `(input_dim, output_dim)` |
| `src.backend.numpy.nn.compute_conv_transpose_padding_args_for_jax` | Function | `(input_shape, kernel_shape, strides, padding, output_padding, dilation_rate)` |
| `src.backend.numpy.nn.conv` | Function | `(inputs, kernel, strides, padding, data_format, dilation_rate)` |
| `src.backend.numpy.nn.conv_transpose` | Function | `(inputs, kernel, strides, padding, output_padding, data_format, dilation_rate)` |
| `src.backend.numpy.nn.convert_to_tensor` | Function | `(x, dtype, sparse, ragged)` |
| `src.backend.numpy.nn.ctc_decode` | Function | `(inputs, sequence_lengths, strategy, beam_width, top_paths, merge_repeated, mask_index)` |
| `src.backend.numpy.nn.ctc_loss` | Function | `(target, output, target_length, output_length, mask_index)` |
| `src.backend.numpy.nn.depthwise_conv` | Function | `(inputs, kernel, strides, padding, data_format, dilation_rate)` |
| `src.backend.numpy.nn.dot_product_attention` | Function | `(query, key, value, bias, mask, scale, is_causal, flash_attention, attn_logits_soft_cap)` |
| `src.backend.numpy.nn.elu` | Function | `(x, alpha)` |
| `src.backend.numpy.nn.gelu` | Function | `(x, approximate)` |
| `src.backend.numpy.nn.glu` | Function | `(x, axis)` |
| `src.backend.numpy.nn.hard_shrink` | Function | `(x, threshold)` |
| `src.backend.numpy.nn.hard_sigmoid` | Function | `(x)` |
| `src.backend.numpy.nn.hard_silu` | Function | `(x)` |
| `src.backend.numpy.nn.hard_tanh` | Function | `(x)` |
| `src.backend.numpy.nn.is_tensor` | Function | `(x)` |
| `src.backend.numpy.nn.leaky_relu` | Function | `(x, negative_slope)` |
| `src.backend.numpy.nn.log_sigmoid` | Function | `(x)` |
| `src.backend.numpy.nn.log_softmax` | Function | `(x, axis)` |
| `src.backend.numpy.nn.max_pool` | Function | `(inputs, pool_size, strides, padding, data_format)` |
| `src.backend.numpy.nn.moments` | Function | `(x, axes, keepdims, synchronized)` |
| `src.backend.numpy.nn.multi_hot` | Function | `(x, num_classes, axis, dtype, sparse)` |
| `src.backend.numpy.nn.one_hot` | Function | `(x, num_classes, axis, dtype, sparse)` |
| `src.backend.numpy.nn.psnr` | Function | `(x1, x2, max_val)` |
| `src.backend.numpy.nn.relu` | Function | `(x)` |
| `src.backend.numpy.nn.relu6` | Function | `(x)` |
| `src.backend.numpy.nn.scipy` | Object | `` |
| `src.backend.numpy.nn.selu` | Function | `(x)` |
| `src.backend.numpy.nn.separable_conv` | Function | `(inputs, depthwise_kernel, pointwise_kernel, strides, padding, data_format, dilation_rate)` |
| `src.backend.numpy.nn.sigmoid` | Function | `(x)` |
| `src.backend.numpy.nn.silu` | Function | `(x)` |
| `src.backend.numpy.nn.soft_shrink` | Function | `(x, threshold)` |
| `src.backend.numpy.nn.softmax` | Function | `(x, axis)` |
| `src.backend.numpy.nn.softplus` | Function | `(x)` |
| `src.backend.numpy.nn.softsign` | Function | `(x)` |
| `src.backend.numpy.nn.sparse_categorical_crossentropy` | Function | `(target, output, from_logits, axis)` |
| `src.backend.numpy.nn.sparse_plus` | Function | `(x)` |
| `src.backend.numpy.nn.sparse_sigmoid` | Function | `(x)` |
| `src.backend.numpy.nn.sparsemax` | Function | `(x, axis)` |
| `src.backend.numpy.nn.squareplus` | Function | `(x, b)` |
| `src.backend.numpy.nn.tanh` | Function | `(x)` |
| `src.backend.numpy.nn.tanh_shrink` | Function | `(x)` |
| `src.backend.numpy.nn.threshold` | Function | `(x, threshold, default_value)` |
| `src.backend.numpy.nn.unfold` | Function | `(input, kernel_size, dilation, padding, stride)` |
| `src.backend.numpy.numpy.abs` | Function | `(x)` |
| `src.backend.numpy.numpy.absolute` | Function | `(x)` |
| `src.backend.numpy.numpy.add` | Function | `(x1, x2)` |
| `src.backend.numpy.numpy.all` | Function | `(x, axis, keepdims)` |
| `src.backend.numpy.numpy.amax` | Function | `(x, axis, keepdims)` |
| `src.backend.numpy.numpy.amin` | Function | `(x, axis, keepdims)` |
| `src.backend.numpy.numpy.angle` | Function | `(x)` |
| `src.backend.numpy.numpy.any` | Function | `(x, axis, keepdims)` |
| `src.backend.numpy.numpy.append` | Function | `(x1, x2, axis)` |
| `src.backend.numpy.numpy.arange` | Function | `(start, stop, step, dtype)` |
| `src.backend.numpy.numpy.arccos` | Function | `(x)` |
| `src.backend.numpy.numpy.arccosh` | Function | `(x)` |
| `src.backend.numpy.numpy.arcsin` | Function | `(x)` |
| `src.backend.numpy.numpy.arcsinh` | Function | `(x)` |
| `src.backend.numpy.numpy.arctan` | Function | `(x)` |
| `src.backend.numpy.numpy.arctan2` | Function | `(x1, x2)` |
| `src.backend.numpy.numpy.arctanh` | Function | `(x)` |
| `src.backend.numpy.numpy.argmax` | Function | `(x, axis, keepdims)` |
| `src.backend.numpy.numpy.argmin` | Function | `(x, axis, keepdims)` |
| `src.backend.numpy.numpy.argpartition` | Function | `(x, kth, axis)` |
| `src.backend.numpy.numpy.argsort` | Function | `(x, axis)` |
| `src.backend.numpy.numpy.array` | Function | `(x, dtype)` |
| `src.backend.numpy.numpy.array_split` | Function | `(x, indices_or_sections, axis)` |
| `src.backend.numpy.numpy.average` | Function | `(x, axis, weights)` |
| `src.backend.numpy.numpy.bartlett` | Function | `(x)` |
| `src.backend.numpy.numpy.bincount` | Function | `(x, weights, minlength, sparse)` |
| `src.backend.numpy.numpy.bitwise_and` | Function | `(x, y)` |
| `src.backend.numpy.numpy.bitwise_invert` | Function | `(x)` |
| `src.backend.numpy.numpy.bitwise_left_shift` | Function | `(x, y)` |
| `src.backend.numpy.numpy.bitwise_not` | Function | `(x)` |
| `src.backend.numpy.numpy.bitwise_or` | Function | `(x, y)` |
| `src.backend.numpy.numpy.bitwise_right_shift` | Function | `(x, y)` |
| `src.backend.numpy.numpy.bitwise_xor` | Function | `(x, y)` |
| `src.backend.numpy.numpy.blackman` | Function | `(x)` |
| `src.backend.numpy.numpy.broadcast_to` | Function | `(x, shape)` |
| `src.backend.numpy.numpy.cbrt` | Function | `(x)` |
| `src.backend.numpy.numpy.ceil` | Function | `(x)` |
| `src.backend.numpy.numpy.clip` | Function | `(x, x_min, x_max)` |
| `src.backend.numpy.numpy.concatenate` | Function | `(xs, axis)` |
| `src.backend.numpy.numpy.config` | Object | `` |
| `src.backend.numpy.numpy.conj` | Function | `(x)` |
| `src.backend.numpy.numpy.conjugate` | Function | `(x)` |
| `src.backend.numpy.numpy.convert_to_tensor` | Function | `(x, dtype, sparse, ragged)` |
| `src.backend.numpy.numpy.copy` | Function | `(x)` |
| `src.backend.numpy.numpy.corrcoef` | Function | `(x)` |
| `src.backend.numpy.numpy.correlate` | Function | `(x1, x2, mode)` |
| `src.backend.numpy.numpy.cos` | Function | `(x)` |
| `src.backend.numpy.numpy.cosh` | Function | `(x)` |
| `src.backend.numpy.numpy.count_nonzero` | Function | `(x, axis)` |
| `src.backend.numpy.numpy.cross` | Function | `(x1, x2, axisa, axisb, axisc, axis)` |
| `src.backend.numpy.numpy.cumprod` | Function | `(x, axis, dtype)` |
| `src.backend.numpy.numpy.cumsum` | Function | `(x, axis, dtype)` |
| `src.backend.numpy.numpy.deg2rad` | Function | `(x)` |
| `src.backend.numpy.numpy.diag` | Function | `(x, k)` |
| `src.backend.numpy.numpy.diagflat` | Function | `(x, k)` |
| `src.backend.numpy.numpy.diagonal` | Function | `(x, offset, axis1, axis2)` |
| `src.backend.numpy.numpy.diff` | Function | `(a, n, axis)` |
| `src.backend.numpy.numpy.digitize` | Function | `(x, bins)` |
| `src.backend.numpy.numpy.divide` | Function | `(x1, x2)` |
| `src.backend.numpy.numpy.divide_no_nan` | Function | `(x1, x2)` |
| `src.backend.numpy.numpy.dot` | Function | `(x1, x2)` |
| `src.backend.numpy.numpy.dtypes` | Object | `` |
| `src.backend.numpy.numpy.einsum` | Function | `(subscripts, operands, kwargs)` |
| `src.backend.numpy.numpy.empty` | Function | `(shape, dtype)` |
| `src.backend.numpy.numpy.empty_like` | Function | `(x, dtype)` |
| `src.backend.numpy.numpy.equal` | Function | `(x1, x2)` |
| `src.backend.numpy.numpy.exp` | Function | `(x)` |
| `src.backend.numpy.numpy.exp2` | Function | `(x)` |
| `src.backend.numpy.numpy.expand_dims` | Function | `(x, axis)` |
| `src.backend.numpy.numpy.expm1` | Function | `(x)` |
| `src.backend.numpy.numpy.eye` | Function | `(N, M, k, dtype)` |
| `src.backend.numpy.numpy.flip` | Function | `(x, axis)` |
| `src.backend.numpy.numpy.floor` | Function | `(x)` |
| `src.backend.numpy.numpy.floor_divide` | Function | `(x1, x2)` |
| `src.backend.numpy.numpy.full` | Function | `(shape, fill_value, dtype)` |
| `src.backend.numpy.numpy.full_like` | Function | `(x, fill_value, dtype)` |
| `src.backend.numpy.numpy.gcd` | Function | `(x1, x2)` |
| `src.backend.numpy.numpy.greater` | Function | `(x1, x2)` |
| `src.backend.numpy.numpy.greater_equal` | Function | `(x1, x2)` |
| `src.backend.numpy.numpy.hamming` | Function | `(x)` |
| `src.backend.numpy.numpy.hanning` | Function | `(x)` |
| `src.backend.numpy.numpy.heaviside` | Function | `(x1, x2)` |
| `src.backend.numpy.numpy.histogram` | Function | `(x, bins, range)` |
| `src.backend.numpy.numpy.hstack` | Function | `(xs)` |
| `src.backend.numpy.numpy.hypot` | Function | `(x1, x2)` |
| `src.backend.numpy.numpy.identity` | Function | `(n, dtype)` |
| `src.backend.numpy.numpy.imag` | Function | `(x)` |
| `src.backend.numpy.numpy.inner` | Function | `(x1, x2)` |
| `src.backend.numpy.numpy.isclose` | Function | `(x1, x2, rtol, atol, equal_nan)` |
| `src.backend.numpy.numpy.isfinite` | Function | `(x)` |
| `src.backend.numpy.numpy.isin` | Function | `(x1, x2, assume_unique, invert)` |
| `src.backend.numpy.numpy.isinf` | Function | `(x)` |
| `src.backend.numpy.numpy.isnan` | Function | `(x)` |
| `src.backend.numpy.numpy.isneginf` | Function | `(x)` |
| `src.backend.numpy.numpy.isposinf` | Function | `(x)` |
| `src.backend.numpy.numpy.isreal` | Function | `(x)` |
| `src.backend.numpy.numpy.kaiser` | Function | `(x, beta)` |
| `src.backend.numpy.numpy.kron` | Function | `(x1, x2)` |
| `src.backend.numpy.numpy.lcm` | Function | `(x1, x2)` |
| `src.backend.numpy.numpy.ldexp` | Function | `(x1, x2)` |
| `src.backend.numpy.numpy.left_shift` | Function | `(x, y)` |
| `src.backend.numpy.numpy.less` | Function | `(x1, x2)` |
| `src.backend.numpy.numpy.less_equal` | Function | `(x1, x2)` |
| `src.backend.numpy.numpy.linspace` | Function | `(start, stop, num, endpoint, retstep, dtype, axis)` |
| `src.backend.numpy.numpy.log` | Function | `(x)` |
| `src.backend.numpy.numpy.log10` | Function | `(x)` |
| `src.backend.numpy.numpy.log1p` | Function | `(x)` |
| `src.backend.numpy.numpy.log2` | Function | `(x)` |
| `src.backend.numpy.numpy.logaddexp` | Function | `(x1, x2)` |
| `src.backend.numpy.numpy.logaddexp2` | Function | `(x1, x2)` |
| `src.backend.numpy.numpy.logical_and` | Function | `(x1, x2)` |
| `src.backend.numpy.numpy.logical_not` | Function | `(x)` |
| `src.backend.numpy.numpy.logical_or` | Function | `(x1, x2)` |
| `src.backend.numpy.numpy.logical_xor` | Function | `(x1, x2)` |
| `src.backend.numpy.numpy.logspace` | Function | `(start, stop, num, endpoint, base, dtype, axis)` |
| `src.backend.numpy.numpy.matmul` | Function | `(x1, x2)` |
| `src.backend.numpy.numpy.max` | Function | `(x, axis, keepdims, initial)` |
| `src.backend.numpy.numpy.maximum` | Function | `(x1, x2)` |
| `src.backend.numpy.numpy.mean` | Function | `(x, axis, keepdims)` |
| `src.backend.numpy.numpy.median` | Function | `(x, axis, keepdims)` |
| `src.backend.numpy.numpy.meshgrid` | Function | `(x, indexing)` |
| `src.backend.numpy.numpy.min` | Function | `(x, axis, keepdims, initial)` |
| `src.backend.numpy.numpy.minimum` | Function | `(x1, x2)` |
| `src.backend.numpy.numpy.mod` | Function | `(x1, x2)` |
| `src.backend.numpy.numpy.moveaxis` | Function | `(x, source, destination)` |
| `src.backend.numpy.numpy.multiply` | Function | `(x1, x2)` |
| `src.backend.numpy.numpy.nan_to_num` | Function | `(x, nan, posinf, neginf)` |
| `src.backend.numpy.numpy.ndim` | Function | `(x)` |
| `src.backend.numpy.numpy.negative` | Function | `(x)` |
| `src.backend.numpy.numpy.nonzero` | Function | `(x)` |
| `src.backend.numpy.numpy.not_equal` | Function | `(x1, x2)` |
| `src.backend.numpy.numpy.ones` | Function | `(shape, dtype)` |
| `src.backend.numpy.numpy.ones_like` | Function | `(x, dtype)` |
| `src.backend.numpy.numpy.outer` | Function | `(x1, x2)` |
| `src.backend.numpy.numpy.pad` | Function | `(x, pad_width, mode, constant_values)` |
| `src.backend.numpy.numpy.power` | Function | `(x1, x2)` |
| `src.backend.numpy.numpy.prod` | Function | `(x, axis, keepdims, dtype)` |
| `src.backend.numpy.numpy.quantile` | Function | `(x, q, axis, method, keepdims)` |
| `src.backend.numpy.numpy.ravel` | Function | `(x)` |
| `src.backend.numpy.numpy.real` | Function | `(x)` |
| `src.backend.numpy.numpy.reciprocal` | Function | `(x)` |
| `src.backend.numpy.numpy.repeat` | Function | `(x, repeats, axis)` |
| `src.backend.numpy.numpy.reshape` | Function | `(x, newshape)` |
| `src.backend.numpy.numpy.right_shift` | Function | `(x, y)` |
| `src.backend.numpy.numpy.roll` | Function | `(x, shift, axis)` |
| `src.backend.numpy.numpy.rot90` | Function | `(array, k, axes)` |
| `src.backend.numpy.numpy.round` | Function | `(x, decimals)` |
| `src.backend.numpy.numpy.searchsorted` | Function | `(sorted_sequence, values, side)` |
| `src.backend.numpy.numpy.select` | Function | `(condlist, choicelist, default)` |
| `src.backend.numpy.numpy.sign` | Function | `(x)` |
| `src.backend.numpy.numpy.signbit` | Function | `(x)` |
| `src.backend.numpy.numpy.sin` | Function | `(x)` |
| `src.backend.numpy.numpy.sinh` | Function | `(x)` |
| `src.backend.numpy.numpy.size` | Function | `(x)` |
| `src.backend.numpy.numpy.slogdet` | Function | `(x)` |
| `src.backend.numpy.numpy.sort` | Function | `(x, axis)` |
| `src.backend.numpy.numpy.split` | Function | `(x, indices_or_sections, axis)` |
| `src.backend.numpy.numpy.sqrt` | Function | `(x)` |
| `src.backend.numpy.numpy.square` | Function | `(x)` |
| `src.backend.numpy.numpy.squeeze` | Function | `(x, axis)` |
| `src.backend.numpy.numpy.stack` | Function | `(x, axis)` |
| `src.backend.numpy.numpy.standardize_axis_for_numpy` | Function | `(axis)` |
| `src.backend.numpy.numpy.standardize_dtype` | Function | `(dtype)` |
| `src.backend.numpy.numpy.std` | Function | `(x, axis, keepdims)` |
| `src.backend.numpy.numpy.subtract` | Function | `(x1, x2)` |
| `src.backend.numpy.numpy.sum` | Function | `(x, axis, keepdims)` |
| `src.backend.numpy.numpy.swapaxes` | Function | `(x, axis1, axis2)` |
| `src.backend.numpy.numpy.take` | Function | `(x, indices, axis)` |
| `src.backend.numpy.numpy.take_along_axis` | Function | `(x, indices, axis)` |
| `src.backend.numpy.numpy.tan` | Function | `(x)` |
| `src.backend.numpy.numpy.tanh` | Function | `(x)` |
| `src.backend.numpy.numpy.tensordot` | Function | `(x1, x2, axes)` |
| `src.backend.numpy.numpy.tile` | Function | `(x, repeats)` |
| `src.backend.numpy.numpy.trace` | Function | `(x, offset, axis1, axis2)` |
| `src.backend.numpy.numpy.transpose` | Function | `(x, axes)` |
| `src.backend.numpy.numpy.trapezoid` | Function | `(y, x, dx, axis)` |
| `src.backend.numpy.numpy.tree` | Object | `` |
| `src.backend.numpy.numpy.tri` | Function | `(N, M, k, dtype)` |
| `src.backend.numpy.numpy.tril` | Function | `(x, k)` |
| `src.backend.numpy.numpy.triu` | Function | `(x, k)` |
| `src.backend.numpy.numpy.true_divide` | Function | `(x1, x2)` |
| `src.backend.numpy.numpy.trunc` | Function | `(x)` |
| `src.backend.numpy.numpy.unravel_index` | Function | `(indices, shape)` |
| `src.backend.numpy.numpy.vander` | Function | `(x, N, increasing)` |
| `src.backend.numpy.numpy.var` | Function | `(x, axis, keepdims)` |
| `src.backend.numpy.numpy.vdot` | Function | `(x1, x2)` |
| `src.backend.numpy.numpy.vectorize` | Function | `(pyfunc, excluded, signature)` |
| `src.backend.numpy.numpy.view` | Function | `(x, dtype)` |
| `src.backend.numpy.numpy.vstack` | Function | `(xs)` |
| `src.backend.numpy.numpy.where` | Function | `(condition, x1, x2)` |
| `src.backend.numpy.numpy.zeros` | Function | `(shape, dtype)` |
| `src.backend.numpy.numpy.zeros_like` | Function | `(x, dtype)` |
| `src.backend.numpy.random.SeedGenerator` | Class | `(seed, name, kwargs)` |
| `src.backend.numpy.random.beta` | Function | `(shape, alpha, beta, dtype, seed)` |
| `src.backend.numpy.random.binomial` | Function | `(shape, counts, probabilities, dtype, seed)` |
| `src.backend.numpy.random.categorical` | Function | `(logits, num_samples, dtype, seed)` |
| `src.backend.numpy.random.draw_seed` | Function | `(seed)` |
| `src.backend.numpy.random.dropout` | Function | `(inputs, rate, noise_shape, seed)` |
| `src.backend.numpy.random.floatx` | Function | `()` |
| `src.backend.numpy.random.gamma` | Function | `(shape, alpha, dtype, seed)` |
| `src.backend.numpy.random.make_default_seed` | Function | `()` |
| `src.backend.numpy.random.normal` | Function | `(shape, mean, stddev, dtype, seed)` |
| `src.backend.numpy.random.randint` | Function | `(shape, minval, maxval, dtype, seed)` |
| `src.backend.numpy.random.shuffle` | Function | `(x, axis, seed)` |
| `src.backend.numpy.random.softmax` | Function | `(x, axis)` |
| `src.backend.numpy.random.truncated_normal` | Function | `(shape, mean, stddev, dtype, seed)` |
| `src.backend.numpy.random.uniform` | Function | `(shape, minval, maxval, dtype, seed)` |
| `src.backend.numpy.random_seed_dtype` | Function | `()` |
| `src.backend.numpy.rnn.cudnn_ok` | Function | `(args, kwargs)` |
| `src.backend.numpy.rnn.gru` | Function | `(args, kwargs)` |
| `src.backend.numpy.rnn.lstm` | Function | `(args, kwargs)` |
| `src.backend.numpy.rnn.numpy_scan` | Function | `(f, init, xs, reverse, mask)` |
| `src.backend.numpy.rnn.rnn` | Function | `(step_function, inputs, initial_states, go_backwards, mask, constants, unroll, input_length, time_major, zero_output_for_mask, return_all_outputs)` |
| `src.backend.numpy.rnn.tree` | Object | `` |
| `src.backend.numpy.rnn.unstack` | Function | `(x, axis)` |
| `src.backend.numpy.shape` | Function | `(x)` |
| `src.backend.numpy.trainer.EpochIterator` | Class | `(x, y, sample_weight, batch_size, steps_per_epoch, shuffle, class_weight, steps_per_execution)` |
| `src.backend.numpy.trainer.KerasTensor` | Class | `(shape, dtype, sparse, ragged, record_history, name, kwargs)` |
| `src.backend.numpy.trainer.NumpyTrainer` | Class | `()` |
| `src.backend.numpy.trainer.backend` | Object | `` |
| `src.backend.numpy.trainer.base_trainer` | Object | `` |
| `src.backend.numpy.trainer.callbacks_module` | Object | `` |
| `src.backend.numpy.trainer.data_adapter_utils` | Object | `` |
| `src.backend.numpy.trainer.is_tensor` | Function | `(x)` |
| `src.backend.numpy.trainer.standardize_dtype` | Function | `(dtype)` |
| `src.backend.numpy.trainer.traceback_utils` | Object | `` |
| `src.backend.numpy.trainer.tree` | Object | `` |
| `src.backend.numpy.vectorized_map` | Function | `(function, elements)` |
| `src.backend.openvino.IS_THREAD_SAFE` | Object | `` |
| `src.backend.openvino.SUPPORTS_RAGGED_TENSORS` | Object | `` |
| `src.backend.openvino.SUPPORTS_SPARSE_TENSORS` | Object | `` |
| `src.backend.openvino.Variable` | Class | `(...)` |
| `src.backend.openvino.cast` | Function | `(x, dtype)` |
| `src.backend.openvino.compute_output_spec` | Function | `(fn, args, kwargs)` |
| `src.backend.openvino.cond` | Function | `(pred, true_fn, false_fn)` |
| `src.backend.openvino.convert_to_numpy` | Function | `(x)` |
| `src.backend.openvino.convert_to_tensor` | Function | `(x, dtype, sparse, ragged)` |
| `src.backend.openvino.core.DTYPES_MAX` | Object | `` |
| `src.backend.openvino.core.DTYPES_MIN` | Object | `` |
| `src.backend.openvino.core.IS_THREAD_SAFE` | Object | `` |
| `src.backend.openvino.core.KerasTensor` | Class | `(shape, dtype, sparse, ragged, record_history, name, kwargs)` |
| `src.backend.openvino.core.KerasVariable` | Class | `(initializer, shape, dtype, trainable, autocast, aggregation, synchronization, name, kwargs)` |
| `src.backend.openvino.core.OPENVINO_DTYPES` | Object | `` |
| `src.backend.openvino.core.OpenVINOKerasTensor` | Class | `(x, data)` |
| `src.backend.openvino.core.SUPPORTS_RAGGED_TENSORS` | Object | `` |
| `src.backend.openvino.core.SUPPORTS_SPARSE_TENSORS` | Object | `` |
| `src.backend.openvino.core.StatelessScope` | Class | `(state_mapping, collect_losses, initialize_variables)` |
| `src.backend.openvino.core.Variable` | Class | `(...)` |
| `src.backend.openvino.core.align_operand_types` | Function | `(x1, x2, op_name)` |
| `src.backend.openvino.core.cast` | Function | `(x, dtype)` |
| `src.backend.openvino.core.compute_output_spec` | Function | `(fn, args, kwargs)` |
| `src.backend.openvino.core.cond` | Function | `(pred, true_fn, false_fn)` |
| `src.backend.openvino.core.convert_to_numpy` | Function | `(x)` |
| `src.backend.openvino.core.convert_to_tensor` | Function | `(x, dtype, sparse, ragged)` |
| `src.backend.openvino.core.custom_gradient` | Function | `(fun)` |
| `src.backend.openvino.core.device_scope` | Function | `(device_name)` |
| `src.backend.openvino.core.dtypes` | Object | `` |
| `src.backend.openvino.core.fori_loop` | Function | `(lower, upper, body_fun, init_val)` |
| `src.backend.openvino.core.get_device` | Function | `()` |
| `src.backend.openvino.core.get_ov_output` | Function | `(x, ov_type)` |
| `src.backend.openvino.core.is_tensor` | Function | `(x)` |
| `src.backend.openvino.core.ov_to_keras_type` | Function | `(ov_type)` |
| `src.backend.openvino.core.random_seed_dtype` | Function | `()` |
| `src.backend.openvino.core.remat` | Function | `(f)` |
| `src.backend.openvino.core.result_type` | Function | `(dtypes)` |
| `src.backend.openvino.core.scan` | Function | `(f, init, xs, length, reverse, unroll)` |
| `src.backend.openvino.core.scatter` | Function | `(indices, values, shape)` |
| `src.backend.openvino.core.scatter_update` | Function | `(inputs, indices, updates)` |
| `src.backend.openvino.core.shape` | Function | `(x)` |
| `src.backend.openvino.core.slice` | Function | `(inputs, start_indices, shape)` |
| `src.backend.openvino.core.slice_update` | Function | `(inputs, start_indices, updates)` |
| `src.backend.openvino.core.standardize_dtype` | Function | `(dtype)` |
| `src.backend.openvino.core.stop_gradient` | Function | `(variable)` |
| `src.backend.openvino.core.tree` | Object | `` |
| `src.backend.openvino.core.unstack` | Function | `(x, num, axis)` |
| `src.backend.openvino.core.vectorized_map` | Function | `(function, elements)` |
| `src.backend.openvino.core.while_loop` | Function | `(cond, body, loop_vars, maximum_iterations)` |
| `src.backend.openvino.cudnn_ok` | Function | `(args, kwargs)` |
| `src.backend.openvino.device_scope` | Function | `(device_name)` |
| `src.backend.openvino.export.OpenvinoExportArchive` | Class | `(...)` |
| `src.backend.openvino.gru` | Function | `(args, kwargs)` |
| `src.backend.openvino.image.affine_transform` | Function | `(images, transform, interpolation, fill_mode, fill_value, data_format)` |
| `src.backend.openvino.image.elastic_transform` | Function | `(images, alpha, sigma, interpolation, fill_mode, fill_value, seed, data_format)` |
| `src.backend.openvino.image.gaussian_blur` | Function | `(images, kernel_size, sigma, data_format)` |
| `src.backend.openvino.image.map_coordinates` | Function | `(inputs, coordinates, order, fill_mode, fill_value)` |
| `src.backend.openvino.image.perspective_transform` | Function | `(images, start_points, end_points, interpolation, fill_value, data_format)` |
| `src.backend.openvino.image.resize` | Function | `(image, size, interpolation, antialias, crop_to_aspect_ratio, pad_to_aspect_ratio, fill_mode, fill_value, data_format)` |
| `src.backend.openvino.image.rgb_to_grayscale` | Function | `(images, data_format)` |
| `src.backend.openvino.image.scale_and_translate` | Function | `(images, output_shape, scale, translation, spatial_dims, method, antialias)` |
| `src.backend.openvino.is_tensor` | Function | `(x)` |
| `src.backend.openvino.layer.OpenvinoLayer` | Class | `(...)` |
| `src.backend.openvino.linalg.cholesky` | Function | `(a, upper)` |
| `src.backend.openvino.linalg.cholesky_inverse` | Function | `(a, upper)` |
| `src.backend.openvino.linalg.det` | Function | `(a)` |
| `src.backend.openvino.linalg.eig` | Function | `(a)` |
| `src.backend.openvino.linalg.eigh` | Function | `(a)` |
| `src.backend.openvino.linalg.inv` | Function | `(a)` |
| `src.backend.openvino.linalg.jvp` | Function | `(fun, primals, tangents, has_aux)` |
| `src.backend.openvino.linalg.lstsq` | Function | `(a, b, rcond)` |
| `src.backend.openvino.linalg.lu_factor` | Function | `(a)` |
| `src.backend.openvino.linalg.norm` | Function | `(x, ord, axis, keepdims)` |
| `src.backend.openvino.linalg.qr` | Function | `(x, mode)` |
| `src.backend.openvino.linalg.solve` | Function | `(a, b)` |
| `src.backend.openvino.linalg.solve_triangular` | Function | `(a, b, lower)` |
| `src.backend.openvino.linalg.svd` | Function | `(x, full_matrices, compute_uv)` |
| `src.backend.openvino.lstm` | Function | `(args, kwargs)` |
| `src.backend.openvino.math.OpenVINOKerasTensor` | Class | `(x, data)` |
| `src.backend.openvino.math.erf` | Function | `(x)` |
| `src.backend.openvino.math.erfinv` | Function | `(x)` |
| `src.backend.openvino.math.extract_sequences` | Function | `(x, sequence_length, sequence_stride)` |
| `src.backend.openvino.math.fft` | Function | `(x)` |
| `src.backend.openvino.math.fft2` | Function | `(x)` |
| `src.backend.openvino.math.get_ov_output` | Function | `(x, ov_type)` |
| `src.backend.openvino.math.in_top_k` | Function | `(targets, predictions, k)` |
| `src.backend.openvino.math.irfft` | Function | `(x, fft_length)` |
| `src.backend.openvino.math.istft` | Function | `(x, sequence_length, sequence_stride, fft_length, length, window, center)` |
| `src.backend.openvino.math.logsumexp` | Function | `(x, axis, keepdims)` |
| `src.backend.openvino.math.norm` | Function | `(x, ord, axis, keepdims)` |
| `src.backend.openvino.math.qr` | Function | `(x, mode)` |
| `src.backend.openvino.math.rfft` | Function | `(x, fft_length)` |
| `src.backend.openvino.math.rsqrt` | Function | `(x)` |
| `src.backend.openvino.math.segment_max` | Function | `(data, segment_ids, num_segments, sorted)` |
| `src.backend.openvino.math.segment_sum` | Function | `(data, segment_ids, num_segments, sorted)` |
| `src.backend.openvino.math.solve` | Function | `(a, b)` |
| `src.backend.openvino.math.stft` | Function | `(x, sequence_length, sequence_stride, fft_length, window, center)` |
| `src.backend.openvino.math.top_k` | Function | `(x, k, sorted)` |
| `src.backend.openvino.name_scope` | Class | `(name, caller, deduplicate, override_parent)` |
| `src.backend.openvino.nn.OPENVINO_DTYPES` | Object | `` |
| `src.backend.openvino.nn.OpenVINOKerasTensor` | Class | `(x, data)` |
| `src.backend.openvino.nn.adaptive_average_pool` | Function | `(inputs, output_size, data_format)` |
| `src.backend.openvino.nn.adaptive_max_pool` | Function | `(inputs, output_size, data_format)` |
| `src.backend.openvino.nn.average_pool` | Function | `(inputs, pool_size, strides, padding, data_format)` |
| `src.backend.openvino.nn.backend` | Object | `` |
| `src.backend.openvino.nn.batch_normalization` | Function | `(x, mean, variance, axis, offset, scale, epsilon)` |
| `src.backend.openvino.nn.binary_crossentropy` | Function | `(target, output, from_logits)` |
| `src.backend.openvino.nn.categorical_crossentropy` | Function | `(target, output, from_logits, axis)` |
| `src.backend.openvino.nn.celu` | Function | `(x, alpha)` |
| `src.backend.openvino.nn.conv` | Function | `(inputs, kernel, strides, padding, data_format, dilation_rate)` |
| `src.backend.openvino.nn.conv_transpose` | Function | `(inputs, kernel, strides, padding, output_padding, data_format, dilation_rate)` |
| `src.backend.openvino.nn.ctc_decode` | Function | `(inputs, sequence_lengths, strategy, beam_width, top_paths, merge_repeated, mask_index)` |
| `src.backend.openvino.nn.ctc_loss` | Function | `(target, output, target_length, output_length, mask_index)` |
| `src.backend.openvino.nn.depthwise_conv` | Function | `(inputs, kernel, strides, padding, data_format, dilation_rate)` |
| `src.backend.openvino.nn.dot_product_attention` | Function | `(query, key, value, bias, mask, scale, is_causal, flash_attention, attn_logits_soft_cap)` |
| `src.backend.openvino.nn.elu` | Function | `(x, alpha)` |
| `src.backend.openvino.nn.gelu` | Function | `(x, approximate)` |
| `src.backend.openvino.nn.get_ov_output` | Function | `(x, ov_type)` |
| `src.backend.openvino.nn.hard_shrink` | Function | `(x, threshold)` |
| `src.backend.openvino.nn.hard_sigmoid` | Function | `(x)` |
| `src.backend.openvino.nn.hard_silu` | Function | `(x)` |
| `src.backend.openvino.nn.hard_tanh` | Function | `(x)` |
| `src.backend.openvino.nn.leaky_relu` | Function | `(x, negative_slope)` |
| `src.backend.openvino.nn.log_sigmoid` | Function | `(x)` |
| `src.backend.openvino.nn.log_softmax` | Function | `(x, axis)` |
| `src.backend.openvino.nn.max_pool` | Function | `(inputs, pool_size, strides, padding, data_format)` |
| `src.backend.openvino.nn.moments` | Function | `(x, axes, keepdims, synchronized)` |
| `src.backend.openvino.nn.multi_hot` | Function | `(x, num_classes, axis, dtype, sparse)` |
| `src.backend.openvino.nn.one_hot` | Function | `(x, num_classes, axis, dtype, sparse)` |
| `src.backend.openvino.nn.psnr` | Function | `(x1, x2, max_val)` |
| `src.backend.openvino.nn.relu` | Function | `(x)` |
| `src.backend.openvino.nn.relu6` | Function | `(x)` |
| `src.backend.openvino.nn.selu` | Function | `(x)` |
| `src.backend.openvino.nn.separable_conv` | Function | `(inputs, depthwise_kernel, pointwise_kernel, strides, padding, data_format, dilation_rate)` |
| `src.backend.openvino.nn.sigmoid` | Function | `(x)` |
| `src.backend.openvino.nn.silu` | Function | `(x)` |
| `src.backend.openvino.nn.soft_shrink` | Function | `(x, threshold)` |
| `src.backend.openvino.nn.softmax` | Function | `(x, axis)` |
| `src.backend.openvino.nn.softplus` | Function | `(x)` |
| `src.backend.openvino.nn.softsign` | Function | `(x)` |
| `src.backend.openvino.nn.sparse_categorical_crossentropy` | Function | `(target, output, from_logits, axis)` |
| `src.backend.openvino.nn.sparse_plus` | Function | `(x)` |
| `src.backend.openvino.nn.sparse_sigmoid` | Function | `(x)` |
| `src.backend.openvino.nn.squareplus` | Function | `(x, b)` |
| `src.backend.openvino.nn.tanh` | Function | `(x)` |
| `src.backend.openvino.nn.tanh_shrink` | Function | `(x)` |
| `src.backend.openvino.nn.threshold` | Function | `(x, threshold, default_value)` |
| `src.backend.openvino.nn.unfold` | Function | `(input, kernel_size, dilation, padding, stride)` |
| `src.backend.openvino.numpy.DTYPES_MAX` | Object | `` |
| `src.backend.openvino.numpy.DTYPES_MIN` | Object | `` |
| `src.backend.openvino.numpy.OPENVINO_DTYPES` | Object | `` |
| `src.backend.openvino.numpy.OpenVINOKerasTensor` | Class | `(x, data)` |
| `src.backend.openvino.numpy.abs` | Function | `(x)` |
| `src.backend.openvino.numpy.absolute` | Function | `(x)` |
| `src.backend.openvino.numpy.add` | Function | `(x1, x2)` |
| `src.backend.openvino.numpy.all` | Function | `(x, axis, keepdims)` |
| `src.backend.openvino.numpy.amax` | Function | `(x, axis, keepdims)` |
| `src.backend.openvino.numpy.amin` | Function | `(x, axis, keepdims)` |
| `src.backend.openvino.numpy.angle` | Function | `(x)` |
| `src.backend.openvino.numpy.any` | Function | `(x, axis, keepdims)` |
| `src.backend.openvino.numpy.append` | Function | `(x1, x2, axis)` |
| `src.backend.openvino.numpy.arange` | Function | `(start, stop, step, dtype)` |
| `src.backend.openvino.numpy.arccos` | Function | `(x)` |
| `src.backend.openvino.numpy.arccosh` | Function | `(x)` |
| `src.backend.openvino.numpy.arcsin` | Function | `(x)` |
| `src.backend.openvino.numpy.arcsinh` | Function | `(x)` |
| `src.backend.openvino.numpy.arctan` | Function | `(x)` |
| `src.backend.openvino.numpy.arctan2` | Function | `(x1, x2)` |
| `src.backend.openvino.numpy.arctanh` | Function | `(x)` |
| `src.backend.openvino.numpy.argmax` | Function | `(x, axis, keepdims)` |
| `src.backend.openvino.numpy.argmin` | Function | `(x, axis, keepdims)` |
| `src.backend.openvino.numpy.argpartition` | Function | `(x, kth, axis)` |
| `src.backend.openvino.numpy.argsort` | Function | `(x, axis)` |
| `src.backend.openvino.numpy.array` | Function | `(x, dtype)` |
| `src.backend.openvino.numpy.array_split` | Function | `(x, indices_or_sections, axis)` |
| `src.backend.openvino.numpy.average` | Function | `(x, axis, weights)` |
| `src.backend.openvino.numpy.bartlett` | Function | `(x)` |
| `src.backend.openvino.numpy.bincount` | Function | `(x, weights, minlength, sparse)` |
| `src.backend.openvino.numpy.blackman` | Function | `(x)` |
| `src.backend.openvino.numpy.broadcast_to` | Function | `(x, shape)` |
| `src.backend.openvino.numpy.cbrt` | Function | `(x)` |
| `src.backend.openvino.numpy.ceil` | Function | `(x)` |
| `src.backend.openvino.numpy.clip` | Function | `(x, x_min, x_max)` |
| `src.backend.openvino.numpy.concatenate` | Function | `(xs, axis)` |
| `src.backend.openvino.numpy.config` | Object | `` |
| `src.backend.openvino.numpy.conj` | Function | `(x)` |
| `src.backend.openvino.numpy.conjugate` | Function | `(x)` |
| `src.backend.openvino.numpy.convert_to_tensor` | Function | `(x, dtype, sparse, ragged)` |
| `src.backend.openvino.numpy.copy` | Function | `(x)` |
| `src.backend.openvino.numpy.corrcoef` | Function | `(x)` |
| `src.backend.openvino.numpy.correlate` | Function | `(x1, x2, mode)` |
| `src.backend.openvino.numpy.cos` | Function | `(x)` |
| `src.backend.openvino.numpy.cosh` | Function | `(x)` |
| `src.backend.openvino.numpy.count_nonzero` | Function | `(x, axis)` |
| `src.backend.openvino.numpy.cross` | Function | `(x1, x2, axisa, axisb, axisc, axis)` |
| `src.backend.openvino.numpy.cumprod` | Function | `(x, axis, dtype)` |
| `src.backend.openvino.numpy.cumsum` | Function | `(x, axis, dtype)` |
| `src.backend.openvino.numpy.deg2rad` | Function | `(x)` |
| `src.backend.openvino.numpy.diag` | Function | `(x, k)` |
| `src.backend.openvino.numpy.diagonal` | Function | `(x, offset, axis1, axis2)` |
| `src.backend.openvino.numpy.diff` | Function | `(a, n, axis)` |
| `src.backend.openvino.numpy.digitize` | Function | `(x, bins)` |
| `src.backend.openvino.numpy.divide` | Function | `(x1, x2)` |
| `src.backend.openvino.numpy.divide_no_nan` | Function | `(x1, x2)` |
| `src.backend.openvino.numpy.dot` | Function | `(x1, x2)` |
| `src.backend.openvino.numpy.dtypes` | Object | `` |
| `src.backend.openvino.numpy.einsum` | Function | `(subscripts, operands, kwargs)` |
| `src.backend.openvino.numpy.empty` | Function | `(shape, dtype)` |
| `src.backend.openvino.numpy.empty_like` | Function | `(x, dtype)` |
| `src.backend.openvino.numpy.equal` | Function | `(x1, x2)` |
| `src.backend.openvino.numpy.exp` | Function | `(x)` |
| `src.backend.openvino.numpy.expand_dims` | Function | `(x, axis)` |
| `src.backend.openvino.numpy.expm1` | Function | `(x)` |
| `src.backend.openvino.numpy.eye` | Function | `(N, M, k, dtype)` |
| `src.backend.openvino.numpy.flip` | Function | `(x, axis)` |
| `src.backend.openvino.numpy.floor` | Function | `(x)` |
| `src.backend.openvino.numpy.floor_divide` | Function | `(x1, x2)` |
| `src.backend.openvino.numpy.full` | Function | `(shape, fill_value, dtype)` |
| `src.backend.openvino.numpy.full_like` | Function | `(x, fill_value, dtype)` |
| `src.backend.openvino.numpy.gcd` | Function | `(x1, x2)` |
| `src.backend.openvino.numpy.get_ov_output` | Function | `(x, ov_type)` |
| `src.backend.openvino.numpy.greater` | Function | `(x1, x2)` |
| `src.backend.openvino.numpy.greater_equal` | Function | `(x1, x2)` |
| `src.backend.openvino.numpy.hamming` | Function | `(x)` |
| `src.backend.openvino.numpy.heaviside` | Function | `(x1, x2)` |
| `src.backend.openvino.numpy.hstack` | Function | `(xs)` |
| `src.backend.openvino.numpy.hypot` | Function | `(x1, x2)` |
| `src.backend.openvino.numpy.identity` | Function | `(n, dtype)` |
| `src.backend.openvino.numpy.imag` | Function | `(x)` |
| `src.backend.openvino.numpy.isclose` | Function | `(x1, x2, rtol, atol, equal_nan)` |
| `src.backend.openvino.numpy.isfinite` | Function | `(x)` |
| `src.backend.openvino.numpy.isin` | Function | `(x1, x2, assume_unique, invert)` |
| `src.backend.openvino.numpy.isinf` | Function | `(x)` |
| `src.backend.openvino.numpy.isnan` | Function | `(x)` |
| `src.backend.openvino.numpy.isneginf` | Function | `(x)` |
| `src.backend.openvino.numpy.isposinf` | Function | `(x)` |
| `src.backend.openvino.numpy.isreal` | Function | `(x)` |
| `src.backend.openvino.numpy.kaiser` | Function | `(x, beta)` |
| `src.backend.openvino.numpy.kron` | Function | `(x1, x2)` |
| `src.backend.openvino.numpy.lcm` | Function | `(x1, x2)` |
| `src.backend.openvino.numpy.ldexp` | Function | `(x1, x2)` |
| `src.backend.openvino.numpy.less` | Function | `(x1, x2)` |
| `src.backend.openvino.numpy.less_equal` | Function | `(x1, x2)` |
| `src.backend.openvino.numpy.linspace` | Function | `(start, stop, num, endpoint, retstep, dtype, axis)` |
| `src.backend.openvino.numpy.log` | Function | `(x)` |
| `src.backend.openvino.numpy.log10` | Function | `(x)` |
| `src.backend.openvino.numpy.log1p` | Function | `(x)` |
| `src.backend.openvino.numpy.log2` | Function | `(x)` |
| `src.backend.openvino.numpy.logaddexp` | Function | `(x1, x2)` |
| `src.backend.openvino.numpy.logaddexp2` | Function | `(x1, x2)` |
| `src.backend.openvino.numpy.logical_and` | Function | `(x1, x2)` |
| `src.backend.openvino.numpy.logical_not` | Function | `(x)` |
| `src.backend.openvino.numpy.logical_or` | Function | `(x1, x2)` |
| `src.backend.openvino.numpy.logical_xor` | Function | `(x1, x2)` |
| `src.backend.openvino.numpy.logspace` | Function | `(start, stop, num, endpoint, base, dtype, axis)` |
| `src.backend.openvino.numpy.matmul` | Function | `(x1, x2)` |
| `src.backend.openvino.numpy.max` | Function | `(x, axis, keepdims, initial)` |
| `src.backend.openvino.numpy.maximum` | Function | `(x1, x2)` |
| `src.backend.openvino.numpy.mean` | Function | `(x, axis, keepdims)` |
| `src.backend.openvino.numpy.median` | Function | `(x, axis, keepdims)` |
| `src.backend.openvino.numpy.meshgrid` | Function | `(x, indexing)` |
| `src.backend.openvino.numpy.min` | Function | `(x, axis, keepdims, initial)` |
| `src.backend.openvino.numpy.minimum` | Function | `(x1, x2)` |
| `src.backend.openvino.numpy.mod` | Function | `(x1, x2)` |
| `src.backend.openvino.numpy.moveaxis` | Function | `(x, source, destination)` |
| `src.backend.openvino.numpy.multiply` | Function | `(x1, x2)` |
| `src.backend.openvino.numpy.nan_to_num` | Function | `(x, nan, posinf, neginf)` |
| `src.backend.openvino.numpy.ndim` | Function | `(x)` |
| `src.backend.openvino.numpy.negative` | Function | `(x)` |
| `src.backend.openvino.numpy.nonzero` | Function | `(x)` |
| `src.backend.openvino.numpy.not_equal` | Function | `(x1, x2)` |
| `src.backend.openvino.numpy.ones` | Function | `(shape, dtype)` |
| `src.backend.openvino.numpy.ones_like` | Function | `(x, dtype)` |
| `src.backend.openvino.numpy.outer` | Function | `(x1, x2)` |
| `src.backend.openvino.numpy.ov_to_keras_type` | Function | `(ov_type)` |
| `src.backend.openvino.numpy.pad` | Function | `(x, pad_width, mode, constant_values)` |
| `src.backend.openvino.numpy.power` | Function | `(x1, x2)` |
| `src.backend.openvino.numpy.prod` | Function | `(x, axis, keepdims, dtype)` |
| `src.backend.openvino.numpy.quantile` | Function | `(x, q, axis, method, keepdims)` |
| `src.backend.openvino.numpy.ravel` | Function | `(x)` |
| `src.backend.openvino.numpy.real` | Function | `(x)` |
| `src.backend.openvino.numpy.reciprocal` | Function | `(x)` |
| `src.backend.openvino.numpy.repeat` | Function | `(x, repeats, axis)` |
| `src.backend.openvino.numpy.reshape` | Function | `(x, newshape)` |
| `src.backend.openvino.numpy.roll` | Function | `(x, shift, axis)` |
| `src.backend.openvino.numpy.round` | Function | `(x, decimals)` |
| `src.backend.openvino.numpy.select` | Function | `(condlist, choicelist, default)` |
| `src.backend.openvino.numpy.sign` | Function | `(x)` |
| `src.backend.openvino.numpy.signbit` | Function | `(x)` |
| `src.backend.openvino.numpy.sin` | Function | `(x)` |
| `src.backend.openvino.numpy.sinh` | Function | `(x)` |
| `src.backend.openvino.numpy.size` | Function | `(x)` |
| `src.backend.openvino.numpy.slogdet` | Function | `(x)` |
| `src.backend.openvino.numpy.sort` | Function | `(x, axis)` |
| `src.backend.openvino.numpy.split` | Function | `(x, indices_or_sections, axis)` |
| `src.backend.openvino.numpy.sqrt` | Function | `(x)` |
| `src.backend.openvino.numpy.square` | Function | `(x)` |
| `src.backend.openvino.numpy.squeeze` | Function | `(x, axis)` |
| `src.backend.openvino.numpy.stack` | Function | `(x, axis)` |
| `src.backend.openvino.numpy.standardize_dtype` | Function | `(dtype)` |
| `src.backend.openvino.numpy.std` | Function | `(x, axis, keepdims)` |
| `src.backend.openvino.numpy.subtract` | Function | `(x1, x2)` |
| `src.backend.openvino.numpy.sum` | Function | `(x, axis, keepdims)` |
| `src.backend.openvino.numpy.swapaxes` | Function | `(x, axis1, axis2)` |
| `src.backend.openvino.numpy.take` | Function | `(x, indices, axis)` |
| `src.backend.openvino.numpy.take_along_axis` | Function | `(x, indices, axis)` |
| `src.backend.openvino.numpy.tan` | Function | `(x)` |
| `src.backend.openvino.numpy.tanh` | Function | `(x)` |
| `src.backend.openvino.numpy.tensordot` | Function | `(x1, x2, axes)` |
| `src.backend.openvino.numpy.tile` | Function | `(x, repeats)` |
| `src.backend.openvino.numpy.trace` | Function | `(x, offset, axis1, axis2)` |
| `src.backend.openvino.numpy.transpose` | Function | `(x, axes)` |
| `src.backend.openvino.numpy.trapezoid` | Function | `(y, x, dx, axis)` |
| `src.backend.openvino.numpy.tri` | Function | `(N, M, k, dtype)` |
| `src.backend.openvino.numpy.tril` | Function | `(x, k)` |
| `src.backend.openvino.numpy.triu` | Function | `(x, k)` |
| `src.backend.openvino.numpy.true_divide` | Function | `(x1, x2)` |
| `src.backend.openvino.numpy.vander` | Function | `(x, N, increasing)` |
| `src.backend.openvino.numpy.var` | Function | `(x, axis, keepdims)` |
| `src.backend.openvino.numpy.vdot` | Function | `(x1, x2)` |
| `src.backend.openvino.numpy.vectorize` | Function | `(pyfunc, excluded, signature)` |
| `src.backend.openvino.numpy.view` | Function | `(x, dtype)` |
| `src.backend.openvino.numpy.vstack` | Function | `(xs)` |
| `src.backend.openvino.numpy.where` | Function | `(condition, x1, x2)` |
| `src.backend.openvino.numpy.zeros` | Function | `(shape, dtype)` |
| `src.backend.openvino.numpy.zeros_like` | Function | `(x, dtype)` |
| `src.backend.openvino.random.OPENVINO_DTYPES` | Object | `` |
| `src.backend.openvino.random.OpenVINOKerasTensor` | Class | `(x, data)` |
| `src.backend.openvino.random.SeedGenerator` | Class | `(seed, name, kwargs)` |
| `src.backend.openvino.random.beta` | Function | `(shape, alpha, beta, dtype, seed)` |
| `src.backend.openvino.random.binomial` | Function | `(shape, counts, probabilities, dtype, seed)` |
| `src.backend.openvino.random.categorical` | Function | `(logits, num_samples, dtype, seed)` |
| `src.backend.openvino.random.convert_to_numpy` | Function | `(x)` |
| `src.backend.openvino.random.draw_seed` | Function | `(seed)` |
| `src.backend.openvino.random.dropout` | Function | `(inputs, rate, noise_shape, seed)` |
| `src.backend.openvino.random.floatx` | Function | `()` |
| `src.backend.openvino.random.gamma` | Function | `(shape, alpha, dtype, seed)` |
| `src.backend.openvino.random.get_ov_output` | Function | `(x, ov_type)` |
| `src.backend.openvino.random.make_default_seed` | Function | `()` |
| `src.backend.openvino.random.normal` | Function | `(shape, mean, stddev, dtype, seed)` |
| `src.backend.openvino.random.randint` | Function | `(shape, minval, maxval, dtype, seed)` |
| `src.backend.openvino.random.shuffle` | Function | `(x, axis, seed)` |
| `src.backend.openvino.random.truncated_normal` | Function | `(shape, mean, stddev, dtype, seed)` |
| `src.backend.openvino.random.uniform` | Function | `(shape, minval, maxval, dtype, seed)` |
| `src.backend.openvino.random_seed_dtype` | Function | `()` |
| `src.backend.openvino.rnn.cudnn_ok` | Function | `(args, kwargs)` |
| `src.backend.openvino.rnn.gru` | Function | `(args, kwargs)` |
| `src.backend.openvino.rnn.lstm` | Function | `(args, kwargs)` |
| `src.backend.openvino.rnn.numpy_scan` | Function | `(f, init, xs, reverse, mask)` |
| `src.backend.openvino.rnn.rnn` | Function | `(step_function, inputs, initial_states, go_backwards, mask, constants, unroll, input_length, time_major, zero_output_for_mask, return_all_outputs)` |
| `src.backend.openvino.rnn.unstack` | Function | `(x, axis)` |
| `src.backend.openvino.shape` | Function | `(x)` |
| `src.backend.openvino.trainer.EpochIterator` | Class | `(x, y, sample_weight, batch_size, steps_per_epoch, shuffle, class_weight, steps_per_execution)` |
| `src.backend.openvino.trainer.OPENVINO_DTYPES` | Object | `` |
| `src.backend.openvino.trainer.OpenVINOKerasTensor` | Class | `(x, data)` |
| `src.backend.openvino.trainer.OpenVINOTrainer` | Class | `()` |
| `src.backend.openvino.trainer.backend` | Object | `` |
| `src.backend.openvino.trainer.base_trainer` | Object | `` |
| `src.backend.openvino.trainer.callbacks_module` | Object | `` |
| `src.backend.openvino.trainer.data_adapter_utils` | Object | `` |
| `src.backend.openvino.trainer.get_device` | Function | `()` |
| `src.backend.openvino.trainer.traceback_utils` | Object | `` |
| `src.backend.openvino.trainer.tree` | Object | `` |
| `src.backend.openvino.vectorized_map` | Function | `(function, elements)` |
| `src.backend.random` | Object | `` |
| `src.backend.random_seed_dtype` | Function | `()` |
| `src.backend.result_type` | Function | `(dtypes)` |
| `src.backend.rnn` | Object | `` |
| `src.backend.scatter` | Function | `(indices, values, shape)` |
| `src.backend.set_epsilon` | Function | `(value)` |
| `src.backend.set_floatx` | Function | `(value)` |
| `src.backend.set_image_data_format` | Function | `(data_format)` |
| `src.backend.set_keras_mask` | Function | `(x, mask)` |
| `src.backend.shape` | Function | `(x)` |
| `src.backend.standardize_data_format` | Function | `(data_format)` |
| `src.backend.standardize_dtype` | Function | `(dtype)` |
| `src.backend.standardize_shape` | Function | `(shape)` |
| `src.backend.stop_gradient` | Function | `(variable)` |
| `src.backend.tensorboard` | Object | `` |
| `src.backend.tensorflow.IS_THREAD_SAFE` | Object | `` |
| `src.backend.tensorflow.SUPPORTS_RAGGED_TENSORS` | Object | `` |
| `src.backend.tensorflow.SUPPORTS_SPARSE_TENSORS` | Object | `` |
| `src.backend.tensorflow.Variable` | Class | `(...)` |
| `src.backend.tensorflow.cast` | Function | `(x, dtype)` |
| `src.backend.tensorflow.compute_output_spec` | Function | `(fn, args, kwargs)` |
| `src.backend.tensorflow.cond` | Function | `(pred, true_fn, false_fn)` |
| `src.backend.tensorflow.convert_to_numpy` | Function | `(x)` |
| `src.backend.tensorflow.convert_to_tensor` | Function | `(x, dtype, sparse, ragged)` |
| `src.backend.tensorflow.core.IS_THREAD_SAFE` | Object | `` |
| `src.backend.tensorflow.core.KerasTensor` | Class | `(shape, dtype, sparse, ragged, record_history, name, kwargs)` |
| `src.backend.tensorflow.core.KerasVariable` | Class | `(initializer, shape, dtype, trainable, autocast, aggregation, synchronization, name, kwargs)` |
| `src.backend.tensorflow.core.SUPPORTS_RAGGED_TENSORS` | Object | `` |
| `src.backend.tensorflow.core.SUPPORTS_SPARSE_TENSORS` | Object | `` |
| `src.backend.tensorflow.core.StatelessScope` | Class | `(state_mapping, collect_losses, initialize_variables)` |
| `src.backend.tensorflow.core.SymbolicScope` | Class | `(...)` |
| `src.backend.tensorflow.core.Variable` | Class | `(...)` |
| `src.backend.tensorflow.core.associative_scan` | Function | `(f, elems, reverse, axis)` |
| `src.backend.tensorflow.core.auto_name` | Function | `(prefix)` |
| `src.backend.tensorflow.core.base_name_scope` | Class | `(name, caller, deduplicate, override_parent)` |
| `src.backend.tensorflow.core.cast` | Function | `(x, dtype)` |
| `src.backend.tensorflow.core.compute_output_spec` | Function | `(fn, args, kwargs)` |
| `src.backend.tensorflow.core.cond` | Function | `(pred, true_fn, false_fn)` |
| `src.backend.tensorflow.core.convert_to_numpy` | Function | `(x)` |
| `src.backend.tensorflow.core.convert_to_tensor` | Function | `(x, dtype, sparse, ragged)` |
| `src.backend.tensorflow.core.custom_gradient` | Function | `(fun)` |
| `src.backend.tensorflow.core.device_scope` | Function | `(device_name)` |
| `src.backend.tensorflow.core.fori_loop` | Function | `(lower, upper, body_fun, init_val)` |
| `src.backend.tensorflow.core.global_state` | Object | `` |
| `src.backend.tensorflow.core.in_stateless_scope` | Function | `()` |
| `src.backend.tensorflow.core.is_int_dtype` | Function | `(dtype)` |
| `src.backend.tensorflow.core.is_tensor` | Function | `(x)` |
| `src.backend.tensorflow.core.map` | Function | `(f, xs)` |
| `src.backend.tensorflow.core.name_scope` | Class | `(name, kwargs)` |
| `src.backend.tensorflow.core.random_seed_dtype` | Function | `()` |
| `src.backend.tensorflow.core.remat` | Function | `(f)` |
| `src.backend.tensorflow.core.scan` | Function | `(f, init, xs, length, reverse, unroll)` |
| `src.backend.tensorflow.core.scatter` | Function | `(indices, values, shape)` |
| `src.backend.tensorflow.core.scatter_update` | Function | `(inputs, indices, updates)` |
| `src.backend.tensorflow.core.shape` | Function | `(x)` |
| `src.backend.tensorflow.core.slice` | Function | `(inputs, start_indices, shape)` |
| `src.backend.tensorflow.core.slice_along_axis` | Function | `(x, start, stop, step, axis)` |
| `src.backend.tensorflow.core.slice_update` | Function | `(inputs, start_indices, updates)` |
| `src.backend.tensorflow.core.sparse_to_dense` | Function | `(x, default_value)` |
| `src.backend.tensorflow.core.standardize_dtype` | Function | `(dtype)` |
| `src.backend.tensorflow.core.stop_gradient` | Function | `(variable)` |
| `src.backend.tensorflow.core.switch` | Function | `(index, branches, operands)` |
| `src.backend.tensorflow.core.tree` | Object | `` |
| `src.backend.tensorflow.core.unstack` | Function | `(x, num, axis)` |
| `src.backend.tensorflow.core.vectorized_map` | Function | `(function, elements)` |
| `src.backend.tensorflow.core.while_loop` | Function | `(cond, body, loop_vars, maximum_iterations)` |
| `src.backend.tensorflow.cudnn_ok` | Function | `(activation, recurrent_activation, unroll, use_bias, reset_after)` |
| `src.backend.tensorflow.device_scope` | Function | `(device_name)` |
| `src.backend.tensorflow.distribution_lib.distribute_value` | Function | `(value, tensor_layout)` |
| `src.backend.tensorflow.distribution_lib.list_devices` | Function | `(device_type)` |
| `src.backend.tensorflow.export.TFExportArchive` | Class | `(...)` |
| `src.backend.tensorflow.gru` | Function | `(inputs, initial_state, mask, kernel, recurrent_kernel, bias, activation, recurrent_activation, return_sequences, go_backwards, unroll, time_major, reset_after)` |
| `src.backend.tensorflow.image.AFFINE_TRANSFORM_FILL_MODES` | Object | `` |
| `src.backend.tensorflow.image.AFFINE_TRANSFORM_INTERPOLATIONS` | Object | `` |
| `src.backend.tensorflow.image.MAP_COORDINATES_FILL_MODES` | Object | `` |
| `src.backend.tensorflow.image.RESIZE_INTERPOLATIONS` | Object | `` |
| `src.backend.tensorflow.image.SCALE_AND_TRANSLATE_METHODS` | Object | `` |
| `src.backend.tensorflow.image.affine_transform` | Function | `(images, transform, interpolation, fill_mode, fill_value, data_format)` |
| `src.backend.tensorflow.image.backend` | Object | `` |
| `src.backend.tensorflow.image.compute_homography_matrix` | Function | `(start_points, end_points)` |
| `src.backend.tensorflow.image.convert_to_tensor` | Function | `(x, dtype, sparse, ragged)` |
| `src.backend.tensorflow.image.draw_seed` | Function | `(seed)` |
| `src.backend.tensorflow.image.elastic_transform` | Function | `(images, alpha, sigma, interpolation, fill_mode, fill_value, seed, data_format)` |
| `src.backend.tensorflow.image.gaussian_blur` | Function | `(images, kernel_size, sigma, data_format)` |
| `src.backend.tensorflow.image.hsv_to_rgb` | Function | `(images, data_format)` |
| `src.backend.tensorflow.image.map_coordinates` | Function | `(inputs, coordinates, order, fill_mode, fill_value)` |
| `src.backend.tensorflow.image.moveaxis` | Function | `(x, source, destination)` |
| `src.backend.tensorflow.image.perspective_transform` | Function | `(images, start_points, end_points, interpolation, fill_value, data_format)` |
| `src.backend.tensorflow.image.resize` | Function | `(images, size, interpolation, antialias, crop_to_aspect_ratio, pad_to_aspect_ratio, fill_mode, fill_value, data_format)` |
| `src.backend.tensorflow.image.rgb_to_grayscale` | Function | `(images, data_format)` |
| `src.backend.tensorflow.image.rgb_to_hsv` | Function | `(images, data_format)` |
| `src.backend.tensorflow.image.scale_and_translate` | Function | `(images, output_shape, scale, translation, spatial_dims, method, antialias)` |
| `src.backend.tensorflow.is_tensor` | Function | `(x)` |
| `src.backend.tensorflow.layer.KerasAutoTrackable` | Class | `(...)` |
| `src.backend.tensorflow.layer.TFLayer` | Class | `(args, kwargs)` |
| `src.backend.tensorflow.layer.tf_utils` | Object | `` |
| `src.backend.tensorflow.layer.tracking` | Object | `` |
| `src.backend.tensorflow.layer.tree` | Object | `` |
| `src.backend.tensorflow.linalg.cast` | Function | `(x, dtype)` |
| `src.backend.tensorflow.linalg.cholesky` | Function | `(a, upper)` |
| `src.backend.tensorflow.linalg.cholesky_inverse` | Function | `(a, upper)` |
| `src.backend.tensorflow.linalg.config` | Object | `` |
| `src.backend.tensorflow.linalg.convert_to_tensor` | Function | `(x, dtype, sparse, ragged)` |
| `src.backend.tensorflow.linalg.det` | Function | `(a)` |
| `src.backend.tensorflow.linalg.dtypes` | Object | `` |
| `src.backend.tensorflow.linalg.eig` | Function | `(a)` |
| `src.backend.tensorflow.linalg.eigh` | Function | `(a)` |
| `src.backend.tensorflow.linalg.inv` | Function | `(a)` |
| `src.backend.tensorflow.linalg.jvp` | Function | `(fun, primals, tangents, has_aux)` |
| `src.backend.tensorflow.linalg.lstsq` | Function | `(a, b, rcond)` |
| `src.backend.tensorflow.linalg.lu_factor` | Function | `(a)` |
| `src.backend.tensorflow.linalg.norm` | Function | `(x, ord, axis, keepdims)` |
| `src.backend.tensorflow.linalg.qr` | Function | `(x, mode)` |
| `src.backend.tensorflow.linalg.solve` | Function | `(a, b)` |
| `src.backend.tensorflow.linalg.solve_triangular` | Function | `(a, b, lower)` |
| `src.backend.tensorflow.linalg.standardize_dtype` | Function | `(dtype)` |
| `src.backend.tensorflow.linalg.svd` | Function | `(x, full_matrices, compute_uv)` |
| `src.backend.tensorflow.lstm` | Function | `(inputs, initial_state_h, initial_state_c, mask, kernel, recurrent_kernel, bias, activation, recurrent_activation, return_sequences, go_backwards, unroll, time_major)` |
| `src.backend.tensorflow.math.cast` | Function | `(x, dtype)` |
| `src.backend.tensorflow.math.config` | Object | `` |
| `src.backend.tensorflow.math.convert_to_tensor` | Function | `(x, dtype, sparse, ragged)` |
| `src.backend.tensorflow.math.dtypes` | Object | `` |
| `src.backend.tensorflow.math.erf` | Function | `(x)` |
| `src.backend.tensorflow.math.erfinv` | Function | `(x)` |
| `src.backend.tensorflow.math.extract_sequences` | Function | `(x, sequence_length, sequence_stride)` |
| `src.backend.tensorflow.math.fft` | Function | `(x)` |
| `src.backend.tensorflow.math.fft2` | Function | `(x)` |
| `src.backend.tensorflow.math.ifft2` | Function | `(x)` |
| `src.backend.tensorflow.math.in_top_k` | Function | `(targets, predictions, k)` |
| `src.backend.tensorflow.math.irfft` | Function | `(x, fft_length)` |
| `src.backend.tensorflow.math.istft` | Function | `(x, sequence_length, sequence_stride, fft_length, length, window, center)` |
| `src.backend.tensorflow.math.logdet` | Function | `(x)` |
| `src.backend.tensorflow.math.logsumexp` | Function | `(x, axis, keepdims)` |
| `src.backend.tensorflow.math.norm` | Function | `(x, ord, axis, keepdims)` |
| `src.backend.tensorflow.math.qr` | Function | `(x, mode)` |
| `src.backend.tensorflow.math.rfft` | Function | `(x, fft_length)` |
| `src.backend.tensorflow.math.rsqrt` | Function | `(x)` |
| `src.backend.tensorflow.math.segment_max` | Function | `(data, segment_ids, num_segments, sorted)` |
| `src.backend.tensorflow.math.segment_sum` | Function | `(data, segment_ids, num_segments, sorted)` |
| `src.backend.tensorflow.math.solve` | Function | `(a, b)` |
| `src.backend.tensorflow.math.standardize_dtype` | Function | `(dtype)` |
| `src.backend.tensorflow.math.stft` | Function | `(x, sequence_length, sequence_stride, fft_length, window, center)` |
| `src.backend.tensorflow.math.top_k` | Function | `(x, k, sorted)` |
| `src.backend.tensorflow.name_scope` | Class | `(name, kwargs)` |
| `src.backend.tensorflow.nn.adaptive_average_pool` | Function | `(inputs, output_size, data_format)` |
| `src.backend.tensorflow.nn.adaptive_max_pool` | Function | `(inputs, output_size, data_format)` |
| `src.backend.tensorflow.nn.average_pool` | Function | `(inputs, pool_size, strides, padding, data_format)` |
| `src.backend.tensorflow.nn.backend` | Object | `` |
| `src.backend.tensorflow.nn.batch_normalization` | Function | `(x, mean, variance, axis, offset, scale, epsilon)` |
| `src.backend.tensorflow.nn.binary_crossentropy` | Function | `(target, output, from_logits)` |
| `src.backend.tensorflow.nn.cast` | Function | `(x, dtype)` |
| `src.backend.tensorflow.nn.categorical_crossentropy` | Function | `(target, output, from_logits, axis)` |
| `src.backend.tensorflow.nn.celu` | Function | `(x, alpha)` |
| `src.backend.tensorflow.nn.compute_adaptive_pooling_window_sizes` | Function | `(input_dim, output_dim)` |
| `src.backend.tensorflow.nn.compute_conv_transpose_output_shape` | Function | `(input_shape, kernel_size, filters, strides, padding, output_padding, data_format, dilation_rate)` |
| `src.backend.tensorflow.nn.conv` | Function | `(inputs, kernel, strides, padding, data_format, dilation_rate)` |
| `src.backend.tensorflow.nn.conv_transpose` | Function | `(inputs, kernel, strides, padding, output_padding, data_format, dilation_rate)` |
| `src.backend.tensorflow.nn.convert_to_tensor` | Function | `(x, dtype, sparse, ragged)` |
| `src.backend.tensorflow.nn.ctc_decode` | Function | `(inputs, sequence_lengths, strategy, beam_width, top_paths, merge_repeated, mask_index)` |
| `src.backend.tensorflow.nn.ctc_loss` | Function | `(target, output, target_length, output_length, mask_index)` |
| `src.backend.tensorflow.nn.depthwise_conv` | Function | `(inputs, kernel, strides, padding, data_format, dilation_rate)` |
| `src.backend.tensorflow.nn.dot_product_attention` | Function | `(query, key, value, bias, mask, scale, is_causal, flash_attention, attn_logits_soft_cap)` |
| `src.backend.tensorflow.nn.elu` | Function | `(x, alpha)` |
| `src.backend.tensorflow.nn.gelu` | Function | `(x, approximate)` |
| `src.backend.tensorflow.nn.glu` | Function | `(x, axis)` |
| `src.backend.tensorflow.nn.hard_shrink` | Function | `(x, threshold)` |
| `src.backend.tensorflow.nn.hard_sigmoid` | Function | `(x)` |
| `src.backend.tensorflow.nn.hard_silu` | Function | `(x)` |
| `src.backend.tensorflow.nn.hard_tanh` | Function | `(x)` |
| `src.backend.tensorflow.nn.leaky_relu` | Function | `(x, negative_slope)` |
| `src.backend.tensorflow.nn.log_sigmoid` | Function | `(x)` |
| `src.backend.tensorflow.nn.log_softmax` | Function | `(x, axis)` |
| `src.backend.tensorflow.nn.max_pool` | Function | `(inputs, pool_size, strides, padding, data_format)` |
| `src.backend.tensorflow.nn.moments` | Function | `(x, axes, keepdims, synchronized)` |
| `src.backend.tensorflow.nn.multi_hot` | Function | `(x, num_classes, axis, dtype, sparse)` |
| `src.backend.tensorflow.nn.one_hot` | Function | `(x, num_classes, axis, dtype, sparse)` |
| `src.backend.tensorflow.nn.psnr` | Function | `(x1, x2, max_val)` |
| `src.backend.tensorflow.nn.relu` | Function | `(x)` |
| `src.backend.tensorflow.nn.relu6` | Function | `(x)` |
| `src.backend.tensorflow.nn.selu` | Function | `(x)` |
| `src.backend.tensorflow.nn.separable_conv` | Function | `(inputs, depthwise_kernel, pointwise_kernel, strides, padding, data_format, dilation_rate)` |
| `src.backend.tensorflow.nn.sigmoid` | Function | `(x)` |
| `src.backend.tensorflow.nn.silu` | Function | `(x)` |
| `src.backend.tensorflow.nn.soft_shrink` | Function | `(x, threshold)` |
| `src.backend.tensorflow.nn.softmax` | Function | `(x, axis)` |
| `src.backend.tensorflow.nn.softplus` | Function | `(x)` |
| `src.backend.tensorflow.nn.softsign` | Function | `(x)` |
| `src.backend.tensorflow.nn.sparse_categorical_crossentropy` | Function | `(target, output, from_logits, axis)` |
| `src.backend.tensorflow.nn.sparse_plus` | Function | `(x)` |
| `src.backend.tensorflow.nn.sparse_sigmoid` | Function | `(x)` |
| `src.backend.tensorflow.nn.sparsemax` | Function | `(x, axis)` |
| `src.backend.tensorflow.nn.squareplus` | Function | `(x, b)` |
| `src.backend.tensorflow.nn.tanh` | Function | `(x)` |
| `src.backend.tensorflow.nn.tanh_shrink` | Function | `(x)` |
| `src.backend.tensorflow.nn.threshold` | Function | `(x, threshold, default_value)` |
| `src.backend.tensorflow.nn.unfold` | Function | `(input, kernel_size, dilation, padding, stride)` |
| `src.backend.tensorflow.numpy.abs` | Function | `(x)` |
| `src.backend.tensorflow.numpy.absolute` | Function | `(x)` |
| `src.backend.tensorflow.numpy.add` | Function | `(x1, x2)` |
| `src.backend.tensorflow.numpy.all` | Function | `(x, axis, keepdims)` |
| `src.backend.tensorflow.numpy.amax` | Function | `(x, axis, keepdims)` |
| `src.backend.tensorflow.numpy.amin` | Function | `(x, axis, keepdims)` |
| `src.backend.tensorflow.numpy.angle` | Function | `(x)` |
| `src.backend.tensorflow.numpy.any` | Function | `(x, axis, keepdims)` |
| `src.backend.tensorflow.numpy.append` | Function | `(x1, x2, axis)` |
| `src.backend.tensorflow.numpy.arange` | Function | `(start, stop, step, dtype)` |
| `src.backend.tensorflow.numpy.arccos` | Function | `(x)` |
| `src.backend.tensorflow.numpy.arccosh` | Function | `(x)` |
| `src.backend.tensorflow.numpy.arcsin` | Function | `(x)` |
| `src.backend.tensorflow.numpy.arcsinh` | Function | `(x)` |
| `src.backend.tensorflow.numpy.arctan` | Function | `(x)` |
| `src.backend.tensorflow.numpy.arctan2` | Function | `(x1, x2)` |
| `src.backend.tensorflow.numpy.arctanh` | Function | `(x)` |
| `src.backend.tensorflow.numpy.argmax` | Function | `(x, axis, keepdims)` |
| `src.backend.tensorflow.numpy.argmin` | Function | `(x, axis, keepdims)` |
| `src.backend.tensorflow.numpy.argpartition` | Function | `(x, kth, axis)` |
| `src.backend.tensorflow.numpy.argsort` | Function | `(x, axis)` |
| `src.backend.tensorflow.numpy.array` | Function | `(x, dtype)` |
| `src.backend.tensorflow.numpy.array_split` | Function | `(x, indices_or_sections, axis)` |
| `src.backend.tensorflow.numpy.average` | Function | `(x, axis, weights)` |
| `src.backend.tensorflow.numpy.bartlett` | Function | `(x)` |
| `src.backend.tensorflow.numpy.bincount` | Function | `(x, weights, minlength, sparse)` |
| `src.backend.tensorflow.numpy.bitwise_and` | Function | `(x, y)` |
| `src.backend.tensorflow.numpy.bitwise_invert` | Function | `(x)` |
| `src.backend.tensorflow.numpy.bitwise_left_shift` | Function | `(x, y)` |
| `src.backend.tensorflow.numpy.bitwise_not` | Function | `(x)` |
| `src.backend.tensorflow.numpy.bitwise_or` | Function | `(x, y)` |
| `src.backend.tensorflow.numpy.bitwise_right_shift` | Function | `(x, y)` |
| `src.backend.tensorflow.numpy.bitwise_xor` | Function | `(x, y)` |
| `src.backend.tensorflow.numpy.blackman` | Function | `(x)` |
| `src.backend.tensorflow.numpy.broadcast_to` | Function | `(x, shape)` |
| `src.backend.tensorflow.numpy.canonicalize_axis` | Function | `(axis, num_dims)` |
| `src.backend.tensorflow.numpy.cast` | Function | `(x, dtype)` |
| `src.backend.tensorflow.numpy.cbrt` | Function | `(x)` |
| `src.backend.tensorflow.numpy.ceil` | Function | `(x)` |
| `src.backend.tensorflow.numpy.clip` | Function | `(x, x_min, x_max)` |
| `src.backend.tensorflow.numpy.concatenate` | Function | `(xs, axis)` |
| `src.backend.tensorflow.numpy.config` | Object | `` |
| `src.backend.tensorflow.numpy.conj` | Function | `(x)` |
| `src.backend.tensorflow.numpy.conjugate` | Function | `(x)` |
| `src.backend.tensorflow.numpy.convert_to_tensor` | Function | `(x, dtype, sparse, ragged)` |
| `src.backend.tensorflow.numpy.copy` | Function | `(x)` |
| `src.backend.tensorflow.numpy.corrcoef` | Function | `(x)` |
| `src.backend.tensorflow.numpy.correlate` | Function | `(x1, x2, mode)` |
| `src.backend.tensorflow.numpy.cos` | Function | `(x)` |
| `src.backend.tensorflow.numpy.cosh` | Function | `(x)` |
| `src.backend.tensorflow.numpy.count_nonzero` | Function | `(x, axis)` |
| `src.backend.tensorflow.numpy.cross` | Function | `(x1, x2, axisa, axisb, axisc, axis)` |
| `src.backend.tensorflow.numpy.cumprod` | Function | `(x, axis, dtype)` |
| `src.backend.tensorflow.numpy.cumsum` | Function | `(x, axis, dtype)` |
| `src.backend.tensorflow.numpy.deg2rad` | Function | `(x)` |
| `src.backend.tensorflow.numpy.diag` | Function | `(x, k)` |
| `src.backend.tensorflow.numpy.diagflat` | Function | `(x, k)` |
| `src.backend.tensorflow.numpy.diagonal` | Function | `(x, offset, axis1, axis2)` |
| `src.backend.tensorflow.numpy.diff` | Function | `(a, n, axis)` |
| `src.backend.tensorflow.numpy.digitize` | Function | `(x, bins)` |
| `src.backend.tensorflow.numpy.divide` | Function | `(x1, x2)` |
| `src.backend.tensorflow.numpy.divide_no_nan` | Function | `(x1, x2)` |
| `src.backend.tensorflow.numpy.dot` | Function | `(x1, x2)` |
| `src.backend.tensorflow.numpy.dtypes` | Object | `` |
| `src.backend.tensorflow.numpy.einsum` | Function | `(subscripts, operands, kwargs)` |
| `src.backend.tensorflow.numpy.empty` | Function | `(shape, dtype)` |
| `src.backend.tensorflow.numpy.empty_like` | Function | `(x, dtype)` |
| `src.backend.tensorflow.numpy.equal` | Function | `(x1, x2)` |
| `src.backend.tensorflow.numpy.exp` | Function | `(x)` |
| `src.backend.tensorflow.numpy.exp2` | Function | `(x)` |
| `src.backend.tensorflow.numpy.expand_dims` | Function | `(x, axis)` |
| `src.backend.tensorflow.numpy.expm1` | Function | `(x)` |
| `src.backend.tensorflow.numpy.eye` | Function | `(N, M, k, dtype)` |
| `src.backend.tensorflow.numpy.flip` | Function | `(x, axis)` |
| `src.backend.tensorflow.numpy.floor` | Function | `(x)` |
| `src.backend.tensorflow.numpy.floor_divide` | Function | `(x1, x2)` |
| `src.backend.tensorflow.numpy.full` | Function | `(shape, fill_value, dtype)` |
| `src.backend.tensorflow.numpy.full_like` | Function | `(x, fill_value, dtype)` |
| `src.backend.tensorflow.numpy.gcd` | Function | `(x1, x2)` |
| `src.backend.tensorflow.numpy.greater` | Function | `(x1, x2)` |
| `src.backend.tensorflow.numpy.greater_equal` | Function | `(x1, x2)` |
| `src.backend.tensorflow.numpy.hamming` | Function | `(x)` |
| `src.backend.tensorflow.numpy.hanning` | Function | `(x)` |
| `src.backend.tensorflow.numpy.heaviside` | Function | `(x1, x2)` |
| `src.backend.tensorflow.numpy.histogram` | Function | `(x, bins, range)` |
| `src.backend.tensorflow.numpy.hstack` | Function | `(xs)` |
| `src.backend.tensorflow.numpy.hypot` | Function | `(x1, x2)` |
| `src.backend.tensorflow.numpy.identity` | Function | `(n, dtype)` |
| `src.backend.tensorflow.numpy.imag` | Function | `(x)` |
| `src.backend.tensorflow.numpy.inner` | Function | `(x1, x2)` |
| `src.backend.tensorflow.numpy.isclose` | Function | `(x1, x2, rtol, atol, equal_nan)` |
| `src.backend.tensorflow.numpy.isfinite` | Function | `(x)` |
| `src.backend.tensorflow.numpy.isin` | Function | `(x1, x2, assume_unique, invert)` |
| `src.backend.tensorflow.numpy.isinf` | Function | `(x)` |
| `src.backend.tensorflow.numpy.isnan` | Function | `(x)` |
| `src.backend.tensorflow.numpy.isneginf` | Function | `(x)` |
| `src.backend.tensorflow.numpy.isposinf` | Function | `(x)` |
| `src.backend.tensorflow.numpy.isreal` | Function | `(x)` |
| `src.backend.tensorflow.numpy.kaiser` | Function | `(x, beta)` |
| `src.backend.tensorflow.numpy.kron` | Function | `(x1, x2)` |
| `src.backend.tensorflow.numpy.lcm` | Function | `(x1, x2)` |
| `src.backend.tensorflow.numpy.ldexp` | Function | `(x1, x2)` |
| `src.backend.tensorflow.numpy.left_shift` | Function | `(x, y)` |
| `src.backend.tensorflow.numpy.less` | Function | `(x1, x2)` |
| `src.backend.tensorflow.numpy.less_equal` | Function | `(x1, x2)` |
| `src.backend.tensorflow.numpy.linspace` | Function | `(start, stop, num, endpoint, retstep, dtype, axis)` |
| `src.backend.tensorflow.numpy.log` | Function | `(x)` |
| `src.backend.tensorflow.numpy.log10` | Function | `(x)` |
| `src.backend.tensorflow.numpy.log1p` | Function | `(x)` |
| `src.backend.tensorflow.numpy.log2` | Function | `(x)` |
| `src.backend.tensorflow.numpy.logaddexp` | Function | `(x1, x2)` |
| `src.backend.tensorflow.numpy.logaddexp2` | Function | `(x1, x2)` |
| `src.backend.tensorflow.numpy.logical_and` | Function | `(x1, x2)` |
| `src.backend.tensorflow.numpy.logical_not` | Function | `(x)` |
| `src.backend.tensorflow.numpy.logical_or` | Function | `(x1, x2)` |
| `src.backend.tensorflow.numpy.logical_xor` | Function | `(x1, x2)` |
| `src.backend.tensorflow.numpy.logspace` | Function | `(start, stop, num, endpoint, base, dtype, axis)` |
| `src.backend.tensorflow.numpy.matmul` | Function | `(x1, x2)` |
| `src.backend.tensorflow.numpy.max` | Function | `(x, axis, keepdims, initial)` |
| `src.backend.tensorflow.numpy.maximum` | Function | `(x1, x2)` |
| `src.backend.tensorflow.numpy.mean` | Function | `(x, axis, keepdims)` |
| `src.backend.tensorflow.numpy.median` | Function | `(x, axis, keepdims)` |
| `src.backend.tensorflow.numpy.meshgrid` | Function | `(x, indexing)` |
| `src.backend.tensorflow.numpy.min` | Function | `(x, axis, keepdims, initial)` |
| `src.backend.tensorflow.numpy.minimum` | Function | `(x1, x2)` |
| `src.backend.tensorflow.numpy.mod` | Function | `(x1, x2)` |
| `src.backend.tensorflow.numpy.moveaxis` | Function | `(x, source, destination)` |
| `src.backend.tensorflow.numpy.multiply` | Function | `(x1, x2)` |
| `src.backend.tensorflow.numpy.nan_to_num` | Function | `(x, nan, posinf, neginf)` |
| `src.backend.tensorflow.numpy.ndim` | Function | `(x)` |
| `src.backend.tensorflow.numpy.negative` | Function | `(x)` |
| `src.backend.tensorflow.numpy.nonzero` | Function | `(x)` |
| `src.backend.tensorflow.numpy.not_equal` | Function | `(x1, x2)` |
| `src.backend.tensorflow.numpy.ones` | Function | `(shape, dtype)` |
| `src.backend.tensorflow.numpy.ones_like` | Function | `(x, dtype)` |
| `src.backend.tensorflow.numpy.outer` | Function | `(x1, x2)` |
| `src.backend.tensorflow.numpy.pad` | Function | `(x, pad_width, mode, constant_values)` |
| `src.backend.tensorflow.numpy.power` | Function | `(x1, x2)` |
| `src.backend.tensorflow.numpy.prod` | Function | `(x, axis, keepdims, dtype)` |
| `src.backend.tensorflow.numpy.quantile` | Function | `(x, q, axis, method, keepdims)` |
| `src.backend.tensorflow.numpy.ravel` | Function | `(x)` |
| `src.backend.tensorflow.numpy.real` | Function | `(x)` |
| `src.backend.tensorflow.numpy.reciprocal` | Function | `(x)` |
| `src.backend.tensorflow.numpy.repeat` | Function | `(x, repeats, axis)` |
| `src.backend.tensorflow.numpy.reshape` | Function | `(x, newshape)` |
| `src.backend.tensorflow.numpy.right_shift` | Function | `(x, y)` |
| `src.backend.tensorflow.numpy.roll` | Function | `(x, shift, axis)` |
| `src.backend.tensorflow.numpy.rot90` | Function | `(array, k, axes)` |
| `src.backend.tensorflow.numpy.round` | Function | `(x, decimals)` |
| `src.backend.tensorflow.numpy.searchsorted` | Function | `(sorted_sequence, values, side)` |
| `src.backend.tensorflow.numpy.select` | Function | `(condlist, choicelist, default)` |
| `src.backend.tensorflow.numpy.shape_op` | Function | `(x)` |
| `src.backend.tensorflow.numpy.sign` | Function | `(x)` |
| `src.backend.tensorflow.numpy.signbit` | Function | `(x)` |
| `src.backend.tensorflow.numpy.sin` | Function | `(x)` |
| `src.backend.tensorflow.numpy.sinh` | Function | `(x)` |
| `src.backend.tensorflow.numpy.size` | Function | `(x)` |
| `src.backend.tensorflow.numpy.slogdet` | Function | `(x)` |
| `src.backend.tensorflow.numpy.sort` | Function | `(x, axis)` |
| `src.backend.tensorflow.numpy.sparse` | Object | `` |
| `src.backend.tensorflow.numpy.split` | Function | `(x, indices_or_sections, axis)` |
| `src.backend.tensorflow.numpy.sqrt` | Function | `(x)` |
| `src.backend.tensorflow.numpy.square` | Function | `(x)` |
| `src.backend.tensorflow.numpy.squeeze` | Function | `(x, axis)` |
| `src.backend.tensorflow.numpy.stack` | Function | `(x, axis)` |
| `src.backend.tensorflow.numpy.standardize_dtype` | Function | `(dtype)` |
| `src.backend.tensorflow.numpy.std` | Function | `(x, axis, keepdims)` |
| `src.backend.tensorflow.numpy.subtract` | Function | `(x1, x2)` |
| `src.backend.tensorflow.numpy.sum` | Function | `(x, axis, keepdims)` |
| `src.backend.tensorflow.numpy.swapaxes` | Function | `(x, axis1, axis2)` |
| `src.backend.tensorflow.numpy.take` | Function | `(x, indices, axis)` |
| `src.backend.tensorflow.numpy.take_along_axis` | Function | `(x, indices, axis)` |
| `src.backend.tensorflow.numpy.tan` | Function | `(x)` |
| `src.backend.tensorflow.numpy.tanh` | Function | `(x)` |
| `src.backend.tensorflow.numpy.tensordot` | Function | `(x1, x2, axes)` |
| `src.backend.tensorflow.numpy.tile` | Function | `(x, repeats)` |
| `src.backend.tensorflow.numpy.to_tuple_or_list` | Function | `(value)` |
| `src.backend.tensorflow.numpy.trace` | Function | `(x, offset, axis1, axis2)` |
| `src.backend.tensorflow.numpy.transpose` | Function | `(x, axes)` |
| `src.backend.tensorflow.numpy.trapezoid` | Function | `(y, x, dx, axis)` |
| `src.backend.tensorflow.numpy.tree` | Object | `` |
| `src.backend.tensorflow.numpy.tri` | Function | `(N, M, k, dtype)` |
| `src.backend.tensorflow.numpy.tril` | Function | `(x, k)` |
| `src.backend.tensorflow.numpy.triu` | Function | `(x, k)` |
| `src.backend.tensorflow.numpy.true_divide` | Function | `(x1, x2)` |
| `src.backend.tensorflow.numpy.trunc` | Function | `(x)` |
| `src.backend.tensorflow.numpy.unravel_index` | Function | `(indices, shape)` |
| `src.backend.tensorflow.numpy.vander` | Function | `(x, N, increasing)` |
| `src.backend.tensorflow.numpy.var` | Function | `(x, axis, keepdims)` |
| `src.backend.tensorflow.numpy.vdot` | Function | `(x1, x2)` |
| `src.backend.tensorflow.numpy.vectorize` | Function | `(pyfunc, excluded, signature)` |
| `src.backend.tensorflow.numpy.vectorize_impl` | Function | `(pyfunc, vmap_fn, excluded, signature)` |
| `src.backend.tensorflow.numpy.view` | Function | `(x, dtype)` |
| `src.backend.tensorflow.numpy.vstack` | Function | `(xs)` |
| `src.backend.tensorflow.numpy.where` | Function | `(condition, x1, x2)` |
| `src.backend.tensorflow.numpy.zeros` | Function | `(shape, dtype)` |
| `src.backend.tensorflow.numpy.zeros_like` | Function | `(x, dtype)` |
| `src.backend.tensorflow.optimizer.KerasAutoTrackable` | Class | `(...)` |
| `src.backend.tensorflow.optimizer.TFOptimizer` | Class | `(args, kwargs)` |
| `src.backend.tensorflow.optimizer.backend` | Object | `` |
| `src.backend.tensorflow.optimizer.base_optimizer` | Object | `` |
| `src.backend.tensorflow.optimizer.filter_empty_gradients` | Function | `(grads_and_vars)` |
| `src.backend.tensorflow.random.SeedGenerator` | Class | `(seed, name, kwargs)` |
| `src.backend.tensorflow.random.beta` | Function | `(shape, alpha, beta, dtype, seed)` |
| `src.backend.tensorflow.random.binomial` | Function | `(shape, counts, probabilities, dtype, seed)` |
| `src.backend.tensorflow.random.categorical` | Function | `(logits, num_samples, dtype, seed)` |
| `src.backend.tensorflow.random.draw_seed` | Function | `(seed)` |
| `src.backend.tensorflow.random.dropout` | Function | `(inputs, rate, noise_shape, seed)` |
| `src.backend.tensorflow.random.floatx` | Function | `()` |
| `src.backend.tensorflow.random.gamma` | Function | `(shape, alpha, dtype, seed)` |
| `src.backend.tensorflow.random.make_default_seed` | Function | `()` |
| `src.backend.tensorflow.random.normal` | Function | `(shape, mean, stddev, dtype, seed)` |
| `src.backend.tensorflow.random.randint` | Function | `(shape, minval, maxval, dtype, seed)` |
| `src.backend.tensorflow.random.shuffle` | Function | `(x, axis, seed)` |
| `src.backend.tensorflow.random.standardize_dtype` | Function | `(dtype)` |
| `src.backend.tensorflow.random.truncated_normal` | Function | `(shape, mean, stddev, dtype, seed)` |
| `src.backend.tensorflow.random.uniform` | Function | `(shape, minval, maxval, dtype, seed)` |
| `src.backend.tensorflow.random_seed_dtype` | Function | `()` |
| `src.backend.tensorflow.rnn.cudnn_ok` | Function | `(activation, recurrent_activation, unroll, use_bias, reset_after)` |
| `src.backend.tensorflow.rnn.gru` | Function | `(inputs, initial_state, mask, kernel, recurrent_kernel, bias, activation, recurrent_activation, return_sequences, go_backwards, unroll, time_major, reset_after)` |
| `src.backend.tensorflow.rnn.lstm` | Function | `(inputs, initial_state_h, initial_state_c, mask, kernel, recurrent_kernel, bias, activation, recurrent_activation, return_sequences, go_backwards, unroll, time_major)` |
| `src.backend.tensorflow.rnn.rnn` | Function | `(step_function, inputs, initial_states, go_backwards, mask, constants, unroll, input_length, time_major, zero_output_for_mask, return_all_outputs)` |
| `src.backend.tensorflow.rnn.tree` | Object | `` |
| `src.backend.tensorflow.scatter` | Function | `(indices, values, shape)` |
| `src.backend.tensorflow.shape` | Function | `(x)` |
| `src.backend.tensorflow.sparse.broadcast_scalar_to_sparse_shape` | Function | `(scalar, sparse)` |
| `src.backend.tensorflow.sparse.densifying_unary` | Function | `(default_value)` |
| `src.backend.tensorflow.sparse.elementwise_binary_intersection` | Function | `(func)` |
| `src.backend.tensorflow.sparse.elementwise_binary_union` | Function | `(sparse_op, densify_mixed)` |
| `src.backend.tensorflow.sparse.elementwise_division` | Function | `(func)` |
| `src.backend.tensorflow.sparse.elementwise_unary` | Function | `(func)` |
| `src.backend.tensorflow.sparse.indexed_slices_intersection_indices_and_values` | Function | `(x1, x2)` |
| `src.backend.tensorflow.sparse.indexed_slices_union_indices_and_values` | Function | `(x1, x2_indices, x2_values)` |
| `src.backend.tensorflow.sparse.ones_bool` | Object | `` |
| `src.backend.tensorflow.sparse.ones_int8` | Object | `` |
| `src.backend.tensorflow.sparse.ones_like_int8` | Object | `` |
| `src.backend.tensorflow.sparse.sparse_intersection_indices_and_values` | Function | `(x1, x2)` |
| `src.backend.tensorflow.sparse.sparse_subtract` | Function | `(x1, x2)` |
| `src.backend.tensorflow.sparse.sparse_to_dense` | Function | `(x, default_value)` |
| `src.backend.tensorflow.sparse.sparse_union_indices_and_values` | Function | `(x1, x2_indices, x2_values)` |
| `src.backend.tensorflow.sparse.sparse_with_values` | Function | `(x, values)` |
| `src.backend.tensorflow.sparse.zeros_int8` | Object | `` |
| `src.backend.tensorflow.sparse.zeros_like_int8` | Object | `` |
| `src.backend.tensorflow.stop_gradient` | Function | `(variable)` |
| `src.backend.tensorflow.tensorboard.start_batch_trace` | Function | `(batch)` |
| `src.backend.tensorflow.tensorboard.start_trace` | Function | `(logdir)` |
| `src.backend.tensorflow.tensorboard.stop_batch_trace` | Function | `(batch_trace_context)` |
| `src.backend.tensorflow.tensorboard.stop_trace` | Function | `(save)` |
| `src.backend.tensorflow.tensorboard.tf` | Object | `` |
| `src.backend.tensorflow.trackable.KerasAutoTrackable` | Class | `(...)` |
| `src.backend.tensorflow.trackable.sticky_attribute_assignment` | Function | `(trackable, name, value)` |
| `src.backend.tensorflow.trackable.tracking` | Object | `` |
| `src.backend.tensorflow.trainer.EpochIterator` | Class | `(x, y, sample_weight, batch_size, steps_per_epoch, shuffle, class_weight, steps_per_execution)` |
| `src.backend.tensorflow.trainer.TFEpochIterator` | Class | `(distribute_strategy, args, kwargs)` |
| `src.backend.tensorflow.trainer.TensorFlowTrainer` | Class | `()` |
| `src.backend.tensorflow.trainer.array_slicing` | Object | `` |
| `src.backend.tensorflow.trainer.base_trainer` | Object | `` |
| `src.backend.tensorflow.trainer.callbacks_module` | Object | `` |
| `src.backend.tensorflow.trainer.concat` | Function | `(tensors, axis)` |
| `src.backend.tensorflow.trainer.config` | Object | `` |
| `src.backend.tensorflow.trainer.convert_to_np_if_not_ragged` | Function | `(x)` |
| `src.backend.tensorflow.trainer.data_adapter_utils` | Object | `` |
| `src.backend.tensorflow.trainer.loss_module` | Object | `` |
| `src.backend.tensorflow.trainer.metrics_module` | Object | `` |
| `src.backend.tensorflow.trainer.optimizers_module` | Object | `` |
| `src.backend.tensorflow.trainer.potentially_ragged_concat` | Function | `(tensors)` |
| `src.backend.tensorflow.trainer.reduce_per_replica` | Function | `(values, strategy, reduction)` |
| `src.backend.tensorflow.trainer.traceback_utils` | Object | `` |
| `src.backend.tensorflow.trainer.tree` | Object | `` |
| `src.backend.tensorflow.vectorized_map` | Function | `(function, elements)` |
| `src.backend.to_torch_dtype` | Function | `(dtype)` |
| `src.backend.torch.IS_THREAD_SAFE` | Object | `` |
| `src.backend.torch.SUPPORTS_RAGGED_TENSORS` | Object | `` |
| `src.backend.torch.SUPPORTS_SPARSE_TENSORS` | Object | `` |
| `src.backend.torch.Variable` | Class | `(...)` |
| `src.backend.torch.cast` | Function | `(x, dtype)` |
| `src.backend.torch.compute_output_spec` | Function | `(fn, args, kwargs)` |
| `src.backend.torch.cond` | Function | `(pred, true_fn, false_fn)` |
| `src.backend.torch.convert_to_numpy` | Function | `(x)` |
| `src.backend.torch.convert_to_tensor` | Function | `(x, dtype, sparse, ragged)` |
| `src.backend.torch.core.CustomGradientFunction` | Class | `(...)` |
| `src.backend.torch.core.DEFAULT_DEVICE` | Object | `` |
| `src.backend.torch.core.IS_THREAD_SAFE` | Object | `` |
| `src.backend.torch.core.KerasTensor` | Class | `(shape, dtype, sparse, ragged, record_history, name, kwargs)` |
| `src.backend.torch.core.KerasVariable` | Class | `(initializer, shape, dtype, trainable, autocast, aggregation, synchronization, name, kwargs)` |
| `src.backend.torch.core.SUPPORTS_RAGGED_TENSORS` | Object | `` |
| `src.backend.torch.core.SUPPORTS_SPARSE_TENSORS` | Object | `` |
| `src.backend.torch.core.StatelessScope` | Class | `(state_mapping, collect_losses, initialize_variables)` |
| `src.backend.torch.core.SymbolicScope` | Class | `(...)` |
| `src.backend.torch.core.TORCH_DTYPES` | Object | `` |
| `src.backend.torch.core.Variable` | Class | `(...)` |
| `src.backend.torch.core.associative_scan` | Function | `(f, elems, reverse, axis)` |
| `src.backend.torch.core.cast` | Function | `(x, dtype)` |
| `src.backend.torch.core.compute_output_spec` | Function | `(fn, args, kwargs)` |
| `src.backend.torch.core.cond` | Function | `(pred, true_fn, false_fn)` |
| `src.backend.torch.core.convert_to_numpy` | Function | `(x)` |
| `src.backend.torch.core.convert_to_tensor` | Function | `(x, dtype, sparse, ragged)` |
| `src.backend.torch.core.custom_gradient` | Class | `(forward_fn)` |
| `src.backend.torch.core.device_scope` | Function | `(device_name)` |
| `src.backend.torch.core.floatx` | Function | `()` |
| `src.backend.torch.core.fori_loop` | Function | `(lower, upper, body_fun, init_val)` |
| `src.backend.torch.core.get_device` | Function | `()` |
| `src.backend.torch.core.get_stateless_scope` | Function | `()` |
| `src.backend.torch.core.global_state` | Object | `` |
| `src.backend.torch.core.in_stateless_scope` | Function | `()` |
| `src.backend.torch.core.is_tensor` | Function | `(x)` |
| `src.backend.torch.core.map` | Function | `(f, xs)` |
| `src.backend.torch.core.random_seed_dtype` | Function | `()` |
| `src.backend.torch.core.remat` | Function | `(f)` |
| `src.backend.torch.core.result_type` | Function | `(dtypes)` |
| `src.backend.torch.core.scan` | Function | `(f, init, xs, length, reverse, unroll)` |
| `src.backend.torch.core.scatter` | Function | `(indices, values, shape)` |
| `src.backend.torch.core.scatter_update` | Function | `(inputs, indices, updates)` |
| `src.backend.torch.core.shape` | Function | `(x)` |
| `src.backend.torch.core.slice` | Function | `(inputs, start_indices, shape)` |
| `src.backend.torch.core.slice_along_axis` | Function | `(x, start, stop, step, axis)` |
| `src.backend.torch.core.slice_update` | Function | `(inputs, start_indices, updates)` |
| `src.backend.torch.core.standardize_dtype` | Function | `(dtype)` |
| `src.backend.torch.core.stop_gradient` | Function | `(variable)` |
| `src.backend.torch.core.switch` | Function | `(index, branches, operands)` |
| `src.backend.torch.core.to_torch_dtype` | Function | `(dtype)` |
| `src.backend.torch.core.tree` | Object | `` |
| `src.backend.torch.core.unstack` | Function | `(x, num, axis)` |
| `src.backend.torch.core.vectorized_map` | Function | `(function, elements)` |
| `src.backend.torch.core.while_loop` | Function | `(cond, body, loop_vars, maximum_iterations)` |
| `src.backend.torch.cudnn_ok` | Function | `(activation, recurrent_activation, unroll, use_bias)` |
| `src.backend.torch.device_scope` | Function | `(device_name)` |
| `src.backend.torch.export.TorchExportArchive` | Class | `(...)` |
| `src.backend.torch.export.convert_spec_to_tensor` | Function | `(spec, replace_none_number)` |
| `src.backend.torch.export.tf` | Object | `` |
| `src.backend.torch.export.torch_xla` | Object | `` |
| `src.backend.torch.export.tree` | Object | `` |
| `src.backend.torch.gru` | Function | `(args, kwargs)` |
| `src.backend.torch.image.AFFINE_TRANSFORM_FILL_MODES` | Object | `` |
| `src.backend.torch.image.AFFINE_TRANSFORM_INTERPOLATIONS` | Object | `` |
| `src.backend.torch.image.RESIZE_INTERPOLATIONS` | Object | `` |
| `src.backend.torch.image.SCALE_AND_TRANSLATE_METHODS` | Object | `` |
| `src.backend.torch.image.UNSUPPORTED_INTERPOLATIONS` | Object | `` |
| `src.backend.torch.image.affine_transform` | Function | `(images, transform, interpolation, fill_mode, fill_value, data_format)` |
| `src.backend.torch.image.backend` | Object | `` |
| `src.backend.torch.image.cast` | Function | `(x, dtype)` |
| `src.backend.torch.image.compute_homography_matrix` | Function | `(start_points, end_points)` |
| `src.backend.torch.image.convert_to_tensor` | Function | `(x, dtype, sparse, ragged)` |
| `src.backend.torch.image.draw_seed` | Function | `(seed)` |
| `src.backend.torch.image.elastic_transform` | Function | `(images, alpha, sigma, interpolation, fill_mode, fill_value, seed, data_format)` |
| `src.backend.torch.image.gaussian_blur` | Function | `(images, kernel_size, sigma, data_format)` |
| `src.backend.torch.image.get_device` | Function | `()` |
| `src.backend.torch.image.hsv_to_rgb` | Function | `(images, data_format)` |
| `src.backend.torch.image.map_coordinates` | Function | `(inputs, coordinates, order, fill_mode, fill_value)` |
| `src.backend.torch.image.perspective_transform` | Function | `(images, start_points, end_points, interpolation, fill_value, data_format)` |
| `src.backend.torch.image.resize` | Function | `(images, size, interpolation, antialias, crop_to_aspect_ratio, pad_to_aspect_ratio, fill_mode, fill_value, data_format)` |
| `src.backend.torch.image.rgb_to_grayscale` | Function | `(images, data_format)` |
| `src.backend.torch.image.rgb_to_hsv` | Function | `(images, data_format)` |
| `src.backend.torch.image.scale_and_translate` | Function | `(images, output_shape, scale, translation, spatial_dims, method, antialias)` |
| `src.backend.torch.image.to_torch_dtype` | Function | `(dtype)` |
| `src.backend.torch.is_tensor` | Function | `(x)` |
| `src.backend.torch.layer.Operation` | Class | `(name)` |
| `src.backend.torch.layer.TorchLayer` | Class | `(...)` |
| `src.backend.torch.layer.in_stateless_scope` | Function | `()` |
| `src.backend.torch.linalg.cast` | Function | `(x, dtype)` |
| `src.backend.torch.linalg.cholesky` | Function | `(x, upper)` |
| `src.backend.torch.linalg.cholesky_inverse` | Function | `(x, upper)` |
| `src.backend.torch.linalg.config` | Object | `` |
| `src.backend.torch.linalg.convert_to_tensor` | Function | `(x, dtype, sparse, ragged)` |
| `src.backend.torch.linalg.det` | Function | `(x)` |
| `src.backend.torch.linalg.dtypes` | Object | `` |
| `src.backend.torch.linalg.eig` | Function | `(x)` |
| `src.backend.torch.linalg.eigh` | Function | `(x)` |
| `src.backend.torch.linalg.inv` | Function | `(x)` |
| `src.backend.torch.linalg.jvp` | Function | `(fun, primals, tangents, has_aux)` |
| `src.backend.torch.linalg.lstsq` | Function | `(a, b, rcond)` |
| `src.backend.torch.linalg.lu_factor` | Function | `(x)` |
| `src.backend.torch.linalg.norm` | Function | `(x, ord, axis, keepdims)` |
| `src.backend.torch.linalg.qr` | Function | `(x, mode)` |
| `src.backend.torch.linalg.solve` | Function | `(a, b)` |
| `src.backend.torch.linalg.solve_triangular` | Function | `(a, b, lower)` |
| `src.backend.torch.linalg.standardize_dtype` | Function | `(dtype)` |
| `src.backend.torch.linalg.svd` | Function | `(x, full_matrices, compute_uv)` |
| `src.backend.torch.lstm` | Function | `(inputs, initial_state_h, initial_state_c, mask, kernel, recurrent_kernel, bias, activation, recurrent_activation, return_sequences, go_backwards, unroll, batch_first)` |
| `src.backend.torch.math.cast` | Function | `(x, dtype)` |
| `src.backend.torch.math.config` | Object | `` |
| `src.backend.torch.math.convert_to_tensor` | Function | `(x, dtype, sparse, ragged)` |
| `src.backend.torch.math.dtypes` | Object | `` |
| `src.backend.torch.math.erf` | Function | `(x)` |
| `src.backend.torch.math.erfinv` | Function | `(x)` |
| `src.backend.torch.math.extract_sequences` | Function | `(x, sequence_length, sequence_stride)` |
| `src.backend.torch.math.fft` | Function | `(x)` |
| `src.backend.torch.math.fft2` | Function | `(x)` |
| `src.backend.torch.math.get_device` | Function | `()` |
| `src.backend.torch.math.ifft2` | Function | `(x)` |
| `src.backend.torch.math.in_top_k` | Function | `(targets, predictions, k)` |
| `src.backend.torch.math.irfft` | Function | `(x, fft_length)` |
| `src.backend.torch.math.istft` | Function | `(x, sequence_length, sequence_stride, fft_length, length, window, center)` |
| `src.backend.torch.math.logdet` | Function | `(x)` |
| `src.backend.torch.math.logsumexp` | Function | `(x, axis, keepdims)` |
| `src.backend.torch.math.norm` | Function | `(x, ord, axis, keepdims)` |
| `src.backend.torch.math.pad` | Function | `(x, pad_width, mode, constant_values)` |
| `src.backend.torch.math.qr` | Function | `(x, mode)` |
| `src.backend.torch.math.rfft` | Function | `(x, fft_length)` |
| `src.backend.torch.math.rsqrt` | Function | `(x)` |
| `src.backend.torch.math.segment_max` | Function | `(data, segment_ids, num_segments, sorted)` |
| `src.backend.torch.math.segment_sum` | Function | `(data, segment_ids, num_segments, sorted)` |
| `src.backend.torch.math.solve` | Function | `(a, b)` |
| `src.backend.torch.math.standardize_dtype` | Function | `(dtype)` |
| `src.backend.torch.math.stft` | Function | `(x, sequence_length, sequence_stride, fft_length, window, center)` |
| `src.backend.torch.math.top_k` | Function | `(x, k, sorted)` |
| `src.backend.torch.name_scope` | Class | `(name, caller, deduplicate, override_parent)` |
| `src.backend.torch.nn.adaptive_average_pool` | Function | `(inputs, output_size, data_format)` |
| `src.backend.torch.nn.adaptive_max_pool` | Function | `(inputs, output_size, data_format)` |
| `src.backend.torch.nn.average_pool` | Function | `(inputs, pool_size, strides, padding, data_format)` |
| `src.backend.torch.nn.backend` | Object | `` |
| `src.backend.torch.nn.batch_normalization` | Function | `(x, mean, variance, axis, offset, scale, epsilon)` |
| `src.backend.torch.nn.binary_crossentropy` | Function | `(target, output, from_logits)` |
| `src.backend.torch.nn.cast` | Function | `(x, dtype)` |
| `src.backend.torch.nn.categorical_crossentropy` | Function | `(target, output, from_logits, axis)` |
| `src.backend.torch.nn.celu` | Function | `(x, alpha)` |
| `src.backend.torch.nn.compute_conv_transpose_padding_args_for_torch` | Function | `(input_shape, kernel_shape, strides, padding, output_padding, dilation_rate)` |
| `src.backend.torch.nn.conv` | Function | `(inputs, kernel, strides, padding, data_format, dilation_rate)` |
| `src.backend.torch.nn.conv_transpose` | Function | `(inputs, kernel, strides, padding, output_padding, data_format, dilation_rate)` |
| `src.backend.torch.nn.convert_to_tensor` | Function | `(x, dtype, sparse, ragged)` |
| `src.backend.torch.nn.ctc_decode` | Function | `(inputs, sequence_lengths, strategy, beam_width, top_paths, merge_repeated, mask_index)` |
| `src.backend.torch.nn.ctc_loss` | Function | `(target, output, target_length, output_length, mask_index)` |
| `src.backend.torch.nn.depthwise_conv` | Function | `(inputs, kernel, strides, padding, data_format, dilation_rate)` |
| `src.backend.torch.nn.dot_product_attention` | Function | `(query, key, value, bias, mask, scale, is_causal, flash_attention, attn_logits_soft_cap)` |
| `src.backend.torch.nn.elu` | Function | `(x, alpha)` |
| `src.backend.torch.nn.expand_dims` | Function | `(x, axis)` |
| `src.backend.torch.nn.gelu` | Function | `(x, approximate)` |
| `src.backend.torch.nn.get_device` | Function | `()` |
| `src.backend.torch.nn.glu` | Function | `(x, axis)` |
| `src.backend.torch.nn.hard_shrink` | Function | `(x, threshold)` |
| `src.backend.torch.nn.hard_sigmoid` | Function | `(x)` |
| `src.backend.torch.nn.hard_silu` | Function | `(x)` |
| `src.backend.torch.nn.hard_tanh` | Function | `(x)` |
| `src.backend.torch.nn.leaky_relu` | Function | `(x, negative_slope)` |
| `src.backend.torch.nn.log_sigmoid` | Function | `(x)` |
| `src.backend.torch.nn.log_softmax` | Function | `(x, axis)` |
| `src.backend.torch.nn.max_pool` | Function | `(inputs, pool_size, strides, padding, data_format)` |
| `src.backend.torch.nn.moments` | Function | `(x, axes, keepdims, synchronized)` |
| `src.backend.torch.nn.multi_hot` | Function | `(x, num_classes, axis, dtype, sparse)` |
| `src.backend.torch.nn.one_hot` | Function | `(x, num_classes, axis, dtype, sparse)` |
| `src.backend.torch.nn.psnr` | Function | `(x1, x2, max_val)` |
| `src.backend.torch.nn.relu` | Function | `(x)` |
| `src.backend.torch.nn.relu6` | Function | `(x)` |
| `src.backend.torch.nn.selu` | Function | `(x)` |
| `src.backend.torch.nn.separable_conv` | Function | `(inputs, depthwise_kernel, pointwise_kernel, strides, padding, data_format, dilation_rate)` |
| `src.backend.torch.nn.sigmoid` | Function | `(x)` |
| `src.backend.torch.nn.silu` | Function | `(x)` |
| `src.backend.torch.nn.soft_shrink` | Function | `(x, threshold)` |
| `src.backend.torch.nn.softmax` | Function | `(x, axis)` |
| `src.backend.torch.nn.softplus` | Function | `(x)` |
| `src.backend.torch.nn.softsign` | Function | `(x)` |
| `src.backend.torch.nn.sparse_categorical_crossentropy` | Function | `(target, output, from_logits, axis)` |
| `src.backend.torch.nn.sparse_plus` | Function | `(x)` |
| `src.backend.torch.nn.sparse_sigmoid` | Function | `(x)` |
| `src.backend.torch.nn.sparsemax` | Function | `(x, axis)` |
| `src.backend.torch.nn.squareplus` | Function | `(x, b)` |
| `src.backend.torch.nn.standardize_tuple` | Function | `(value, n, name, allow_zero)` |
| `src.backend.torch.nn.tanh` | Function | `(x)` |
| `src.backend.torch.nn.tanh_shrink` | Function | `(x)` |
| `src.backend.torch.nn.threshold` | Function | `(x, threshold, default_value)` |
| `src.backend.torch.nn.unfold` | Function | `(input, kernel_size, dilation, padding, stride)` |
| `src.backend.torch.nn.where` | Function | `(condition, x1, x2)` |
| `src.backend.torch.numpy.KerasTensor` | Class | `(shape, dtype, sparse, ragged, record_history, name, kwargs)` |
| `src.backend.torch.numpy.TORCH_INT_TYPES` | Object | `` |
| `src.backend.torch.numpy.abs` | Function | `(x)` |
| `src.backend.torch.numpy.absolute` | Function | `(x)` |
| `src.backend.torch.numpy.add` | Function | `(x1, x2)` |
| `src.backend.torch.numpy.all` | Function | `(x, axis, keepdims)` |
| `src.backend.torch.numpy.amax` | Function | `(x, axis, keepdims)` |
| `src.backend.torch.numpy.amin` | Function | `(x, axis, keepdims)` |
| `src.backend.torch.numpy.angle` | Function | `(x)` |
| `src.backend.torch.numpy.any` | Function | `(x, axis, keepdims)` |
| `src.backend.torch.numpy.append` | Function | `(x1, x2, axis)` |
| `src.backend.torch.numpy.arange` | Function | `(start, stop, step, dtype)` |
| `src.backend.torch.numpy.arccos` | Function | `(x)` |
| `src.backend.torch.numpy.arccosh` | Function | `(x)` |
| `src.backend.torch.numpy.arcsin` | Function | `(x)` |
| `src.backend.torch.numpy.arcsinh` | Function | `(x)` |
| `src.backend.torch.numpy.arctan` | Function | `(x)` |
| `src.backend.torch.numpy.arctan2` | Function | `(x1, x2)` |
| `src.backend.torch.numpy.arctanh` | Function | `(x)` |
| `src.backend.torch.numpy.argmax` | Function | `(x, axis, keepdims)` |
| `src.backend.torch.numpy.argmin` | Function | `(x, axis, keepdims)` |
| `src.backend.torch.numpy.argpartition` | Function | `(x, kth, axis)` |
| `src.backend.torch.numpy.argsort` | Function | `(x, axis)` |
| `src.backend.torch.numpy.array` | Function | `(x, dtype)` |
| `src.backend.torch.numpy.array_split` | Function | `(x, indices_or_sections, axis)` |
| `src.backend.torch.numpy.average` | Function | `(x, axis, weights)` |
| `src.backend.torch.numpy.bartlett` | Function | `(x)` |
| `src.backend.torch.numpy.bincount` | Function | `(x, weights, minlength, sparse)` |
| `src.backend.torch.numpy.bitwise_and` | Function | `(x, y)` |
| `src.backend.torch.numpy.bitwise_invert` | Function | `(x)` |
| `src.backend.torch.numpy.bitwise_left_shift` | Function | `(x, y)` |
| `src.backend.torch.numpy.bitwise_not` | Function | `(x)` |
| `src.backend.torch.numpy.bitwise_or` | Function | `(x, y)` |
| `src.backend.torch.numpy.bitwise_right_shift` | Function | `(x, y)` |
| `src.backend.torch.numpy.bitwise_xor` | Function | `(x, y)` |
| `src.backend.torch.numpy.blackman` | Function | `(x)` |
| `src.backend.torch.numpy.broadcast_to` | Function | `(x, shape)` |
| `src.backend.torch.numpy.canonicalize_axis` | Function | `(axis, num_dims)` |
| `src.backend.torch.numpy.cast` | Function | `(x, dtype)` |
| `src.backend.torch.numpy.cbrt` | Function | `(x)` |
| `src.backend.torch.numpy.ceil` | Function | `(x)` |
| `src.backend.torch.numpy.clip` | Function | `(x, x_min, x_max)` |
| `src.backend.torch.numpy.concatenate` | Function | `(xs, axis)` |
| `src.backend.torch.numpy.config` | Object | `` |
| `src.backend.torch.numpy.conj` | Function | `(x)` |
| `src.backend.torch.numpy.conjugate` | Function | `(x)` |
| `src.backend.torch.numpy.convert_to_tensor` | Function | `(x, dtype, sparse, ragged)` |
| `src.backend.torch.numpy.copy` | Function | `(x)` |
| `src.backend.torch.numpy.corrcoef` | Function | `(x)` |
| `src.backend.torch.numpy.correlate` | Function | `(x1, x2, mode)` |
| `src.backend.torch.numpy.cos` | Function | `(x)` |
| `src.backend.torch.numpy.cosh` | Function | `(x)` |
| `src.backend.torch.numpy.count_nonzero` | Function | `(x, axis)` |
| `src.backend.torch.numpy.cross` | Function | `(x1, x2, axisa, axisb, axisc, axis)` |
| `src.backend.torch.numpy.cumprod` | Function | `(x, axis, dtype)` |
| `src.backend.torch.numpy.cumsum` | Function | `(x, axis, dtype)` |
| `src.backend.torch.numpy.deg2rad` | Function | `(x)` |
| `src.backend.torch.numpy.diag` | Function | `(x, k)` |
| `src.backend.torch.numpy.diagflat` | Function | `(x, k)` |
| `src.backend.torch.numpy.diagonal` | Function | `(x, offset, axis1, axis2)` |
| `src.backend.torch.numpy.diff` | Function | `(a, n, axis)` |
| `src.backend.torch.numpy.digitize` | Function | `(x, bins)` |
| `src.backend.torch.numpy.divide` | Function | `(x1, x2)` |
| `src.backend.torch.numpy.divide_no_nan` | Function | `(x1, x2)` |
| `src.backend.torch.numpy.dot` | Function | `(x1, x2)` |
| `src.backend.torch.numpy.dtypes` | Object | `` |
| `src.backend.torch.numpy.einsum` | Function | `(subscripts, operands, kwargs)` |
| `src.backend.torch.numpy.empty` | Function | `(shape, dtype)` |
| `src.backend.torch.numpy.empty_like` | Function | `(x, dtype)` |
| `src.backend.torch.numpy.equal` | Function | `(x1, x2)` |
| `src.backend.torch.numpy.exp` | Function | `(x)` |
| `src.backend.torch.numpy.exp2` | Function | `(x)` |
| `src.backend.torch.numpy.expand_dims` | Function | `(x, axis)` |
| `src.backend.torch.numpy.expm1` | Function | `(x)` |
| `src.backend.torch.numpy.eye` | Function | `(N, M, k, dtype)` |
| `src.backend.torch.numpy.flip` | Function | `(x, axis)` |
| `src.backend.torch.numpy.floor` | Function | `(x)` |
| `src.backend.torch.numpy.floor_divide` | Function | `(x1, x2)` |
| `src.backend.torch.numpy.full` | Function | `(shape, fill_value, dtype)` |
| `src.backend.torch.numpy.full_like` | Function | `(x, fill_value, dtype)` |
| `src.backend.torch.numpy.gcd` | Function | `(x1, x2)` |
| `src.backend.torch.numpy.get_device` | Function | `()` |
| `src.backend.torch.numpy.greater` | Function | `(x1, x2)` |
| `src.backend.torch.numpy.greater_equal` | Function | `(x1, x2)` |
| `src.backend.torch.numpy.hamming` | Function | `(x)` |
| `src.backend.torch.numpy.hanning` | Function | `(x)` |
| `src.backend.torch.numpy.heaviside` | Function | `(x1, x2)` |
| `src.backend.torch.numpy.histogram` | Function | `(x, bins, range)` |
| `src.backend.torch.numpy.hstack` | Function | `(xs)` |
| `src.backend.torch.numpy.hypot` | Function | `(x1, x2)` |
| `src.backend.torch.numpy.identity` | Function | `(n, dtype)` |
| `src.backend.torch.numpy.imag` | Function | `(x)` |
| `src.backend.torch.numpy.inner` | Function | `(x1, x2)` |
| `src.backend.torch.numpy.is_tensor` | Function | `(x)` |
| `src.backend.torch.numpy.isclose` | Function | `(x1, x2, rtol, atol, equal_nan)` |
| `src.backend.torch.numpy.isfinite` | Function | `(x)` |
| `src.backend.torch.numpy.isin` | Function | `(x1, x2, assume_unique, invert)` |
| `src.backend.torch.numpy.isinf` | Function | `(x)` |
| `src.backend.torch.numpy.isnan` | Function | `(x)` |
| `src.backend.torch.numpy.isneginf` | Function | `(x)` |
| `src.backend.torch.numpy.isposinf` | Function | `(x)` |
| `src.backend.torch.numpy.isreal` | Function | `(x)` |
| `src.backend.torch.numpy.kaiser` | Function | `(x, beta)` |
| `src.backend.torch.numpy.kron` | Function | `(x1, x2)` |
| `src.backend.torch.numpy.lcm` | Function | `(x1, x2)` |
| `src.backend.torch.numpy.ldexp` | Function | `(x1, x2)` |
| `src.backend.torch.numpy.left_shift` | Function | `(x, y)` |
| `src.backend.torch.numpy.less` | Function | `(x1, x2)` |
| `src.backend.torch.numpy.less_equal` | Function | `(x1, x2)` |
| `src.backend.torch.numpy.linspace` | Function | `(start, stop, num, endpoint, retstep, dtype, axis)` |
| `src.backend.torch.numpy.log` | Function | `(x)` |
| `src.backend.torch.numpy.log10` | Function | `(x)` |
| `src.backend.torch.numpy.log1p` | Function | `(x)` |
| `src.backend.torch.numpy.log2` | Function | `(x)` |
| `src.backend.torch.numpy.logaddexp` | Function | `(x1, x2)` |
| `src.backend.torch.numpy.logaddexp2` | Function | `(x1, x2)` |
| `src.backend.torch.numpy.logical_and` | Function | `(x1, x2)` |
| `src.backend.torch.numpy.logical_not` | Function | `(x)` |
| `src.backend.torch.numpy.logical_or` | Function | `(x1, x2)` |
| `src.backend.torch.numpy.logical_xor` | Function | `(x1, x2)` |
| `src.backend.torch.numpy.logspace` | Function | `(start, stop, num, endpoint, base, dtype, axis)` |
| `src.backend.torch.numpy.matmul` | Function | `(x1, x2)` |
| `src.backend.torch.numpy.max` | Function | `(x, axis, keepdims, initial)` |
| `src.backend.torch.numpy.maximum` | Function | `(x1, x2)` |
| `src.backend.torch.numpy.mean` | Function | `(x, axis, keepdims)` |
| `src.backend.torch.numpy.median` | Function | `(x, axis, keepdims)` |
| `src.backend.torch.numpy.meshgrid` | Function | `(x, indexing)` |
| `src.backend.torch.numpy.min` | Function | `(x, axis, keepdims, initial)` |
| `src.backend.torch.numpy.minimum` | Function | `(x1, x2)` |
| `src.backend.torch.numpy.mod` | Function | `(x1, x2)` |
| `src.backend.torch.numpy.moveaxis` | Function | `(x, source, destination)` |
| `src.backend.torch.numpy.multiply` | Function | `(x1, x2)` |
| `src.backend.torch.numpy.nan_to_num` | Function | `(x, nan, posinf, neginf)` |
| `src.backend.torch.numpy.ndim` | Function | `(x)` |
| `src.backend.torch.numpy.negative` | Function | `(x)` |
| `src.backend.torch.numpy.nonzero` | Function | `(x)` |
| `src.backend.torch.numpy.not_equal` | Function | `(x1, x2)` |
| `src.backend.torch.numpy.ones` | Function | `(shape, dtype)` |
| `src.backend.torch.numpy.ones_like` | Function | `(x, dtype)` |
| `src.backend.torch.numpy.outer` | Function | `(x1, x2)` |
| `src.backend.torch.numpy.pad` | Function | `(x, pad_width, mode, constant_values)` |
| `src.backend.torch.numpy.power` | Function | `(x1, x2)` |
| `src.backend.torch.numpy.prod` | Function | `(x, axis, keepdims, dtype)` |
| `src.backend.torch.numpy.quantile` | Function | `(x, q, axis, method, keepdims)` |
| `src.backend.torch.numpy.ravel` | Function | `(x)` |
| `src.backend.torch.numpy.real` | Function | `(x)` |
| `src.backend.torch.numpy.reciprocal` | Function | `(x)` |
| `src.backend.torch.numpy.repeat` | Function | `(x, repeats, axis)` |
| `src.backend.torch.numpy.reshape` | Function | `(x, newshape)` |
| `src.backend.torch.numpy.right_shift` | Function | `(x, y)` |
| `src.backend.torch.numpy.roll` | Function | `(x, shift, axis)` |
| `src.backend.torch.numpy.rot90` | Function | `(array, k, axes)` |
| `src.backend.torch.numpy.round` | Function | `(x, decimals)` |
| `src.backend.torch.numpy.searchsorted` | Function | `(sorted_sequence, values, side)` |
| `src.backend.torch.numpy.select` | Function | `(condlist, choicelist, default)` |
| `src.backend.torch.numpy.sign` | Function | `(x)` |
| `src.backend.torch.numpy.signbit` | Function | `(x)` |
| `src.backend.torch.numpy.sin` | Function | `(x)` |
| `src.backend.torch.numpy.sinh` | Function | `(x)` |
| `src.backend.torch.numpy.size` | Function | `(x)` |
| `src.backend.torch.numpy.slogdet` | Function | `(x)` |
| `src.backend.torch.numpy.sort` | Function | `(x, axis)` |
| `src.backend.torch.numpy.split` | Function | `(x, indices_or_sections, axis)` |
| `src.backend.torch.numpy.sqrt` | Function | `(x)` |
| `src.backend.torch.numpy.square` | Function | `(x)` |
| `src.backend.torch.numpy.squeeze` | Function | `(x, axis)` |
| `src.backend.torch.numpy.stack` | Function | `(x, axis)` |
| `src.backend.torch.numpy.standardize_dtype` | Function | `(dtype)` |
| `src.backend.torch.numpy.std` | Function | `(x, axis, keepdims)` |
| `src.backend.torch.numpy.subtract` | Function | `(x1, x2)` |
| `src.backend.torch.numpy.sum` | Function | `(x, axis, keepdims)` |
| `src.backend.torch.numpy.swapaxes` | Function | `(x, axis1, axis2)` |
| `src.backend.torch.numpy.take` | Function | `(x, indices, axis)` |
| `src.backend.torch.numpy.take_along_axis` | Function | `(x, indices, axis)` |
| `src.backend.torch.numpy.tan` | Function | `(x)` |
| `src.backend.torch.numpy.tanh` | Function | `(x)` |
| `src.backend.torch.numpy.tensordot` | Function | `(x1, x2, axes)` |
| `src.backend.torch.numpy.tile` | Function | `(x, repeats)` |
| `src.backend.torch.numpy.to_torch_dtype` | Function | `(dtype)` |
| `src.backend.torch.numpy.to_tuple_or_list` | Function | `(value)` |
| `src.backend.torch.numpy.trace` | Function | `(x, offset, axis1, axis2)` |
| `src.backend.torch.numpy.transpose` | Function | `(x, axes)` |
| `src.backend.torch.numpy.trapezoid` | Function | `(y, x, dx, axis)` |
| `src.backend.torch.numpy.tri` | Function | `(N, M, k, dtype)` |
| `src.backend.torch.numpy.tril` | Function | `(x, k)` |
| `src.backend.torch.numpy.triu` | Function | `(x, k)` |
| `src.backend.torch.numpy.true_divide` | Function | `(x1, x2)` |
| `src.backend.torch.numpy.trunc` | Function | `(x)` |
| `src.backend.torch.numpy.unravel_index` | Function | `(indices, shape)` |
| `src.backend.torch.numpy.vander` | Function | `(x, N, increasing)` |
| `src.backend.torch.numpy.var` | Function | `(x, axis, keepdims)` |
| `src.backend.torch.numpy.vdot` | Function | `(x1, x2)` |
| `src.backend.torch.numpy.vectorize` | Function | `(pyfunc, excluded, signature)` |
| `src.backend.torch.numpy.vectorize_impl` | Function | `(pyfunc, vmap_fn, excluded, signature)` |
| `src.backend.torch.numpy.view` | Function | `(x, dtype)` |
| `src.backend.torch.numpy.vstack` | Function | `(xs)` |
| `src.backend.torch.numpy.where` | Function | `(condition, x1, x2)` |
| `src.backend.torch.numpy.zeros` | Function | `(shape, dtype)` |
| `src.backend.torch.numpy.zeros_like` | Function | `(x, dtype)` |
| `src.backend.torch.optimizers.TorchOptimizer` | Class | `(...)` |
| `src.backend.torch.optimizers.torch_adadelta.Adadelta` | Class | `(...)` |
| `src.backend.torch.optimizers.torch_adadelta.ops` | Object | `` |
| `src.backend.torch.optimizers.torch_adadelta.optimizers` | Object | `` |
| `src.backend.torch.optimizers.torch_adadelta.torch_parallel_optimizer` | Object | `` |
| `src.backend.torch.optimizers.torch_adagrad.Adagrad` | Class | `(...)` |
| `src.backend.torch.optimizers.torch_adagrad.ops` | Object | `` |
| `src.backend.torch.optimizers.torch_adagrad.optimizers` | Object | `` |
| `src.backend.torch.optimizers.torch_adagrad.torch_parallel_optimizer` | Object | `` |
| `src.backend.torch.optimizers.torch_adam.Adam` | Class | `(...)` |
| `src.backend.torch.optimizers.torch_adam.ops` | Object | `` |
| `src.backend.torch.optimizers.torch_adam.optimizers` | Object | `` |
| `src.backend.torch.optimizers.torch_adam.torch_parallel_optimizer` | Object | `` |
| `src.backend.torch.optimizers.torch_adamax.Adamax` | Class | `(...)` |
| `src.backend.torch.optimizers.torch_adamax.ops` | Object | `` |
| `src.backend.torch.optimizers.torch_adamax.optimizers` | Object | `` |
| `src.backend.torch.optimizers.torch_adamax.torch_parallel_optimizer` | Object | `` |
| `src.backend.torch.optimizers.torch_adamw.AdamW` | Class | `(...)` |
| `src.backend.torch.optimizers.torch_adamw.optimizers` | Object | `` |
| `src.backend.torch.optimizers.torch_adamw.torch_adam` | Object | `` |
| `src.backend.torch.optimizers.torch_lion.Lion` | Class | `(...)` |
| `src.backend.torch.optimizers.torch_lion.ops` | Object | `` |
| `src.backend.torch.optimizers.torch_lion.optimizers` | Object | `` |
| `src.backend.torch.optimizers.torch_lion.torch_parallel_optimizer` | Object | `` |
| `src.backend.torch.optimizers.torch_nadam.Nadam` | Class | `(...)` |
| `src.backend.torch.optimizers.torch_nadam.core` | Object | `` |
| `src.backend.torch.optimizers.torch_nadam.ops` | Object | `` |
| `src.backend.torch.optimizers.torch_nadam.optimizers` | Object | `` |
| `src.backend.torch.optimizers.torch_nadam.torch_parallel_optimizer` | Object | `` |
| `src.backend.torch.optimizers.torch_optimizer.BaseOptimizer` | Class | `(learning_rate, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps, name, kwargs)` |
| `src.backend.torch.optimizers.torch_optimizer.TorchOptimizer` | Class | `(...)` |
| `src.backend.torch.optimizers.torch_optimizer.optimizers` | Object | `` |
| `src.backend.torch.optimizers.torch_optimizer.torch_utils` | Object | `` |
| `src.backend.torch.optimizers.torch_parallel_optimizer.BaseOptimizer` | Class | `(learning_rate, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps, name, kwargs)` |
| `src.backend.torch.optimizers.torch_parallel_optimizer.TorchParallelOptimizer` | Class | `(...)` |
| `src.backend.torch.optimizers.torch_parallel_optimizer.torch_utils` | Object | `` |
| `src.backend.torch.optimizers.torch_rmsprop.RMSprop` | Class | `(...)` |
| `src.backend.torch.optimizers.torch_rmsprop.ops` | Object | `` |
| `src.backend.torch.optimizers.torch_rmsprop.optimizers` | Object | `` |
| `src.backend.torch.optimizers.torch_rmsprop.torch_parallel_optimizer` | Object | `` |
| `src.backend.torch.optimizers.torch_sgd.SGD` | Class | `(...)` |
| `src.backend.torch.optimizers.torch_sgd.optimizers` | Object | `` |
| `src.backend.torch.optimizers.torch_sgd.torch_parallel_optimizer` | Object | `` |
| `src.backend.torch.random.SeedGenerator` | Class | `(seed, name, kwargs)` |
| `src.backend.torch.random.beta` | Function | `(shape, alpha, beta, dtype, seed)` |
| `src.backend.torch.random.binomial` | Function | `(shape, counts, probabilities, dtype, seed)` |
| `src.backend.torch.random.categorical` | Function | `(logits, num_samples, dtype, seed)` |
| `src.backend.torch.random.convert_to_tensor` | Function | `(x, dtype, sparse, ragged)` |
| `src.backend.torch.random.draw_seed` | Function | `(seed)` |
| `src.backend.torch.random.dropout` | Function | `(inputs, rate, noise_shape, seed)` |
| `src.backend.torch.random.floatx` | Function | `()` |
| `src.backend.torch.random.gamma` | Function | `(shape, alpha, dtype, seed)` |
| `src.backend.torch.random.get_device` | Function | `()` |
| `src.backend.torch.random.make_default_seed` | Function | `()` |
| `src.backend.torch.random.normal` | Function | `(shape, mean, stddev, dtype, seed)` |
| `src.backend.torch.random.randint` | Function | `(shape, minval, maxval, dtype, seed)` |
| `src.backend.torch.random.shuffle` | Function | `(x, axis, seed)` |
| `src.backend.torch.random.to_torch_dtype` | Function | `(dtype)` |
| `src.backend.torch.random.torch_seed_generator` | Function | `(seed)` |
| `src.backend.torch.random.truncated_normal` | Function | `(shape, mean, stddev, dtype, seed)` |
| `src.backend.torch.random.uniform` | Function | `(shape, minval, maxval, dtype, seed)` |
| `src.backend.torch.random_seed_dtype` | Function | `()` |
| `src.backend.torch.rnn.convert_to_tensor` | Function | `(x, dtype, sparse, ragged)` |
| `src.backend.torch.rnn.cudnn_ok` | Function | `(activation, recurrent_activation, unroll, use_bias)` |
| `src.backend.torch.rnn.get_device` | Function | `()` |
| `src.backend.torch.rnn.gru` | Function | `(args, kwargs)` |
| `src.backend.torch.rnn.lstm` | Function | `(inputs, initial_state_h, initial_state_c, mask, kernel, recurrent_kernel, bias, activation, recurrent_activation, return_sequences, go_backwards, unroll, batch_first)` |
| `src.backend.torch.rnn.prepare_lstm_weights` | Function | `(lstm, kernel, recurrent_kernel, bias, device)` |
| `src.backend.torch.rnn.rnn` | Function | `(step_function, inputs, initial_states, go_backwards, mask, constants, unroll, input_length, time_major, zero_output_for_mask, return_all_outputs)` |
| `src.backend.torch.rnn.tree` | Object | `` |
| `src.backend.torch.scatter` | Function | `(indices, values, shape)` |
| `src.backend.torch.shape` | Function | `(x)` |
| `src.backend.torch.stop_gradient` | Function | `(variable)` |
| `src.backend.torch.to_torch_dtype` | Function | `(dtype)` |
| `src.backend.torch.trainer.EpochIterator` | Class | `(x, y, sample_weight, batch_size, steps_per_epoch, shuffle, class_weight, steps_per_execution)` |
| `src.backend.torch.trainer.TorchEpochIterator` | Class | `(...)` |
| `src.backend.torch.trainer.TorchTrainer` | Class | `()` |
| `src.backend.torch.trainer.array_slicing` | Object | `` |
| `src.backend.torch.trainer.backend` | Object | `` |
| `src.backend.torch.trainer.base_trainer` | Object | `` |
| `src.backend.torch.trainer.callbacks_module` | Object | `` |
| `src.backend.torch.trainer.config` | Object | `` |
| `src.backend.torch.trainer.data_adapter_utils` | Object | `` |
| `src.backend.torch.trainer.optimizers_module` | Object | `` |
| `src.backend.torch.trainer.traceback_utils` | Object | `` |
| `src.backend.torch.trainer.tree` | Object | `` |
| `src.backend.torch.vectorized_map` | Function | `(function, elements)` |
| `src.backend.vectorized_map` | Function | `(function, elements)` |
| `src.callbacks.BackupAndRestore` | Class | `(backup_dir, save_freq, double_checkpoint, delete_checkpoint)` |
| `src.callbacks.CSVLogger` | Class | `(filename, separator, append)` |
| `src.callbacks.Callback` | Class | `()` |
| `src.callbacks.CallbackList` | Class | `(callbacks, add_history, add_progbar, model, params)` |
| `src.callbacks.EarlyStopping` | Class | `(monitor, min_delta, patience, verbose, mode, baseline, restore_best_weights, start_from_epoch)` |
| `src.callbacks.History` | Class | `()` |
| `src.callbacks.LambdaCallback` | Class | `(on_epoch_begin, on_epoch_end, on_train_begin, on_train_end, on_train_batch_begin, on_train_batch_end, kwargs)` |
| `src.callbacks.LearningRateScheduler` | Class | `(schedule, verbose)` |
| `src.callbacks.ModelCheckpoint` | Class | `(filepath, monitor, verbose, save_best_only, save_weights_only, mode, save_freq, initial_value_threshold)` |
| `src.callbacks.MonitorCallback` | Class | `(monitor, mode, baseline, min_delta)` |
| `src.callbacks.OrbaxCheckpoint` | Class | `(directory, monitor, verbose, save_best_only, save_weights_only, mode, save_freq, initial_value_threshold, max_to_keep, save_on_background)` |
| `src.callbacks.ProgbarLogger` | Class | `()` |
| `src.callbacks.ReduceLROnPlateau` | Class | `(monitor, factor, patience, verbose, mode, min_delta, cooldown, min_lr, kwargs)` |
| `src.callbacks.RemoteMonitor` | Class | `(root, path, field, headers, send_as_json)` |
| `src.callbacks.SwapEMAWeights` | Class | `(swap_on_epoch)` |
| `src.callbacks.TensorBoard` | Class | `(log_dir, histogram_freq, write_graph, write_images, write_steps_per_second, update_freq, profile_batch, embeddings_freq, embeddings_metadata)` |
| `src.callbacks.TerminateOnNaN` | Class | `(raise_error: bool)` |
| `src.callbacks.backup_and_restore.BackupAndRestore` | Class | `(backup_dir, save_freq, double_checkpoint, delete_checkpoint)` |
| `src.callbacks.backup_and_restore.Callback` | Class | `()` |
| `src.callbacks.backup_and_restore.file_utils` | Object | `` |
| `src.callbacks.backup_and_restore.keras_export` | Class | `(path)` |
| `src.callbacks.callback.Callback` | Class | `()` |
| `src.callbacks.callback.backend` | Object | `` |
| `src.callbacks.callback.keras_export` | Class | `(path)` |
| `src.callbacks.callback.utils` | Object | `` |
| `src.callbacks.callback_list.Callback` | Class | `()` |
| `src.callbacks.callback_list.CallbackList` | Class | `(callbacks, add_history, add_progbar, model, params)` |
| `src.callbacks.callback_list.History` | Class | `()` |
| `src.callbacks.callback_list.ProgbarLogger` | Class | `()` |
| `src.callbacks.callback_list.backend` | Object | `` |
| `src.callbacks.callback_list.keras_export` | Class | `(path)` |
| `src.callbacks.callback_list.python_utils` | Object | `` |
| `src.callbacks.callback_list.tree` | Object | `` |
| `src.callbacks.callback_list.utils` | Object | `` |
| `src.callbacks.csv_logger.CSVLogger` | Class | `(filename, separator, append)` |
| `src.callbacks.csv_logger.Callback` | Class | `()` |
| `src.callbacks.csv_logger.file_utils` | Object | `` |
| `src.callbacks.csv_logger.keras_export` | Class | `(path)` |
| `src.callbacks.early_stopping.EarlyStopping` | Class | `(monitor, min_delta, patience, verbose, mode, baseline, restore_best_weights, start_from_epoch)` |
| `src.callbacks.early_stopping.MonitorCallback` | Class | `(monitor, mode, baseline, min_delta)` |
| `src.callbacks.early_stopping.io_utils` | Object | `` |
| `src.callbacks.early_stopping.keras_export` | Class | `(path)` |
| `src.callbacks.history.Callback` | Class | `()` |
| `src.callbacks.history.History` | Class | `()` |
| `src.callbacks.history.keras_export` | Class | `(path)` |
| `src.callbacks.lambda_callback.Callback` | Class | `()` |
| `src.callbacks.lambda_callback.LambdaCallback` | Class | `(on_epoch_begin, on_epoch_end, on_train_begin, on_train_end, on_train_batch_begin, on_train_batch_end, kwargs)` |
| `src.callbacks.lambda_callback.keras_export` | Class | `(path)` |
| `src.callbacks.learning_rate_scheduler.Callback` | Class | `()` |
| `src.callbacks.learning_rate_scheduler.LearningRateScheduler` | Class | `(schedule, verbose)` |
| `src.callbacks.learning_rate_scheduler.backend` | Object | `` |
| `src.callbacks.learning_rate_scheduler.io_utils` | Object | `` |
| `src.callbacks.learning_rate_scheduler.keras_export` | Class | `(path)` |
| `src.callbacks.model_checkpoint.ModelCheckpoint` | Class | `(filepath, monitor, verbose, save_best_only, save_weights_only, mode, save_freq, initial_value_threshold)` |
| `src.callbacks.model_checkpoint.MonitorCallback` | Class | `(monitor, mode, baseline, min_delta)` |
| `src.callbacks.model_checkpoint.backend` | Object | `` |
| `src.callbacks.model_checkpoint.file_utils` | Object | `` |
| `src.callbacks.model_checkpoint.io_utils` | Object | `` |
| `src.callbacks.model_checkpoint.keras_export` | Class | `(path)` |
| `src.callbacks.monitor_callback.Callback` | Class | `()` |
| `src.callbacks.monitor_callback.MonitorCallback` | Class | `(monitor, mode, baseline, min_delta)` |
| `src.callbacks.monitor_callback.compile_utils` | Object | `` |
| `src.callbacks.monitor_callback.ops` | Object | `` |
| `src.callbacks.orbax_checkpoint.MonitorCallback` | Class | `(monitor, mode, baseline, min_delta)` |
| `src.callbacks.orbax_checkpoint.OrbaxCheckpoint` | Class | `(directory, monitor, verbose, save_best_only, save_weights_only, mode, save_freq, initial_value_threshold, max_to_keep, save_on_background)` |
| `src.callbacks.orbax_checkpoint.backend` | Object | `` |
| `src.callbacks.orbax_checkpoint.ocp` | Object | `` |
| `src.callbacks.orbax_checkpoint.print_msg` | Function | `(message, line_break)` |
| `src.callbacks.orbax_checkpoint.tree` | Object | `` |
| `src.callbacks.progbar_logger.Callback` | Class | `()` |
| `src.callbacks.progbar_logger.Progbar` | Class | `(target, width, verbose, interval, stateful_metrics, unit_name)` |
| `src.callbacks.progbar_logger.ProgbarLogger` | Class | `()` |
| `src.callbacks.progbar_logger.io_utils` | Object | `` |
| `src.callbacks.progbar_logger.keras_export` | Class | `(path)` |
| `src.callbacks.reduce_lr_on_plateau.MonitorCallback` | Class | `(monitor, mode, baseline, min_delta)` |
| `src.callbacks.reduce_lr_on_plateau.ReduceLROnPlateau` | Class | `(monitor, factor, patience, verbose, mode, min_delta, cooldown, min_lr, kwargs)` |
| `src.callbacks.reduce_lr_on_plateau.backend` | Object | `` |
| `src.callbacks.reduce_lr_on_plateau.io_utils` | Object | `` |
| `src.callbacks.reduce_lr_on_plateau.keras_export` | Class | `(path)` |
| `src.callbacks.remote_monitor.Callback` | Class | `()` |
| `src.callbacks.remote_monitor.RemoteMonitor` | Class | `(root, path, field, headers, send_as_json)` |
| `src.callbacks.remote_monitor.keras_export` | Class | `(path)` |
| `src.callbacks.swap_ema_weights.Callback` | Class | `()` |
| `src.callbacks.swap_ema_weights.SwapEMAWeights` | Class | `(swap_on_epoch)` |
| `src.callbacks.swap_ema_weights.backend` | Object | `` |
| `src.callbacks.swap_ema_weights.keras_export` | Class | `(path)` |
| `src.callbacks.swap_ema_weights.ops` | Object | `` |
| `src.callbacks.tensorboard.Callback` | Class | `()` |
| `src.callbacks.tensorboard.Embedding` | Class | `(input_dim, output_dim, embeddings_initializer, embeddings_regularizer, embeddings_constraint, mask_zero, weights, lora_rank, lora_alpha, quantization_config, kwargs)` |
| `src.callbacks.tensorboard.Optimizer` | Class | `(...)` |
| `src.callbacks.tensorboard.TensorBoard` | Class | `(log_dir, histogram_freq, write_graph, write_images, write_steps_per_second, update_freq, profile_batch, embeddings_freq, embeddings_metadata)` |
| `src.callbacks.tensorboard.backend` | Object | `` |
| `src.callbacks.tensorboard.file_utils` | Object | `` |
| `src.callbacks.tensorboard.keras_export` | Class | `(path)` |
| `src.callbacks.tensorboard.keras_model_summary` | Function | `(name, data, step)` |
| `src.callbacks.tensorboard.ops` | Object | `` |
| `src.callbacks.tensorboard.tree` | Object | `` |
| `src.callbacks.terminate_on_nan.Callback` | Class | `()` |
| `src.callbacks.terminate_on_nan.TerminateOnNaN` | Class | `(raise_error: bool)` |
| `src.callbacks.terminate_on_nan.io_utils` | Object | `` |
| `src.callbacks.terminate_on_nan.keras_export` | Class | `(path)` |
| `src.constraints.ALL_OBJECTS` | Object | `` |
| `src.constraints.ALL_OBJECTS_DICT` | Object | `` |
| `src.constraints.Constraint` | Class | `(...)` |
| `src.constraints.MaxNorm` | Class | `(max_value, axis)` |
| `src.constraints.MinMaxNorm` | Class | `(min_value, max_value, rate, axis)` |
| `src.constraints.NonNeg` | Class | `(...)` |
| `src.constraints.UnitNorm` | Class | `(axis)` |
| `src.constraints.constraints.Constraint` | Class | `(...)` |
| `src.constraints.constraints.MaxNorm` | Class | `(max_value, axis)` |
| `src.constraints.constraints.MinMaxNorm` | Class | `(min_value, max_value, rate, axis)` |
| `src.constraints.constraints.NonNeg` | Class | `(...)` |
| `src.constraints.constraints.UnitNorm` | Class | `(axis)` |
| `src.constraints.constraints.backend` | Object | `` |
| `src.constraints.constraints.keras_export` | Class | `(path)` |
| `src.constraints.constraints.ops` | Object | `` |
| `src.constraints.deserialize` | Function | `(config, custom_objects)` |
| `src.constraints.get` | Function | `(identifier)` |
| `src.constraints.keras_export` | Class | `(path)` |
| `src.constraints.serialization_lib` | Object | `` |
| `src.constraints.serialize` | Function | `(constraint)` |
| `src.constraints.to_snake_case` | Function | `(name)` |
| `src.datasets.boston_housing.get_file` | Function | `(fname, origin, untar, md5_hash, file_hash, cache_subdir, hash_algorithm, extract, archive_format, cache_dir, force_download)` |
| `src.datasets.boston_housing.keras_export` | Class | `(path)` |
| `src.datasets.boston_housing.load_data` | Function | `(path, test_split, seed)` |
| `src.datasets.california_housing.get_file` | Function | `(fname, origin, untar, md5_hash, file_hash, cache_subdir, hash_algorithm, extract, archive_format, cache_dir, force_download)` |
| `src.datasets.california_housing.keras_export` | Class | `(path)` |
| `src.datasets.california_housing.load_data` | Function | `(version, path, test_split, seed)` |
| `src.datasets.cifar.load_batch` | Function | `(fpath, label_key)` |
| `src.datasets.cifar10.backend` | Object | `` |
| `src.datasets.cifar10.get_file` | Function | `(fname, origin, untar, md5_hash, file_hash, cache_subdir, hash_algorithm, extract, archive_format, cache_dir, force_download)` |
| `src.datasets.cifar10.keras_export` | Class | `(path)` |
| `src.datasets.cifar10.load_batch` | Function | `(fpath, label_key)` |
| `src.datasets.cifar10.load_data` | Function | `()` |
| `src.datasets.cifar100.backend` | Object | `` |
| `src.datasets.cifar100.get_file` | Function | `(fname, origin, untar, md5_hash, file_hash, cache_subdir, hash_algorithm, extract, archive_format, cache_dir, force_download)` |
| `src.datasets.cifar100.keras_export` | Class | `(path)` |
| `src.datasets.cifar100.load_batch` | Function | `(fpath, label_key)` |
| `src.datasets.cifar100.load_data` | Function | `(label_mode)` |
| `src.datasets.fashion_mnist.get_file` | Function | `(fname, origin, untar, md5_hash, file_hash, cache_subdir, hash_algorithm, extract, archive_format, cache_dir, force_download)` |
| `src.datasets.fashion_mnist.keras_export` | Class | `(path)` |
| `src.datasets.fashion_mnist.load_data` | Function | `()` |
| `src.datasets.imdb.get_file` | Function | `(fname, origin, untar, md5_hash, file_hash, cache_subdir, hash_algorithm, extract, archive_format, cache_dir, force_download)` |
| `src.datasets.imdb.get_word_index` | Function | `(path)` |
| `src.datasets.imdb.keras_export` | Class | `(path)` |
| `src.datasets.imdb.load_data` | Function | `(path, num_words, skip_top, maxlen, seed, start_char, oov_char, index_from, kwargs)` |
| `src.datasets.imdb.remove_long_seq` | Function | `(maxlen, seq, label)` |
| `src.datasets.mnist.get_file` | Function | `(fname, origin, untar, md5_hash, file_hash, cache_subdir, hash_algorithm, extract, archive_format, cache_dir, force_download)` |
| `src.datasets.mnist.keras_export` | Class | `(path)` |
| `src.datasets.mnist.load_data` | Function | `(path)` |
| `src.datasets.reuters.get_file` | Function | `(fname, origin, untar, md5_hash, file_hash, cache_subdir, hash_algorithm, extract, archive_format, cache_dir, force_download)` |
| `src.datasets.reuters.get_label_names` | Function | `()` |
| `src.datasets.reuters.get_word_index` | Function | `(path)` |
| `src.datasets.reuters.keras_export` | Class | `(path)` |
| `src.datasets.reuters.load_data` | Function | `(path, num_words, skip_top, maxlen, test_split, seed, start_char, oov_char, index_from)` |
| `src.datasets.reuters.remove_long_seq` | Function | `(maxlen, seq, label)` |
| `src.distillation.distillation_loss.DistillationLoss` | Class | `(...)` |
| `src.distillation.distillation_loss.FeatureDistillation` | Class | `(loss, teacher_layer_name, student_layer_name)` |
| `src.distillation.distillation_loss.LogitsDistillation` | Class | `(temperature, loss)` |
| `src.distillation.distillation_loss.keras_export` | Class | `(path)` |
| `src.distillation.distillation_loss.serialization_lib` | Object | `` |
| `src.distillation.distillation_loss.tracking` | Object | `` |
| `src.distillation.distillation_loss.tree` | Object | `` |
| `src.distillation.distiller.Distiller` | Class | `(teacher, student, distillation_losses, distillation_loss_weights, student_loss_weight, name, kwargs)` |
| `src.distillation.distiller.Model` | Class | `(args, kwargs)` |
| `src.distillation.distiller.keras_export` | Class | `(path)` |
| `src.distillation.distiller.serialization_lib` | Object | `` |
| `src.distillation.distiller.tree` | Object | `` |
| `src.distribution.DataParallel` | Class | `(device_mesh, devices, auto_shard_dataset)` |
| `src.distribution.DeviceMesh` | Class | `(shape, axis_names, devices)` |
| `src.distribution.Distribution` | Class | `(device_mesh, batch_dim_name, auto_shard_dataset)` |
| `src.distribution.LayoutMap` | Class | `(device_mesh)` |
| `src.distribution.ModelParallel` | Class | `(layout_map, batch_dim_name, auto_shard_dataset, kwargs)` |
| `src.distribution.TensorLayout` | Class | `(axes, device_mesh)` |
| `src.distribution.distribute_tensor` | Function | `(tensor, layout)` |
| `src.distribution.distribution` | Function | `()` |
| `src.distribution.distribution_lib.DEFAULT_BATCH_DIM_NAME` | Object | `` |
| `src.distribution.distribution_lib.DataParallel` | Class | `(device_mesh, devices, auto_shard_dataset)` |
| `src.distribution.distribution_lib.DeviceMesh` | Class | `(shape, axis_names, devices)` |
| `src.distribution.distribution_lib.Distribution` | Class | `(device_mesh, batch_dim_name, auto_shard_dataset)` |
| `src.distribution.distribution_lib.GLOBAL_ATTRIBUTE_NAME` | Object | `` |
| `src.distribution.distribution_lib.KerasTensor` | Class | `(shape, dtype, sparse, ragged, record_history, name, kwargs)` |
| `src.distribution.distribution_lib.LayoutMap` | Class | `(device_mesh)` |
| `src.distribution.distribution_lib.ModelParallel` | Class | `(layout_map, batch_dim_name, auto_shard_dataset, kwargs)` |
| `src.distribution.distribution_lib.TensorLayout` | Class | `(axes, device_mesh)` |
| `src.distribution.distribution_lib.distribute_tensor` | Function | `(tensor, layout)` |
| `src.distribution.distribution_lib.distribution` | Function | `()` |
| `src.distribution.distribution_lib.distribution_lib` | Object | `` |
| `src.distribution.distribution_lib.get_device_count` | Function | `(device_type)` |
| `src.distribution.distribution_lib.global_state` | Object | `` |
| `src.distribution.distribution_lib.initialize` | Function | `(job_addresses, num_processes, process_id)` |
| `src.distribution.distribution_lib.keras_export` | Class | `(path)` |
| `src.distribution.distribution_lib.list_devices` | Function | `(device_type)` |
| `src.distribution.distribution_lib.set_distribution` | Function | `(value)` |
| `src.distribution.initialize` | Function | `(job_addresses, num_processes, process_id)` |
| `src.distribution.list_devices` | Function | `(device_type)` |
| `src.distribution.set_distribution` | Function | `(value)` |
| `src.dtype_policies.ALL_OBJECTS` | Object | `` |
| `src.dtype_policies.ALL_OBJECTS_DICT` | Object | `` |
| `src.dtype_policies.DTypePolicy` | Class | `(name)` |
| `src.dtype_policies.DTypePolicyMap` | Class | `(default_policy, policy_map)` |
| `src.dtype_policies.FloatDTypePolicy` | Class | `(...)` |
| `src.dtype_policies.GPTQDTypePolicy` | Class | `(mode, source_name)` |
| `src.dtype_policies.QUANTIZATION_MODES` | Object | `` |
| `src.dtype_policies.QuantizedDTypePolicy` | Class | `(mode, source_name)` |
| `src.dtype_policies.QuantizedFloat8DTypePolicy` | Class | `(mode, source_name, amax_history_length)` |
| `src.dtype_policies.backend` | Object | `` |
| `src.dtype_policies.deserialize` | Function | `(config, custom_objects)` |
| `src.dtype_policies.dtype_policy.DTypePolicy` | Class | `(name)` |
| `src.dtype_policies.dtype_policy.FloatDTypePolicy` | Class | `(...)` |
| `src.dtype_policies.dtype_policy.GPTQDTypePolicy` | Class | `(mode, source_name)` |
| `src.dtype_policies.dtype_policy.QUANTIZATION_MODES` | Object | `` |
| `src.dtype_policies.dtype_policy.QuantizedDTypePolicy` | Class | `(mode, source_name)` |
| `src.dtype_policies.dtype_policy.QuantizedFloat8DTypePolicy` | Class | `(mode, source_name, amax_history_length)` |
| `src.dtype_policies.dtype_policy.backend` | Object | `` |
| `src.dtype_policies.dtype_policy.dtype_policy` | Function | `()` |
| `src.dtype_policies.dtype_policy.global_state` | Object | `` |
| `src.dtype_policies.dtype_policy.keras_export` | Class | `(path)` |
| `src.dtype_policies.dtype_policy.ops` | Object | `` |
| `src.dtype_policies.dtype_policy.set_dtype_policy` | Function | `(policy)` |
| `src.dtype_policies.dtype_policy_map.DTypePolicy` | Class | `(name)` |
| `src.dtype_policies.dtype_policy_map.DTypePolicyMap` | Class | `(default_policy, policy_map)` |
| `src.dtype_policies.dtype_policy_map.dtype_policies` | Object | `` |
| `src.dtype_policies.dtype_policy_map.keras_export` | Class | `(path)` |
| `src.dtype_policies.get` | Function | `(identifier)` |
| `src.dtype_policies.keras_export` | Class | `(path)` |
| `src.dtype_policies.serialize` | Function | `(dtype_policy)` |
| `src.export.ExportArchive` | Class | `()` |
| `src.export.LiteRTExporter` | Class | `(model, input_signature, kwargs)` |
| `src.export.TFSMLayer` | Class | `(filepath, call_endpoint, call_training_endpoint, trainable, name, dtype)` |
| `src.export.export_litert` | Function | `(model, filepath, input_signature, kwargs)` |
| `src.export.export_onnx` | Function | `(model, filepath, verbose, input_signature, opset_version, kwargs)` |
| `src.export.export_openvino` | Function | `(model, filepath, verbose, input_signature, kwargs)` |
| `src.export.export_saved_model` | Function | `(model, filepath, verbose, input_signature, kwargs)` |
| `src.export.export_utils.backend` | Object | `` |
| `src.export.export_utils.convert_spec_to_tensor` | Function | `(spec, replace_none_number)` |
| `src.export.export_utils.get_input_signature` | Function | `(model)` |
| `src.export.export_utils.layers` | Object | `` |
| `src.export.export_utils.make_input_spec` | Function | `(x)` |
| `src.export.export_utils.make_tf_tensor_spec` | Function | `(x, dynamic_batch)` |
| `src.export.export_utils.models` | Object | `` |
| `src.export.export_utils.ops` | Object | `` |
| `src.export.export_utils.tf` | Object | `` |
| `src.export.export_utils.tree` | Object | `` |
| `src.export.litert.LiteRTExporter` | Class | `(model, input_signature, kwargs)` |
| `src.export.litert.export_litert` | Function | `(model, filepath, input_signature, kwargs)` |
| `src.export.litert.get_input_signature` | Function | `(model)` |
| `src.export.litert.io_utils` | Object | `` |
| `src.export.litert.layers` | Object | `` |
| `src.export.litert.models` | Object | `` |
| `src.export.litert.tf` | Object | `` |
| `src.export.litert.tree` | Object | `` |
| `src.export.onnx.DEFAULT_ENDPOINT_NAME` | Object | `` |
| `src.export.onnx.ExportArchive` | Class | `()` |
| `src.export.onnx.backend` | Object | `` |
| `src.export.onnx.convert_spec_to_tensor` | Function | `(spec, replace_none_number)` |
| `src.export.onnx.export_onnx` | Function | `(model, filepath, verbose, input_signature, opset_version, kwargs)` |
| `src.export.onnx.get_concrete_fn` | Function | `(model, input_signature, kwargs)` |
| `src.export.onnx.get_input_signature` | Function | `(model)` |
| `src.export.onnx.io_utils` | Object | `` |
| `src.export.onnx.make_tf_tensor_spec` | Function | `(x, dynamic_batch)` |
| `src.export.onnx.patch_tf2onnx` | Function | `()` |
| `src.export.onnx.tree` | Object | `` |
| `src.export.openvino.DEFAULT_ENDPOINT_NAME` | Object | `` |
| `src.export.openvino.ExportArchive` | Class | `()` |
| `src.export.openvino.backend` | Object | `` |
| `src.export.openvino.collect_names` | Function | `(structure)` |
| `src.export.openvino.convert_spec_to_tensor` | Function | `(spec, replace_none_number)` |
| `src.export.openvino.export_openvino` | Function | `(model, filepath, verbose, input_signature, kwargs)` |
| `src.export.openvino.get_concrete_fn` | Function | `(model, input_signature, kwargs)` |
| `src.export.openvino.get_input_signature` | Function | `(model)` |
| `src.export.openvino.io_utils` | Object | `` |
| `src.export.openvino.make_tf_tensor_spec` | Function | `(x, dynamic_batch)` |
| `src.export.openvino.set_names` | Function | `(model, inputs)` |
| `src.export.openvino.tree` | Object | `` |
| `src.export.saved_model.BackendExportArchive` | Class | `(...)` |
| `src.export.saved_model.DEFAULT_ENDPOINT_NAME` | Object | `` |
| `src.export.saved_model.ExportArchive` | Class | `()` |
| `src.export.saved_model.backend` | Object | `` |
| `src.export.saved_model.export_saved_model` | Function | `(model, filepath, verbose, input_signature, kwargs)` |
| `src.export.saved_model.get_input_signature` | Function | `(model)` |
| `src.export.saved_model.io_utils` | Object | `` |
| `src.export.saved_model.keras_export` | Class | `(path)` |
| `src.export.saved_model.layers` | Object | `` |
| `src.export.saved_model.make_tf_tensor_spec` | Function | `(x, dynamic_batch)` |
| `src.export.saved_model.tf` | Object | `` |
| `src.export.saved_model.tree` | Object | `` |
| `src.export.tf2onnx_lib.patch_tf2onnx` | Function | `()` |
| `src.export.tfsm_layer.TFSMLayer` | Class | `(filepath, call_endpoint, call_training_endpoint, trainable, name, dtype)` |
| `src.export.tfsm_layer.backend` | Object | `` |
| `src.export.tfsm_layer.keras_export` | Class | `(path)` |
| `src.export.tfsm_layer.layers` | Object | `` |
| `src.export.tfsm_layer.serialization_lib` | Object | `` |
| `src.export.tfsm_layer.tf` | Object | `` |
| `src.initializers.ALL_OBJECTS` | Object | `` |
| `src.initializers.ALL_OBJECTS_DICT` | Object | `` |
| `src.initializers.Constant` | Class | `(value)` |
| `src.initializers.GlorotNormal` | Class | `(seed)` |
| `src.initializers.GlorotUniform` | Class | `(seed)` |
| `src.initializers.HeNormal` | Class | `(seed)` |
| `src.initializers.HeUniform` | Class | `(seed)` |
| `src.initializers.Identity` | Class | `(gain)` |
| `src.initializers.Initializer` | Class | `(...)` |
| `src.initializers.LecunNormal` | Class | `(seed)` |
| `src.initializers.LecunUniform` | Class | `(seed)` |
| `src.initializers.Ones` | Class | `(...)` |
| `src.initializers.Orthogonal` | Class | `(gain, seed)` |
| `src.initializers.RandomNormal` | Class | `(mean, stddev, seed)` |
| `src.initializers.RandomUniform` | Class | `(minval, maxval, seed)` |
| `src.initializers.STFT` | Class | `(side, window, scaling, periodic)` |
| `src.initializers.TruncatedNormal` | Class | `(mean, stddev, seed)` |
| `src.initializers.VarianceScaling` | Class | `(scale, mode, distribution, seed)` |
| `src.initializers.Zeros` | Class | `(...)` |
| `src.initializers.backend` | Object | `` |
| `src.initializers.constant_initializers.Constant` | Class | `(value)` |
| `src.initializers.constant_initializers.Identity` | Class | `(gain)` |
| `src.initializers.constant_initializers.Initializer` | Class | `(...)` |
| `src.initializers.constant_initializers.Ones` | Class | `(...)` |
| `src.initializers.constant_initializers.STFT` | Class | `(side, window, scaling, periodic)` |
| `src.initializers.constant_initializers.Zeros` | Class | `(...)` |
| `src.initializers.constant_initializers.keras_export` | Class | `(path)` |
| `src.initializers.constant_initializers.ops` | Object | `` |
| `src.initializers.constant_initializers.scipy` | Object | `` |
| `src.initializers.constant_initializers.serialization_lib` | Object | `` |
| `src.initializers.constant_initializers.standardize_dtype` | Function | `(dtype)` |
| `src.initializers.deserialize` | Function | `(config, custom_objects)` |
| `src.initializers.get` | Function | `(identifier)` |
| `src.initializers.initializer.Initializer` | Class | `(...)` |
| `src.initializers.initializer.keras_export` | Class | `(path)` |
| `src.initializers.keras_export` | Class | `(path)` |
| `src.initializers.ops` | Object | `` |
| `src.initializers.random_initializers.GlorotNormal` | Class | `(seed)` |
| `src.initializers.random_initializers.GlorotUniform` | Class | `(seed)` |
| `src.initializers.random_initializers.HeNormal` | Class | `(seed)` |
| `src.initializers.random_initializers.HeUniform` | Class | `(seed)` |
| `src.initializers.random_initializers.Initializer` | Class | `(...)` |
| `src.initializers.random_initializers.LecunNormal` | Class | `(seed)` |
| `src.initializers.random_initializers.LecunUniform` | Class | `(seed)` |
| `src.initializers.random_initializers.Orthogonal` | Class | `(gain, seed)` |
| `src.initializers.random_initializers.RandomInitializer` | Class | `(seed)` |
| `src.initializers.random_initializers.RandomNormal` | Class | `(mean, stddev, seed)` |
| `src.initializers.random_initializers.RandomUniform` | Class | `(minval, maxval, seed)` |
| `src.initializers.random_initializers.TruncatedNormal` | Class | `(mean, stddev, seed)` |
| `src.initializers.random_initializers.VarianceScaling` | Class | `(scale, mode, distribution, seed)` |
| `src.initializers.random_initializers.compute_fans` | Function | `(shape)` |
| `src.initializers.random_initializers.keras_export` | Class | `(path)` |
| `src.initializers.random_initializers.ops` | Object | `` |
| `src.initializers.random_initializers.random` | Object | `` |
| `src.initializers.random_initializers.serialization_lib` | Object | `` |
| `src.initializers.serialization_lib` | Object | `` |
| `src.initializers.serialize` | Function | `(initializer)` |
| `src.initializers.to_snake_case` | Function | `(name)` |
| `src.layers.Activation` | Class | `(activation, kwargs)` |
| `src.layers.ActivityRegularization` | Class | `(l1, l2, kwargs)` |
| `src.layers.AdaptiveAveragePooling1D` | Class | `(output_size, data_format, kwargs)` |
| `src.layers.AdaptiveAveragePooling2D` | Class | `(output_size, data_format, kwargs)` |
| `src.layers.AdaptiveAveragePooling3D` | Class | `(output_size, data_format, kwargs)` |
| `src.layers.AdaptiveMaxPooling1D` | Class | `(output_size, data_format, kwargs)` |
| `src.layers.AdaptiveMaxPooling2D` | Class | `(output_size, data_format, kwargs)` |
| `src.layers.AdaptiveMaxPooling3D` | Class | `(output_size, data_format, kwargs)` |
| `src.layers.Add` | Class | `(...)` |
| `src.layers.AdditiveAttention` | Class | `(use_scale, dropout, kwargs)` |
| `src.layers.AlphaDropout` | Class | `(rate, noise_shape, seed, kwargs)` |
| `src.layers.Attention` | Class | `(use_scale, score_mode, dropout, seed, kwargs)` |
| `src.layers.AugMix` | Class | `(value_range, num_chains, chain_depth, factor, alpha, all_ops, interpolation, seed, data_format, kwargs)` |
| `src.layers.AutoContrast` | Class | `(value_range, kwargs)` |
| `src.layers.Average` | Class | `(...)` |
| `src.layers.AveragePooling1D` | Class | `(pool_size, strides, padding, data_format, name, kwargs)` |
| `src.layers.AveragePooling2D` | Class | `(pool_size, strides, padding, data_format, name, kwargs)` |
| `src.layers.AveragePooling3D` | Class | `(pool_size, strides, padding, data_format, name, kwargs)` |
| `src.layers.BatchNormalization` | Class | `(axis, momentum, epsilon, center, scale, beta_initializer, gamma_initializer, moving_mean_initializer, moving_variance_initializer, beta_regularizer, gamma_regularizer, beta_constraint, gamma_constraint, synchronized, kwargs)` |
| `src.layers.Bidirectional` | Class | `(layer, merge_mode, weights, backward_layer, kwargs)` |
| `src.layers.CategoryEncoding` | Class | `(num_tokens, output_mode, sparse, kwargs)` |
| `src.layers.CenterCrop` | Class | `(height, width, data_format, kwargs)` |
| `src.layers.Concatenate` | Class | `(axis, kwargs)` |
| `src.layers.Conv1D` | Class | `(filters, kernel_size, strides, padding, data_format, dilation_rate, groups, activation, use_bias, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint, kwargs)` |
| `src.layers.Conv1DTranspose` | Class | `(filters, kernel_size, strides, padding, output_padding, data_format, dilation_rate, activation, use_bias, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint, kwargs)` |
| `src.layers.Conv2D` | Class | `(filters, kernel_size, strides, padding, data_format, dilation_rate, groups, activation, use_bias, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint, kwargs)` |
| `src.layers.Conv2DTranspose` | Class | `(filters, kernel_size, strides, padding, output_padding, data_format, dilation_rate, activation, use_bias, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint, kwargs)` |
| `src.layers.Conv3D` | Class | `(filters, kernel_size, strides, padding, data_format, dilation_rate, groups, activation, use_bias, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint, kwargs)` |
| `src.layers.Conv3DTranspose` | Class | `(filters, kernel_size, strides, padding, data_format, output_padding, dilation_rate, activation, use_bias, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint, kwargs)` |
| `src.layers.ConvLSTM1D` | Class | `(filters, kernel_size, strides, padding, data_format, dilation_rate, activation, recurrent_activation, use_bias, kernel_initializer, recurrent_initializer, bias_initializer, unit_forget_bias, kernel_regularizer, recurrent_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, recurrent_constraint, bias_constraint, dropout, recurrent_dropout, seed, return_sequences, return_state, go_backwards, stateful, kwargs)` |
| `src.layers.ConvLSTM2D` | Class | `(filters, kernel_size, strides, padding, data_format, dilation_rate, activation, recurrent_activation, use_bias, kernel_initializer, recurrent_initializer, bias_initializer, unit_forget_bias, kernel_regularizer, recurrent_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, recurrent_constraint, bias_constraint, dropout, recurrent_dropout, seed, return_sequences, return_state, go_backwards, stateful, kwargs)` |
| `src.layers.ConvLSTM3D` | Class | `(filters, kernel_size, strides, padding, data_format, dilation_rate, activation, recurrent_activation, use_bias, kernel_initializer, recurrent_initializer, bias_initializer, unit_forget_bias, kernel_regularizer, recurrent_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, recurrent_constraint, bias_constraint, dropout, recurrent_dropout, seed, return_sequences, return_state, go_backwards, stateful, kwargs)` |
| `src.layers.Cropping1D` | Class | `(cropping, kwargs)` |
| `src.layers.Cropping2D` | Class | `(cropping, data_format, kwargs)` |
| `src.layers.Cropping3D` | Class | `(cropping, data_format, kwargs)` |
| `src.layers.CutMix` | Class | `(factor, seed, data_format, kwargs)` |
| `src.layers.Dense` | Class | `(units, activation, use_bias, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint, lora_rank, lora_alpha, quantization_config, kwargs)` |
| `src.layers.DepthwiseConv1D` | Class | `(kernel_size, strides, padding, depth_multiplier, data_format, dilation_rate, activation, use_bias, depthwise_initializer, bias_initializer, depthwise_regularizer, bias_regularizer, activity_regularizer, depthwise_constraint, bias_constraint, kwargs)` |
| `src.layers.DepthwiseConv2D` | Class | `(kernel_size, strides, padding, depth_multiplier, data_format, dilation_rate, activation, use_bias, depthwise_initializer, bias_initializer, depthwise_regularizer, bias_regularizer, activity_regularizer, depthwise_constraint, bias_constraint, kwargs)` |
| `src.layers.Discretization` | Class | `(bin_boundaries, num_bins, epsilon, output_mode, sparse, dtype, name)` |
| `src.layers.Dot` | Class | `(axes, normalize, kwargs)` |
| `src.layers.Dropout` | Class | `(rate, noise_shape, seed, kwargs)` |
| `src.layers.ELU` | Class | `(alpha, kwargs)` |
| `src.layers.EinsumDense` | Class | `(equation, output_shape, activation, bias_axes, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, kernel_constraint, bias_constraint, lora_rank, lora_alpha, gptq_unpacked_column_size, quantization_config, kwargs)` |
| `src.layers.Embedding` | Class | `(input_dim, output_dim, embeddings_initializer, embeddings_regularizer, embeddings_constraint, mask_zero, weights, lora_rank, lora_alpha, quantization_config, kwargs)` |
| `src.layers.Equalization` | Class | `(value_range, bins, data_format, kwargs)` |
| `src.layers.Flatten` | Class | `(data_format, kwargs)` |
| `src.layers.GRU` | Class | `(units, activation, recurrent_activation, use_bias, kernel_initializer, recurrent_initializer, bias_initializer, kernel_regularizer, recurrent_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, recurrent_constraint, bias_constraint, dropout, recurrent_dropout, seed, return_sequences, return_state, go_backwards, stateful, unroll, reset_after, use_cudnn, kwargs)` |
| `src.layers.GRUCell` | Class | `(units, activation, recurrent_activation, use_bias, kernel_initializer, recurrent_initializer, bias_initializer, kernel_regularizer, recurrent_regularizer, bias_regularizer, kernel_constraint, recurrent_constraint, bias_constraint, dropout, recurrent_dropout, reset_after, seed, kwargs)` |
| `src.layers.GaussianDropout` | Class | `(rate, seed, kwargs)` |
| `src.layers.GaussianNoise` | Class | `(stddev, seed, kwargs)` |
| `src.layers.GlobalAveragePooling1D` | Class | `(data_format, keepdims, kwargs)` |
| `src.layers.GlobalAveragePooling2D` | Class | `(data_format, keepdims, kwargs)` |
| `src.layers.GlobalAveragePooling3D` | Class | `(data_format, keepdims, kwargs)` |
| `src.layers.GlobalMaxPooling1D` | Class | `(data_format, keepdims, kwargs)` |
| `src.layers.GlobalMaxPooling2D` | Class | `(data_format, keepdims, kwargs)` |
| `src.layers.GlobalMaxPooling3D` | Class | `(data_format, keepdims, kwargs)` |
| `src.layers.GroupNormalization` | Class | `(groups, axis, epsilon, center, scale, beta_initializer, gamma_initializer, beta_regularizer, gamma_regularizer, beta_constraint, gamma_constraint, kwargs)` |
| `src.layers.GroupedQueryAttention` | Class | `(head_dim, num_query_heads, num_key_value_heads, dropout, use_bias, flash_attention, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint, seed, kwargs)` |
| `src.layers.HashedCrossing` | Class | `(num_bins, output_mode, sparse, name, dtype, kwargs)` |
| `src.layers.Hashing` | Class | `(num_bins, mask_value, salt, output_mode, sparse, kwargs)` |
| `src.layers.Identity` | Class | `(kwargs)` |
| `src.layers.IndexLookup` | Class | `(max_tokens, num_oov_indices, mask_token, oov_token, vocabulary_dtype, vocabulary, idf_weights, invert, output_mode, sparse, pad_to_max_tokens, name, kwargs)` |
| `src.layers.Input` | Function | `(shape, batch_size, dtype, sparse, ragged, batch_shape, name, tensor, optional)` |
| `src.layers.InputLayer` | Class | `(shape, batch_size, dtype, sparse, ragged, batch_shape, input_tensor, optional, name, kwargs)` |
| `src.layers.InputSpec` | Class | `(dtype, shape, ndim, max_ndim, min_ndim, axes, allow_last_axis_squeeze, name, optional)` |
| `src.layers.IntegerLookup` | Class | `(max_tokens, num_oov_indices, mask_token, oov_token, vocabulary, vocabulary_dtype, idf_weights, invert, output_mode, sparse, pad_to_max_tokens, name, kwargs)` |
| `src.layers.LSTM` | Class | `(units, activation, recurrent_activation, use_bias, kernel_initializer, recurrent_initializer, bias_initializer, unit_forget_bias, kernel_regularizer, recurrent_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, recurrent_constraint, bias_constraint, dropout, recurrent_dropout, seed, return_sequences, return_state, go_backwards, stateful, unroll, use_cudnn, kwargs)` |
| `src.layers.LSTMCell` | Class | `(units, activation, recurrent_activation, use_bias, kernel_initializer, recurrent_initializer, bias_initializer, unit_forget_bias, kernel_regularizer, recurrent_regularizer, bias_regularizer, kernel_constraint, recurrent_constraint, bias_constraint, dropout, recurrent_dropout, seed, kwargs)` |
| `src.layers.Lambda` | Class | `(function, output_shape, mask, arguments, kwargs)` |
| `src.layers.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.layers.LayerNormalization` | Class | `(axis, epsilon, center, scale, beta_initializer, gamma_initializer, beta_regularizer, gamma_regularizer, beta_constraint, gamma_constraint, kwargs)` |
| `src.layers.LeakyReLU` | Class | `(negative_slope, kwargs)` |
| `src.layers.Masking` | Class | `(mask_value, kwargs)` |
| `src.layers.MaxNumBoundingBoxes` | Class | `(max_number, fill_value, kwargs)` |
| `src.layers.MaxPooling1D` | Class | `(pool_size, strides, padding, data_format, name, kwargs)` |
| `src.layers.MaxPooling2D` | Class | `(pool_size, strides, padding, data_format, name, kwargs)` |
| `src.layers.MaxPooling3D` | Class | `(pool_size, strides, padding, data_format, name, kwargs)` |
| `src.layers.Maximum` | Class | `(...)` |
| `src.layers.MelSpectrogram` | Class | `(fft_length, sequence_stride, sequence_length, window, sampling_rate, num_mel_bins, min_freq, max_freq, power_to_db, top_db, mag_exp, min_power, ref_power, kwargs)` |
| `src.layers.Minimum` | Class | `(...)` |
| `src.layers.MixUp` | Class | `(alpha, data_format, seed, kwargs)` |
| `src.layers.MultiHeadAttention` | Class | `(num_heads, key_dim, value_dim, dropout, use_bias, output_shape, attention_axes, flash_attention, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint, seed, kwargs)` |
| `src.layers.Multiply` | Class | `(...)` |
| `src.layers.Normalization` | Class | `(axis, mean, variance, invert, kwargs)` |
| `src.layers.PReLU` | Class | `(alpha_initializer, alpha_regularizer, alpha_constraint, shared_axes, kwargs)` |
| `src.layers.Permute` | Class | `(dims, kwargs)` |
| `src.layers.Pipeline` | Class | `(layers, name)` |
| `src.layers.RMSNormalization` | Class | `(axis, epsilon, kwargs)` |
| `src.layers.RNN` | Class | `(cell, return_sequences, return_state, go_backwards, stateful, unroll, zero_output_for_mask, kwargs)` |
| `src.layers.RandAugment` | Class | `(value_range, num_ops, factor, interpolation, seed, data_format, kwargs)` |
| `src.layers.RandomBrightness` | Class | `(factor, value_range, seed, kwargs)` |
| `src.layers.RandomColorDegeneration` | Class | `(factor, value_range, data_format, seed, kwargs)` |
| `src.layers.RandomColorJitter` | Class | `(value_range, brightness_factor, contrast_factor, saturation_factor, hue_factor, seed, data_format, kwargs)` |
| `src.layers.RandomContrast` | Class | `(factor, value_range, seed, kwargs)` |
| `src.layers.RandomCrop` | Class | `(height, width, seed, data_format, name, kwargs)` |
| `src.layers.RandomElasticTransform` | Class | `(factor, scale, interpolation, fill_mode, fill_value, value_range, seed, data_format, kwargs)` |
| `src.layers.RandomErasing` | Class | `(factor, scale, fill_value, value_range, seed, data_format, kwargs)` |
| `src.layers.RandomFlip` | Class | `(mode, seed, data_format, kwargs)` |
| `src.layers.RandomGaussianBlur` | Class | `(factor, kernel_size, sigma, value_range, data_format, seed, kwargs)` |
| `src.layers.RandomGrayscale` | Class | `(factor, data_format, seed, kwargs)` |
| `src.layers.RandomHue` | Class | `(factor, value_range, data_format, seed, kwargs)` |
| `src.layers.RandomInvert` | Class | `(factor, value_range, seed, data_format, kwargs)` |
| `src.layers.RandomPerspective` | Class | `(factor, scale, interpolation, fill_value, seed, data_format, kwargs)` |
| `src.layers.RandomPosterization` | Class | `(factor, value_range, data_format, seed, kwargs)` |
| `src.layers.RandomRotation` | Class | `(factor, fill_mode, interpolation, seed, fill_value, data_format, kwargs)` |
| `src.layers.RandomSaturation` | Class | `(factor, value_range, data_format, seed, kwargs)` |
| `src.layers.RandomSharpness` | Class | `(factor, value_range, data_format, seed, kwargs)` |
| `src.layers.RandomShear` | Class | `(x_factor, y_factor, interpolation, fill_mode, fill_value, data_format, seed, kwargs)` |
| `src.layers.RandomTranslation` | Class | `(height_factor, width_factor, fill_mode, interpolation, seed, fill_value, data_format, kwargs)` |
| `src.layers.RandomZoom` | Class | `(height_factor, width_factor, fill_mode, interpolation, seed, fill_value, data_format, kwargs)` |
| `src.layers.ReLU` | Class | `(max_value, negative_slope, threshold, kwargs)` |
| `src.layers.RepeatVector` | Class | `(n, kwargs)` |
| `src.layers.Rescaling` | Class | `(scale, offset, kwargs)` |
| `src.layers.Reshape` | Class | `(target_shape, kwargs)` |
| `src.layers.Resizing` | Class | `(height, width, interpolation, crop_to_aspect_ratio, pad_to_aspect_ratio, fill_mode, fill_value, antialias, data_format, kwargs)` |
| `src.layers.ReversibleEmbedding` | Class | `(input_dim, output_dim, tie_weights, embeddings_initializer, embeddings_regularizer, embeddings_constraint, mask_zero, reverse_dtype, logit_soft_cap, kwargs)` |
| `src.layers.STFTSpectrogram` | Class | `(mode, frame_length, frame_step, fft_length, window, periodic, scaling, padding, expand_dims, data_format, kwargs)` |
| `src.layers.SeparableConv1D` | Class | `(filters, kernel_size, strides, padding, data_format, dilation_rate, depth_multiplier, activation, use_bias, depthwise_initializer, pointwise_initializer, bias_initializer, depthwise_regularizer, pointwise_regularizer, bias_regularizer, activity_regularizer, depthwise_constraint, pointwise_constraint, bias_constraint, kwargs)` |
| `src.layers.SeparableConv2D` | Class | `(filters, kernel_size, strides, padding, data_format, dilation_rate, depth_multiplier, activation, use_bias, depthwise_initializer, pointwise_initializer, bias_initializer, depthwise_regularizer, pointwise_regularizer, bias_regularizer, activity_regularizer, depthwise_constraint, pointwise_constraint, bias_constraint, kwargs)` |
| `src.layers.SimpleRNN` | Class | `(units, activation, use_bias, kernel_initializer, recurrent_initializer, bias_initializer, kernel_regularizer, recurrent_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, recurrent_constraint, bias_constraint, dropout, recurrent_dropout, return_sequences, return_state, go_backwards, stateful, unroll, seed, kwargs)` |
| `src.layers.SimpleRNNCell` | Class | `(units, activation, use_bias, kernel_initializer, recurrent_initializer, bias_initializer, kernel_regularizer, recurrent_regularizer, bias_regularizer, kernel_constraint, recurrent_constraint, bias_constraint, dropout, recurrent_dropout, seed, kwargs)` |
| `src.layers.Softmax` | Class | `(axis, kwargs)` |
| `src.layers.Solarization` | Class | `(addition_factor, threshold_factor, value_range, seed, kwargs)` |
| `src.layers.SpatialDropout1D` | Class | `(rate, seed, name, dtype)` |
| `src.layers.SpatialDropout2D` | Class | `(rate, data_format, seed, name, dtype)` |
| `src.layers.SpatialDropout3D` | Class | `(rate, data_format, seed, name, dtype)` |
| `src.layers.SpectralNormalization` | Class | `(layer, power_iterations, kwargs)` |
| `src.layers.StackedRNNCells` | Class | `(cells, kwargs)` |
| `src.layers.StringLookup` | Class | `(max_tokens, num_oov_indices, mask_token, oov_token, vocabulary, idf_weights, invert, output_mode, pad_to_max_tokens, sparse, encoding, name, kwargs)` |
| `src.layers.Subtract` | Class | `(...)` |
| `src.layers.TextVectorization` | Class | `(max_tokens, standardize, split, ngrams, output_mode, output_sequence_length, pad_to_max_tokens, vocabulary, idf_weights, sparse, ragged, encoding, name, kwargs)` |
| `src.layers.TimeDistributed` | Class | `(layer, kwargs)` |
| `src.layers.UnitNormalization` | Class | `(axis, kwargs)` |
| `src.layers.UpSampling1D` | Class | `(size, kwargs)` |
| `src.layers.UpSampling2D` | Class | `(size, data_format, interpolation, kwargs)` |
| `src.layers.UpSampling3D` | Class | `(size, data_format, kwargs)` |
| `src.layers.Wrapper` | Class | `(layer, kwargs)` |
| `src.layers.ZeroPadding1D` | Class | `(padding, data_format, kwargs)` |
| `src.layers.ZeroPadding2D` | Class | `(padding, data_format, kwargs)` |
| `src.layers.ZeroPadding3D` | Class | `(padding, data_format, kwargs)` |
| `src.layers.activations.ELU` | Class | `(alpha, kwargs)` |
| `src.layers.activations.LeakyReLU` | Class | `(negative_slope, kwargs)` |
| `src.layers.activations.PReLU` | Class | `(alpha_initializer, alpha_regularizer, alpha_constraint, shared_axes, kwargs)` |
| `src.layers.activations.ReLU` | Class | `(max_value, negative_slope, threshold, kwargs)` |
| `src.layers.activations.Softmax` | Class | `(axis, kwargs)` |
| `src.layers.activations.activation.Activation` | Class | `(activation, kwargs)` |
| `src.layers.activations.activation.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.layers.activations.activation.activations` | Object | `` |
| `src.layers.activations.activation.keras_export` | Class | `(path)` |
| `src.layers.activations.elu.ELU` | Class | `(alpha, kwargs)` |
| `src.layers.activations.elu.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.layers.activations.elu.activations` | Object | `` |
| `src.layers.activations.elu.keras_export` | Class | `(path)` |
| `src.layers.activations.leaky_relu.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.layers.activations.leaky_relu.LeakyReLU` | Class | `(negative_slope, kwargs)` |
| `src.layers.activations.leaky_relu.activations` | Object | `` |
| `src.layers.activations.leaky_relu.keras_export` | Class | `(path)` |
| `src.layers.activations.prelu.InputSpec` | Class | `(dtype, shape, ndim, max_ndim, min_ndim, axes, allow_last_axis_squeeze, name, optional)` |
| `src.layers.activations.prelu.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.layers.activations.prelu.PReLU` | Class | `(alpha_initializer, alpha_regularizer, alpha_constraint, shared_axes, kwargs)` |
| `src.layers.activations.prelu.activations` | Object | `` |
| `src.layers.activations.prelu.constraints` | Object | `` |
| `src.layers.activations.prelu.initializers` | Object | `` |
| `src.layers.activations.prelu.keras_export` | Class | `(path)` |
| `src.layers.activations.prelu.regularizers` | Object | `` |
| `src.layers.activations.relu.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.layers.activations.relu.ReLU` | Class | `(max_value, negative_slope, threshold, kwargs)` |
| `src.layers.activations.relu.activations` | Object | `` |
| `src.layers.activations.relu.keras_export` | Class | `(path)` |
| `src.layers.activations.softmax.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.layers.activations.softmax.Softmax` | Class | `(axis, kwargs)` |
| `src.layers.activations.softmax.activations` | Object | `` |
| `src.layers.activations.softmax.backend` | Object | `` |
| `src.layers.activations.softmax.keras_export` | Class | `(path)` |
| `src.layers.add` | Function | `(inputs, kwargs)` |
| `src.layers.attention.additive_attention.AdditiveAttention` | Class | `(use_scale, dropout, kwargs)` |
| `src.layers.attention.additive_attention.Attention` | Class | `(use_scale, score_mode, dropout, seed, kwargs)` |
| `src.layers.attention.additive_attention.keras_export` | Class | `(path)` |
| `src.layers.attention.additive_attention.ops` | Object | `` |
| `src.layers.attention.attention.Attention` | Class | `(use_scale, score_mode, dropout, seed, kwargs)` |
| `src.layers.attention.attention.KerasTensor` | Class | `(shape, dtype, sparse, ragged, record_history, name, kwargs)` |
| `src.layers.attention.attention.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.layers.attention.attention.backend` | Object | `` |
| `src.layers.attention.attention.keras_export` | Class | `(path)` |
| `src.layers.attention.attention.ops` | Object | `` |
| `src.layers.attention.grouped_query_attention.Dropout` | Class | `(rate, noise_shape, seed, kwargs)` |
| `src.layers.attention.grouped_query_attention.EinsumDense` | Class | `(equation, output_shape, activation, bias_axes, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, kernel_constraint, bias_constraint, lora_rank, lora_alpha, gptq_unpacked_column_size, quantization_config, kwargs)` |
| `src.layers.attention.grouped_query_attention.GroupedQueryAttention` | Class | `(head_dim, num_query_heads, num_key_value_heads, dropout, use_bias, flash_attention, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint, seed, kwargs)` |
| `src.layers.attention.grouped_query_attention.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.layers.attention.grouped_query_attention.Softmax` | Class | `(axis, kwargs)` |
| `src.layers.attention.grouped_query_attention.constraints` | Object | `` |
| `src.layers.attention.grouped_query_attention.initializers` | Object | `` |
| `src.layers.attention.grouped_query_attention.is_flash_attention_enabled` | Function | `()` |
| `src.layers.attention.grouped_query_attention.keras_export` | Class | `(path)` |
| `src.layers.attention.grouped_query_attention.ops` | Object | `` |
| `src.layers.attention.grouped_query_attention.regularizers` | Object | `` |
| `src.layers.attention.multi_head_attention.Dropout` | Class | `(rate, noise_shape, seed, kwargs)` |
| `src.layers.attention.multi_head_attention.EinsumDense` | Class | `(equation, output_shape, activation, bias_axes, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, kernel_constraint, bias_constraint, lora_rank, lora_alpha, gptq_unpacked_column_size, quantization_config, kwargs)` |
| `src.layers.attention.multi_head_attention.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.layers.attention.multi_head_attention.MultiHeadAttention` | Class | `(num_heads, key_dim, value_dim, dropout, use_bias, output_shape, attention_axes, flash_attention, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint, seed, kwargs)` |
| `src.layers.attention.multi_head_attention.Softmax` | Class | `(axis, kwargs)` |
| `src.layers.attention.multi_head_attention.backend` | Object | `` |
| `src.layers.attention.multi_head_attention.constraints` | Object | `` |
| `src.layers.attention.multi_head_attention.initializers` | Object | `` |
| `src.layers.attention.multi_head_attention.is_flash_attention_enabled` | Function | `()` |
| `src.layers.attention.multi_head_attention.keras_export` | Class | `(path)` |
| `src.layers.attention.multi_head_attention.ops` | Object | `` |
| `src.layers.attention.multi_head_attention.regularizers` | Object | `` |
| `src.layers.average` | Function | `(inputs, kwargs)` |
| `src.layers.concatenate` | Function | `(inputs, axis, kwargs)` |
| `src.layers.convolutional.base_conv.BaseConv` | Class | `(rank, filters, kernel_size, strides, padding, data_format, dilation_rate, groups, activation, use_bias, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint, lora_rank, lora_alpha, kwargs)` |
| `src.layers.convolutional.base_conv.InputSpec` | Class | `(dtype, shape, ndim, max_ndim, min_ndim, axes, allow_last_axis_squeeze, name, optional)` |
| `src.layers.convolutional.base_conv.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.layers.convolutional.base_conv.activations` | Object | `` |
| `src.layers.convolutional.base_conv.compute_conv_output_shape` | Function | `(input_shape, filters, kernel_size, strides, padding, data_format, dilation_rate)` |
| `src.layers.convolutional.base_conv.constraints` | Object | `` |
| `src.layers.convolutional.base_conv.initializers` | Object | `` |
| `src.layers.convolutional.base_conv.ops` | Object | `` |
| `src.layers.convolutional.base_conv.regularizers` | Object | `` |
| `src.layers.convolutional.base_conv.standardize_data_format` | Function | `(data_format)` |
| `src.layers.convolutional.base_conv.standardize_padding` | Function | `(value, allow_causal)` |
| `src.layers.convolutional.base_conv.standardize_tuple` | Function | `(value, n, name, allow_zero)` |
| `src.layers.convolutional.base_conv_transpose.BaseConvTranspose` | Class | `(rank, filters, kernel_size, strides, padding, output_padding, data_format, dilation_rate, activation, use_bias, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint, trainable, name, kwargs)` |
| `src.layers.convolutional.base_conv_transpose.InputSpec` | Class | `(dtype, shape, ndim, max_ndim, min_ndim, axes, allow_last_axis_squeeze, name, optional)` |
| `src.layers.convolutional.base_conv_transpose.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.layers.convolutional.base_conv_transpose.activations` | Object | `` |
| `src.layers.convolutional.base_conv_transpose.compute_conv_transpose_output_shape` | Function | `(input_shape, kernel_size, filters, strides, padding, output_padding, data_format, dilation_rate)` |
| `src.layers.convolutional.base_conv_transpose.constraints` | Object | `` |
| `src.layers.convolutional.base_conv_transpose.initializers` | Object | `` |
| `src.layers.convolutional.base_conv_transpose.ops` | Object | `` |
| `src.layers.convolutional.base_conv_transpose.regularizers` | Object | `` |
| `src.layers.convolutional.base_conv_transpose.standardize_data_format` | Function | `(data_format)` |
| `src.layers.convolutional.base_conv_transpose.standardize_padding` | Function | `(value, allow_causal)` |
| `src.layers.convolutional.base_conv_transpose.standardize_tuple` | Function | `(value, n, name, allow_zero)` |
| `src.layers.convolutional.base_depthwise_conv.BaseDepthwiseConv` | Class | `(rank, depth_multiplier, kernel_size, strides, padding, data_format, dilation_rate, activation, use_bias, depthwise_initializer, bias_initializer, depthwise_regularizer, bias_regularizer, activity_regularizer, depthwise_constraint, bias_constraint, trainable, name, kwargs)` |
| `src.layers.convolutional.base_depthwise_conv.InputSpec` | Class | `(dtype, shape, ndim, max_ndim, min_ndim, axes, allow_last_axis_squeeze, name, optional)` |
| `src.layers.convolutional.base_depthwise_conv.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.layers.convolutional.base_depthwise_conv.activations` | Object | `` |
| `src.layers.convolutional.base_depthwise_conv.compute_conv_output_shape` | Function | `(input_shape, filters, kernel_size, strides, padding, data_format, dilation_rate)` |
| `src.layers.convolutional.base_depthwise_conv.constraints` | Object | `` |
| `src.layers.convolutional.base_depthwise_conv.initializers` | Object | `` |
| `src.layers.convolutional.base_depthwise_conv.ops` | Object | `` |
| `src.layers.convolutional.base_depthwise_conv.regularizers` | Object | `` |
| `src.layers.convolutional.base_depthwise_conv.standardize_data_format` | Function | `(data_format)` |
| `src.layers.convolutional.base_depthwise_conv.standardize_padding` | Function | `(value, allow_causal)` |
| `src.layers.convolutional.base_depthwise_conv.standardize_tuple` | Function | `(value, n, name, allow_zero)` |
| `src.layers.convolutional.base_separable_conv.BaseSeparableConv` | Class | `(rank, depth_multiplier, filters, kernel_size, strides, padding, data_format, dilation_rate, activation, use_bias, depthwise_initializer, pointwise_initializer, bias_initializer, depthwise_regularizer, pointwise_regularizer, bias_regularizer, activity_regularizer, depthwise_constraint, pointwise_constraint, bias_constraint, trainable, name, kwargs)` |
| `src.layers.convolutional.base_separable_conv.InputSpec` | Class | `(dtype, shape, ndim, max_ndim, min_ndim, axes, allow_last_axis_squeeze, name, optional)` |
| `src.layers.convolutional.base_separable_conv.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.layers.convolutional.base_separable_conv.activations` | Object | `` |
| `src.layers.convolutional.base_separable_conv.compute_conv_output_shape` | Function | `(input_shape, filters, kernel_size, strides, padding, data_format, dilation_rate)` |
| `src.layers.convolutional.base_separable_conv.constraints` | Object | `` |
| `src.layers.convolutional.base_separable_conv.initializers` | Object | `` |
| `src.layers.convolutional.base_separable_conv.ops` | Object | `` |
| `src.layers.convolutional.base_separable_conv.regularizers` | Object | `` |
| `src.layers.convolutional.base_separable_conv.standardize_data_format` | Function | `(data_format)` |
| `src.layers.convolutional.base_separable_conv.standardize_padding` | Function | `(value, allow_causal)` |
| `src.layers.convolutional.base_separable_conv.standardize_tuple` | Function | `(value, n, name, allow_zero)` |
| `src.layers.convolutional.conv1d.BaseConv` | Class | `(rank, filters, kernel_size, strides, padding, data_format, dilation_rate, groups, activation, use_bias, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint, lora_rank, lora_alpha, kwargs)` |
| `src.layers.convolutional.conv1d.Conv1D` | Class | `(filters, kernel_size, strides, padding, data_format, dilation_rate, groups, activation, use_bias, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint, kwargs)` |
| `src.layers.convolutional.conv1d.keras_export` | Class | `(path)` |
| `src.layers.convolutional.conv1d.ops` | Object | `` |
| `src.layers.convolutional.conv1d_transpose.BaseConvTranspose` | Class | `(rank, filters, kernel_size, strides, padding, output_padding, data_format, dilation_rate, activation, use_bias, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint, trainable, name, kwargs)` |
| `src.layers.convolutional.conv1d_transpose.Conv1DTranspose` | Class | `(filters, kernel_size, strides, padding, output_padding, data_format, dilation_rate, activation, use_bias, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint, kwargs)` |
| `src.layers.convolutional.conv1d_transpose.keras_export` | Class | `(path)` |
| `src.layers.convolutional.conv2d.BaseConv` | Class | `(rank, filters, kernel_size, strides, padding, data_format, dilation_rate, groups, activation, use_bias, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint, lora_rank, lora_alpha, kwargs)` |
| `src.layers.convolutional.conv2d.Conv2D` | Class | `(filters, kernel_size, strides, padding, data_format, dilation_rate, groups, activation, use_bias, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint, kwargs)` |
| `src.layers.convolutional.conv2d.keras_export` | Class | `(path)` |
| `src.layers.convolutional.conv2d_transpose.BaseConvTranspose` | Class | `(rank, filters, kernel_size, strides, padding, output_padding, data_format, dilation_rate, activation, use_bias, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint, trainable, name, kwargs)` |
| `src.layers.convolutional.conv2d_transpose.Conv2DTranspose` | Class | `(filters, kernel_size, strides, padding, output_padding, data_format, dilation_rate, activation, use_bias, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint, kwargs)` |
| `src.layers.convolutional.conv2d_transpose.keras_export` | Class | `(path)` |
| `src.layers.convolutional.conv3d.BaseConv` | Class | `(rank, filters, kernel_size, strides, padding, data_format, dilation_rate, groups, activation, use_bias, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint, lora_rank, lora_alpha, kwargs)` |
| `src.layers.convolutional.conv3d.Conv3D` | Class | `(filters, kernel_size, strides, padding, data_format, dilation_rate, groups, activation, use_bias, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint, kwargs)` |
| `src.layers.convolutional.conv3d.keras_export` | Class | `(path)` |
| `src.layers.convolutional.conv3d_transpose.BaseConvTranspose` | Class | `(rank, filters, kernel_size, strides, padding, output_padding, data_format, dilation_rate, activation, use_bias, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint, trainable, name, kwargs)` |
| `src.layers.convolutional.conv3d_transpose.Conv3DTranspose` | Class | `(filters, kernel_size, strides, padding, data_format, output_padding, dilation_rate, activation, use_bias, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint, kwargs)` |
| `src.layers.convolutional.conv3d_transpose.keras_export` | Class | `(path)` |
| `src.layers.convolutional.depthwise_conv1d.BaseDepthwiseConv` | Class | `(rank, depth_multiplier, kernel_size, strides, padding, data_format, dilation_rate, activation, use_bias, depthwise_initializer, bias_initializer, depthwise_regularizer, bias_regularizer, activity_regularizer, depthwise_constraint, bias_constraint, trainable, name, kwargs)` |
| `src.layers.convolutional.depthwise_conv1d.DepthwiseConv1D` | Class | `(kernel_size, strides, padding, depth_multiplier, data_format, dilation_rate, activation, use_bias, depthwise_initializer, bias_initializer, depthwise_regularizer, bias_regularizer, activity_regularizer, depthwise_constraint, bias_constraint, kwargs)` |
| `src.layers.convolutional.depthwise_conv1d.keras_export` | Class | `(path)` |
| `src.layers.convolutional.depthwise_conv2d.BaseDepthwiseConv` | Class | `(rank, depth_multiplier, kernel_size, strides, padding, data_format, dilation_rate, activation, use_bias, depthwise_initializer, bias_initializer, depthwise_regularizer, bias_regularizer, activity_regularizer, depthwise_constraint, bias_constraint, trainable, name, kwargs)` |
| `src.layers.convolutional.depthwise_conv2d.DepthwiseConv2D` | Class | `(kernel_size, strides, padding, depth_multiplier, data_format, dilation_rate, activation, use_bias, depthwise_initializer, bias_initializer, depthwise_regularizer, bias_regularizer, activity_regularizer, depthwise_constraint, bias_constraint, kwargs)` |
| `src.layers.convolutional.depthwise_conv2d.keras_export` | Class | `(path)` |
| `src.layers.convolutional.separable_conv1d.BaseSeparableConv` | Class | `(rank, depth_multiplier, filters, kernel_size, strides, padding, data_format, dilation_rate, activation, use_bias, depthwise_initializer, pointwise_initializer, bias_initializer, depthwise_regularizer, pointwise_regularizer, bias_regularizer, activity_regularizer, depthwise_constraint, pointwise_constraint, bias_constraint, trainable, name, kwargs)` |
| `src.layers.convolutional.separable_conv1d.SeparableConv1D` | Class | `(filters, kernel_size, strides, padding, data_format, dilation_rate, depth_multiplier, activation, use_bias, depthwise_initializer, pointwise_initializer, bias_initializer, depthwise_regularizer, pointwise_regularizer, bias_regularizer, activity_regularizer, depthwise_constraint, pointwise_constraint, bias_constraint, kwargs)` |
| `src.layers.convolutional.separable_conv1d.keras_export` | Class | `(path)` |
| `src.layers.convolutional.separable_conv2d.BaseSeparableConv` | Class | `(rank, depth_multiplier, filters, kernel_size, strides, padding, data_format, dilation_rate, activation, use_bias, depthwise_initializer, pointwise_initializer, bias_initializer, depthwise_regularizer, pointwise_regularizer, bias_regularizer, activity_regularizer, depthwise_constraint, pointwise_constraint, bias_constraint, trainable, name, kwargs)` |
| `src.layers.convolutional.separable_conv2d.SeparableConv2D` | Class | `(filters, kernel_size, strides, padding, data_format, dilation_rate, depth_multiplier, activation, use_bias, depthwise_initializer, pointwise_initializer, bias_initializer, depthwise_regularizer, pointwise_regularizer, bias_regularizer, activity_regularizer, depthwise_constraint, pointwise_constraint, bias_constraint, kwargs)` |
| `src.layers.convolutional.separable_conv2d.keras_export` | Class | `(path)` |
| `src.layers.core.dense.Dense` | Class | `(units, activation, use_bias, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint, lora_rank, lora_alpha, quantization_config, kwargs)` |
| `src.layers.core.dense.InputSpec` | Class | `(dtype, shape, ndim, max_ndim, min_ndim, axes, allow_last_axis_squeeze, name, optional)` |
| `src.layers.core.dense.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.layers.core.dense.QuantizationConfig` | Class | `(weight_quantizer, activation_quantizer)` |
| `src.layers.core.dense.activations` | Object | `` |
| `src.layers.core.dense.constraints` | Object | `` |
| `src.layers.core.dense.dequantize_with_sz_map` | Function | `(weights_matrix, scale, zero, g_idx)` |
| `src.layers.core.dense.initializers` | Object | `` |
| `src.layers.core.dense.keras_export` | Class | `(path)` |
| `src.layers.core.dense.ops` | Object | `` |
| `src.layers.core.dense.quantizers` | Object | `` |
| `src.layers.core.dense.regularizers` | Object | `` |
| `src.layers.core.dense.serialization_lib` | Object | `` |
| `src.layers.core.einsum_dense.EinsumDense` | Class | `(equation, output_shape, activation, bias_axes, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, kernel_constraint, bias_constraint, lora_rank, lora_alpha, gptq_unpacked_column_size, quantization_config, kwargs)` |
| `src.layers.core.einsum_dense.InputSpec` | Class | `(dtype, shape, ndim, max_ndim, min_ndim, axes, allow_last_axis_squeeze, name, optional)` |
| `src.layers.core.einsum_dense.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.layers.core.einsum_dense.QuantizationConfig` | Class | `(weight_quantizer, activation_quantizer)` |
| `src.layers.core.einsum_dense.activations` | Object | `` |
| `src.layers.core.einsum_dense.backend` | Object | `` |
| `src.layers.core.einsum_dense.constraints` | Object | `` |
| `src.layers.core.einsum_dense.dequantize_with_sz_map` | Function | `(weights_matrix, scale, zero, g_idx)` |
| `src.layers.core.einsum_dense.dtype_policies` | Object | `` |
| `src.layers.core.einsum_dense.initializers` | Object | `` |
| `src.layers.core.einsum_dense.keras_export` | Class | `(path)` |
| `src.layers.core.einsum_dense.ops` | Object | `` |
| `src.layers.core.einsum_dense.quantizers` | Object | `` |
| `src.layers.core.einsum_dense.regularizers` | Object | `` |
| `src.layers.core.einsum_dense.serialization_lib` | Object | `` |
| `src.layers.core.embedding.Embedding` | Class | `(input_dim, output_dim, embeddings_initializer, embeddings_regularizer, embeddings_constraint, mask_zero, weights, lora_rank, lora_alpha, quantization_config, kwargs)` |
| `src.layers.core.embedding.KerasTensor` | Class | `(shape, dtype, sparse, ragged, record_history, name, kwargs)` |
| `src.layers.core.embedding.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.layers.core.embedding.QuantizationConfig` | Class | `(weight_quantizer, activation_quantizer)` |
| `src.layers.core.embedding.backend` | Object | `` |
| `src.layers.core.embedding.constraints` | Object | `` |
| `src.layers.core.embedding.dtype_policies` | Object | `` |
| `src.layers.core.embedding.initializers` | Object | `` |
| `src.layers.core.embedding.keras_export` | Class | `(path)` |
| `src.layers.core.embedding.ops` | Object | `` |
| `src.layers.core.embedding.quantizers` | Object | `` |
| `src.layers.core.embedding.regularizers` | Object | `` |
| `src.layers.core.embedding.serialization_lib` | Object | `` |
| `src.layers.core.identity.Identity` | Class | `(kwargs)` |
| `src.layers.core.identity.KerasTensor` | Class | `(shape, dtype, sparse, ragged, record_history, name, kwargs)` |
| `src.layers.core.identity.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.layers.core.identity.keras_export` | Class | `(path)` |
| `src.layers.core.identity.tree` | Object | `` |
| `src.layers.core.input_layer.Input` | Function | `(shape, batch_size, dtype, sparse, ragged, batch_shape, name, tensor, optional)` |
| `src.layers.core.input_layer.InputLayer` | Class | `(shape, batch_size, dtype, sparse, ragged, batch_shape, input_tensor, optional, name, kwargs)` |
| `src.layers.core.input_layer.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.layers.core.input_layer.Node` | Class | `(operation, call_args, call_kwargs, outputs)` |
| `src.layers.core.input_layer.backend` | Object | `` |
| `src.layers.core.input_layer.keras_export` | Class | `(path)` |
| `src.layers.core.lambda_layer.Lambda` | Class | `(function, output_shape, mask, arguments, kwargs)` |
| `src.layers.core.lambda_layer.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.layers.core.lambda_layer.backend` | Object | `` |
| `src.layers.core.lambda_layer.keras_export` | Class | `(path)` |
| `src.layers.core.lambda_layer.python_utils` | Object | `` |
| `src.layers.core.lambda_layer.serialization_lib` | Object | `` |
| `src.layers.core.lambda_layer.tree` | Object | `` |
| `src.layers.core.masking.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.layers.core.masking.Masking` | Class | `(mask_value, kwargs)` |
| `src.layers.core.masking.backend` | Object | `` |
| `src.layers.core.masking.deserialize_keras_object` | Function | `(config, custom_objects, safe_mode, kwargs)` |
| `src.layers.core.masking.keras_export` | Class | `(path)` |
| `src.layers.core.masking.ops` | Object | `` |
| `src.layers.core.reversible_embedding.KerasTensor` | Class | `(shape, dtype, sparse, ragged, record_history, name, kwargs)` |
| `src.layers.core.reversible_embedding.QuantizationConfig` | Class | `(weight_quantizer, activation_quantizer)` |
| `src.layers.core.reversible_embedding.ReversibleEmbedding` | Class | `(input_dim, output_dim, tie_weights, embeddings_initializer, embeddings_regularizer, embeddings_constraint, mask_zero, reverse_dtype, logit_soft_cap, kwargs)` |
| `src.layers.core.reversible_embedding.dtype_policies` | Object | `` |
| `src.layers.core.reversible_embedding.keras_export` | Class | `(path)` |
| `src.layers.core.reversible_embedding.layers` | Object | `` |
| `src.layers.core.reversible_embedding.ops` | Object | `` |
| `src.layers.core.reversible_embedding.quantizers` | Object | `` |
| `src.layers.core.wrapper.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.layers.core.wrapper.Wrapper` | Class | `(layer, kwargs)` |
| `src.layers.core.wrapper.keras_export` | Class | `(path)` |
| `src.layers.core.wrapper.serialization_lib` | Object | `` |
| `src.layers.deserialize` | Function | `(config, custom_objects)` |
| `src.layers.dot` | Function | `(inputs, axes, kwargs)` |
| `src.layers.input_spec.InputSpec` | Class | `(dtype, shape, ndim, max_ndim, min_ndim, axes, allow_last_axis_squeeze, name, optional)` |
| `src.layers.input_spec.assert_input_compatibility` | Function | `(input_spec, inputs, layer_name)` |
| `src.layers.input_spec.backend` | Object | `` |
| `src.layers.input_spec.keras_export` | Class | `(path)` |
| `src.layers.input_spec.tree` | Object | `` |
| `src.layers.keras_export` | Class | `(path)` |
| `src.layers.layer.BackendLayer` | Class | `(...)` |
| `src.layers.layer.CallContext` | Class | `(entry_layer)` |
| `src.layers.layer.CallSpec` | Class | `(signature, call_context_args, args, kwargs)` |
| `src.layers.layer.DTypePolicyMap` | Class | `(default_policy, policy_map)` |
| `src.layers.layer.KerasTensor` | Class | `(shape, dtype, sparse, ragged, record_history, name, kwargs)` |
| `src.layers.layer.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.layers.layer.Metric` | Class | `(dtype, name)` |
| `src.layers.layer.Node` | Class | `(operation, call_args, call_kwargs, outputs)` |
| `src.layers.layer.Operation` | Class | `(name)` |
| `src.layers.layer.any_symbolic_tensors` | Function | `(args, kwargs)` |
| `src.layers.layer.backend` | Object | `` |
| `src.layers.layer.constraints` | Object | `` |
| `src.layers.layer.current_path` | Function | `()` |
| `src.layers.layer.distribution_lib` | Object | `` |
| `src.layers.layer.dtype_policies` | Object | `` |
| `src.layers.layer.get_arguments_dict` | Function | `(fn, args, kwargs)` |
| `src.layers.layer.get_current_remat_mode` | Function | `()` |
| `src.layers.layer.get_shapes_dict` | Function | `(call_spec)` |
| `src.layers.layer.global_state` | Object | `` |
| `src.layers.layer.in_symbolic_scope` | Function | `()` |
| `src.layers.layer.initializers` | Object | `` |
| `src.layers.layer.input_spec` | Object | `` |
| `src.layers.layer.is_backend_tensor_or_symbolic` | Function | `(x, allow_none)` |
| `src.layers.layer.is_nnx_enabled` | Function | `()` |
| `src.layers.layer.is_shape_tuple` | Function | `(s)` |
| `src.layers.layer.keras_export` | Class | `(path)` |
| `src.layers.layer.might_have_unbuilt_state` | Function | `(layer)` |
| `src.layers.layer.python_utils` | Object | `` |
| `src.layers.layer.regularizers` | Object | `` |
| `src.layers.layer.remat` | Object | `` |
| `src.layers.layer.summary_utils` | Object | `` |
| `src.layers.layer.traceback_utils` | Object | `` |
| `src.layers.layer.tracking` | Object | `` |
| `src.layers.layer.tree` | Object | `` |
| `src.layers.layer.update_shapes_dict_for_target_fn` | Function | `(target_fn, shapes_dict, call_spec, class_name)` |
| `src.layers.layer.utils` | Object | `` |
| `src.layers.layer.validate_and_resolve_config` | Function | `(mode, config)` |
| `src.layers.maximum` | Function | `(inputs, kwargs)` |
| `src.layers.merging.add.Add` | Class | `(...)` |
| `src.layers.merging.add.Merge` | Class | `(kwargs)` |
| `src.layers.merging.add.add` | Function | `(inputs, kwargs)` |
| `src.layers.merging.add.keras_export` | Class | `(path)` |
| `src.layers.merging.add.ops` | Object | `` |
| `src.layers.merging.average.Average` | Class | `(...)` |
| `src.layers.merging.average.Merge` | Class | `(kwargs)` |
| `src.layers.merging.average.average` | Function | `(inputs, kwargs)` |
| `src.layers.merging.average.keras_export` | Class | `(path)` |
| `src.layers.merging.average.ops` | Object | `` |
| `src.layers.merging.base_merge.KerasTensor` | Class | `(shape, dtype, sparse, ragged, record_history, name, kwargs)` |
| `src.layers.merging.base_merge.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.layers.merging.base_merge.Merge` | Class | `(kwargs)` |
| `src.layers.merging.base_merge.backend` | Object | `` |
| `src.layers.merging.base_merge.ops` | Object | `` |
| `src.layers.merging.concatenate.Concatenate` | Class | `(axis, kwargs)` |
| `src.layers.merging.concatenate.Merge` | Class | `(kwargs)` |
| `src.layers.merging.concatenate.concatenate` | Function | `(inputs, axis, kwargs)` |
| `src.layers.merging.concatenate.keras_export` | Class | `(path)` |
| `src.layers.merging.concatenate.ops` | Object | `` |
| `src.layers.merging.dot.Dot` | Class | `(axes, normalize, kwargs)` |
| `src.layers.merging.dot.Merge` | Class | `(kwargs)` |
| `src.layers.merging.dot.batch_dot` | Function | `(x, y, axes)` |
| `src.layers.merging.dot.dot` | Function | `(inputs, axes, kwargs)` |
| `src.layers.merging.dot.keras_export` | Class | `(path)` |
| `src.layers.merging.dot.normalize` | Function | `(x, axis, order)` |
| `src.layers.merging.dot.ops` | Object | `` |
| `src.layers.merging.maximum.Maximum` | Class | `(...)` |
| `src.layers.merging.maximum.Merge` | Class | `(kwargs)` |
| `src.layers.merging.maximum.keras_export` | Class | `(path)` |
| `src.layers.merging.maximum.maximum` | Function | `(inputs, kwargs)` |
| `src.layers.merging.maximum.ops` | Object | `` |
| `src.layers.merging.minimum.Merge` | Class | `(kwargs)` |
| `src.layers.merging.minimum.Minimum` | Class | `(...)` |
| `src.layers.merging.minimum.keras_export` | Class | `(path)` |
| `src.layers.merging.minimum.minimum` | Function | `(inputs, kwargs)` |
| `src.layers.merging.minimum.ops` | Object | `` |
| `src.layers.merging.multiply.Merge` | Class | `(kwargs)` |
| `src.layers.merging.multiply.Multiply` | Class | `(...)` |
| `src.layers.merging.multiply.backend` | Object | `` |
| `src.layers.merging.multiply.keras_export` | Class | `(path)` |
| `src.layers.merging.multiply.multiply` | Function | `(inputs, kwargs)` |
| `src.layers.merging.multiply.ops` | Object | `` |
| `src.layers.merging.subtract.Merge` | Class | `(kwargs)` |
| `src.layers.merging.subtract.Subtract` | Class | `(...)` |
| `src.layers.merging.subtract.keras_export` | Class | `(path)` |
| `src.layers.merging.subtract.ops` | Object | `` |
| `src.layers.merging.subtract.subtract` | Function | `(inputs, kwargs)` |
| `src.layers.minimum` | Function | `(inputs, kwargs)` |
| `src.layers.multiply` | Function | `(inputs, kwargs)` |
| `src.layers.normalization.batch_normalization.BatchNormalization` | Class | `(axis, momentum, epsilon, center, scale, beta_initializer, gamma_initializer, moving_mean_initializer, moving_variance_initializer, beta_regularizer, gamma_regularizer, beta_constraint, gamma_constraint, synchronized, kwargs)` |
| `src.layers.normalization.batch_normalization.InputSpec` | Class | `(dtype, shape, ndim, max_ndim, min_ndim, axes, allow_last_axis_squeeze, name, optional)` |
| `src.layers.normalization.batch_normalization.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.layers.normalization.batch_normalization.backend` | Object | `` |
| `src.layers.normalization.batch_normalization.constraints` | Object | `` |
| `src.layers.normalization.batch_normalization.initializers` | Object | `` |
| `src.layers.normalization.batch_normalization.keras_export` | Class | `(path)` |
| `src.layers.normalization.batch_normalization.ops` | Object | `` |
| `src.layers.normalization.batch_normalization.regularizers` | Object | `` |
| `src.layers.normalization.group_normalization.GroupNormalization` | Class | `(groups, axis, epsilon, center, scale, beta_initializer, gamma_initializer, beta_regularizer, gamma_regularizer, beta_constraint, gamma_constraint, kwargs)` |
| `src.layers.normalization.group_normalization.InputSpec` | Class | `(dtype, shape, ndim, max_ndim, min_ndim, axes, allow_last_axis_squeeze, name, optional)` |
| `src.layers.normalization.group_normalization.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.layers.normalization.group_normalization.backend` | Object | `` |
| `src.layers.normalization.group_normalization.constraints` | Object | `` |
| `src.layers.normalization.group_normalization.initializers` | Object | `` |
| `src.layers.normalization.group_normalization.keras_export` | Class | `(path)` |
| `src.layers.normalization.group_normalization.ops` | Object | `` |
| `src.layers.normalization.group_normalization.regularizers` | Object | `` |
| `src.layers.normalization.layer_normalization.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.layers.normalization.layer_normalization.LayerNormalization` | Class | `(axis, epsilon, center, scale, beta_initializer, gamma_initializer, beta_regularizer, gamma_regularizer, beta_constraint, gamma_constraint, kwargs)` |
| `src.layers.normalization.layer_normalization.constraints` | Object | `` |
| `src.layers.normalization.layer_normalization.initializers` | Object | `` |
| `src.layers.normalization.layer_normalization.keras_export` | Class | `(path)` |
| `src.layers.normalization.layer_normalization.ops` | Object | `` |
| `src.layers.normalization.layer_normalization.regularizers` | Object | `` |
| `src.layers.normalization.rms_normalization.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.layers.normalization.rms_normalization.RMSNormalization` | Class | `(axis, epsilon, kwargs)` |
| `src.layers.normalization.rms_normalization.keras_export` | Class | `(path)` |
| `src.layers.normalization.rms_normalization.ops` | Object | `` |
| `src.layers.normalization.spectral_normalization.InputSpec` | Class | `(dtype, shape, ndim, max_ndim, min_ndim, axes, allow_last_axis_squeeze, name, optional)` |
| `src.layers.normalization.spectral_normalization.SpectralNormalization` | Class | `(layer, power_iterations, kwargs)` |
| `src.layers.normalization.spectral_normalization.Wrapper` | Class | `(layer, kwargs)` |
| `src.layers.normalization.spectral_normalization.initializers` | Object | `` |
| `src.layers.normalization.spectral_normalization.keras_export` | Class | `(path)` |
| `src.layers.normalization.spectral_normalization.normalize` | Function | `(x, axis, order)` |
| `src.layers.normalization.spectral_normalization.ops` | Object | `` |
| `src.layers.normalization.unit_normalization.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.layers.normalization.unit_normalization.UnitNormalization` | Class | `(axis, kwargs)` |
| `src.layers.normalization.unit_normalization.keras_export` | Class | `(path)` |
| `src.layers.normalization.unit_normalization.ops` | Object | `` |
| `src.layers.pooling.adaptive_average_pooling1d.AdaptiveAveragePooling1D` | Class | `(output_size, data_format, kwargs)` |
| `src.layers.pooling.adaptive_average_pooling1d.BaseAdaptiveAveragePooling` | Class | `(...)` |
| `src.layers.pooling.adaptive_average_pooling1d.keras_export` | Class | `(path)` |
| `src.layers.pooling.adaptive_average_pooling2d.AdaptiveAveragePooling2D` | Class | `(output_size, data_format, kwargs)` |
| `src.layers.pooling.adaptive_average_pooling2d.BaseAdaptiveAveragePooling` | Class | `(...)` |
| `src.layers.pooling.adaptive_average_pooling2d.keras_export` | Class | `(path)` |
| `src.layers.pooling.adaptive_average_pooling3d.AdaptiveAveragePooling3D` | Class | `(output_size, data_format, kwargs)` |
| `src.layers.pooling.adaptive_average_pooling3d.BaseAdaptiveAveragePooling` | Class | `(...)` |
| `src.layers.pooling.adaptive_average_pooling3d.keras_export` | Class | `(path)` |
| `src.layers.pooling.adaptive_max_pooling1d.AdaptiveMaxPooling1D` | Class | `(output_size, data_format, kwargs)` |
| `src.layers.pooling.adaptive_max_pooling1d.BaseAdaptiveMaxPooling` | Class | `(...)` |
| `src.layers.pooling.adaptive_max_pooling1d.keras_export` | Class | `(path)` |
| `src.layers.pooling.adaptive_max_pooling2d.AdaptiveMaxPooling2D` | Class | `(output_size, data_format, kwargs)` |
| `src.layers.pooling.adaptive_max_pooling2d.BaseAdaptiveMaxPooling` | Class | `(...)` |
| `src.layers.pooling.adaptive_max_pooling2d.keras_export` | Class | `(path)` |
| `src.layers.pooling.adaptive_max_pooling3d.AdaptiveMaxPooling3D` | Class | `(output_size, data_format, kwargs)` |
| `src.layers.pooling.adaptive_max_pooling3d.BaseAdaptiveMaxPooling` | Class | `(...)` |
| `src.layers.pooling.adaptive_max_pooling3d.keras_export` | Class | `(path)` |
| `src.layers.pooling.average_pooling1d.AveragePooling1D` | Class | `(pool_size, strides, padding, data_format, name, kwargs)` |
| `src.layers.pooling.average_pooling1d.BasePooling` | Class | `(pool_size, strides, pool_dimensions, pool_mode, padding, data_format, name, kwargs)` |
| `src.layers.pooling.average_pooling1d.keras_export` | Class | `(path)` |
| `src.layers.pooling.average_pooling2d.AveragePooling2D` | Class | `(pool_size, strides, padding, data_format, name, kwargs)` |
| `src.layers.pooling.average_pooling2d.BasePooling` | Class | `(pool_size, strides, pool_dimensions, pool_mode, padding, data_format, name, kwargs)` |
| `src.layers.pooling.average_pooling2d.keras_export` | Class | `(path)` |
| `src.layers.pooling.average_pooling3d.AveragePooling3D` | Class | `(pool_size, strides, padding, data_format, name, kwargs)` |
| `src.layers.pooling.average_pooling3d.BasePooling` | Class | `(pool_size, strides, pool_dimensions, pool_mode, padding, data_format, name, kwargs)` |
| `src.layers.pooling.average_pooling3d.keras_export` | Class | `(path)` |
| `src.layers.pooling.base_adaptive_pooling.BaseAdaptiveAveragePooling` | Class | `(...)` |
| `src.layers.pooling.base_adaptive_pooling.BaseAdaptiveMaxPooling` | Class | `(...)` |
| `src.layers.pooling.base_adaptive_pooling.BaseAdaptivePooling` | Class | `(output_size, data_format, kwargs)` |
| `src.layers.pooling.base_adaptive_pooling.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.layers.pooling.base_adaptive_pooling.config` | Object | `` |
| `src.layers.pooling.base_adaptive_pooling.ops` | Object | `` |
| `src.layers.pooling.base_global_pooling.BaseGlobalPooling` | Class | `(pool_dimensions, data_format, keepdims, kwargs)` |
| `src.layers.pooling.base_global_pooling.InputSpec` | Class | `(dtype, shape, ndim, max_ndim, min_ndim, axes, allow_last_axis_squeeze, name, optional)` |
| `src.layers.pooling.base_global_pooling.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.layers.pooling.base_global_pooling.backend` | Object | `` |
| `src.layers.pooling.base_pooling.BasePooling` | Class | `(pool_size, strides, pool_dimensions, pool_mode, padding, data_format, name, kwargs)` |
| `src.layers.pooling.base_pooling.InputSpec` | Class | `(dtype, shape, ndim, max_ndim, min_ndim, axes, allow_last_axis_squeeze, name, optional)` |
| `src.layers.pooling.base_pooling.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.layers.pooling.base_pooling.argument_validation` | Object | `` |
| `src.layers.pooling.base_pooling.backend` | Object | `` |
| `src.layers.pooling.base_pooling.compute_pooling_output_shape` | Function | `(input_shape, pool_size, strides, padding, data_format)` |
| `src.layers.pooling.base_pooling.ops` | Object | `` |
| `src.layers.pooling.global_average_pooling1d.BaseGlobalPooling` | Class | `(pool_dimensions, data_format, keepdims, kwargs)` |
| `src.layers.pooling.global_average_pooling1d.GlobalAveragePooling1D` | Class | `(data_format, keepdims, kwargs)` |
| `src.layers.pooling.global_average_pooling1d.backend` | Object | `` |
| `src.layers.pooling.global_average_pooling1d.keras_export` | Class | `(path)` |
| `src.layers.pooling.global_average_pooling1d.ops` | Object | `` |
| `src.layers.pooling.global_average_pooling2d.BaseGlobalPooling` | Class | `(pool_dimensions, data_format, keepdims, kwargs)` |
| `src.layers.pooling.global_average_pooling2d.GlobalAveragePooling2D` | Class | `(data_format, keepdims, kwargs)` |
| `src.layers.pooling.global_average_pooling2d.keras_export` | Class | `(path)` |
| `src.layers.pooling.global_average_pooling2d.ops` | Object | `` |
| `src.layers.pooling.global_average_pooling3d.BaseGlobalPooling` | Class | `(pool_dimensions, data_format, keepdims, kwargs)` |
| `src.layers.pooling.global_average_pooling3d.GlobalAveragePooling3D` | Class | `(data_format, keepdims, kwargs)` |
| `src.layers.pooling.global_average_pooling3d.keras_export` | Class | `(path)` |
| `src.layers.pooling.global_average_pooling3d.ops` | Object | `` |
| `src.layers.pooling.global_max_pooling1d.BaseGlobalPooling` | Class | `(pool_dimensions, data_format, keepdims, kwargs)` |
| `src.layers.pooling.global_max_pooling1d.GlobalMaxPooling1D` | Class | `(data_format, keepdims, kwargs)` |
| `src.layers.pooling.global_max_pooling1d.keras_export` | Class | `(path)` |
| `src.layers.pooling.global_max_pooling1d.ops` | Object | `` |
| `src.layers.pooling.global_max_pooling2d.BaseGlobalPooling` | Class | `(pool_dimensions, data_format, keepdims, kwargs)` |
| `src.layers.pooling.global_max_pooling2d.GlobalMaxPooling2D` | Class | `(data_format, keepdims, kwargs)` |
| `src.layers.pooling.global_max_pooling2d.keras_export` | Class | `(path)` |
| `src.layers.pooling.global_max_pooling2d.ops` | Object | `` |
| `src.layers.pooling.global_max_pooling3d.BaseGlobalPooling` | Class | `(pool_dimensions, data_format, keepdims, kwargs)` |
| `src.layers.pooling.global_max_pooling3d.GlobalMaxPooling3D` | Class | `(data_format, keepdims, kwargs)` |
| `src.layers.pooling.global_max_pooling3d.keras_export` | Class | `(path)` |
| `src.layers.pooling.global_max_pooling3d.ops` | Object | `` |
| `src.layers.pooling.max_pooling1d.BasePooling` | Class | `(pool_size, strides, pool_dimensions, pool_mode, padding, data_format, name, kwargs)` |
| `src.layers.pooling.max_pooling1d.MaxPooling1D` | Class | `(pool_size, strides, padding, data_format, name, kwargs)` |
| `src.layers.pooling.max_pooling1d.keras_export` | Class | `(path)` |
| `src.layers.pooling.max_pooling2d.BasePooling` | Class | `(pool_size, strides, pool_dimensions, pool_mode, padding, data_format, name, kwargs)` |
| `src.layers.pooling.max_pooling2d.MaxPooling2D` | Class | `(pool_size, strides, padding, data_format, name, kwargs)` |
| `src.layers.pooling.max_pooling2d.keras_export` | Class | `(path)` |
| `src.layers.pooling.max_pooling3d.BasePooling` | Class | `(pool_size, strides, pool_dimensions, pool_mode, padding, data_format, name, kwargs)` |
| `src.layers.pooling.max_pooling3d.MaxPooling3D` | Class | `(pool_size, strides, padding, data_format, name, kwargs)` |
| `src.layers.pooling.max_pooling3d.keras_export` | Class | `(path)` |
| `src.layers.preprocessing.category_encoding.CategoryEncoding` | Class | `(num_tokens, output_mode, sparse, kwargs)` |
| `src.layers.preprocessing.category_encoding.DataLayer` | Class | `(kwargs)` |
| `src.layers.preprocessing.category_encoding.KerasTensor` | Class | `(shape, dtype, sparse, ragged, record_history, name, kwargs)` |
| `src.layers.preprocessing.category_encoding.backend_utils` | Object | `` |
| `src.layers.preprocessing.category_encoding.keras_export` | Class | `(path)` |
| `src.layers.preprocessing.category_encoding.numerical_utils` | Object | `` |
| `src.layers.preprocessing.data_layer.DataLayer` | Class | `(kwargs)` |
| `src.layers.preprocessing.data_layer.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.layers.preprocessing.data_layer.SeedGenerator` | Class | `(seed, name, kwargs)` |
| `src.layers.preprocessing.data_layer.backend_utils` | Object | `` |
| `src.layers.preprocessing.data_layer.jax_utils` | Object | `` |
| `src.layers.preprocessing.data_layer.tracking` | Object | `` |
| `src.layers.preprocessing.data_layer.tree` | Object | `` |
| `src.layers.preprocessing.discretization.DataLayer` | Class | `(kwargs)` |
| `src.layers.preprocessing.discretization.Discretization` | Class | `(bin_boundaries, num_bins, epsilon, output_mode, sparse, dtype, name)` |
| `src.layers.preprocessing.discretization.argument_validation` | Object | `` |
| `src.layers.preprocessing.discretization.backend` | Object | `` |
| `src.layers.preprocessing.discretization.compress_summary` | Function | `(summary, epsilon)` |
| `src.layers.preprocessing.discretization.get_bin_boundaries` | Function | `(summary, num_bins)` |
| `src.layers.preprocessing.discretization.keras_export` | Class | `(path)` |
| `src.layers.preprocessing.discretization.merge_summaries` | Function | `(prev_summary, next_summary, epsilon)` |
| `src.layers.preprocessing.discretization.numerical_utils` | Object | `` |
| `src.layers.preprocessing.discretization.summarize` | Function | `(values, epsilon)` |
| `src.layers.preprocessing.discretization.tf` | Object | `` |
| `src.layers.preprocessing.feature_space.Cross` | Class | `(feature_names, crossing_dim, output_mode)` |
| `src.layers.preprocessing.feature_space.DataLayer` | Class | `(kwargs)` |
| `src.layers.preprocessing.feature_space.Feature` | Class | `(dtype, preprocessor, output_mode)` |
| `src.layers.preprocessing.feature_space.FeatureSpace` | Class | `(features, output_mode, crosses, crossing_dim, hashing_dim, num_discretization_bins, name)` |
| `src.layers.preprocessing.feature_space.KerasSaveable` | Class | `(...)` |
| `src.layers.preprocessing.feature_space.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.layers.preprocessing.feature_space.TFDConcat` | Class | `(axis, kwargs)` |
| `src.layers.preprocessing.feature_space.TFDIdentity` | Class | `(...)` |
| `src.layers.preprocessing.feature_space.auto_name` | Function | `(prefix)` |
| `src.layers.preprocessing.feature_space.backend` | Object | `` |
| `src.layers.preprocessing.feature_space.backend_utils` | Object | `` |
| `src.layers.preprocessing.feature_space.keras_export` | Class | `(path)` |
| `src.layers.preprocessing.feature_space.layers` | Object | `` |
| `src.layers.preprocessing.feature_space.saving_lib` | Object | `` |
| `src.layers.preprocessing.feature_space.serialization_lib` | Object | `` |
| `src.layers.preprocessing.feature_space.tf` | Object | `` |
| `src.layers.preprocessing.feature_space.tree` | Object | `` |
| `src.layers.preprocessing.hashed_crossing.HashedCrossing` | Class | `(num_bins, output_mode, sparse, name, dtype, kwargs)` |
| `src.layers.preprocessing.hashed_crossing.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.layers.preprocessing.hashed_crossing.argument_validation` | Object | `` |
| `src.layers.preprocessing.hashed_crossing.backend` | Object | `` |
| `src.layers.preprocessing.hashed_crossing.backend_utils` | Object | `` |
| `src.layers.preprocessing.hashed_crossing.keras_export` | Class | `(path)` |
| `src.layers.preprocessing.hashed_crossing.numerical_utils` | Object | `` |
| `src.layers.preprocessing.hashed_crossing.tf` | Object | `` |
| `src.layers.preprocessing.hashed_crossing.tf_utils` | Object | `` |
| `src.layers.preprocessing.hashing.Hashing` | Class | `(num_bins, mask_value, salt, output_mode, sparse, kwargs)` |
| `src.layers.preprocessing.hashing.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.layers.preprocessing.hashing.backend` | Object | `` |
| `src.layers.preprocessing.hashing.backend_utils` | Object | `` |
| `src.layers.preprocessing.hashing.keras_export` | Class | `(path)` |
| `src.layers.preprocessing.hashing.numerical_utils` | Object | `` |
| `src.layers.preprocessing.hashing.tf` | Object | `` |
| `src.layers.preprocessing.hashing.tf_utils` | Object | `` |
| `src.layers.preprocessing.image_preprocessing.aug_mix.AUGMENT_LAYERS` | Object | `` |
| `src.layers.preprocessing.image_preprocessing.aug_mix.AUGMENT_LAYERS_ALL` | Object | `` |
| `src.layers.preprocessing.image_preprocessing.aug_mix.AugMix` | Class | `(value_range, num_chains, chain_depth, factor, alpha, all_ops, interpolation, seed, data_format, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.aug_mix.BaseImagePreprocessingLayer` | Class | `(factor, bounding_box_format, data_format, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.aug_mix.SeedGenerator` | Class | `(seed, name, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.aug_mix.backend_utils` | Object | `` |
| `src.layers.preprocessing.image_preprocessing.aug_mix.keras_export` | Class | `(path)` |
| `src.layers.preprocessing.image_preprocessing.aug_mix.layers` | Object | `` |
| `src.layers.preprocessing.image_preprocessing.auto_contrast.AutoContrast` | Class | `(value_range, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.auto_contrast.BaseImagePreprocessingLayer` | Class | `(factor, bounding_box_format, data_format, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.auto_contrast.backend` | Object | `` |
| `src.layers.preprocessing.image_preprocessing.auto_contrast.keras_export` | Class | `(path)` |
| `src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer.BaseImagePreprocessingLayer` | Class | `(factor, bounding_box_format, data_format, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer.DataLayer` | Class | `(kwargs)` |
| `src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer.backend_config` | Object | `` |
| `src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer.densify_bounding_boxes` | Function | `(bounding_boxes, is_batched, max_boxes, boxes_default_value, labels_default_value, backend)` |
| `src.layers.preprocessing.image_preprocessing.bounding_boxes.bounding_box.BoundingBox` | Class | `()` |
| `src.layers.preprocessing.image_preprocessing.bounding_boxes.bounding_box.SUPPORTED_FORMATS` | Object | `` |
| `src.layers.preprocessing.image_preprocessing.bounding_boxes.bounding_box.backend_utils` | Object | `` |
| `src.layers.preprocessing.image_preprocessing.bounding_boxes.converters.BoundingBox` | Class | `()` |
| `src.layers.preprocessing.image_preprocessing.bounding_boxes.converters.affine_transform` | Function | `(boxes, angle, translate_x, translate_y, scale, shear_x, shear_y, height, width, center_x, center_y, bounding_box_format)` |
| `src.layers.preprocessing.image_preprocessing.bounding_boxes.converters.backend` | Object | `` |
| `src.layers.preprocessing.image_preprocessing.bounding_boxes.converters.backend_utils` | Object | `` |
| `src.layers.preprocessing.image_preprocessing.bounding_boxes.converters.clip_to_image_size` | Function | `(bounding_boxes, height, width, bounding_box_format)` |
| `src.layers.preprocessing.image_preprocessing.bounding_boxes.converters.convert_format` | Function | `(boxes, source, target, height, width, dtype)` |
| `src.layers.preprocessing.image_preprocessing.bounding_boxes.converters.crop` | Function | `(boxes, top, left, height, width, bounding_box_format)` |
| `src.layers.preprocessing.image_preprocessing.bounding_boxes.converters.decode_deltas_to_boxes` | Function | `(anchors, boxes_delta, anchor_format, box_format, encoded_format, variance, image_shape)` |
| `src.layers.preprocessing.image_preprocessing.bounding_boxes.converters.encode_box_to_deltas` | Function | `(anchors, boxes, anchor_format, box_format, encoding_format, variance, image_shape)` |
| `src.layers.preprocessing.image_preprocessing.bounding_boxes.converters.keras_export` | Class | `(path)` |
| `src.layers.preprocessing.image_preprocessing.bounding_boxes.converters.ops` | Object | `` |
| `src.layers.preprocessing.image_preprocessing.bounding_boxes.converters.pad` | Function | `(boxes, top, left, height, width, bounding_box_format)` |
| `src.layers.preprocessing.image_preprocessing.bounding_boxes.formats.CENTER_XYWH` | Class | `(...)` |
| `src.layers.preprocessing.image_preprocessing.bounding_boxes.formats.REL_XYWH` | Class | `(...)` |
| `src.layers.preprocessing.image_preprocessing.bounding_boxes.formats.REL_XYXY` | Class | `(...)` |
| `src.layers.preprocessing.image_preprocessing.bounding_boxes.formats.REL_YXYX` | Class | `(...)` |
| `src.layers.preprocessing.image_preprocessing.bounding_boxes.formats.XYWH` | Class | `(...)` |
| `src.layers.preprocessing.image_preprocessing.bounding_boxes.formats.XYXY` | Class | `(...)` |
| `src.layers.preprocessing.image_preprocessing.bounding_boxes.formats.YXYX` | Class | `(...)` |
| `src.layers.preprocessing.image_preprocessing.bounding_boxes.iou.backend` | Object | `` |
| `src.layers.preprocessing.image_preprocessing.bounding_boxes.iou.compute_ciou` | Function | `(boxes1, boxes2, bounding_box_format, image_shape)` |
| `src.layers.preprocessing.image_preprocessing.bounding_boxes.iou.compute_iou` | Function | `(boxes1, boxes2, bounding_box_format, use_masking, mask_val, image_shape)` |
| `src.layers.preprocessing.image_preprocessing.bounding_boxes.iou.converters` | Object | `` |
| `src.layers.preprocessing.image_preprocessing.bounding_boxes.iou.keras_export` | Class | `(path)` |
| `src.layers.preprocessing.image_preprocessing.bounding_boxes.iou.ops` | Object | `` |
| `src.layers.preprocessing.image_preprocessing.bounding_boxes.validation.current_backend` | Object | `` |
| `src.layers.preprocessing.image_preprocessing.bounding_boxes.validation.densify_bounding_boxes` | Function | `(bounding_boxes, is_batched, max_boxes, boxes_default_value, labels_default_value, backend)` |
| `src.layers.preprocessing.image_preprocessing.bounding_boxes.validation.tf_utils` | Object | `` |
| `src.layers.preprocessing.image_preprocessing.bounding_boxes.validation.validate_bounding_boxes` | Function | `(bounding_boxes)` |
| `src.layers.preprocessing.image_preprocessing.center_crop.BaseImagePreprocessingLayer` | Class | `(factor, bounding_box_format, data_format, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.center_crop.CenterCrop` | Class | `(height, width, data_format, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.center_crop.clip_to_image_size` | Function | `(bounding_boxes, height, width, bounding_box_format)` |
| `src.layers.preprocessing.image_preprocessing.center_crop.convert_format` | Function | `(boxes, source, target, height, width, dtype)` |
| `src.layers.preprocessing.image_preprocessing.center_crop.image_utils` | Object | `` |
| `src.layers.preprocessing.image_preprocessing.center_crop.keras_export` | Class | `(path)` |
| `src.layers.preprocessing.image_preprocessing.cut_mix.BaseImagePreprocessingLayer` | Class | `(factor, bounding_box_format, data_format, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.cut_mix.CutMix` | Class | `(factor, seed, data_format, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.cut_mix.SeedGenerator` | Class | `(seed, name, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.cut_mix.keras_export` | Class | `(path)` |
| `src.layers.preprocessing.image_preprocessing.equalization.BaseImagePreprocessingLayer` | Class | `(factor, bounding_box_format, data_format, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.equalization.Equalization` | Class | `(value_range, bins, data_format, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.equalization.backend` | Object | `` |
| `src.layers.preprocessing.image_preprocessing.equalization.keras_export` | Class | `(path)` |
| `src.layers.preprocessing.image_preprocessing.max_num_bounding_box.BaseImagePreprocessingLayer` | Class | `(factor, bounding_box_format, data_format, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.max_num_bounding_box.MaxNumBoundingBoxes` | Class | `(max_number, fill_value, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.max_num_bounding_box.keras_export` | Class | `(path)` |
| `src.layers.preprocessing.image_preprocessing.mix_up.BaseImagePreprocessingLayer` | Class | `(factor, bounding_box_format, data_format, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.mix_up.MixUp` | Class | `(alpha, data_format, seed, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.mix_up.SeedGenerator` | Class | `(seed, name, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.mix_up.backend_utils` | Object | `` |
| `src.layers.preprocessing.image_preprocessing.mix_up.keras_export` | Class | `(path)` |
| `src.layers.preprocessing.image_preprocessing.mix_up.ops` | Object | `` |
| `src.layers.preprocessing.image_preprocessing.rand_augment.BaseImagePreprocessingLayer` | Class | `(factor, bounding_box_format, data_format, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.rand_augment.RandAugment` | Class | `(value_range, num_ops, factor, interpolation, seed, data_format, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.rand_augment.SeedGenerator` | Class | `(seed, name, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.rand_augment.backend_utils` | Object | `` |
| `src.layers.preprocessing.image_preprocessing.rand_augment.keras_export` | Class | `(path)` |
| `src.layers.preprocessing.image_preprocessing.rand_augment.layers` | Object | `` |
| `src.layers.preprocessing.image_preprocessing.random_brightness.BaseImagePreprocessingLayer` | Class | `(factor, bounding_box_format, data_format, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.random_brightness.RandomBrightness` | Class | `(factor, value_range, seed, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.random_brightness.SeedGenerator` | Class | `(seed, name, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.random_brightness.keras_export` | Class | `(path)` |
| `src.layers.preprocessing.image_preprocessing.random_color_degeneration.BaseImagePreprocessingLayer` | Class | `(factor, bounding_box_format, data_format, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.random_color_degeneration.RandomColorDegeneration` | Class | `(factor, value_range, data_format, seed, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.random_color_degeneration.SeedGenerator` | Class | `(seed, name, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.random_color_degeneration.keras_export` | Class | `(path)` |
| `src.layers.preprocessing.image_preprocessing.random_color_jitter.BaseImagePreprocessingLayer` | Class | `(factor, bounding_box_format, data_format, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.random_color_jitter.RandomColorJitter` | Class | `(value_range, brightness_factor, contrast_factor, saturation_factor, hue_factor, seed, data_format, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.random_color_jitter.SeedGenerator` | Class | `(seed, name, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.random_color_jitter.backend_utils` | Object | `` |
| `src.layers.preprocessing.image_preprocessing.random_color_jitter.keras_export` | Class | `(path)` |
| `src.layers.preprocessing.image_preprocessing.random_color_jitter.random_brightness` | Object | `` |
| `src.layers.preprocessing.image_preprocessing.random_color_jitter.random_contrast` | Object | `` |
| `src.layers.preprocessing.image_preprocessing.random_color_jitter.random_hue` | Object | `` |
| `src.layers.preprocessing.image_preprocessing.random_color_jitter.random_saturation` | Object | `` |
| `src.layers.preprocessing.image_preprocessing.random_contrast.BaseImagePreprocessingLayer` | Class | `(factor, bounding_box_format, data_format, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.random_contrast.RandomContrast` | Class | `(factor, value_range, seed, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.random_contrast.SeedGenerator` | Class | `(seed, name, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.random_contrast.keras_export` | Class | `(path)` |
| `src.layers.preprocessing.image_preprocessing.random_crop.BaseImagePreprocessingLayer` | Class | `(factor, bounding_box_format, data_format, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.random_crop.RandomCrop` | Class | `(height, width, seed, data_format, name, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.random_crop.SeedGenerator` | Class | `(seed, name, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.random_crop.backend` | Object | `` |
| `src.layers.preprocessing.image_preprocessing.random_crop.convert_format` | Function | `(boxes, source, target, height, width, dtype)` |
| `src.layers.preprocessing.image_preprocessing.random_crop.densify_bounding_boxes` | Function | `(bounding_boxes, is_batched, max_boxes, boxes_default_value, labels_default_value, backend)` |
| `src.layers.preprocessing.image_preprocessing.random_crop.keras_export` | Class | `(path)` |
| `src.layers.preprocessing.image_preprocessing.random_elastic_transform.BaseImagePreprocessingLayer` | Class | `(factor, bounding_box_format, data_format, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.random_elastic_transform.RandomElasticTransform` | Class | `(factor, scale, interpolation, fill_mode, fill_value, value_range, seed, data_format, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.random_elastic_transform.SeedGenerator` | Class | `(seed, name, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.random_elastic_transform.keras_export` | Class | `(path)` |
| `src.layers.preprocessing.image_preprocessing.random_erasing.BaseImagePreprocessingLayer` | Class | `(factor, bounding_box_format, data_format, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.random_erasing.RandomErasing` | Class | `(factor, scale, fill_value, value_range, seed, data_format, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.random_erasing.SeedGenerator` | Class | `(seed, name, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.random_erasing.keras_export` | Class | `(path)` |
| `src.layers.preprocessing.image_preprocessing.random_flip.BaseImagePreprocessingLayer` | Class | `(factor, bounding_box_format, data_format, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.random_flip.HORIZONTAL` | Object | `` |
| `src.layers.preprocessing.image_preprocessing.random_flip.HORIZONTAL_AND_VERTICAL` | Object | `` |
| `src.layers.preprocessing.image_preprocessing.random_flip.RandomFlip` | Class | `(mode, seed, data_format, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.random_flip.SeedGenerator` | Class | `(seed, name, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.random_flip.VERTICAL` | Object | `` |
| `src.layers.preprocessing.image_preprocessing.random_flip.backend_utils` | Object | `` |
| `src.layers.preprocessing.image_preprocessing.random_flip.clip_to_image_size` | Function | `(bounding_boxes, height, width, bounding_box_format)` |
| `src.layers.preprocessing.image_preprocessing.random_flip.convert_format` | Function | `(boxes, source, target, height, width, dtype)` |
| `src.layers.preprocessing.image_preprocessing.random_flip.keras_export` | Class | `(path)` |
| `src.layers.preprocessing.image_preprocessing.random_gaussian_blur.BaseImagePreprocessingLayer` | Class | `(factor, bounding_box_format, data_format, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.random_gaussian_blur.RandomGaussianBlur` | Class | `(factor, kernel_size, sigma, value_range, data_format, seed, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.random_gaussian_blur.SeedGenerator` | Class | `(seed, name, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.random_gaussian_blur.keras_export` | Class | `(path)` |
| `src.layers.preprocessing.image_preprocessing.random_grayscale.BaseImagePreprocessingLayer` | Class | `(factor, bounding_box_format, data_format, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.random_grayscale.RandomGrayscale` | Class | `(factor, data_format, seed, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.random_grayscale.backend` | Object | `` |
| `src.layers.preprocessing.image_preprocessing.random_grayscale.keras_export` | Class | `(path)` |
| `src.layers.preprocessing.image_preprocessing.random_hue.BaseImagePreprocessingLayer` | Class | `(factor, bounding_box_format, data_format, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.random_hue.RandomHue` | Class | `(factor, value_range, data_format, seed, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.random_hue.keras_export` | Class | `(path)` |
| `src.layers.preprocessing.image_preprocessing.random_invert.BaseImagePreprocessingLayer` | Class | `(factor, bounding_box_format, data_format, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.random_invert.RandomInvert` | Class | `(factor, value_range, seed, data_format, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.random_invert.keras_export` | Class | `(path)` |
| `src.layers.preprocessing.image_preprocessing.random_perspective.BaseImagePreprocessingLayer` | Class | `(factor, bounding_box_format, data_format, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.random_perspective.RandomPerspective` | Class | `(factor, scale, interpolation, fill_value, seed, data_format, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.random_perspective.SeedGenerator` | Class | `(seed, name, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.random_perspective.backend_utils` | Object | `` |
| `src.layers.preprocessing.image_preprocessing.random_perspective.clip_to_image_size` | Function | `(bounding_boxes, height, width, bounding_box_format)` |
| `src.layers.preprocessing.image_preprocessing.random_perspective.convert_format` | Function | `(boxes, source, target, height, width, dtype)` |
| `src.layers.preprocessing.image_preprocessing.random_perspective.keras_export` | Class | `(path)` |
| `src.layers.preprocessing.image_preprocessing.random_posterization.BaseImagePreprocessingLayer` | Class | `(factor, bounding_box_format, data_format, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.random_posterization.RandomPosterization` | Class | `(factor, value_range, data_format, seed, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.random_posterization.keras_export` | Class | `(path)` |
| `src.layers.preprocessing.image_preprocessing.random_rotation.BaseImagePreprocessingLayer` | Class | `(factor, bounding_box_format, data_format, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.random_rotation.RandomRotation` | Class | `(factor, fill_mode, interpolation, seed, fill_value, data_format, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.random_rotation.SeedGenerator` | Class | `(seed, name, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.random_rotation.converters` | Object | `` |
| `src.layers.preprocessing.image_preprocessing.random_rotation.keras_export` | Class | `(path)` |
| `src.layers.preprocessing.image_preprocessing.random_saturation.BaseImagePreprocessingLayer` | Class | `(factor, bounding_box_format, data_format, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.random_saturation.RandomSaturation` | Class | `(factor, value_range, data_format, seed, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.random_saturation.SeedGenerator` | Class | `(seed, name, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.random_saturation.epsilon` | Function | `()` |
| `src.layers.preprocessing.image_preprocessing.random_saturation.keras_export` | Class | `(path)` |
| `src.layers.preprocessing.image_preprocessing.random_sharpness.BaseImagePreprocessingLayer` | Class | `(factor, bounding_box_format, data_format, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.random_sharpness.RandomSharpness` | Class | `(factor, value_range, data_format, seed, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.random_sharpness.SeedGenerator` | Class | `(seed, name, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.random_sharpness.keras_export` | Class | `(path)` |
| `src.layers.preprocessing.image_preprocessing.random_shear.BaseImagePreprocessingLayer` | Class | `(factor, bounding_box_format, data_format, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.random_shear.RandomShear` | Class | `(x_factor, y_factor, interpolation, fill_mode, fill_value, data_format, seed, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.random_shear.SeedGenerator` | Class | `(seed, name, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.random_shear.backend_utils` | Object | `` |
| `src.layers.preprocessing.image_preprocessing.random_shear.clip_to_image_size` | Function | `(bounding_boxes, height, width, bounding_box_format)` |
| `src.layers.preprocessing.image_preprocessing.random_shear.convert_format` | Function | `(boxes, source, target, height, width, dtype)` |
| `src.layers.preprocessing.image_preprocessing.random_shear.keras_export` | Class | `(path)` |
| `src.layers.preprocessing.image_preprocessing.random_translation.BaseImagePreprocessingLayer` | Class | `(factor, bounding_box_format, data_format, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.random_translation.RandomTranslation` | Class | `(height_factor, width_factor, fill_mode, interpolation, seed, fill_value, data_format, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.random_translation.SeedGenerator` | Class | `(seed, name, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.random_translation.backend_utils` | Object | `` |
| `src.layers.preprocessing.image_preprocessing.random_translation.clip_to_image_size` | Function | `(bounding_boxes, height, width, bounding_box_format)` |
| `src.layers.preprocessing.image_preprocessing.random_translation.convert_format` | Function | `(boxes, source, target, height, width, dtype)` |
| `src.layers.preprocessing.image_preprocessing.random_translation.keras_export` | Class | `(path)` |
| `src.layers.preprocessing.image_preprocessing.random_zoom.BaseImagePreprocessingLayer` | Class | `(factor, bounding_box_format, data_format, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.random_zoom.RandomZoom` | Class | `(height_factor, width_factor, fill_mode, interpolation, seed, fill_value, data_format, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.random_zoom.SeedGenerator` | Class | `(seed, name, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.random_zoom.backend` | Object | `` |
| `src.layers.preprocessing.image_preprocessing.random_zoom.backend_utils` | Object | `` |
| `src.layers.preprocessing.image_preprocessing.random_zoom.clip_to_image_size` | Function | `(bounding_boxes, height, width, bounding_box_format)` |
| `src.layers.preprocessing.image_preprocessing.random_zoom.convert_format` | Function | `(boxes, source, target, height, width, dtype)` |
| `src.layers.preprocessing.image_preprocessing.random_zoom.keras_export` | Class | `(path)` |
| `src.layers.preprocessing.image_preprocessing.resizing.BaseImagePreprocessingLayer` | Class | `(factor, bounding_box_format, data_format, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.resizing.Resizing` | Class | `(height, width, interpolation, crop_to_aspect_ratio, pad_to_aspect_ratio, fill_mode, fill_value, antialias, data_format, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.resizing.backend` | Object | `` |
| `src.layers.preprocessing.image_preprocessing.resizing.clip_to_image_size` | Function | `(bounding_boxes, height, width, bounding_box_format)` |
| `src.layers.preprocessing.image_preprocessing.resizing.convert_format` | Function | `(boxes, source, target, height, width, dtype)` |
| `src.layers.preprocessing.image_preprocessing.resizing.keras_export` | Class | `(path)` |
| `src.layers.preprocessing.image_preprocessing.solarization.BaseImagePreprocessingLayer` | Class | `(factor, bounding_box_format, data_format, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.solarization.SeedGenerator` | Class | `(seed, name, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.solarization.Solarization` | Class | `(addition_factor, threshold_factor, value_range, seed, kwargs)` |
| `src.layers.preprocessing.image_preprocessing.solarization.backend` | Object | `` |
| `src.layers.preprocessing.image_preprocessing.solarization.keras_export` | Class | `(path)` |
| `src.layers.preprocessing.index_lookup.IndexLookup` | Class | `(max_tokens, num_oov_indices, mask_token, oov_token, vocabulary_dtype, vocabulary, idf_weights, invert, output_mode, sparse, pad_to_max_tokens, name, kwargs)` |
| `src.layers.preprocessing.index_lookup.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.layers.preprocessing.index_lookup.argument_validation` | Object | `` |
| `src.layers.preprocessing.index_lookup.backend` | Object | `` |
| `src.layers.preprocessing.index_lookup.get_null_initializer` | Function | `(key_dtype, value_dtype)` |
| `src.layers.preprocessing.index_lookup.listify_tensors` | Function | `(x)` |
| `src.layers.preprocessing.index_lookup.numerical_utils` | Object | `` |
| `src.layers.preprocessing.index_lookup.serialization_lib` | Object | `` |
| `src.layers.preprocessing.index_lookup.tf` | Object | `` |
| `src.layers.preprocessing.index_lookup.tf_utils` | Object | `` |
| `src.layers.preprocessing.integer_lookup.IndexLookup` | Class | `(max_tokens, num_oov_indices, mask_token, oov_token, vocabulary_dtype, vocabulary, idf_weights, invert, output_mode, sparse, pad_to_max_tokens, name, kwargs)` |
| `src.layers.preprocessing.integer_lookup.IntegerLookup` | Class | `(max_tokens, num_oov_indices, mask_token, oov_token, vocabulary, vocabulary_dtype, idf_weights, invert, output_mode, sparse, pad_to_max_tokens, name, kwargs)` |
| `src.layers.preprocessing.integer_lookup.backend` | Object | `` |
| `src.layers.preprocessing.integer_lookup.backend_utils` | Object | `` |
| `src.layers.preprocessing.integer_lookup.keras_export` | Class | `(path)` |
| `src.layers.preprocessing.integer_lookup.tf` | Object | `` |
| `src.layers.preprocessing.mel_spectrogram.DataLayer` | Class | `(kwargs)` |
| `src.layers.preprocessing.mel_spectrogram.MelSpectrogram` | Class | `(fft_length, sequence_stride, sequence_length, window, sampling_rate, num_mel_bins, min_freq, max_freq, power_to_db, top_db, mag_exp, min_power, ref_power, kwargs)` |
| `src.layers.preprocessing.mel_spectrogram.keras_export` | Class | `(path)` |
| `src.layers.preprocessing.normalization.DataLayer` | Class | `(kwargs)` |
| `src.layers.preprocessing.normalization.Normalization` | Class | `(axis, mean, variance, invert, kwargs)` |
| `src.layers.preprocessing.normalization.PyDataset` | Class | `(workers, use_multiprocessing, max_queue_size)` |
| `src.layers.preprocessing.normalization.backend` | Object | `` |
| `src.layers.preprocessing.normalization.keras_export` | Class | `(path)` |
| `src.layers.preprocessing.normalization.ops` | Object | `` |
| `src.layers.preprocessing.normalization.tf` | Object | `` |
| `src.layers.preprocessing.pipeline.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.layers.preprocessing.pipeline.Pipeline` | Class | `(layers, name)` |
| `src.layers.preprocessing.pipeline.keras_export` | Class | `(path)` |
| `src.layers.preprocessing.pipeline.serialization_lib` | Object | `` |
| `src.layers.preprocessing.pipeline.tree` | Object | `` |
| `src.layers.preprocessing.rescaling.DataLayer` | Class | `(kwargs)` |
| `src.layers.preprocessing.rescaling.Rescaling` | Class | `(scale, offset, kwargs)` |
| `src.layers.preprocessing.rescaling.backend` | Object | `` |
| `src.layers.preprocessing.rescaling.keras_export` | Class | `(path)` |
| `src.layers.preprocessing.rescaling.serialization_lib` | Object | `` |
| `src.layers.preprocessing.stft_spectrogram.STFTSpectrogram` | Class | `(mode, frame_length, frame_step, fft_length, window, periodic, scaling, padding, expand_dims, data_format, kwargs)` |
| `src.layers.preprocessing.stft_spectrogram.backend` | Object | `` |
| `src.layers.preprocessing.stft_spectrogram.initializers` | Object | `` |
| `src.layers.preprocessing.stft_spectrogram.keras_export` | Class | `(path)` |
| `src.layers.preprocessing.stft_spectrogram.layers` | Object | `` |
| `src.layers.preprocessing.stft_spectrogram.ops` | Object | `` |
| `src.layers.preprocessing.stft_spectrogram.scipy` | Object | `` |
| `src.layers.preprocessing.string_lookup.IndexLookup` | Class | `(max_tokens, num_oov_indices, mask_token, oov_token, vocabulary_dtype, vocabulary, idf_weights, invert, output_mode, sparse, pad_to_max_tokens, name, kwargs)` |
| `src.layers.preprocessing.string_lookup.StringLookup` | Class | `(max_tokens, num_oov_indices, mask_token, oov_token, vocabulary, idf_weights, invert, output_mode, pad_to_max_tokens, sparse, encoding, name, kwargs)` |
| `src.layers.preprocessing.string_lookup.backend` | Object | `` |
| `src.layers.preprocessing.string_lookup.backend_utils` | Object | `` |
| `src.layers.preprocessing.string_lookup.keras_export` | Class | `(path)` |
| `src.layers.preprocessing.string_lookup.tf` | Object | `` |
| `src.layers.preprocessing.text_vectorization.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.layers.preprocessing.text_vectorization.StringLookup` | Class | `(max_tokens, num_oov_indices, mask_token, oov_token, vocabulary, idf_weights, invert, output_mode, pad_to_max_tokens, sparse, encoding, name, kwargs)` |
| `src.layers.preprocessing.text_vectorization.TextVectorization` | Class | `(max_tokens, standardize, split, ngrams, output_mode, output_sequence_length, pad_to_max_tokens, vocabulary, idf_weights, sparse, ragged, encoding, name, kwargs)` |
| `src.layers.preprocessing.text_vectorization.argument_validation` | Object | `` |
| `src.layers.preprocessing.text_vectorization.backend` | Object | `` |
| `src.layers.preprocessing.text_vectorization.backend_utils` | Object | `` |
| `src.layers.preprocessing.text_vectorization.keras_export` | Class | `(path)` |
| `src.layers.preprocessing.text_vectorization.listify_tensors` | Function | `(x)` |
| `src.layers.preprocessing.text_vectorization.serialization_lib` | Object | `` |
| `src.layers.preprocessing.text_vectorization.tf` | Object | `` |
| `src.layers.preprocessing.text_vectorization.tf_utils` | Object | `` |
| `src.layers.regularization.activity_regularization.ActivityRegularization` | Class | `(l1, l2, kwargs)` |
| `src.layers.regularization.activity_regularization.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.layers.regularization.activity_regularization.keras_export` | Class | `(path)` |
| `src.layers.regularization.activity_regularization.regularizers` | Object | `` |
| `src.layers.regularization.alpha_dropout.AlphaDropout` | Class | `(rate, noise_shape, seed, kwargs)` |
| `src.layers.regularization.alpha_dropout.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.layers.regularization.alpha_dropout.backend` | Object | `` |
| `src.layers.regularization.alpha_dropout.keras_export` | Class | `(path)` |
| `src.layers.regularization.alpha_dropout.ops` | Object | `` |
| `src.layers.regularization.dropout.Dropout` | Class | `(rate, noise_shape, seed, kwargs)` |
| `src.layers.regularization.dropout.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.layers.regularization.dropout.backend` | Object | `` |
| `src.layers.regularization.dropout.keras_export` | Class | `(path)` |
| `src.layers.regularization.gaussian_dropout.GaussianDropout` | Class | `(rate, seed, kwargs)` |
| `src.layers.regularization.gaussian_dropout.backend` | Object | `` |
| `src.layers.regularization.gaussian_dropout.keras_export` | Class | `(path)` |
| `src.layers.regularization.gaussian_dropout.layers` | Object | `` |
| `src.layers.regularization.gaussian_dropout.ops` | Object | `` |
| `src.layers.regularization.gaussian_noise.GaussianNoise` | Class | `(stddev, seed, kwargs)` |
| `src.layers.regularization.gaussian_noise.backend` | Object | `` |
| `src.layers.regularization.gaussian_noise.keras_export` | Class | `(path)` |
| `src.layers.regularization.gaussian_noise.layers` | Object | `` |
| `src.layers.regularization.gaussian_noise.ops` | Object | `` |
| `src.layers.regularization.spatial_dropout.BaseSpatialDropout` | Class | `(rate, seed, name, dtype)` |
| `src.layers.regularization.spatial_dropout.Dropout` | Class | `(rate, noise_shape, seed, kwargs)` |
| `src.layers.regularization.spatial_dropout.InputSpec` | Class | `(dtype, shape, ndim, max_ndim, min_ndim, axes, allow_last_axis_squeeze, name, optional)` |
| `src.layers.regularization.spatial_dropout.SpatialDropout1D` | Class | `(rate, seed, name, dtype)` |
| `src.layers.regularization.spatial_dropout.SpatialDropout2D` | Class | `(rate, data_format, seed, name, dtype)` |
| `src.layers.regularization.spatial_dropout.SpatialDropout3D` | Class | `(rate, data_format, seed, name, dtype)` |
| `src.layers.regularization.spatial_dropout.backend` | Object | `` |
| `src.layers.regularization.spatial_dropout.keras_export` | Class | `(path)` |
| `src.layers.regularization.spatial_dropout.ops` | Object | `` |
| `src.layers.reshaping.cropping1d.Cropping1D` | Class | `(cropping, kwargs)` |
| `src.layers.reshaping.cropping1d.InputSpec` | Class | `(dtype, shape, ndim, max_ndim, min_ndim, axes, allow_last_axis_squeeze, name, optional)` |
| `src.layers.reshaping.cropping1d.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.layers.reshaping.cropping1d.argument_validation` | Object | `` |
| `src.layers.reshaping.cropping1d.keras_export` | Class | `(path)` |
| `src.layers.reshaping.cropping2d.Cropping2D` | Class | `(cropping, data_format, kwargs)` |
| `src.layers.reshaping.cropping2d.InputSpec` | Class | `(dtype, shape, ndim, max_ndim, min_ndim, axes, allow_last_axis_squeeze, name, optional)` |
| `src.layers.reshaping.cropping2d.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.layers.reshaping.cropping2d.argument_validation` | Object | `` |
| `src.layers.reshaping.cropping2d.backend` | Object | `` |
| `src.layers.reshaping.cropping2d.keras_export` | Class | `(path)` |
| `src.layers.reshaping.cropping3d.Cropping3D` | Class | `(cropping, data_format, kwargs)` |
| `src.layers.reshaping.cropping3d.InputSpec` | Class | `(dtype, shape, ndim, max_ndim, min_ndim, axes, allow_last_axis_squeeze, name, optional)` |
| `src.layers.reshaping.cropping3d.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.layers.reshaping.cropping3d.argument_validation` | Object | `` |
| `src.layers.reshaping.cropping3d.backend` | Object | `` |
| `src.layers.reshaping.cropping3d.keras_export` | Class | `(path)` |
| `src.layers.reshaping.flatten.Flatten` | Class | `(data_format, kwargs)` |
| `src.layers.reshaping.flatten.InputSpec` | Class | `(dtype, shape, ndim, max_ndim, min_ndim, axes, allow_last_axis_squeeze, name, optional)` |
| `src.layers.reshaping.flatten.KerasTensor` | Class | `(shape, dtype, sparse, ragged, record_history, name, kwargs)` |
| `src.layers.reshaping.flatten.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.layers.reshaping.flatten.backend` | Object | `` |
| `src.layers.reshaping.flatten.keras_export` | Class | `(path)` |
| `src.layers.reshaping.flatten.ops` | Object | `` |
| `src.layers.reshaping.permute.InputSpec` | Class | `(dtype, shape, ndim, max_ndim, min_ndim, axes, allow_last_axis_squeeze, name, optional)` |
| `src.layers.reshaping.permute.KerasTensor` | Class | `(shape, dtype, sparse, ragged, record_history, name, kwargs)` |
| `src.layers.reshaping.permute.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.layers.reshaping.permute.Permute` | Class | `(dims, kwargs)` |
| `src.layers.reshaping.permute.keras_export` | Class | `(path)` |
| `src.layers.reshaping.permute.ops` | Object | `` |
| `src.layers.reshaping.repeat_vector.InputSpec` | Class | `(dtype, shape, ndim, max_ndim, min_ndim, axes, allow_last_axis_squeeze, name, optional)` |
| `src.layers.reshaping.repeat_vector.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.layers.reshaping.repeat_vector.RepeatVector` | Class | `(n, kwargs)` |
| `src.layers.reshaping.repeat_vector.keras_export` | Class | `(path)` |
| `src.layers.reshaping.repeat_vector.ops` | Object | `` |
| `src.layers.reshaping.reshape.KerasTensor` | Class | `(shape, dtype, sparse, ragged, record_history, name, kwargs)` |
| `src.layers.reshaping.reshape.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.layers.reshaping.reshape.Reshape` | Class | `(target_shape, kwargs)` |
| `src.layers.reshaping.reshape.keras_export` | Class | `(path)` |
| `src.layers.reshaping.reshape.operation_utils` | Object | `` |
| `src.layers.reshaping.reshape.ops` | Object | `` |
| `src.layers.reshaping.up_sampling1d.InputSpec` | Class | `(dtype, shape, ndim, max_ndim, min_ndim, axes, allow_last_axis_squeeze, name, optional)` |
| `src.layers.reshaping.up_sampling1d.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.layers.reshaping.up_sampling1d.UpSampling1D` | Class | `(size, kwargs)` |
| `src.layers.reshaping.up_sampling1d.keras_export` | Class | `(path)` |
| `src.layers.reshaping.up_sampling1d.ops` | Object | `` |
| `src.layers.reshaping.up_sampling2d.InputSpec` | Class | `(dtype, shape, ndim, max_ndim, min_ndim, axes, allow_last_axis_squeeze, name, optional)` |
| `src.layers.reshaping.up_sampling2d.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.layers.reshaping.up_sampling2d.UpSampling2D` | Class | `(size, data_format, interpolation, kwargs)` |
| `src.layers.reshaping.up_sampling2d.argument_validation` | Object | `` |
| `src.layers.reshaping.up_sampling2d.backend` | Object | `` |
| `src.layers.reshaping.up_sampling2d.keras_export` | Class | `(path)` |
| `src.layers.reshaping.up_sampling2d.ops` | Object | `` |
| `src.layers.reshaping.up_sampling3d.InputSpec` | Class | `(dtype, shape, ndim, max_ndim, min_ndim, axes, allow_last_axis_squeeze, name, optional)` |
| `src.layers.reshaping.up_sampling3d.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.layers.reshaping.up_sampling3d.UpSampling3D` | Class | `(size, data_format, kwargs)` |
| `src.layers.reshaping.up_sampling3d.argument_validation` | Object | `` |
| `src.layers.reshaping.up_sampling3d.backend` | Object | `` |
| `src.layers.reshaping.up_sampling3d.keras_export` | Class | `(path)` |
| `src.layers.reshaping.up_sampling3d.ops` | Object | `` |
| `src.layers.reshaping.zero_padding1d.InputSpec` | Class | `(dtype, shape, ndim, max_ndim, min_ndim, axes, allow_last_axis_squeeze, name, optional)` |
| `src.layers.reshaping.zero_padding1d.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.layers.reshaping.zero_padding1d.ZeroPadding1D` | Class | `(padding, data_format, kwargs)` |
| `src.layers.reshaping.zero_padding1d.argument_validation` | Object | `` |
| `src.layers.reshaping.zero_padding1d.backend` | Object | `` |
| `src.layers.reshaping.zero_padding1d.keras_export` | Class | `(path)` |
| `src.layers.reshaping.zero_padding1d.ops` | Object | `` |
| `src.layers.reshaping.zero_padding2d.InputSpec` | Class | `(dtype, shape, ndim, max_ndim, min_ndim, axes, allow_last_axis_squeeze, name, optional)` |
| `src.layers.reshaping.zero_padding2d.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.layers.reshaping.zero_padding2d.ZeroPadding2D` | Class | `(padding, data_format, kwargs)` |
| `src.layers.reshaping.zero_padding2d.argument_validation` | Object | `` |
| `src.layers.reshaping.zero_padding2d.backend` | Object | `` |
| `src.layers.reshaping.zero_padding2d.keras_export` | Class | `(path)` |
| `src.layers.reshaping.zero_padding2d.ops` | Object | `` |
| `src.layers.reshaping.zero_padding3d.InputSpec` | Class | `(dtype, shape, ndim, max_ndim, min_ndim, axes, allow_last_axis_squeeze, name, optional)` |
| `src.layers.reshaping.zero_padding3d.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.layers.reshaping.zero_padding3d.ZeroPadding3D` | Class | `(padding, data_format, kwargs)` |
| `src.layers.reshaping.zero_padding3d.argument_validation` | Object | `` |
| `src.layers.reshaping.zero_padding3d.backend` | Object | `` |
| `src.layers.reshaping.zero_padding3d.keras_export` | Class | `(path)` |
| `src.layers.reshaping.zero_padding3d.ops` | Object | `` |
| `src.layers.rnn.bidirectional.Bidirectional` | Class | `(layer, merge_mode, weights, backward_layer, kwargs)` |
| `src.layers.rnn.bidirectional.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.layers.rnn.bidirectional.keras_export` | Class | `(path)` |
| `src.layers.rnn.bidirectional.ops` | Object | `` |
| `src.layers.rnn.bidirectional.serialization_lib` | Object | `` |
| `src.layers.rnn.bidirectional.utils` | Object | `` |
| `src.layers.rnn.conv_lstm.ConvLSTM` | Class | `(rank, filters, kernel_size, strides, padding, data_format, dilation_rate, activation, recurrent_activation, use_bias, kernel_initializer, recurrent_initializer, bias_initializer, unit_forget_bias, kernel_regularizer, recurrent_regularizer, bias_regularizer, kernel_constraint, recurrent_constraint, bias_constraint, dropout, recurrent_dropout, seed, return_sequences, return_state, go_backwards, stateful, kwargs)` |
| `src.layers.rnn.conv_lstm.ConvLSTMCell` | Class | `(rank, filters, kernel_size, strides, padding, data_format, dilation_rate, activation, recurrent_activation, use_bias, kernel_initializer, recurrent_initializer, bias_initializer, unit_forget_bias, kernel_regularizer, recurrent_regularizer, bias_regularizer, kernel_constraint, recurrent_constraint, bias_constraint, dropout, recurrent_dropout, seed, kwargs)` |
| `src.layers.rnn.conv_lstm.DropoutRNNCell` | Class | `(...)` |
| `src.layers.rnn.conv_lstm.InputSpec` | Class | `(dtype, shape, ndim, max_ndim, min_ndim, axes, allow_last_axis_squeeze, name, optional)` |
| `src.layers.rnn.conv_lstm.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.layers.rnn.conv_lstm.RNN` | Class | `(cell, return_sequences, return_state, go_backwards, stateful, unroll, zero_output_for_mask, kwargs)` |
| `src.layers.rnn.conv_lstm.activations` | Object | `` |
| `src.layers.rnn.conv_lstm.argument_validation` | Object | `` |
| `src.layers.rnn.conv_lstm.backend` | Object | `` |
| `src.layers.rnn.conv_lstm.constraints` | Object | `` |
| `src.layers.rnn.conv_lstm.initializers` | Object | `` |
| `src.layers.rnn.conv_lstm.operation_utils` | Object | `` |
| `src.layers.rnn.conv_lstm.ops` | Object | `` |
| `src.layers.rnn.conv_lstm.regularizers` | Object | `` |
| `src.layers.rnn.conv_lstm.tree` | Object | `` |
| `src.layers.rnn.conv_lstm1d.ConvLSTM` | Class | `(rank, filters, kernel_size, strides, padding, data_format, dilation_rate, activation, recurrent_activation, use_bias, kernel_initializer, recurrent_initializer, bias_initializer, unit_forget_bias, kernel_regularizer, recurrent_regularizer, bias_regularizer, kernel_constraint, recurrent_constraint, bias_constraint, dropout, recurrent_dropout, seed, return_sequences, return_state, go_backwards, stateful, kwargs)` |
| `src.layers.rnn.conv_lstm1d.ConvLSTM1D` | Class | `(filters, kernel_size, strides, padding, data_format, dilation_rate, activation, recurrent_activation, use_bias, kernel_initializer, recurrent_initializer, bias_initializer, unit_forget_bias, kernel_regularizer, recurrent_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, recurrent_constraint, bias_constraint, dropout, recurrent_dropout, seed, return_sequences, return_state, go_backwards, stateful, kwargs)` |
| `src.layers.rnn.conv_lstm1d.keras_export` | Class | `(path)` |
| `src.layers.rnn.conv_lstm2d.ConvLSTM` | Class | `(rank, filters, kernel_size, strides, padding, data_format, dilation_rate, activation, recurrent_activation, use_bias, kernel_initializer, recurrent_initializer, bias_initializer, unit_forget_bias, kernel_regularizer, recurrent_regularizer, bias_regularizer, kernel_constraint, recurrent_constraint, bias_constraint, dropout, recurrent_dropout, seed, return_sequences, return_state, go_backwards, stateful, kwargs)` |
| `src.layers.rnn.conv_lstm2d.ConvLSTM2D` | Class | `(filters, kernel_size, strides, padding, data_format, dilation_rate, activation, recurrent_activation, use_bias, kernel_initializer, recurrent_initializer, bias_initializer, unit_forget_bias, kernel_regularizer, recurrent_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, recurrent_constraint, bias_constraint, dropout, recurrent_dropout, seed, return_sequences, return_state, go_backwards, stateful, kwargs)` |
| `src.layers.rnn.conv_lstm2d.keras_export` | Class | `(path)` |
| `src.layers.rnn.conv_lstm3d.ConvLSTM` | Class | `(rank, filters, kernel_size, strides, padding, data_format, dilation_rate, activation, recurrent_activation, use_bias, kernel_initializer, recurrent_initializer, bias_initializer, unit_forget_bias, kernel_regularizer, recurrent_regularizer, bias_regularizer, kernel_constraint, recurrent_constraint, bias_constraint, dropout, recurrent_dropout, seed, return_sequences, return_state, go_backwards, stateful, kwargs)` |
| `src.layers.rnn.conv_lstm3d.ConvLSTM3D` | Class | `(filters, kernel_size, strides, padding, data_format, dilation_rate, activation, recurrent_activation, use_bias, kernel_initializer, recurrent_initializer, bias_initializer, unit_forget_bias, kernel_regularizer, recurrent_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, recurrent_constraint, bias_constraint, dropout, recurrent_dropout, seed, return_sequences, return_state, go_backwards, stateful, kwargs)` |
| `src.layers.rnn.conv_lstm3d.keras_export` | Class | `(path)` |
| `src.layers.rnn.dropout_rnn_cell.DropoutRNNCell` | Class | `(...)` |
| `src.layers.rnn.dropout_rnn_cell.backend` | Object | `` |
| `src.layers.rnn.dropout_rnn_cell.ops` | Object | `` |
| `src.layers.rnn.gru.DropoutRNNCell` | Class | `(...)` |
| `src.layers.rnn.gru.GRU` | Class | `(units, activation, recurrent_activation, use_bias, kernel_initializer, recurrent_initializer, bias_initializer, kernel_regularizer, recurrent_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, recurrent_constraint, bias_constraint, dropout, recurrent_dropout, seed, return_sequences, return_state, go_backwards, stateful, unroll, reset_after, use_cudnn, kwargs)` |
| `src.layers.rnn.gru.GRUCell` | Class | `(units, activation, recurrent_activation, use_bias, kernel_initializer, recurrent_initializer, bias_initializer, kernel_regularizer, recurrent_regularizer, bias_regularizer, kernel_constraint, recurrent_constraint, bias_constraint, dropout, recurrent_dropout, reset_after, seed, kwargs)` |
| `src.layers.rnn.gru.InputSpec` | Class | `(dtype, shape, ndim, max_ndim, min_ndim, axes, allow_last_axis_squeeze, name, optional)` |
| `src.layers.rnn.gru.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.layers.rnn.gru.RNN` | Class | `(cell, return_sequences, return_state, go_backwards, stateful, unroll, zero_output_for_mask, kwargs)` |
| `src.layers.rnn.gru.activations` | Object | `` |
| `src.layers.rnn.gru.backend` | Object | `` |
| `src.layers.rnn.gru.constraints` | Object | `` |
| `src.layers.rnn.gru.initializers` | Object | `` |
| `src.layers.rnn.gru.keras_export` | Class | `(path)` |
| `src.layers.rnn.gru.ops` | Object | `` |
| `src.layers.rnn.gru.regularizers` | Object | `` |
| `src.layers.rnn.gru.tree` | Object | `` |
| `src.layers.rnn.lstm.DropoutRNNCell` | Class | `(...)` |
| `src.layers.rnn.lstm.InputSpec` | Class | `(dtype, shape, ndim, max_ndim, min_ndim, axes, allow_last_axis_squeeze, name, optional)` |
| `src.layers.rnn.lstm.LSTM` | Class | `(units, activation, recurrent_activation, use_bias, kernel_initializer, recurrent_initializer, bias_initializer, unit_forget_bias, kernel_regularizer, recurrent_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, recurrent_constraint, bias_constraint, dropout, recurrent_dropout, seed, return_sequences, return_state, go_backwards, stateful, unroll, use_cudnn, kwargs)` |
| `src.layers.rnn.lstm.LSTMCell` | Class | `(units, activation, recurrent_activation, use_bias, kernel_initializer, recurrent_initializer, bias_initializer, unit_forget_bias, kernel_regularizer, recurrent_regularizer, bias_regularizer, kernel_constraint, recurrent_constraint, bias_constraint, dropout, recurrent_dropout, seed, kwargs)` |
| `src.layers.rnn.lstm.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.layers.rnn.lstm.RNN` | Class | `(cell, return_sequences, return_state, go_backwards, stateful, unroll, zero_output_for_mask, kwargs)` |
| `src.layers.rnn.lstm.activations` | Object | `` |
| `src.layers.rnn.lstm.backend` | Object | `` |
| `src.layers.rnn.lstm.constraints` | Object | `` |
| `src.layers.rnn.lstm.initializers` | Object | `` |
| `src.layers.rnn.lstm.keras_export` | Class | `(path)` |
| `src.layers.rnn.lstm.ops` | Object | `` |
| `src.layers.rnn.lstm.regularizers` | Object | `` |
| `src.layers.rnn.lstm.tree` | Object | `` |
| `src.layers.rnn.rnn.DropoutRNNCell` | Class | `(...)` |
| `src.layers.rnn.rnn.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.layers.rnn.rnn.RNN` | Class | `(cell, return_sequences, return_state, go_backwards, stateful, unroll, zero_output_for_mask, kwargs)` |
| `src.layers.rnn.rnn.StackedRNNCells` | Class | `(cells, kwargs)` |
| `src.layers.rnn.rnn.backend` | Object | `` |
| `src.layers.rnn.rnn.keras_export` | Class | `(path)` |
| `src.layers.rnn.rnn.ops` | Object | `` |
| `src.layers.rnn.rnn.serialization_lib` | Object | `` |
| `src.layers.rnn.rnn.tracking` | Object | `` |
| `src.layers.rnn.rnn.tree` | Object | `` |
| `src.layers.rnn.simple_rnn.DropoutRNNCell` | Class | `(...)` |
| `src.layers.rnn.simple_rnn.InputSpec` | Class | `(dtype, shape, ndim, max_ndim, min_ndim, axes, allow_last_axis_squeeze, name, optional)` |
| `src.layers.rnn.simple_rnn.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.layers.rnn.simple_rnn.RNN` | Class | `(cell, return_sequences, return_state, go_backwards, stateful, unroll, zero_output_for_mask, kwargs)` |
| `src.layers.rnn.simple_rnn.SimpleRNN` | Class | `(units, activation, use_bias, kernel_initializer, recurrent_initializer, bias_initializer, kernel_regularizer, recurrent_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, recurrent_constraint, bias_constraint, dropout, recurrent_dropout, return_sequences, return_state, go_backwards, stateful, unroll, seed, kwargs)` |
| `src.layers.rnn.simple_rnn.SimpleRNNCell` | Class | `(units, activation, use_bias, kernel_initializer, recurrent_initializer, bias_initializer, kernel_regularizer, recurrent_regularizer, bias_regularizer, kernel_constraint, recurrent_constraint, bias_constraint, dropout, recurrent_dropout, seed, kwargs)` |
| `src.layers.rnn.simple_rnn.activations` | Object | `` |
| `src.layers.rnn.simple_rnn.backend` | Object | `` |
| `src.layers.rnn.simple_rnn.constraints` | Object | `` |
| `src.layers.rnn.simple_rnn.initializers` | Object | `` |
| `src.layers.rnn.simple_rnn.keras_export` | Class | `(path)` |
| `src.layers.rnn.simple_rnn.ops` | Object | `` |
| `src.layers.rnn.simple_rnn.regularizers` | Object | `` |
| `src.layers.rnn.stacked_rnn_cells.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.layers.rnn.stacked_rnn_cells.StackedRNNCells` | Class | `(cells, kwargs)` |
| `src.layers.rnn.stacked_rnn_cells.keras_export` | Class | `(path)` |
| `src.layers.rnn.stacked_rnn_cells.ops` | Object | `` |
| `src.layers.rnn.stacked_rnn_cells.serialization_lib` | Object | `` |
| `src.layers.rnn.stacked_rnn_cells.tree` | Object | `` |
| `src.layers.rnn.time_distributed.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.layers.rnn.time_distributed.TimeDistributed` | Class | `(layer, kwargs)` |
| `src.layers.rnn.time_distributed.Wrapper` | Class | `(layer, kwargs)` |
| `src.layers.rnn.time_distributed.backend` | Object | `` |
| `src.layers.rnn.time_distributed.keras_export` | Class | `(path)` |
| `src.layers.rnn.time_distributed.ops` | Object | `` |
| `src.layers.serialization_lib` | Object | `` |
| `src.layers.serialize` | Function | `(layer)` |
| `src.layers.subtract` | Function | `(inputs, kwargs)` |
| `src.legacy.backend.abs` | Function | `(x)` |
| `src.legacy.backend.all` | Function | `(x, axis, keepdims)` |
| `src.legacy.backend.any` | Function | `(x, axis, keepdims)` |
| `src.legacy.backend.arange` | Function | `(start, stop, step, dtype)` |
| `src.legacy.backend.argmax` | Function | `(x, axis)` |
| `src.legacy.backend.argmin` | Function | `(x, axis)` |
| `src.legacy.backend.backend` | Object | `` |
| `src.legacy.backend.batch_dot` | Function | `(x, y, axes)` |
| `src.legacy.backend.batch_flatten` | Function | `(x)` |
| `src.legacy.backend.batch_get_value` | Function | `(tensors)` |
| `src.legacy.backend.batch_normalization` | Function | `(x, mean, var, beta, gamma, axis, epsilon)` |
| `src.legacy.backend.batch_set_value` | Function | `(tuples)` |
| `src.legacy.backend.bias_add` | Function | `(x, bias, data_format)` |
| `src.legacy.backend.binary_crossentropy` | Function | `(target, output, from_logits)` |
| `src.legacy.backend.binary_focal_crossentropy` | Function | `(target, output, apply_class_balancing, alpha, gamma, from_logits)` |
| `src.legacy.backend.cast` | Function | `(x, dtype)` |
| `src.legacy.backend.cast_to_floatx` | Function | `(x)` |
| `src.legacy.backend.categorical_crossentropy` | Function | `(target, output, from_logits, axis)` |
| `src.legacy.backend.categorical_focal_crossentropy` | Function | `(target, output, alpha, gamma, from_logits, axis)` |
| `src.legacy.backend.clip` | Function | `(x, min_value, max_value)` |
| `src.legacy.backend.concatenate` | Function | `(tensors, axis)` |
| `src.legacy.backend.constant` | Function | `(value, dtype, shape, name)` |
| `src.legacy.backend.conv1d` | Function | `(x, kernel, strides, padding, data_format, dilation_rate)` |
| `src.legacy.backend.conv2d` | Function | `(x, kernel, strides, padding, data_format, dilation_rate)` |
| `src.legacy.backend.conv2d_transpose` | Function | `(x, kernel, output_shape, strides, padding, data_format, dilation_rate)` |
| `src.legacy.backend.conv3d` | Function | `(x, kernel, strides, padding, data_format, dilation_rate)` |
| `src.legacy.backend.cos` | Function | `(x)` |
| `src.legacy.backend.count_params` | Function | `(x)` |
| `src.legacy.backend.ctc_batch_cost` | Function | `(y_true, y_pred, input_length, label_length)` |
| `src.legacy.backend.ctc_decode` | Function | `(y_pred, input_length, greedy, beam_width, top_paths)` |
| `src.legacy.backend.ctc_label_dense_to_sparse` | Function | `(labels, label_lengths)` |
| `src.legacy.backend.cumprod` | Function | `(x, axis)` |
| `src.legacy.backend.cumsum` | Function | `(x, axis)` |
| `src.legacy.backend.depthwise_conv2d` | Function | `(x, depthwise_kernel, strides, padding, data_format, dilation_rate)` |
| `src.legacy.backend.dot` | Function | `(x, y)` |
| `src.legacy.backend.dropout` | Function | `(x, level, noise_shape, seed)` |
| `src.legacy.backend.dtype` | Function | `(x)` |
| `src.legacy.backend.elu` | Function | `(x, alpha)` |
| `src.legacy.backend.equal` | Function | `(x, y)` |
| `src.legacy.backend.eval` | Function | `(x)` |
| `src.legacy.backend.exp` | Function | `(x)` |
| `src.legacy.backend.expand_dims` | Function | `(x, axis)` |
| `src.legacy.backend.eye` | Function | `(size, dtype, name)` |
| `src.legacy.backend.flatten` | Function | `(x)` |
| `src.legacy.backend.foldl` | Function | `(fn, elems, initializer, name)` |
| `src.legacy.backend.foldr` | Function | `(fn, elems, initializer, name)` |
| `src.legacy.backend.gather` | Function | `(reference, indices)` |
| `src.legacy.backend.get_value` | Function | `(x)` |
| `src.legacy.backend.gradients` | Function | `(loss, variables)` |
| `src.legacy.backend.greater` | Function | `(x, y)` |
| `src.legacy.backend.greater_equal` | Function | `(x, y)` |
| `src.legacy.backend.hard_sigmoid` | Function | `(x)` |
| `src.legacy.backend.in_top_k` | Function | `(predictions, targets, k)` |
| `src.legacy.backend.int_shape` | Function | `(x)` |
| `src.legacy.backend.is_sparse` | Function | `(tensor)` |
| `src.legacy.backend.keras_export` | Class | `(path)` |
| `src.legacy.backend.l2_normalize` | Function | `(x, axis)` |
| `src.legacy.backend.less` | Function | `(x, y)` |
| `src.legacy.backend.less_equal` | Function | `(x, y)` |
| `src.legacy.backend.log` | Function | `(x)` |
| `src.legacy.backend.map_fn` | Function | `(fn, elems, name, dtype)` |
| `src.legacy.backend.max` | Function | `(x, axis, keepdims)` |
| `src.legacy.backend.maximum` | Function | `(x, y)` |
| `src.legacy.backend.mean` | Function | `(x, axis, keepdims)` |
| `src.legacy.backend.min` | Function | `(x, axis, keepdims)` |
| `src.legacy.backend.minimum` | Function | `(x, y)` |
| `src.legacy.backend.moving_average_update` | Function | `(x, value, momentum)` |
| `src.legacy.backend.name_scope` | Function | `(name)` |
| `src.legacy.backend.ndim` | Function | `(x)` |
| `src.legacy.backend.not_equal` | Function | `(x, y)` |
| `src.legacy.backend.one_hot` | Function | `(indices, num_classes)` |
| `src.legacy.backend.ones` | Function | `(shape, dtype, name)` |
| `src.legacy.backend.ones_like` | Function | `(x, dtype, name)` |
| `src.legacy.backend.permute_dimensions` | Function | `(x, pattern)` |
| `src.legacy.backend.pool2d` | Function | `(x, pool_size, strides, padding, data_format, pool_mode)` |
| `src.legacy.backend.pool3d` | Function | `(x, pool_size, strides, padding, data_format, pool_mode)` |
| `src.legacy.backend.pow` | Function | `(x, a)` |
| `src.legacy.backend.prod` | Function | `(x, axis, keepdims)` |
| `src.legacy.backend.py_all` | Object | `` |
| `src.legacy.backend.py_any` | Object | `` |
| `src.legacy.backend.random_bernoulli` | Function | `(shape, p, dtype, seed)` |
| `src.legacy.backend.random_normal` | Function | `(shape, mean, stddev, dtype, seed)` |
| `src.legacy.backend.random_normal_variable` | Function | `(shape, mean, scale, dtype, name, seed)` |
| `src.legacy.backend.random_uniform` | Function | `(shape, minval, maxval, dtype, seed)` |
| `src.legacy.backend.random_uniform_variable` | Function | `(shape, low, high, dtype, name, seed)` |
| `src.legacy.backend.relu` | Function | `(x, alpha, max_value, threshold)` |
| `src.legacy.backend.repeat` | Function | `(x, n)` |
| `src.legacy.backend.repeat_elements` | Function | `(x, rep, axis)` |
| `src.legacy.backend.reshape` | Function | `(x, shape)` |
| `src.legacy.backend.resize_images` | Function | `(x, height_factor, width_factor, data_format, interpolation)` |
| `src.legacy.backend.resize_volumes` | Function | `(x, depth_factor, height_factor, width_factor, data_format)` |
| `src.legacy.backend.reverse` | Function | `(x, axes)` |
| `src.legacy.backend.rnn` | Function | `(step_function, inputs, initial_states, go_backwards, mask, constants, unroll, input_length, time_major, zero_output_for_mask, return_all_outputs)` |
| `src.legacy.backend.round` | Function | `(x)` |
| `src.legacy.backend.separable_conv2d` | Function | `(x, depthwise_kernel, pointwise_kernel, strides, padding, data_format, dilation_rate)` |
| `src.legacy.backend.set_value` | Function | `(x, value)` |
| `src.legacy.backend.shape` | Function | `(x)` |
| `src.legacy.backend.sigmoid` | Function | `(x)` |
| `src.legacy.backend.sign` | Function | `(x)` |
| `src.legacy.backend.sin` | Function | `(x)` |
| `src.legacy.backend.softmax` | Function | `(x, axis)` |
| `src.legacy.backend.softplus` | Function | `(x)` |
| `src.legacy.backend.softsign` | Function | `(x)` |
| `src.legacy.backend.sparse_categorical_crossentropy` | Function | `(target, output, from_logits, axis, ignore_class)` |
| `src.legacy.backend.spatial_2d_padding` | Function | `(x, padding, data_format)` |
| `src.legacy.backend.spatial_3d_padding` | Function | `(x, padding, data_format)` |
| `src.legacy.backend.sqrt` | Function | `(x)` |
| `src.legacy.backend.square` | Function | `(x)` |
| `src.legacy.backend.squeeze` | Function | `(x, axis)` |
| `src.legacy.backend.stack` | Function | `(x, axis)` |
| `src.legacy.backend.std` | Function | `(x, axis, keepdims)` |
| `src.legacy.backend.stop_gradient` | Function | `(variables)` |
| `src.legacy.backend.sum` | Function | `(x, axis, keepdims)` |
| `src.legacy.backend.switch` | Function | `(condition, then_expression, else_expression)` |
| `src.legacy.backend.tanh` | Function | `(x)` |
| `src.legacy.backend.temporal_padding` | Function | `(x, padding)` |
| `src.legacy.backend.tf` | Object | `` |
| `src.legacy.backend.tile` | Function | `(x, n)` |
| `src.legacy.backend.to_dense` | Function | `(tensor)` |
| `src.legacy.backend.transpose` | Function | `(x)` |
| `src.legacy.backend.truncated_normal` | Function | `(shape, mean, stddev, dtype, seed)` |
| `src.legacy.backend.update` | Function | `(x, new_x)` |
| `src.legacy.backend.update_add` | Function | `(x, increment)` |
| `src.legacy.backend.update_sub` | Function | `(x, decrement)` |
| `src.legacy.backend.var` | Function | `(x, axis, keepdims)` |
| `src.legacy.backend.variable` | Function | `(value, dtype, name, constraint)` |
| `src.legacy.backend.zeros` | Function | `(shape, dtype, name)` |
| `src.legacy.backend.zeros_like` | Function | `(x, dtype, name)` |
| `src.legacy.layers.AlphaDropout` | Class | `(rate, noise_shape, seed, kwargs)` |
| `src.legacy.layers.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.legacy.layers.RandomHeight` | Class | `(factor, interpolation, seed, kwargs)` |
| `src.legacy.layers.RandomWidth` | Class | `(factor, interpolation, seed, kwargs)` |
| `src.legacy.layers.ThresholdedReLU` | Class | `(theta, kwargs)` |
| `src.legacy.layers.backend` | Object | `` |
| `src.legacy.layers.keras_export` | Class | `(path)` |
| `src.legacy.layers.tf` | Object | `` |
| `src.legacy.losses.Reduction` | Class | `(...)` |
| `src.legacy.losses.keras_export` | Class | `(path)` |
| `src.legacy.preprocessing.image.BatchFromFilesMixin` | Class | `(...)` |
| `src.legacy.preprocessing.image.DataFrameIterator` | Class | `(dataframe, directory, image_data_generator, x_col, y_col, weight_col, target_size, color_mode, classes, class_mode, batch_size, shuffle, seed, data_format, save_to_dir, save_prefix, save_format, subset, interpolation, keep_aspect_ratio, dtype, validate_filenames)` |
| `src.legacy.preprocessing.image.DirectoryIterator` | Class | `(directory, image_data_generator, target_size, color_mode, classes, class_mode, batch_size, shuffle, seed, data_format, save_to_dir, save_prefix, save_format, follow_links, subset, interpolation, keep_aspect_ratio, dtype)` |
| `src.legacy.preprocessing.image.ImageDataGenerator` | Class | `(featurewise_center, samplewise_center, featurewise_std_normalization, samplewise_std_normalization, zca_whitening, zca_epsilon, rotation_range, width_shift_range, height_shift_range, brightness_range, shear_range, zoom_range, channel_shift_range, fill_mode, cval, horizontal_flip, vertical_flip, rescale, preprocessing_function, data_format, validation_split, interpolation_order, dtype)` |
| `src.legacy.preprocessing.image.Iterator` | Class | `(n, batch_size, shuffle, seed, kwargs)` |
| `src.legacy.preprocessing.image.NumpyArrayIterator` | Class | `(x, y, image_data_generator, batch_size, shuffle, sample_weight, seed, data_format, save_to_dir, save_prefix, save_format, subset, ignore_class_split, dtype)` |
| `src.legacy.preprocessing.image.PyDataset` | Class | `(workers, use_multiprocessing, max_queue_size)` |
| `src.legacy.preprocessing.image.apply_affine_transform` | Function | `(x, theta, tx, ty, shear, zx, zy, row_axis, col_axis, channel_axis, fill_mode, cval, order)` |
| `src.legacy.preprocessing.image.apply_brightness_shift` | Function | `(x, brightness, scale)` |
| `src.legacy.preprocessing.image.apply_channel_shift` | Function | `(x, intensity, channel_axis)` |
| `src.legacy.preprocessing.image.backend` | Object | `` |
| `src.legacy.preprocessing.image.flip_axis` | Function | `(x, axis)` |
| `src.legacy.preprocessing.image.image_utils` | Object | `` |
| `src.legacy.preprocessing.image.io_utils` | Object | `` |
| `src.legacy.preprocessing.image.keras_export` | Class | `(path)` |
| `src.legacy.preprocessing.image.random_brightness` | Function | `(x, brightness_range, scale)` |
| `src.legacy.preprocessing.image.random_channel_shift` | Function | `(x, intensity_range, channel_axis)` |
| `src.legacy.preprocessing.image.random_rotation` | Function | `(x, rg, row_axis, col_axis, channel_axis, fill_mode, cval, interpolation_order)` |
| `src.legacy.preprocessing.image.random_shear` | Function | `(x, intensity, row_axis, col_axis, channel_axis, fill_mode, cval, interpolation_order)` |
| `src.legacy.preprocessing.image.random_shift` | Function | `(x, wrg, hrg, row_axis, col_axis, channel_axis, fill_mode, cval, interpolation_order)` |
| `src.legacy.preprocessing.image.random_zoom` | Function | `(x, zoom_range, row_axis, col_axis, channel_axis, fill_mode, cval, interpolation_order)` |
| `src.legacy.preprocessing.image.scipy` | Object | `` |
| `src.legacy.preprocessing.image.transform_matrix_offset_center` | Function | `(matrix, x, y)` |
| `src.legacy.preprocessing.image.validate_filename` | Function | `(filename, white_list_formats)` |
| `src.legacy.preprocessing.sequence.PyDataset` | Class | `(workers, use_multiprocessing, max_queue_size)` |
| `src.legacy.preprocessing.sequence.TimeseriesGenerator` | Class | `(data, targets, length, sampling_rate, stride, start_index, end_index, shuffle, reverse, batch_size, kwargs)` |
| `src.legacy.preprocessing.sequence.keras_export` | Class | `(path)` |
| `src.legacy.preprocessing.sequence.make_sampling_table` | Function | `(size, sampling_factor)` |
| `src.legacy.preprocessing.sequence.skipgrams` | Function | `(sequence, vocabulary_size, window_size, negative_samples, shuffle, categorical, sampling_table, seed)` |
| `src.legacy.preprocessing.text.Tokenizer` | Class | `(num_words, filters, lower, split, char_level, oov_token, analyzer, kwargs)` |
| `src.legacy.preprocessing.text.hashing_trick` | Function | `(text, n, hash_function, filters, lower, split, analyzer)` |
| `src.legacy.preprocessing.text.keras_export` | Class | `(path)` |
| `src.legacy.preprocessing.text.one_hot` | Function | `(input_text, n, filters, lower, split, analyzer)` |
| `src.legacy.preprocessing.text.text_to_word_sequence` | Function | `(input_text, filters, lower, split)` |
| `src.legacy.preprocessing.text.tokenizer_from_json` | Function | `(json_string)` |
| `src.legacy.saving.json_utils.Encoder` | Class | `(...)` |
| `src.legacy.saving.json_utils.decode` | Function | `(json_string)` |
| `src.legacy.saving.json_utils.decode_and_deserialize` | Function | `(json_string, module_objects, custom_objects)` |
| `src.legacy.saving.json_utils.get_json_type` | Function | `(obj)` |
| `src.legacy.saving.json_utils.serialization` | Object | `` |
| `src.legacy.saving.json_utils.serialization_lib` | Object | `` |
| `src.legacy.saving.json_utils.tf` | Object | `` |
| `src.legacy.saving.legacy_h5_format.HDF5_OBJECT_HEADER_LIMIT` | Object | `` |
| `src.legacy.saving.legacy_h5_format.backend` | Object | `` |
| `src.legacy.saving.legacy_h5_format.global_state` | Object | `` |
| `src.legacy.saving.legacy_h5_format.io_utils` | Object | `` |
| `src.legacy.saving.legacy_h5_format.json_utils` | Object | `` |
| `src.legacy.saving.legacy_h5_format.load_attributes_from_hdf5_group` | Function | `(group, name)` |
| `src.legacy.saving.legacy_h5_format.load_model_from_hdf5` | Function | `(filepath, custom_objects, compile, safe_mode)` |
| `src.legacy.saving.legacy_h5_format.load_optimizer_weights_from_hdf5_group` | Function | `(hdf5_group)` |
| `src.legacy.saving.legacy_h5_format.load_subset_weights_from_hdf5_group` | Function | `(f)` |
| `src.legacy.saving.legacy_h5_format.load_weights_from_hdf5_group` | Function | `(f, model, skip_mismatch)` |
| `src.legacy.saving.legacy_h5_format.load_weights_from_hdf5_group_by_name` | Function | `(f, model, skip_mismatch)` |
| `src.legacy.saving.legacy_h5_format.object_registration` | Object | `` |
| `src.legacy.saving.legacy_h5_format.save_attributes_to_hdf5_group` | Function | `(group, name, data)` |
| `src.legacy.saving.legacy_h5_format.save_model_to_hdf5` | Function | `(model, filepath, overwrite, include_optimizer)` |
| `src.legacy.saving.legacy_h5_format.save_optimizer_weights_to_hdf5_group` | Function | `(hdf5_group, optimizer)` |
| `src.legacy.saving.legacy_h5_format.save_subset_weights_to_hdf5_group` | Function | `(f, weights)` |
| `src.legacy.saving.legacy_h5_format.save_weights_to_hdf5_group` | Function | `(f, model)` |
| `src.legacy.saving.legacy_h5_format.saving_options` | Object | `` |
| `src.legacy.saving.legacy_h5_format.saving_utils` | Object | `` |
| `src.legacy.saving.legacy_h5_format.serialization_lib` | Object | `` |
| `src.legacy.saving.saving_options.global_state` | Object | `` |
| `src.legacy.saving.saving_options.keras_option_scope` | Function | `(use_legacy_config)` |
| `src.legacy.saving.saving_utils.LAMBDA_DEP_ARGS` | Object | `` |
| `src.legacy.saving.saving_utils.MODULE_OBJECTS` | Object | `` |
| `src.legacy.saving.saving_utils.backend` | Object | `` |
| `src.legacy.saving.saving_utils.compile_args_from_training_config` | Function | `(training_config, custom_objects)` |
| `src.legacy.saving.saving_utils.losses` | Object | `` |
| `src.legacy.saving.saving_utils.metrics_module` | Object | `` |
| `src.legacy.saving.saving_utils.model_from_config` | Function | `(config, custom_objects)` |
| `src.legacy.saving.saving_utils.model_metadata` | Function | `(model, include_optimizer, require_config)` |
| `src.legacy.saving.saving_utils.object_registration` | Object | `` |
| `src.legacy.saving.saving_utils.serialization` | Object | `` |
| `src.legacy.saving.saving_utils.tree` | Object | `` |
| `src.legacy.saving.saving_utils.try_build_compiled_arguments` | Function | `(model)` |
| `src.legacy.saving.serialization.DisableSharedObjectScope` | Class | `(...)` |
| `src.legacy.saving.serialization.NoopLoadingScope` | Class | `(...)` |
| `src.legacy.saving.serialization.SHARED_OBJECT_DISABLED` | Object | `` |
| `src.legacy.saving.serialization.SHARED_OBJECT_KEY` | Object | `` |
| `src.legacy.saving.serialization.SHARED_OBJECT_LOADING` | Object | `` |
| `src.legacy.saving.serialization.SHARED_OBJECT_SAVING` | Object | `` |
| `src.legacy.saving.serialization.SharedObjectConfig` | Class | `(base_config, object_id, kwargs)` |
| `src.legacy.saving.serialization.SharedObjectLoadingScope` | Class | `(...)` |
| `src.legacy.saving.serialization.SharedObjectSavingScope` | Class | `(...)` |
| `src.legacy.saving.serialization.class_and_config_for_serialized_keras_object` | Function | `(config, module_objects, custom_objects, printable_module_name)` |
| `src.legacy.saving.serialization.deserialize_keras_object` | Function | `(identifier, module_objects, custom_objects, printable_module_name)` |
| `src.legacy.saving.serialization.is_default` | Function | `(method)` |
| `src.legacy.saving.serialization.keras_export` | Class | `(path)` |
| `src.legacy.saving.serialization.object_registration` | Object | `` |
| `src.legacy.saving.serialization.serialize_keras_class_and_config` | Function | `(cls_name, cls_config, obj, shared_object_id)` |
| `src.legacy.saving.serialization.serialize_keras_object` | Function | `(instance)` |
| `src.legacy.saving.serialization.skip_failed_serialization` | Function | `()` |
| `src.legacy.saving.serialization.validate_config` | Function | `(config)` |
| `src.losses.ALL_OBJECTS` | Object | `` |
| `src.losses.ALL_OBJECTS_DICT` | Object | `` |
| `src.losses.BinaryCrossentropy` | Class | `(from_logits, label_smoothing, axis, reduction, name, dtype)` |
| `src.losses.BinaryFocalCrossentropy` | Class | `(apply_class_balancing, alpha, gamma, from_logits, label_smoothing, axis, reduction, name, dtype)` |
| `src.losses.CTC` | Class | `(reduction, name, dtype)` |
| `src.losses.CategoricalCrossentropy` | Class | `(from_logits, label_smoothing, axis, reduction, name, dtype)` |
| `src.losses.CategoricalFocalCrossentropy` | Class | `(alpha, gamma, from_logits, label_smoothing, axis, reduction, name, dtype)` |
| `src.losses.CategoricalHinge` | Class | `(reduction, name, dtype)` |
| `src.losses.Circle` | Class | `(gamma, margin, remove_diagonal, reduction, name, dtype)` |
| `src.losses.CosineSimilarity` | Class | `(axis, reduction, name, dtype)` |
| `src.losses.Dice` | Class | `(reduction, name, axis, dtype)` |
| `src.losses.Hinge` | Class | `(reduction, name, dtype)` |
| `src.losses.Huber` | Class | `(delta, reduction, name, dtype)` |
| `src.losses.KLDivergence` | Class | `(reduction, name, dtype)` |
| `src.losses.LogCosh` | Class | `(reduction, name, dtype)` |
| `src.losses.Loss` | Class | `(name, reduction, dtype)` |
| `src.losses.LossFunctionWrapper` | Class | `(fn, reduction, name, dtype, kwargs)` |
| `src.losses.MeanAbsoluteError` | Class | `(reduction, name, dtype)` |
| `src.losses.MeanAbsolutePercentageError` | Class | `(reduction, name, dtype)` |
| `src.losses.MeanSquaredError` | Class | `(reduction, name, dtype)` |
| `src.losses.MeanSquaredLogarithmicError` | Class | `(reduction, name, dtype)` |
| `src.losses.Poisson` | Class | `(reduction, name, dtype)` |
| `src.losses.SparseCategoricalCrossentropy` | Class | `(from_logits, ignore_class, reduction, axis, name, dtype)` |
| `src.losses.SquaredHinge` | Class | `(reduction, name, dtype)` |
| `src.losses.Tversky` | Class | `(alpha, beta, reduction, name, axis, dtype)` |
| `src.losses.binary_crossentropy` | Function | `(y_true, y_pred, from_logits, label_smoothing, axis)` |
| `src.losses.binary_focal_crossentropy` | Function | `(y_true, y_pred, apply_class_balancing, alpha, gamma, from_logits, label_smoothing, axis)` |
| `src.losses.categorical_crossentropy` | Function | `(y_true, y_pred, from_logits, label_smoothing, axis)` |
| `src.losses.categorical_focal_crossentropy` | Function | `(y_true, y_pred, alpha, gamma, from_logits, label_smoothing, axis)` |
| `src.losses.categorical_hinge` | Function | `(y_true, y_pred)` |
| `src.losses.circle` | Function | `(y_true, y_pred, ref_labels, ref_embeddings, remove_diagonal, gamma, margin)` |
| `src.losses.cosine_similarity` | Function | `(y_true, y_pred, axis)` |
| `src.losses.ctc` | Function | `(y_true, y_pred)` |
| `src.losses.deserialize` | Function | `(name, custom_objects)` |
| `src.losses.dice` | Function | `(y_true, y_pred, axis)` |
| `src.losses.get` | Function | `(identifier)` |
| `src.losses.hinge` | Function | `(y_true, y_pred)` |
| `src.losses.huber` | Function | `(y_true, y_pred, delta)` |
| `src.losses.keras_export` | Class | `(path)` |
| `src.losses.kl_divergence` | Function | `(y_true, y_pred)` |
| `src.losses.log_cosh` | Function | `(y_true, y_pred)` |
| `src.losses.loss.KerasSaveable` | Class | `(...)` |
| `src.losses.loss.Loss` | Class | `(name, reduction, dtype)` |
| `src.losses.loss.apply_mask` | Function | `(sample_weight, mask, dtype, reduction)` |
| `src.losses.loss.auto_name` | Function | `(prefix)` |
| `src.losses.loss.backend` | Object | `` |
| `src.losses.loss.dtype_policies` | Object | `` |
| `src.losses.loss.keras_export` | Class | `(path)` |
| `src.losses.loss.ops` | Object | `` |
| `src.losses.loss.reduce_values` | Function | `(values, sample_weight, reduction)` |
| `src.losses.loss.reduce_weighted_values` | Function | `(values, sample_weight, mask, reduction, dtype)` |
| `src.losses.loss.scale_loss_for_distribution` | Function | `(value)` |
| `src.losses.loss.squeeze_or_expand_to_same_rank` | Function | `(x1, x2, expand_rank_1)` |
| `src.losses.loss.standardize_reduction` | Function | `(reduction)` |
| `src.losses.loss.tree` | Object | `` |
| `src.losses.loss.unscale_loss_for_distribution` | Function | `(value)` |
| `src.losses.losses.BinaryCrossentropy` | Class | `(from_logits, label_smoothing, axis, reduction, name, dtype)` |
| `src.losses.losses.BinaryFocalCrossentropy` | Class | `(apply_class_balancing, alpha, gamma, from_logits, label_smoothing, axis, reduction, name, dtype)` |
| `src.losses.losses.CTC` | Class | `(reduction, name, dtype)` |
| `src.losses.losses.CategoricalCrossentropy` | Class | `(from_logits, label_smoothing, axis, reduction, name, dtype)` |
| `src.losses.losses.CategoricalFocalCrossentropy` | Class | `(alpha, gamma, from_logits, label_smoothing, axis, reduction, name, dtype)` |
| `src.losses.losses.CategoricalGeneralizedCrossEntropy` | Class | `(q, reduction, name, dtype)` |
| `src.losses.losses.CategoricalHinge` | Class | `(reduction, name, dtype)` |
| `src.losses.losses.Circle` | Class | `(gamma, margin, remove_diagonal, reduction, name, dtype)` |
| `src.losses.losses.CosineSimilarity` | Class | `(axis, reduction, name, dtype)` |
| `src.losses.losses.Dice` | Class | `(reduction, name, axis, dtype)` |
| `src.losses.losses.Hinge` | Class | `(reduction, name, dtype)` |
| `src.losses.losses.Huber` | Class | `(delta, reduction, name, dtype)` |
| `src.losses.losses.KLDivergence` | Class | `(reduction, name, dtype)` |
| `src.losses.losses.LogCosh` | Class | `(reduction, name, dtype)` |
| `src.losses.losses.Loss` | Class | `(name, reduction, dtype)` |
| `src.losses.losses.LossFunctionWrapper` | Class | `(fn, reduction, name, dtype, kwargs)` |
| `src.losses.losses.MeanAbsoluteError` | Class | `(reduction, name, dtype)` |
| `src.losses.losses.MeanAbsolutePercentageError` | Class | `(reduction, name, dtype)` |
| `src.losses.losses.MeanSquaredError` | Class | `(reduction, name, dtype)` |
| `src.losses.losses.MeanSquaredLogarithmicError` | Class | `(reduction, name, dtype)` |
| `src.losses.losses.Poisson` | Class | `(reduction, name, dtype)` |
| `src.losses.losses.SparseCategoricalCrossentropy` | Class | `(from_logits, ignore_class, reduction, axis, name, dtype)` |
| `src.losses.losses.SquaredHinge` | Class | `(reduction, name, dtype)` |
| `src.losses.losses.Tversky` | Class | `(alpha, beta, reduction, name, axis, dtype)` |
| `src.losses.losses.backend` | Object | `` |
| `src.losses.losses.binary_crossentropy` | Function | `(y_true, y_pred, from_logits, label_smoothing, axis)` |
| `src.losses.losses.binary_focal_crossentropy` | Function | `(y_true, y_pred, apply_class_balancing, alpha, gamma, from_logits, label_smoothing, axis)` |
| `src.losses.losses.build_pos_neg_masks` | Function | `(query_labels, key_labels, remove_diagonal)` |
| `src.losses.losses.categorical_crossentropy` | Function | `(y_true, y_pred, from_logits, label_smoothing, axis)` |
| `src.losses.losses.categorical_focal_crossentropy` | Function | `(y_true, y_pred, alpha, gamma, from_logits, label_smoothing, axis)` |
| `src.losses.losses.categorical_generalized_cross_entropy` | Function | `(y_true, y_pred, q)` |
| `src.losses.losses.categorical_hinge` | Function | `(y_true, y_pred)` |
| `src.losses.losses.circle` | Function | `(y_true, y_pred, ref_labels, ref_embeddings, remove_diagonal, gamma, margin)` |
| `src.losses.losses.convert_binary_labels_to_hinge` | Function | `(y_true)` |
| `src.losses.losses.cosine_similarity` | Function | `(y_true, y_pred, axis)` |
| `src.losses.losses.ctc` | Function | `(y_true, y_pred)` |
| `src.losses.losses.dice` | Function | `(y_true, y_pred, axis)` |
| `src.losses.losses.hinge` | Function | `(y_true, y_pred)` |
| `src.losses.losses.huber` | Function | `(y_true, y_pred, delta)` |
| `src.losses.losses.keras_export` | Class | `(path)` |
| `src.losses.losses.kl_divergence` | Function | `(y_true, y_pred)` |
| `src.losses.losses.log_cosh` | Function | `(y_true, y_pred)` |
| `src.losses.losses.mean_absolute_error` | Function | `(y_true, y_pred)` |
| `src.losses.losses.mean_absolute_percentage_error` | Function | `(y_true, y_pred)` |
| `src.losses.losses.mean_squared_error` | Function | `(y_true, y_pred)` |
| `src.losses.losses.mean_squared_logarithmic_error` | Function | `(y_true, y_pred)` |
| `src.losses.losses.normalize` | Function | `(x, axis, order)` |
| `src.losses.losses.ops` | Object | `` |
| `src.losses.losses.poisson` | Function | `(y_true, y_pred)` |
| `src.losses.losses.serialization_lib` | Object | `` |
| `src.losses.losses.sparse_categorical_crossentropy` | Function | `(y_true, y_pred, from_logits, ignore_class, axis)` |
| `src.losses.losses.squared_hinge` | Function | `(y_true, y_pred)` |
| `src.losses.losses.squeeze_or_expand_to_same_rank` | Function | `(x1, x2, expand_rank_1)` |
| `src.losses.losses.tree` | Object | `` |
| `src.losses.losses.tversky` | Function | `(y_true, y_pred, alpha, beta, axis)` |
| `src.losses.mean_absolute_error` | Function | `(y_true, y_pred)` |
| `src.losses.mean_absolute_percentage_error` | Function | `(y_true, y_pred)` |
| `src.losses.mean_squared_error` | Function | `(y_true, y_pred)` |
| `src.losses.mean_squared_logarithmic_error` | Function | `(y_true, y_pred)` |
| `src.losses.poisson` | Function | `(y_true, y_pred)` |
| `src.losses.serialization_lib` | Object | `` |
| `src.losses.serialize` | Function | `(loss)` |
| `src.losses.sparse_categorical_crossentropy` | Function | `(y_true, y_pred, from_logits, ignore_class, axis)` |
| `src.losses.squared_hinge` | Function | `(y_true, y_pred)` |
| `src.losses.tversky` | Function | `(y_true, y_pred, alpha, beta, axis)` |
| `src.metrics.ALL_OBJECTS` | Object | `` |
| `src.metrics.ALL_OBJECTS_DICT` | Object | `` |
| `src.metrics.AUC` | Class | `(num_thresholds, curve, summation_method, name, dtype, thresholds, multi_label, num_labels, label_weights, from_logits)` |
| `src.metrics.Accuracy` | Class | `(name, dtype)` |
| `src.metrics.BinaryAccuracy` | Class | `(name, dtype, threshold)` |
| `src.metrics.BinaryCrossentropy` | Class | `(name, dtype, from_logits, label_smoothing)` |
| `src.metrics.BinaryIoU` | Class | `(target_class_ids, threshold, name, dtype)` |
| `src.metrics.CategoricalAccuracy` | Class | `(name, dtype)` |
| `src.metrics.CategoricalCrossentropy` | Class | `(name, dtype, from_logits, label_smoothing, axis)` |
| `src.metrics.CategoricalHinge` | Class | `(name, dtype)` |
| `src.metrics.ConcordanceCorrelation` | Class | `(name, dtype, axis)` |
| `src.metrics.CosineSimilarity` | Class | `(name, dtype, axis)` |
| `src.metrics.F1Score` | Class | `(average, threshold, name, dtype)` |
| `src.metrics.FBetaScore` | Class | `(average, beta, threshold, name, dtype)` |
| `src.metrics.FalseNegatives` | Class | `(thresholds, name, dtype)` |
| `src.metrics.FalsePositives` | Class | `(thresholds, name, dtype)` |
| `src.metrics.Hinge` | Class | `(name, dtype)` |
| `src.metrics.IoU` | Class | `(num_classes, target_class_ids, name, dtype, ignore_class, sparse_y_true, sparse_y_pred, axis)` |
| `src.metrics.KLDivergence` | Class | `(name, dtype)` |
| `src.metrics.LogCoshError` | Class | `(name, dtype)` |
| `src.metrics.Mean` | Class | `(name, dtype)` |
| `src.metrics.MeanAbsoluteError` | Class | `(name, dtype)` |
| `src.metrics.MeanAbsolutePercentageError` | Class | `(name, dtype)` |
| `src.metrics.MeanIoU` | Class | `(num_classes, name, dtype, ignore_class, sparse_y_true, sparse_y_pred, axis)` |
| `src.metrics.MeanMetricWrapper` | Class | `(fn, name, dtype, kwargs)` |
| `src.metrics.MeanSquaredError` | Class | `(name, dtype)` |
| `src.metrics.MeanSquaredLogarithmicError` | Class | `(name, dtype)` |
| `src.metrics.Metric` | Class | `(dtype, name)` |
| `src.metrics.OneHotIoU` | Class | `(num_classes, target_class_ids, name, dtype, ignore_class, sparse_y_pred, axis)` |
| `src.metrics.OneHotMeanIoU` | Class | `(num_classes, name, dtype, ignore_class, sparse_y_pred, axis)` |
| `src.metrics.PearsonCorrelation` | Class | `(name, dtype, axis)` |
| `src.metrics.Poisson` | Class | `(name, dtype)` |
| `src.metrics.Precision` | Class | `(thresholds, top_k, class_id, name, dtype)` |
| `src.metrics.PrecisionAtRecall` | Class | `(recall, num_thresholds, class_id, name, dtype)` |
| `src.metrics.R2Score` | Class | `(class_aggregation, num_regressors, name, dtype)` |
| `src.metrics.Recall` | Class | `(thresholds, top_k, class_id, name, dtype)` |
| `src.metrics.RecallAtPrecision` | Class | `(precision, num_thresholds, class_id, name, dtype)` |
| `src.metrics.RootMeanSquaredError` | Class | `(name, dtype)` |
| `src.metrics.SensitivityAtSpecificity` | Class | `(specificity, num_thresholds, class_id, name, dtype)` |
| `src.metrics.SparseCategoricalAccuracy` | Class | `(name, dtype)` |
| `src.metrics.SparseCategoricalCrossentropy` | Class | `(name, dtype, from_logits, axis)` |
| `src.metrics.SparseTopKCategoricalAccuracy` | Class | `(k, name, dtype, from_sorted_ids)` |
| `src.metrics.SpecificityAtSensitivity` | Class | `(sensitivity, num_thresholds, class_id, name, dtype)` |
| `src.metrics.SquaredHinge` | Class | `(name, dtype)` |
| `src.metrics.Sum` | Class | `(name, dtype)` |
| `src.metrics.TopKCategoricalAccuracy` | Class | `(k, name, dtype)` |
| `src.metrics.TrueNegatives` | Class | `(thresholds, name, dtype)` |
| `src.metrics.TruePositives` | Class | `(thresholds, name, dtype)` |
| `src.metrics.accuracy_metrics.Accuracy` | Class | `(name, dtype)` |
| `src.metrics.accuracy_metrics.BinaryAccuracy` | Class | `(name, dtype, threshold)` |
| `src.metrics.accuracy_metrics.CategoricalAccuracy` | Class | `(name, dtype)` |
| `src.metrics.accuracy_metrics.SparseCategoricalAccuracy` | Class | `(name, dtype)` |
| `src.metrics.accuracy_metrics.SparseTopKCategoricalAccuracy` | Class | `(k, name, dtype, from_sorted_ids)` |
| `src.metrics.accuracy_metrics.TopKCategoricalAccuracy` | Class | `(k, name, dtype)` |
| `src.metrics.accuracy_metrics.accuracy` | Function | `(y_true, y_pred)` |
| `src.metrics.accuracy_metrics.backend` | Object | `` |
| `src.metrics.accuracy_metrics.binary_accuracy` | Function | `(y_true, y_pred, threshold)` |
| `src.metrics.accuracy_metrics.categorical_accuracy` | Function | `(y_true, y_pred)` |
| `src.metrics.accuracy_metrics.keras_export` | Class | `(path)` |
| `src.metrics.accuracy_metrics.ops` | Object | `` |
| `src.metrics.accuracy_metrics.reduction_metrics` | Object | `` |
| `src.metrics.accuracy_metrics.sparse_categorical_accuracy` | Function | `(y_true, y_pred)` |
| `src.metrics.accuracy_metrics.sparse_top_k_categorical_accuracy` | Function | `(y_true, y_pred, k, from_sorted_ids)` |
| `src.metrics.accuracy_metrics.squeeze_or_expand_to_same_rank` | Function | `(x1, x2, expand_rank_1)` |
| `src.metrics.accuracy_metrics.top_k_categorical_accuracy` | Function | `(y_true, y_pred, k)` |
| `src.metrics.confusion_metrics.AUC` | Class | `(num_thresholds, curve, summation_method, name, dtype, thresholds, multi_label, num_labels, label_weights, from_logits)` |
| `src.metrics.confusion_metrics.FalseNegatives` | Class | `(thresholds, name, dtype)` |
| `src.metrics.confusion_metrics.FalsePositives` | Class | `(thresholds, name, dtype)` |
| `src.metrics.confusion_metrics.Metric` | Class | `(dtype, name)` |
| `src.metrics.confusion_metrics.Precision` | Class | `(thresholds, top_k, class_id, name, dtype)` |
| `src.metrics.confusion_metrics.PrecisionAtRecall` | Class | `(recall, num_thresholds, class_id, name, dtype)` |
| `src.metrics.confusion_metrics.Recall` | Class | `(thresholds, top_k, class_id, name, dtype)` |
| `src.metrics.confusion_metrics.RecallAtPrecision` | Class | `(precision, num_thresholds, class_id, name, dtype)` |
| `src.metrics.confusion_metrics.SensitivityAtSpecificity` | Class | `(specificity, num_thresholds, class_id, name, dtype)` |
| `src.metrics.confusion_metrics.SensitivitySpecificityBase` | Class | `(value, num_thresholds, class_id, name, dtype)` |
| `src.metrics.confusion_metrics.SpecificityAtSensitivity` | Class | `(sensitivity, num_thresholds, class_id, name, dtype)` |
| `src.metrics.confusion_metrics.TrueNegatives` | Class | `(thresholds, name, dtype)` |
| `src.metrics.confusion_metrics.TruePositives` | Class | `(thresholds, name, dtype)` |
| `src.metrics.confusion_metrics.activations` | Object | `` |
| `src.metrics.confusion_metrics.backend` | Object | `` |
| `src.metrics.confusion_metrics.initializers` | Object | `` |
| `src.metrics.confusion_metrics.keras_export` | Class | `(path)` |
| `src.metrics.confusion_metrics.metrics_utils` | Object | `` |
| `src.metrics.confusion_metrics.ops` | Object | `` |
| `src.metrics.confusion_metrics.to_list` | Function | `(x)` |
| `src.metrics.correlation_metrics.ConcordanceCorrelation` | Class | `(name, dtype, axis)` |
| `src.metrics.correlation_metrics.PearsonCorrelation` | Class | `(name, dtype, axis)` |
| `src.metrics.correlation_metrics.backend` | Object | `` |
| `src.metrics.correlation_metrics.concordance_correlation` | Function | `(y_true, y_pred, axis)` |
| `src.metrics.correlation_metrics.keras_export` | Class | `(path)` |
| `src.metrics.correlation_metrics.ops` | Object | `` |
| `src.metrics.correlation_metrics.pearson_correlation` | Function | `(y_true, y_pred, axis)` |
| `src.metrics.correlation_metrics.reduction_metrics` | Object | `` |
| `src.metrics.correlation_metrics.squeeze_or_expand_to_same_rank` | Function | `(x1, x2, expand_rank_1)` |
| `src.metrics.deserialize` | Function | `(config, custom_objects)` |
| `src.metrics.f_score_metrics.F1Score` | Class | `(average, threshold, name, dtype)` |
| `src.metrics.f_score_metrics.FBetaScore` | Class | `(average, beta, threshold, name, dtype)` |
| `src.metrics.f_score_metrics.Metric` | Class | `(dtype, name)` |
| `src.metrics.f_score_metrics.backend` | Object | `` |
| `src.metrics.f_score_metrics.initializers` | Object | `` |
| `src.metrics.f_score_metrics.keras_export` | Class | `(path)` |
| `src.metrics.f_score_metrics.ops` | Object | `` |
| `src.metrics.get` | Function | `(identifier)` |
| `src.metrics.hinge_metrics.CategoricalHinge` | Class | `(name, dtype)` |
| `src.metrics.hinge_metrics.Hinge` | Class | `(name, dtype)` |
| `src.metrics.hinge_metrics.SquaredHinge` | Class | `(name, dtype)` |
| `src.metrics.hinge_metrics.categorical_hinge` | Function | `(y_true, y_pred)` |
| `src.metrics.hinge_metrics.hinge` | Function | `(y_true, y_pred)` |
| `src.metrics.hinge_metrics.keras_export` | Class | `(path)` |
| `src.metrics.hinge_metrics.reduction_metrics` | Object | `` |
| `src.metrics.hinge_metrics.squared_hinge` | Function | `(y_true, y_pred)` |
| `src.metrics.iou_metrics.BinaryIoU` | Class | `(target_class_ids, threshold, name, dtype)` |
| `src.metrics.iou_metrics.IoU` | Class | `(num_classes, target_class_ids, name, dtype, ignore_class, sparse_y_true, sparse_y_pred, axis)` |
| `src.metrics.iou_metrics.MeanIoU` | Class | `(num_classes, name, dtype, ignore_class, sparse_y_true, sparse_y_pred, axis)` |
| `src.metrics.iou_metrics.Metric` | Class | `(dtype, name)` |
| `src.metrics.iou_metrics.OneHotIoU` | Class | `(num_classes, target_class_ids, name, dtype, ignore_class, sparse_y_pred, axis)` |
| `src.metrics.iou_metrics.OneHotMeanIoU` | Class | `(num_classes, name, dtype, ignore_class, sparse_y_pred, axis)` |
| `src.metrics.iou_metrics.backend` | Object | `` |
| `src.metrics.iou_metrics.confusion_matrix` | Function | `(labels, predictions, num_classes, weights, dtype)` |
| `src.metrics.iou_metrics.initializers` | Object | `` |
| `src.metrics.iou_metrics.keras_export` | Class | `(path)` |
| `src.metrics.iou_metrics.ops` | Object | `` |
| `src.metrics.keras_export` | Class | `(path)` |
| `src.metrics.metric.KerasSaveable` | Class | `(...)` |
| `src.metrics.metric.Metric` | Class | `(dtype, name)` |
| `src.metrics.metric.Tracker` | Class | `(config, exclusions)` |
| `src.metrics.metric.auto_name` | Function | `(prefix)` |
| `src.metrics.metric.backend` | Object | `` |
| `src.metrics.metric.dtype_policies` | Object | `` |
| `src.metrics.metric.initializers` | Object | `` |
| `src.metrics.metric.keras_export` | Class | `(path)` |
| `src.metrics.metric.ops` | Object | `` |
| `src.metrics.metrics_utils.AUCCurve` | Class | `(...)` |
| `src.metrics.metrics_utils.AUCSummationMethod` | Class | `(...)` |
| `src.metrics.metrics_utils.ConfusionMatrix` | Class | `(...)` |
| `src.metrics.metrics_utils.NEG_INF` | Object | `` |
| `src.metrics.metrics_utils.assert_thresholds_range` | Function | `(thresholds)` |
| `src.metrics.metrics_utils.backend` | Object | `` |
| `src.metrics.metrics_utils.confusion_matrix` | Function | `(labels, predictions, num_classes, weights, dtype)` |
| `src.metrics.metrics_utils.is_evenly_distributed_thresholds` | Function | `(thresholds)` |
| `src.metrics.metrics_utils.ops` | Object | `` |
| `src.metrics.metrics_utils.parse_init_thresholds` | Function | `(thresholds, default_threshold)` |
| `src.metrics.metrics_utils.squeeze_or_expand_to_same_rank` | Function | `(x1, x2, expand_rank_1)` |
| `src.metrics.metrics_utils.to_list` | Function | `(x)` |
| `src.metrics.metrics_utils.update_confusion_matrix_variables` | Function | `(variables_to_update, y_true, y_pred, thresholds, top_k, class_id, sample_weight, multi_label, label_weights, thresholds_distributed_evenly)` |
| `src.metrics.probabilistic_metrics.BinaryCrossentropy` | Class | `(name, dtype, from_logits, label_smoothing)` |
| `src.metrics.probabilistic_metrics.CategoricalCrossentropy` | Class | `(name, dtype, from_logits, label_smoothing, axis)` |
| `src.metrics.probabilistic_metrics.KLDivergence` | Class | `(name, dtype)` |
| `src.metrics.probabilistic_metrics.Poisson` | Class | `(name, dtype)` |
| `src.metrics.probabilistic_metrics.SparseCategoricalCrossentropy` | Class | `(name, dtype, from_logits, axis)` |
| `src.metrics.probabilistic_metrics.binary_crossentropy` | Function | `(y_true, y_pred, from_logits, label_smoothing, axis)` |
| `src.metrics.probabilistic_metrics.categorical_crossentropy` | Function | `(y_true, y_pred, from_logits, label_smoothing, axis)` |
| `src.metrics.probabilistic_metrics.keras_export` | Class | `(path)` |
| `src.metrics.probabilistic_metrics.kl_divergence` | Function | `(y_true, y_pred)` |
| `src.metrics.probabilistic_metrics.poisson` | Function | `(y_true, y_pred)` |
| `src.metrics.probabilistic_metrics.reduction_metrics` | Object | `` |
| `src.metrics.probabilistic_metrics.sparse_categorical_crossentropy` | Function | `(y_true, y_pred, from_logits, ignore_class, axis)` |
| `src.metrics.reduction_metrics.Mean` | Class | `(name, dtype)` |
| `src.metrics.reduction_metrics.MeanMetricWrapper` | Class | `(fn, name, dtype, kwargs)` |
| `src.metrics.reduction_metrics.Metric` | Class | `(dtype, name)` |
| `src.metrics.reduction_metrics.Sum` | Class | `(name, dtype)` |
| `src.metrics.reduction_metrics.backend` | Object | `` |
| `src.metrics.reduction_metrics.initializers` | Object | `` |
| `src.metrics.reduction_metrics.keras_export` | Class | `(path)` |
| `src.metrics.reduction_metrics.losses` | Object | `` |
| `src.metrics.reduction_metrics.ops` | Object | `` |
| `src.metrics.reduction_metrics.reduce_to_samplewise_values` | Function | `(values, sample_weight, reduce_fn, dtype)` |
| `src.metrics.reduction_metrics.serialization_lib` | Object | `` |
| `src.metrics.regression_metrics.CosineSimilarity` | Class | `(name, dtype, axis)` |
| `src.metrics.regression_metrics.LogCoshError` | Class | `(name, dtype)` |
| `src.metrics.regression_metrics.MeanAbsoluteError` | Class | `(name, dtype)` |
| `src.metrics.regression_metrics.MeanAbsolutePercentageError` | Class | `(name, dtype)` |
| `src.metrics.regression_metrics.MeanSquaredError` | Class | `(name, dtype)` |
| `src.metrics.regression_metrics.MeanSquaredLogarithmicError` | Class | `(name, dtype)` |
| `src.metrics.regression_metrics.R2Score` | Class | `(class_aggregation, num_regressors, name, dtype)` |
| `src.metrics.regression_metrics.RootMeanSquaredError` | Class | `(name, dtype)` |
| `src.metrics.regression_metrics.cosine_similarity` | Function | `(y_true, y_pred, axis)` |
| `src.metrics.regression_metrics.initializers` | Object | `` |
| `src.metrics.regression_metrics.keras_export` | Class | `(path)` |
| `src.metrics.regression_metrics.log_cosh` | Function | `(y_true, y_pred)` |
| `src.metrics.regression_metrics.mean_absolute_error` | Function | `(y_true, y_pred)` |
| `src.metrics.regression_metrics.mean_absolute_percentage_error` | Function | `(y_true, y_pred)` |
| `src.metrics.regression_metrics.mean_squared_error` | Function | `(y_true, y_pred)` |
| `src.metrics.regression_metrics.mean_squared_logarithmic_error` | Function | `(y_true, y_pred)` |
| `src.metrics.regression_metrics.normalize` | Function | `(x, axis, order)` |
| `src.metrics.regression_metrics.ops` | Object | `` |
| `src.metrics.regression_metrics.reduction_metrics` | Object | `` |
| `src.metrics.regression_metrics.squeeze_or_expand_to_same_rank` | Function | `(x1, x2, expand_rank_1)` |
| `src.metrics.serialization_lib` | Object | `` |
| `src.metrics.serialize` | Function | `(metric)` |
| `src.metrics.to_snake_case` | Function | `(name)` |
| `src.models.Functional` | Class | `(inputs, outputs, name, kwargs)` |
| `src.models.Model` | Class | `(args, kwargs)` |
| `src.models.Sequential` | Class | `(layers, trainable, name)` |
| `src.models.cloning.Functional` | Class | `(inputs, outputs, name, kwargs)` |
| `src.models.cloning.Input` | Function | `(shape, batch_size, dtype, sparse, ragged, batch_shape, name, tensor, optional)` |
| `src.models.cloning.InputLayer` | Class | `(shape, batch_size, dtype, sparse, ragged, batch_shape, input_tensor, optional, name, kwargs)` |
| `src.models.cloning.Sequential` | Class | `(layers, trainable, name)` |
| `src.models.cloning.backend` | Object | `` |
| `src.models.cloning.clone_model` | Function | `(model, input_tensors, clone_function, call_function, recursive, kwargs)` |
| `src.models.cloning.functional_like_constructor` | Function | `(cls)` |
| `src.models.cloning.keras_export` | Class | `(path)` |
| `src.models.cloning.serialization_lib` | Object | `` |
| `src.models.cloning.tree` | Object | `` |
| `src.models.cloning.utils` | Object | `` |
| `src.models.functional.Function` | Class | `(inputs, outputs, name)` |
| `src.models.functional.Functional` | Class | `(inputs, outputs, name, kwargs)` |
| `src.models.functional.Input` | Function | `(shape, batch_size, dtype, sparse, ragged, batch_shape, name, tensor, optional)` |
| `src.models.functional.InputLayer` | Class | `(shape, batch_size, dtype, sparse, ragged, batch_shape, input_tensor, optional, name, kwargs)` |
| `src.models.functional.InputSpec` | Class | `(dtype, shape, ndim, max_ndim, min_ndim, axes, allow_last_axis_squeeze, name, optional)` |
| `src.models.functional.KerasHistory` | Class | `(...)` |
| `src.models.functional.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.models.functional.Model` | Class | `(args, kwargs)` |
| `src.models.functional.Node` | Class | `(operation, call_args, call_kwargs, outputs)` |
| `src.models.functional.Operation` | Class | `(name)` |
| `src.models.functional.backend` | Object | `` |
| `src.models.functional.clone_graph_nodes` | Function | `(inputs, outputs)` |
| `src.models.functional.clone_keras_tensors` | Function | `(tensors, kt_id_mapping)` |
| `src.models.functional.clone_single_keras_tensor` | Function | `(x)` |
| `src.models.functional.deserialize_node` | Function | `(node_data, created_layers)` |
| `src.models.functional.find_nodes_by_inputs_and_outputs` | Function | `(inputs, outputs)` |
| `src.models.functional.functional_from_config` | Function | `(cls, config, custom_objects)` |
| `src.models.functional.functional_like_constructor` | Function | `(cls)` |
| `src.models.functional.global_state` | Object | `` |
| `src.models.functional.is_input_keras_tensor` | Function | `(x)` |
| `src.models.functional.legacy_serialization` | Object | `` |
| `src.models.functional.make_node_key` | Function | `(op, node_index)` |
| `src.models.functional.operation_fn` | Function | `(operation, call_context_args)` |
| `src.models.functional.ops` | Object | `` |
| `src.models.functional.saving_utils` | Object | `` |
| `src.models.functional.serialization_lib` | Object | `` |
| `src.models.functional.serialize_node` | Function | `(node, own_nodes)` |
| `src.models.functional.tracking` | Object | `` |
| `src.models.functional.tree` | Object | `` |
| `src.models.functional.unpack_singleton` | Function | `(x)` |
| `src.models.model.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.models.model.Model` | Class | `(args, kwargs)` |
| `src.models.model.Trainer` | Class | `()` |
| `src.models.model.backend` | Object | `` |
| `src.models.model.base_trainer` | Object | `` |
| `src.models.model.functional_init_arguments` | Function | `(args, kwargs)` |
| `src.models.model.gptq_quantize` | Function | `(config, quantization_layer_structure, filters)` |
| `src.models.model.inject_functional_model_class` | Function | `(cls)` |
| `src.models.model.keras_export` | Class | `(path)` |
| `src.models.model.map_saveable_variables` | Function | `(saveable, store, visited_saveables)` |
| `src.models.model.model_from_json` | Function | `(json_string, custom_objects)` |
| `src.models.model.saving_api` | Object | `` |
| `src.models.model.should_quantize_layer` | Function | `(layer, filters)` |
| `src.models.model.summary_utils` | Object | `` |
| `src.models.model.traceback_utils` | Object | `` |
| `src.models.model.utils` | Object | `` |
| `src.models.sequential.Functional` | Class | `(inputs, outputs, name, kwargs)` |
| `src.models.sequential.InputLayer` | Class | `(shape, batch_size, dtype, sparse, ragged, batch_shape, input_tensor, optional, name, kwargs)` |
| `src.models.sequential.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.models.sequential.Model` | Class | `(args, kwargs)` |
| `src.models.sequential.Sequential` | Class | `(layers, trainable, name)` |
| `src.models.sequential.backend` | Object | `` |
| `src.models.sequential.global_state` | Object | `` |
| `src.models.sequential.keras_export` | Class | `(path)` |
| `src.models.sequential.legacy_serialization` | Object | `` |
| `src.models.sequential.saving_utils` | Object | `` |
| `src.models.sequential.serialization_lib` | Object | `` |
| `src.models.sequential.standardize_shape` | Function | `(shape)` |
| `src.models.sequential.tree` | Object | `` |
| `src.models.variable_mapping.KerasSaveable` | Class | `(...)` |
| `src.models.variable_mapping.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.models.variable_mapping.Metric` | Class | `(dtype, name)` |
| `src.models.variable_mapping.Optimizer` | Class | `(...)` |
| `src.models.variable_mapping.map_container_variables` | Function | `(container, store, visited_saveables)` |
| `src.models.variable_mapping.map_saveable_variables` | Function | `(saveable, store, visited_saveables)` |
| `src.models.variable_mapping.saving_lib` | Object | `` |
| `src.ops.Abs` | Class | `(...)` |
| `src.ops.Absolute` | Class | `(...)` |
| `src.ops.AdaptiveAveragePool` | Class | `(output_size, data_format, name)` |
| `src.ops.AdaptiveMaxPool` | Class | `(output_size, data_format, name)` |
| `src.ops.Add` | Class | `(...)` |
| `src.ops.All` | Class | `(axis, keepdims, name)` |
| `src.ops.Amax` | Class | `(axis, keepdims, name)` |
| `src.ops.Amin` | Class | `(axis, keepdims, name)` |
| `src.ops.Angle` | Class | `(...)` |
| `src.ops.Any` | Class | `(axis, keepdims, name)` |
| `src.ops.Append` | Class | `(axis, name)` |
| `src.ops.Arange` | Class | `(dtype, name)` |
| `src.ops.Arccos` | Class | `(...)` |
| `src.ops.Arccosh` | Class | `(...)` |
| `src.ops.Arcsin` | Class | `(...)` |
| `src.ops.Arcsinh` | Class | `(...)` |
| `src.ops.Arctan` | Class | `(...)` |
| `src.ops.Arctan2` | Class | `(...)` |
| `src.ops.Arctanh` | Class | `(...)` |
| `src.ops.Argmax` | Class | `(axis, keepdims, name)` |
| `src.ops.Argmin` | Class | `(axis, keepdims, name)` |
| `src.ops.Argpartition` | Class | `(kth, axis, name)` |
| `src.ops.Argsort` | Class | `(axis, name)` |
| `src.ops.Array` | Class | `(dtype, name)` |
| `src.ops.ArraySplit` | Class | `(indices_or_sections, axis, name)` |
| `src.ops.AssociativeScan` | Class | `(reverse, axis, name)` |
| `src.ops.Average` | Class | `(axis, name)` |
| `src.ops.AveragePool` | Class | `(pool_size, strides, padding, data_format, name)` |
| `src.ops.Bartlett` | Class | `(...)` |
| `src.ops.BatchNorm` | Class | `(axis, epsilon, name)` |
| `src.ops.BinaryCrossentropy` | Class | `(from_logits, name)` |
| `src.ops.Bincount` | Class | `(weights, minlength, sparse, name)` |
| `src.ops.BitwiseAnd` | Class | `(...)` |
| `src.ops.BitwiseInvert` | Class | `(...)` |
| `src.ops.BitwiseLeftShift` | Class | `(...)` |
| `src.ops.BitwiseNot` | Class | `(...)` |
| `src.ops.BitwiseOr` | Class | `(...)` |
| `src.ops.BitwiseRightShift` | Class | `(...)` |
| `src.ops.BitwiseXor` | Class | `(...)` |
| `src.ops.Blackman` | Class | `(...)` |
| `src.ops.BroadcastTo` | Class | `(shape, name)` |
| `src.ops.CTCDecode` | Class | `(strategy, beam_width, top_paths, merge_repeated, mask_index, name)` |
| `src.ops.CTCLoss` | Class | `(mask_index, name)` |
| `src.ops.Cast` | Class | `(dtype, name)` |
| `src.ops.CategoricalCrossentropy` | Class | `(from_logits, axis, name)` |
| `src.ops.Cbrt` | Class | `(...)` |
| `src.ops.Ceil` | Class | `(...)` |
| `src.ops.Celu` | Class | `(alpha, name)` |
| `src.ops.Cholesky` | Class | `(upper, name)` |
| `src.ops.CholeskyInverse` | Class | `(upper, name)` |
| `src.ops.Clip` | Class | `(x_min, x_max, name)` |
| `src.ops.Concatenate` | Class | `(axis, name)` |
| `src.ops.Cond` | Class | `(...)` |
| `src.ops.Conj` | Class | `(...)` |
| `src.ops.Conjugate` | Class | `(...)` |
| `src.ops.Conv` | Class | `(strides, padding, data_format, dilation_rate, name)` |
| `src.ops.ConvTranspose` | Class | `(strides, padding, output_padding, data_format, dilation_rate, name)` |
| `src.ops.ConvertToTensor` | Class | `(dtype, sparse, ragged, name)` |
| `src.ops.Copy` | Class | `(...)` |
| `src.ops.Corrcoef` | Class | `(...)` |
| `src.ops.Correlate` | Class | `(mode, name)` |
| `src.ops.Cos` | Class | `(...)` |
| `src.ops.Cosh` | Class | `(...)` |
| `src.ops.CountNonzero` | Class | `(axis, name)` |
| `src.ops.Cross` | Class | `(axisa, axisb, axisc, axis, name)` |
| `src.ops.Cumprod` | Class | `(axis, dtype, name)` |
| `src.ops.Cumsum` | Class | `(axis, dtype, name)` |
| `src.ops.Deg2rad` | Class | `(...)` |
| `src.ops.DepthwiseConv` | Class | `(strides, padding, data_format, dilation_rate, name)` |
| `src.ops.Det` | Class | `(...)` |
| `src.ops.Diag` | Class | `(k, name)` |
| `src.ops.Diagflat` | Class | `(k, name)` |
| `src.ops.Diagonal` | Class | `(offset, axis1, axis2, name)` |
| `src.ops.Diff` | Class | `(n, axis, name)` |
| `src.ops.Digitize` | Class | `(...)` |
| `src.ops.Divide` | Class | `(...)` |
| `src.ops.DivideNoNan` | Class | `(...)` |
| `src.ops.Dot` | Class | `(...)` |
| `src.ops.DotProductAttention` | Class | `(is_causal, flash_attention, attn_logits_soft_cap, name)` |
| `src.ops.Eig` | Class | `(...)` |
| `src.ops.Eigh` | Class | `(...)` |
| `src.ops.Einsum` | Class | `(subscripts, name)` |
| `src.ops.Elu` | Class | `(alpha, name)` |
| `src.ops.EmptyLike` | Class | `(dtype, name)` |
| `src.ops.Equal` | Class | `(...)` |
| `src.ops.Erf` | Class | `(...)` |
| `src.ops.Erfinv` | Class | `(...)` |
| `src.ops.Exp` | Class | `(...)` |
| `src.ops.Exp2` | Class | `(...)` |
| `src.ops.ExpandDims` | Class | `(axis, name)` |
| `src.ops.Expm1` | Class | `(...)` |
| `src.ops.ExtractSequences` | Class | `(sequence_length, sequence_stride, name)` |
| `src.ops.FFT` | Class | `(...)` |
| `src.ops.FFT2` | Class | `(...)` |
| `src.ops.Flip` | Class | `(axis, name)` |
| `src.ops.Floor` | Class | `(...)` |
| `src.ops.FloorDivide` | Class | `(...)` |
| `src.ops.ForiLoop` | Class | `(lower, upper, body_fun, name)` |
| `src.ops.Full` | Class | `(shape, dtype, name)` |
| `src.ops.FullLike` | Class | `(dtype, name)` |
| `src.ops.Gcd` | Class | `(...)` |
| `src.ops.Gelu` | Class | `(approximate, name)` |
| `src.ops.GetItem` | Class | `(...)` |
| `src.ops.Glu` | Class | `(axis, name)` |
| `src.ops.Greater` | Class | `(...)` |
| `src.ops.GreaterEqual` | Class | `(...)` |
| `src.ops.Hamming` | Class | `(...)` |
| `src.ops.Hanning` | Class | `(...)` |
| `src.ops.HardShrink` | Class | `(threshold, name)` |
| `src.ops.HardSigmoid` | Class | `(...)` |
| `src.ops.HardSilu` | Class | `(...)` |
| `src.ops.HardTanh` | Class | `(...)` |
| `src.ops.Heaviside` | Class | `(...)` |
| `src.ops.Histogram` | Class | `(bins, range, name)` |
| `src.ops.Hstack` | Class | `(...)` |
| `src.ops.Hypot` | Class | `(...)` |
| `src.ops.IFFT2` | Class | `(...)` |
| `src.ops.IRFFT` | Class | `(fft_length, name)` |
| `src.ops.ISTFT` | Class | `(sequence_length, sequence_stride, fft_length, length, window, center, name)` |
| `src.ops.Imag` | Class | `(...)` |
| `src.ops.InTopK` | Class | `(k, name)` |
| `src.ops.Inner` | Class | `(...)` |
| `src.ops.Inv` | Class | `(...)` |
| `src.ops.IsIn` | Class | `(assume_unique, invert, name)` |
| `src.ops.Isclose` | Class | `(equal_nan, name)` |
| `src.ops.Isfinite` | Class | `(...)` |
| `src.ops.Isinf` | Class | `(...)` |
| `src.ops.Isnan` | Class | `(...)` |
| `src.ops.Isneginf` | Class | `(...)` |
| `src.ops.Isposinf` | Class | `(...)` |
| `src.ops.Isreal` | Class | `(...)` |
| `src.ops.JVP` | Class | `(has_aux, name)` |
| `src.ops.Kaiser` | Class | `(beta, name)` |
| `src.ops.KerasTensor` | Class | `(shape, dtype, sparse, ragged, record_history, name, kwargs)` |
| `src.ops.Kron` | Class | `(...)` |
| `src.ops.LayerNorm` | Class | `(axis, epsilon, rms_scaling, name)` |
| `src.ops.Lcm` | Class | `(...)` |
| `src.ops.Ldexp` | Class | `(...)` |
| `src.ops.LeakyRelu` | Class | `(negative_slope, name)` |
| `src.ops.LeftShift` | Class | `(...)` |
| `src.ops.Less` | Class | `(...)` |
| `src.ops.LessEqual` | Class | `(...)` |
| `src.ops.Linspace` | Class | `(num, endpoint, retstep, dtype, axis, name)` |
| `src.ops.Log` | Class | `(...)` |
| `src.ops.Log10` | Class | `(...)` |
| `src.ops.Log1p` | Class | `(...)` |
| `src.ops.Log2` | Class | `(...)` |
| `src.ops.LogSigmoid` | Class | `(...)` |
| `src.ops.LogSoftmax` | Class | `(axis, name)` |
| `src.ops.Logaddexp` | Class | `(...)` |
| `src.ops.Logaddexp2` | Class | `(...)` |
| `src.ops.Logdet` | Class | `(...)` |
| `src.ops.LogicalAnd` | Class | `(...)` |
| `src.ops.LogicalNot` | Class | `(...)` |
| `src.ops.LogicalOr` | Class | `(...)` |
| `src.ops.LogicalXor` | Class | `(...)` |
| `src.ops.Logspace` | Class | `(num, endpoint, base, dtype, axis, name)` |
| `src.ops.Logsumexp` | Class | `(axis, keepdims, name)` |
| `src.ops.Lstsq` | Class | `(rcond, name)` |
| `src.ops.LuFactor` | Class | `(...)` |
| `src.ops.Map` | Class | `(...)` |
| `src.ops.Matmul` | Class | `(...)` |
| `src.ops.Max` | Class | `(axis, keepdims, initial, name)` |
| `src.ops.MaxPool` | Class | `(pool_size, strides, padding, data_format, name)` |
| `src.ops.Maximum` | Class | `(...)` |
| `src.ops.Mean` | Class | `(axis, keepdims, name)` |
| `src.ops.Median` | Class | `(axis, keepdims, name)` |
| `src.ops.Meshgrid` | Class | `(indexing, name)` |
| `src.ops.Min` | Class | `(axis, keepdims, initial, name)` |
| `src.ops.Minimum` | Class | `(...)` |
| `src.ops.Mod` | Class | `(...)` |
| `src.ops.Moments` | Class | `(axes, keepdims, synchronized, name)` |
| `src.ops.Moveaxis` | Class | `(source, destination, name)` |
| `src.ops.MultiHot` | Class | `(num_classes, axis, dtype, sparse, name, kwargs)` |
| `src.ops.Multiply` | Class | `(...)` |
| `src.ops.NanToNum` | Class | `(nan, posinf, neginf, name)` |
| `src.ops.Ndim` | Class | `(...)` |
| `src.ops.Negative` | Class | `(...)` |
| `src.ops.Nonzero` | Class | `(...)` |
| `src.ops.Norm` | Class | `(ord, axis, keepdims, name)` |
| `src.ops.Normalize` | Class | `(axis, order, epsilon, name)` |
| `src.ops.NotEqual` | Class | `(...)` |
| `src.ops.OneHot` | Class | `(num_classes, axis, dtype, sparse, name)` |
| `src.ops.OnesLike` | Class | `(dtype, name)` |
| `src.ops.Operation` | Class | `(name)` |
| `src.ops.Outer` | Class | `(...)` |
| `src.ops.PSNR` | Class | `(max_val, name)` |
| `src.ops.Pad` | Class | `(pad_width, mode, name)` |
| `src.ops.Polar` | Class | `(...)` |
| `src.ops.Power` | Class | `(...)` |
| `src.ops.Prod` | Class | `(axis, keepdims, dtype, name)` |
| `src.ops.Qr` | Class | `(mode, name)` |
| `src.ops.Quantile` | Class | `(axis, method, keepdims, name)` |
| `src.ops.RFFT` | Class | `(fft_length, name)` |
| `src.ops.RMSNorm` | Class | `(axis, epsilon, name)` |
| `src.ops.Ravel` | Class | `(...)` |
| `src.ops.Real` | Class | `(...)` |
| `src.ops.Reciprocal` | Class | `(...)` |
| `src.ops.Relu` | Class | `(...)` |
| `src.ops.Relu6` | Class | `(...)` |
| `src.ops.Repeat` | Class | `(repeats, axis, name)` |
| `src.ops.Reshape` | Class | `(newshape, name)` |
| `src.ops.RightShift` | Class | `(...)` |
| `src.ops.Roll` | Class | `(shift, axis, name)` |
| `src.ops.Rot90` | Class | `(k, axes, name)` |
| `src.ops.Round` | Class | `(decimals, name)` |
| `src.ops.Rsqrt` | Class | `(...)` |
| `src.ops.STFT` | Class | `(sequence_length, sequence_stride, fft_length, window, center, name)` |
| `src.ops.SVD` | Class | `(full_matrices, compute_uv, name)` |
| `src.ops.SaturateCast` | Class | `(dtype, name)` |
| `src.ops.Scan` | Class | `(length, reverse, unroll, name)` |
| `src.ops.Scatter` | Class | `(shape, name)` |
| `src.ops.ScatterUpdate` | Class | `(...)` |
| `src.ops.SearchSorted` | Class | `(side, name)` |
| `src.ops.SegmentMax` | Class | `(...)` |
| `src.ops.SegmentReduction` | Class | `(num_segments, sorted, name)` |
| `src.ops.SegmentSum` | Class | `(...)` |
| `src.ops.Select` | Class | `(...)` |
| `src.ops.Selu` | Class | `(...)` |
| `src.ops.SeparableConv` | Class | `(strides, padding, data_format, dilation_rate, name)` |
| `src.ops.Sigmoid` | Class | `(...)` |
| `src.ops.Sign` | Class | `(...)` |
| `src.ops.Signbit` | Class | `(...)` |
| `src.ops.Silu` | Class | `(...)` |
| `src.ops.Sin` | Class | `(...)` |
| `src.ops.Sinh` | Class | `(...)` |
| `src.ops.Size` | Class | `(...)` |
| `src.ops.Slice` | Class | `(shape, name)` |
| `src.ops.SliceUpdate` | Class | `(...)` |
| `src.ops.Slogdet` | Class | `(...)` |
| `src.ops.SoftShrink` | Class | `(threshold, name)` |
| `src.ops.Softmax` | Class | `(axis, name)` |
| `src.ops.Softplus` | Class | `(...)` |
| `src.ops.Softsign` | Class | `(...)` |
| `src.ops.Solve` | Class | `(...)` |
| `src.ops.SolveTriangular` | Class | `(lower, name)` |
| `src.ops.Sort` | Class | `(axis, name)` |
| `src.ops.SparseCategoricalCrossentropy` | Class | `(from_logits, axis, name)` |
| `src.ops.SparsePlus` | Class | `(...)` |
| `src.ops.SparseSigmoid` | Class | `(...)` |
| `src.ops.Sparsemax` | Class | `(axis, name)` |
| `src.ops.Split` | Class | `(indices_or_sections, axis, name)` |
| `src.ops.Sqrt` | Class | `(...)` |
| `src.ops.Square` | Class | `(...)` |
| `src.ops.Squareplus` | Class | `(b, name)` |
| `src.ops.Squeeze` | Class | `(axis, name)` |
| `src.ops.Stack` | Class | `(axis, name)` |
| `src.ops.Std` | Class | `(axis, keepdims, name)` |
| `src.ops.StopGradient` | Class | `(...)` |
| `src.ops.Subtract` | Class | `(...)` |
| `src.ops.Sum` | Class | `(axis, keepdims, name)` |
| `src.ops.Swapaxes` | Class | `(axis1, axis2, name)` |
| `src.ops.Switch` | Class | `(...)` |
| `src.ops.Take` | Class | `(axis, name)` |
| `src.ops.TakeAlongAxis` | Class | `(axis, name)` |
| `src.ops.Tan` | Class | `(...)` |
| `src.ops.Tanh` | Class | `(...)` |
| `src.ops.TanhShrink` | Class | `(...)` |
| `src.ops.Tensordot` | Class | `(axes, name)` |
| `src.ops.Threshold` | Class | `(threshold, default_value, name)` |
| `src.ops.Tile` | Class | `(repeats, name)` |
| `src.ops.TopK` | Class | `(k, sorted, name)` |
| `src.ops.Trace` | Class | `(offset, axis1, axis2, name)` |
| `src.ops.Transpose` | Class | `(axes, name)` |
| `src.ops.Trapezoid` | Class | `(x, dx, axis, name)` |
| `src.ops.Tril` | Class | `(k, name)` |
| `src.ops.Triu` | Class | `(k, name)` |
| `src.ops.TrueDivide` | Class | `(...)` |
| `src.ops.Trunc` | Class | `(...)` |
| `src.ops.Unfold` | Class | `(kernel_size, dilation, padding, stride, name)` |
| `src.ops.UnravelIndex` | Class | `(shape, name)` |
| `src.ops.Unstack` | Class | `(num, axis, name)` |
| `src.ops.Vander` | Class | `(N, increasing, name)` |
| `src.ops.Var` | Class | `(axis, keepdims, name)` |
| `src.ops.Vdot` | Class | `(...)` |
| `src.ops.VectorizedMap` | Class | `(function, name)` |
| `src.ops.View` | Class | `(dtype, name)` |
| `src.ops.ViewAsComplex` | Class | `(...)` |
| `src.ops.ViewAsReal` | Class | `(...)` |
| `src.ops.Vstack` | Class | `(...)` |
| `src.ops.Where` | Class | `(...)` |
| `src.ops.WhileLoop` | Class | `(cond, body, maximum_iterations, name)` |
| `src.ops.ZerosLike` | Class | `(dtype, name)` |
| `src.ops.abs` | Function | `(x)` |
| `src.ops.absolute` | Function | `(x)` |
| `src.ops.adaptive_average_pool` | Function | `(inputs, output_size, data_format)` |
| `src.ops.adaptive_max_pool` | Function | `(inputs, output_size, data_format)` |
| `src.ops.add` | Function | `(x1, x2)` |
| `src.ops.all` | Function | `(x, axis, keepdims)` |
| `src.ops.amax` | Function | `(x, axis, keepdims)` |
| `src.ops.amin` | Function | `(x, axis, keepdims)` |
| `src.ops.angle` | Function | `(x)` |
| `src.ops.any` | Function | `(x, axis, keepdims)` |
| `src.ops.any_symbolic_tensors` | Function | `(args, kwargs)` |
| `src.ops.append` | Function | `(x1, x2, axis)` |
| `src.ops.arange` | Function | `(start, stop, step, dtype)` |
| `src.ops.arccos` | Function | `(x)` |
| `src.ops.arccosh` | Function | `(x)` |
| `src.ops.arcsin` | Function | `(x)` |
| `src.ops.arcsinh` | Function | `(x)` |
| `src.ops.arctan` | Function | `(x)` |
| `src.ops.arctan2` | Function | `(x1, x2)` |
| `src.ops.arctanh` | Function | `(x)` |
| `src.ops.argmax` | Function | `(x, axis, keepdims)` |
| `src.ops.argmin` | Function | `(x, axis, keepdims)` |
| `src.ops.argpartition` | Function | `(x, kth, axis)` |
| `src.ops.argsort` | Function | `(x, axis)` |
| `src.ops.array` | Function | `(x, dtype)` |
| `src.ops.array_split` | Function | `(x, indices_or_sections, axis)` |
| `src.ops.associative_scan` | Function | `(f, elems, reverse, axis)` |
| `src.ops.average` | Function | `(x, axis, weights)` |
| `src.ops.average_pool` | Function | `(inputs, pool_size, strides, padding, data_format)` |
| `src.ops.backend` | Object | `` |
| `src.ops.bartlett` | Function | `(x)` |
| `src.ops.batch_normalization` | Function | `(x, mean, variance, axis, offset, scale, epsilon)` |
| `src.ops.binary_crossentropy` | Function | `(target, output, from_logits)` |
| `src.ops.bincount` | Function | `(x, weights, minlength, sparse)` |
| `src.ops.bitwise_and` | Function | `(x, y)` |
| `src.ops.bitwise_invert` | Function | `(x)` |
| `src.ops.bitwise_left_shift` | Function | `(x, y)` |
| `src.ops.bitwise_not` | Function | `(x)` |
| `src.ops.bitwise_or` | Function | `(x, y)` |
| `src.ops.bitwise_right_shift` | Function | `(x, y)` |
| `src.ops.bitwise_xor` | Function | `(x, y)` |
| `src.ops.blackman` | Function | `(x)` |
| `src.ops.broadcast_shapes` | Function | `(shape1, shape2)` |
| `src.ops.broadcast_to` | Function | `(x, shape)` |
| `src.ops.builtins` | Object | `` |
| `src.ops.canonicalize_axis` | Function | `(axis, num_dims)` |
| `src.ops.cast` | Function | `(x, dtype)` |
| `src.ops.categorical_crossentropy` | Function | `(target, output, from_logits, axis)` |
| `src.ops.cbrt` | Function | `(x)` |
| `src.ops.ceil` | Function | `(x)` |
| `src.ops.celu` | Function | `(x, alpha)` |
| `src.ops.cholesky` | Function | `(x, upper)` |
| `src.ops.cholesky_inverse` | Function | `(x, upper)` |
| `src.ops.clip` | Function | `(x, x_min, x_max)` |
| `src.ops.compute_conv_transpose_output_shape` | Function | `(input_shape, kernel_size, filters, strides, padding, output_padding, data_format, dilation_rate)` |
| `src.ops.concatenate` | Function | `(xs, axis)` |
| `src.ops.cond` | Function | `(pred, true_fn, false_fn)` |
| `src.ops.config` | Object | `` |
| `src.ops.conj` | Function | `(x)` |
| `src.ops.conjugate` | Function | `(x)` |
| `src.ops.conv` | Function | `(inputs, kernel, strides, padding, data_format, dilation_rate)` |
| `src.ops.conv_transpose` | Function | `(inputs, kernel, strides, padding, output_padding, data_format, dilation_rate)` |
| `src.ops.convert_to_numpy` | Function | `(x)` |
| `src.ops.convert_to_tensor` | Function | `(x, dtype, sparse, ragged)` |
| `src.ops.copy` | Function | `(x)` |
| `src.ops.core.AssociativeScan` | Class | `(reverse, axis, name)` |
| `src.ops.core.Cast` | Class | `(dtype, name)` |
| `src.ops.core.Cond` | Class | `(...)` |
| `src.ops.core.ConvertToTensor` | Class | `(dtype, sparse, ragged, name)` |
| `src.ops.core.ForiLoop` | Class | `(lower, upper, body_fun, name)` |
| `src.ops.core.KerasTensor` | Class | `(shape, dtype, sparse, ragged, record_history, name, kwargs)` |
| `src.ops.core.Map` | Class | `(...)` |
| `src.ops.core.Operation` | Class | `(name)` |
| `src.ops.core.SaturateCast` | Class | `(dtype, name)` |
| `src.ops.core.Scan` | Class | `(length, reverse, unroll, name)` |
| `src.ops.core.Scatter` | Class | `(shape, name)` |
| `src.ops.core.ScatterUpdate` | Class | `(...)` |
| `src.ops.core.Slice` | Class | `(shape, name)` |
| `src.ops.core.SliceUpdate` | Class | `(...)` |
| `src.ops.core.StopGradient` | Class | `(...)` |
| `src.ops.core.Switch` | Class | `(...)` |
| `src.ops.core.Unstack` | Class | `(num, axis, name)` |
| `src.ops.core.VectorizedMap` | Class | `(function, name)` |
| `src.ops.core.WhileLoop` | Class | `(cond, body, maximum_iterations, name)` |
| `src.ops.core.any_symbolic_tensors` | Function | `(args, kwargs)` |
| `src.ops.core.associative_scan` | Function | `(f, elems, reverse, axis)` |
| `src.ops.core.backend` | Object | `` |
| `src.ops.core.cast` | Function | `(x, dtype)` |
| `src.ops.core.cond` | Function | `(pred, true_fn, false_fn)` |
| `src.ops.core.convert_to_numpy` | Function | `(x)` |
| `src.ops.core.convert_to_tensor` | Function | `(x, dtype, sparse, ragged)` |
| `src.ops.core.custom_gradient` | Function | `(f)` |
| `src.ops.core.dtype` | Function | `(x)` |
| `src.ops.core.fori_loop` | Function | `(lower, upper, body_fun, init_val)` |
| `src.ops.core.is_tensor` | Function | `(x)` |
| `src.ops.core.keras_export` | Class | `(path)` |
| `src.ops.core.map` | Function | `(f, xs)` |
| `src.ops.core.saturate_cast` | Function | `(x, dtype)` |
| `src.ops.core.scan` | Function | `(f, init, xs, length, reverse, unroll)` |
| `src.ops.core.scatter` | Function | `(indices, values, shape)` |
| `src.ops.core.scatter_update` | Function | `(inputs, indices, updates)` |
| `src.ops.core.serialization_lib` | Object | `` |
| `src.ops.core.shape` | Function | `(x)` |
| `src.ops.core.slice` | Function | `(inputs, start_indices, shape)` |
| `src.ops.core.slice_along_axis` | Function | `(x, start, stop, step, axis)` |
| `src.ops.core.slice_update` | Function | `(inputs, start_indices, updates)` |
| `src.ops.core.stop_gradient` | Function | `(variable)` |
| `src.ops.core.switch` | Function | `(index, branches, operands)` |
| `src.ops.core.traceback_utils` | Object | `` |
| `src.ops.core.tree` | Object | `` |
| `src.ops.core.unstack` | Function | `(x, num, axis)` |
| `src.ops.core.vectorized_map` | Function | `(function, elements)` |
| `src.ops.core.while_loop` | Function | `(cond, body, loop_vars, maximum_iterations)` |
| `src.ops.corrcoef` | Function | `(x)` |
| `src.ops.correlate` | Function | `(x1, x2, mode)` |
| `src.ops.cos` | Function | `(x)` |
| `src.ops.cosh` | Function | `(x)` |
| `src.ops.count_nonzero` | Function | `(x, axis)` |
| `src.ops.cross` | Function | `(x1, x2, axisa, axisb, axisc, axis)` |
| `src.ops.ctc_decode` | Function | `(inputs, sequence_lengths, strategy, beam_width, top_paths, merge_repeated, mask_index)` |
| `src.ops.ctc_loss` | Function | `(target, output, target_length, output_length, mask_index)` |
| `src.ops.cumprod` | Function | `(x, axis, dtype)` |
| `src.ops.cumsum` | Function | `(x, axis, dtype)` |
| `src.ops.custom_gradient` | Function | `(f)` |
| `src.ops.deg2rad` | Function | `(x)` |
| `src.ops.depthwise_conv` | Function | `(inputs, kernel, strides, padding, data_format, dilation_rate)` |
| `src.ops.det` | Function | `(x)` |
| `src.ops.diag` | Function | `(x, k)` |
| `src.ops.diagflat` | Function | `(x, k)` |
| `src.ops.diagonal` | Function | `(x, offset, axis1, axis2)` |
| `src.ops.diff` | Function | `(a, n, axis)` |
| `src.ops.digitize` | Function | `(x, bins)` |
| `src.ops.divide` | Function | `(x1, x2)` |
| `src.ops.divide_no_nan` | Function | `(x1, x2)` |
| `src.ops.dot` | Function | `(x1, x2)` |
| `src.ops.dot_product_attention` | Function | `(query, key, value, bias, mask, scale, is_causal, flash_attention, attn_logits_soft_cap)` |
| `src.ops.dtype` | Function | `(x)` |
| `src.ops.dtypes` | Object | `` |
| `src.ops.eig` | Function | `(x)` |
| `src.ops.eigh` | Function | `(x)` |
| `src.ops.einops.KerasTensor` | Class | `(shape, dtype, sparse, ragged, record_history, name, kwargs)` |
| `src.ops.einops.Operation` | Class | `(name)` |
| `src.ops.einops.Rearrange` | Class | `(...)` |
| `src.ops.einops.any_symbolic_tensors` | Function | `(args, kwargs)` |
| `src.ops.einops.keras_export` | Class | `(path)` |
| `src.ops.einops.prod` | Function | `(x, axis, keepdims, dtype)` |
| `src.ops.einops.rearrange` | Function | `(tensor, pattern, axes_lengths)` |
| `src.ops.einops.reshape` | Function | `(x, newshape)` |
| `src.ops.einops.shape` | Function | `(x)` |
| `src.ops.einops.transpose` | Function | `(x, axes)` |
| `src.ops.einsum` | Function | `(subscripts, operands, kwargs)` |
| `src.ops.elu` | Function | `(x, alpha)` |
| `src.ops.empty` | Function | `(shape, dtype)` |
| `src.ops.empty_like` | Function | `(x, dtype)` |
| `src.ops.equal` | Function | `(x1, x2)` |
| `src.ops.erf` | Function | `(x)` |
| `src.ops.erfinv` | Function | `(x)` |
| `src.ops.exp` | Function | `(x)` |
| `src.ops.exp2` | Function | `(x)` |
| `src.ops.expand_dims` | Function | `(x, axis)` |
| `src.ops.expm1` | Function | `(x)` |
| `src.ops.extract_sequences` | Function | `(x, sequence_length, sequence_stride)` |
| `src.ops.eye` | Function | `(N, M, k, dtype)` |
| `src.ops.fft` | Function | `(x)` |
| `src.ops.fft2` | Function | `(x)` |
| `src.ops.flip` | Function | `(x, axis)` |
| `src.ops.floor` | Function | `(x)` |
| `src.ops.floor_divide` | Function | `(x1, x2)` |
| `src.ops.fori_loop` | Function | `(lower, upper, body_fun, init_val)` |
| `src.ops.full` | Function | `(shape, fill_value, dtype)` |
| `src.ops.full_like` | Function | `(x, fill_value, dtype)` |
| `src.ops.function.Function` | Class | `(inputs, outputs, name)` |
| `src.ops.function.KerasTensor` | Class | `(shape, dtype, sparse, ragged, record_history, name, kwargs)` |
| `src.ops.function.Operation` | Class | `(name)` |
| `src.ops.function.backend` | Function | `()` |
| `src.ops.function.is_nnx_enabled` | Function | `()` |
| `src.ops.function.keras_export` | Class | `(path)` |
| `src.ops.function.make_node_key` | Function | `(op, node_index)` |
| `src.ops.function.map_graph` | Function | `(inputs, outputs)` |
| `src.ops.function.tree` | Object | `` |
| `src.ops.gcd` | Function | `(x1, x2)` |
| `src.ops.gelu` | Function | `(x, approximate)` |
| `src.ops.get_item` | Function | `(x, key)` |
| `src.ops.glu` | Function | `(x, axis)` |
| `src.ops.greater` | Function | `(x1, x2)` |
| `src.ops.greater_equal` | Function | `(x1, x2)` |
| `src.ops.hamming` | Function | `(x)` |
| `src.ops.hanning` | Function | `(x)` |
| `src.ops.hard_shrink` | Function | `(x, threshold)` |
| `src.ops.hard_sigmoid` | Function | `(x)` |
| `src.ops.hard_silu` | Function | `(x)` |
| `src.ops.hard_tanh` | Function | `(x)` |
| `src.ops.heaviside` | Function | `(x1, x2)` |
| `src.ops.histogram` | Function | `(x, bins, range)` |
| `src.ops.hstack` | Function | `(xs)` |
| `src.ops.hypot` | Function | `(x1, x2)` |
| `src.ops.identity` | Function | `(n, dtype)` |
| `src.ops.ifft2` | Function | `(x)` |
| `src.ops.imag` | Function | `(x)` |
| `src.ops.image.AffineTransform` | Class | `(interpolation, fill_mode, fill_value, data_format, name)` |
| `src.ops.image.CropImages` | Class | `(top_cropping, left_cropping, bottom_cropping, right_cropping, target_height, target_width, data_format, name)` |
| `src.ops.image.ElasticTransform` | Class | `(alpha, sigma, interpolation, fill_mode, fill_value, seed, data_format, name)` |
| `src.ops.image.ExtractPatches` | Class | `(size, strides, dilation_rate, padding, data_format, name)` |
| `src.ops.image.ExtractPatches3D` | Class | `(size, strides, dilation_rate, padding, data_format, name)` |
| `src.ops.image.GaussianBlur` | Class | `(kernel_size, sigma, data_format, name)` |
| `src.ops.image.HSVToRGB` | Class | `(data_format, name)` |
| `src.ops.image.KerasTensor` | Class | `(shape, dtype, sparse, ragged, record_history, name, kwargs)` |
| `src.ops.image.MapCoordinates` | Class | `(order, fill_mode, fill_value, name)` |
| `src.ops.image.Operation` | Class | `(name)` |
| `src.ops.image.PadImages` | Class | `(top_padding, left_padding, bottom_padding, right_padding, target_height, target_width, data_format, name)` |
| `src.ops.image.PerspectiveTransform` | Class | `(interpolation, fill_value, data_format, name)` |
| `src.ops.image.RGBToGrayscale` | Class | `(data_format, name)` |
| `src.ops.image.RGBToHSV` | Class | `(data_format, name)` |
| `src.ops.image.Resize` | Class | `(size, interpolation, antialias, crop_to_aspect_ratio, pad_to_aspect_ratio, fill_mode, fill_value, data_format, name)` |
| `src.ops.image.ScaleAndTranslate` | Class | `(spatial_dims, method, antialias, name)` |
| `src.ops.image.affine_transform` | Function | `(images, transform, interpolation, fill_mode, fill_value, data_format)` |
| `src.ops.image.any_symbolic_tensors` | Function | `(args, kwargs)` |
| `src.ops.image.backend` | Object | `` |
| `src.ops.image.compute_conv_output_shape` | Function | `(input_shape, filters, kernel_size, strides, padding, data_format, dilation_rate)` |
| `src.ops.image.crop_images` | Function | `(images, top_cropping, left_cropping, bottom_cropping, right_cropping, target_height, target_width, data_format)` |
| `src.ops.image.elastic_transform` | Function | `(images, alpha, sigma, interpolation, fill_mode, fill_value, seed, data_format)` |
| `src.ops.image.extract_patches` | Function | `(images, size, strides, dilation_rate, padding, data_format)` |
| `src.ops.image.extract_patches_3d` | Function | `(volumes, size, strides, dilation_rate, padding, data_format)` |
| `src.ops.image.gaussian_blur` | Function | `(images, kernel_size, sigma, data_format)` |
| `src.ops.image.hsv_to_rgb` | Function | `(images, data_format)` |
| `src.ops.image.keras_export` | Class | `(path)` |
| `src.ops.image.map_coordinates` | Function | `(inputs, coordinates, order, fill_mode, fill_value)` |
| `src.ops.image.ops` | Object | `` |
| `src.ops.image.pad_images` | Function | `(images, top_padding, left_padding, bottom_padding, right_padding, target_height, target_width, data_format)` |
| `src.ops.image.perspective_transform` | Function | `(images, start_points, end_points, interpolation, fill_value, data_format)` |
| `src.ops.image.resize` | Function | `(images, size, interpolation, antialias, crop_to_aspect_ratio, pad_to_aspect_ratio, fill_mode, fill_value, data_format)` |
| `src.ops.image.rgb_to_grayscale` | Function | `(images, data_format)` |
| `src.ops.image.rgb_to_hsv` | Function | `(images, data_format)` |
| `src.ops.image.scale_and_translate` | Function | `(images, output_shape, scale, translation, spatial_dims, method, antialias)` |
| `src.ops.in_top_k` | Function | `(targets, predictions, k)` |
| `src.ops.inner` | Function | `(x1, x2)` |
| `src.ops.inv` | Function | `(x)` |
| `src.ops.irfft` | Function | `(x, fft_length)` |
| `src.ops.is_continuous_axis` | Function | `(axis)` |
| `src.ops.is_tensor` | Function | `(x)` |
| `src.ops.isclose` | Function | `(x1, x2, rtol, atol, equal_nan)` |
| `src.ops.isfinite` | Function | `(x)` |
| `src.ops.isin` | Function | `(x1, x2, assume_unique, invert)` |
| `src.ops.isinf` | Function | `(x)` |
| `src.ops.isnan` | Function | `(x)` |
| `src.ops.isneginf` | Function | `(x)` |
| `src.ops.isposinf` | Function | `(x)` |
| `src.ops.isreal` | Function | `(x)` |
| `src.ops.istft` | Function | `(x, sequence_length, sequence_stride, fft_length, length, window, center)` |
| `src.ops.jvp` | Function | `(fun, primals, tangents, has_aux)` |
| `src.ops.kaiser` | Function | `(x, beta)` |
| `src.ops.keras_export` | Class | `(path)` |
| `src.ops.kron` | Function | `(x1, x2)` |
| `src.ops.layer_normalization` | Function | `(x, gamma, beta, axis, epsilon, kwargs)` |
| `src.ops.lcm` | Function | `(x1, x2)` |
| `src.ops.ldexp` | Function | `(x1, x2)` |
| `src.ops.leaky_relu` | Function | `(x, negative_slope)` |
| `src.ops.left_shift` | Function | `(x, y)` |
| `src.ops.less` | Function | `(x1, x2)` |
| `src.ops.less_equal` | Function | `(x1, x2)` |
| `src.ops.linalg.Cholesky` | Class | `(upper, name)` |
| `src.ops.linalg.CholeskyInverse` | Class | `(upper, name)` |
| `src.ops.linalg.Det` | Class | `(...)` |
| `src.ops.linalg.Eig` | Class | `(...)` |
| `src.ops.linalg.Eigh` | Class | `(...)` |
| `src.ops.linalg.Inv` | Class | `(...)` |
| `src.ops.linalg.JVP` | Class | `(has_aux, name)` |
| `src.ops.linalg.KerasTensor` | Class | `(shape, dtype, sparse, ragged, record_history, name, kwargs)` |
| `src.ops.linalg.Lstsq` | Class | `(rcond, name)` |
| `src.ops.linalg.LuFactor` | Class | `(...)` |
| `src.ops.linalg.Norm` | Class | `(ord, axis, keepdims, name)` |
| `src.ops.linalg.Operation` | Class | `(name)` |
| `src.ops.linalg.Qr` | Class | `(mode, name)` |
| `src.ops.linalg.SVD` | Class | `(full_matrices, compute_uv, name)` |
| `src.ops.linalg.Solve` | Class | `(...)` |
| `src.ops.linalg.SolveTriangular` | Class | `(lower, name)` |
| `src.ops.linalg.any_symbolic_tensors` | Function | `(args, kwargs)` |
| `src.ops.linalg.backend` | Object | `` |
| `src.ops.linalg.cholesky` | Function | `(x, upper)` |
| `src.ops.linalg.cholesky_inverse` | Function | `(x, upper)` |
| `src.ops.linalg.det` | Function | `(x)` |
| `src.ops.linalg.eig` | Function | `(x)` |
| `src.ops.linalg.eigh` | Function | `(x)` |
| `src.ops.linalg.inv` | Function | `(x)` |
| `src.ops.linalg.jvp` | Function | `(fun, primals, tangents, has_aux)` |
| `src.ops.linalg.keras_export` | Class | `(path)` |
| `src.ops.linalg.lstsq` | Function | `(a, b, rcond)` |
| `src.ops.linalg.lu_factor` | Function | `(x)` |
| `src.ops.linalg.norm` | Function | `(x, ord, axis, keepdims)` |
| `src.ops.linalg.qr` | Function | `(x, mode)` |
| `src.ops.linalg.reduce_shape` | Function | `(shape, axis, keepdims)` |
| `src.ops.linalg.solve` | Function | `(a, b)` |
| `src.ops.linalg.solve_triangular` | Function | `(a, b, lower)` |
| `src.ops.linalg.svd` | Function | `(x, full_matrices, compute_uv)` |
| `src.ops.linalg.tree` | Object | `` |
| `src.ops.linspace` | Function | `(start, stop, num, endpoint, retstep, dtype, axis)` |
| `src.ops.log` | Function | `(x)` |
| `src.ops.log10` | Function | `(x)` |
| `src.ops.log1p` | Function | `(x)` |
| `src.ops.log2` | Function | `(x)` |
| `src.ops.log_sigmoid` | Function | `(x)` |
| `src.ops.log_softmax` | Function | `(x, axis)` |
| `src.ops.logaddexp` | Function | `(x1, x2)` |
| `src.ops.logaddexp2` | Function | `(x1, x2)` |
| `src.ops.logdet` | Function | `(x)` |
| `src.ops.logical_and` | Function | `(x1, x2)` |
| `src.ops.logical_not` | Function | `(x)` |
| `src.ops.logical_or` | Function | `(x1, x2)` |
| `src.ops.logical_xor` | Function | `(x1, x2)` |
| `src.ops.logspace` | Function | `(start, stop, num, endpoint, base, dtype, axis)` |
| `src.ops.logsumexp` | Function | `(x, axis, keepdims)` |
| `src.ops.lstsq` | Function | `(a, b, rcond)` |
| `src.ops.lu_factor` | Function | `(x)` |
| `src.ops.map` | Function | `(f, xs)` |
| `src.ops.math.Erf` | Class | `(...)` |
| `src.ops.math.Erfinv` | Class | `(...)` |
| `src.ops.math.ExtractSequences` | Class | `(sequence_length, sequence_stride, name)` |
| `src.ops.math.FFT` | Class | `(...)` |
| `src.ops.math.FFT2` | Class | `(...)` |
| `src.ops.math.IFFT2` | Class | `(...)` |
| `src.ops.math.IRFFT` | Class | `(fft_length, name)` |
| `src.ops.math.ISTFT` | Class | `(sequence_length, sequence_stride, fft_length, length, window, center, name)` |
| `src.ops.math.InTopK` | Class | `(k, name)` |
| `src.ops.math.KerasTensor` | Class | `(shape, dtype, sparse, ragged, record_history, name, kwargs)` |
| `src.ops.math.Logdet` | Class | `(...)` |
| `src.ops.math.Logsumexp` | Class | `(axis, keepdims, name)` |
| `src.ops.math.Operation` | Class | `(name)` |
| `src.ops.math.RFFT` | Class | `(fft_length, name)` |
| `src.ops.math.Rsqrt` | Class | `(...)` |
| `src.ops.math.STFT` | Class | `(sequence_length, sequence_stride, fft_length, window, center, name)` |
| `src.ops.math.SegmentMax` | Class | `(...)` |
| `src.ops.math.SegmentReduction` | Class | `(num_segments, sorted, name)` |
| `src.ops.math.SegmentSum` | Class | `(...)` |
| `src.ops.math.TopK` | Class | `(k, sorted, name)` |
| `src.ops.math.ViewAsComplex` | Class | `(...)` |
| `src.ops.math.ViewAsReal` | Class | `(...)` |
| `src.ops.math.any_symbolic_tensors` | Function | `(args, kwargs)` |
| `src.ops.math.backend` | Object | `` |
| `src.ops.math.erf` | Function | `(x)` |
| `src.ops.math.erfinv` | Function | `(x)` |
| `src.ops.math.extract_sequences` | Function | `(x, sequence_length, sequence_stride)` |
| `src.ops.math.fft` | Function | `(x)` |
| `src.ops.math.fft2` | Function | `(x)` |
| `src.ops.math.ifft2` | Function | `(x)` |
| `src.ops.math.in_top_k` | Function | `(targets, predictions, k)` |
| `src.ops.math.irfft` | Function | `(x, fft_length)` |
| `src.ops.math.istft` | Function | `(x, sequence_length, sequence_stride, fft_length, length, window, center)` |
| `src.ops.math.keras_export` | Class | `(path)` |
| `src.ops.math.logdet` | Function | `(x)` |
| `src.ops.math.logsumexp` | Function | `(x, axis, keepdims)` |
| `src.ops.math.reduce_shape` | Function | `(shape, axis, keepdims)` |
| `src.ops.math.rfft` | Function | `(x, fft_length)` |
| `src.ops.math.rsqrt` | Function | `(x)` |
| `src.ops.math.segment_max` | Function | `(data, segment_ids, num_segments, sorted)` |
| `src.ops.math.segment_sum` | Function | `(data, segment_ids, num_segments, sorted)` |
| `src.ops.math.stft` | Function | `(x, sequence_length, sequence_stride, fft_length, window, center)` |
| `src.ops.math.top_k` | Function | `(x, k, sorted)` |
| `src.ops.math.view_as_complex` | Function | `(x)` |
| `src.ops.math.view_as_real` | Function | `(x)` |
| `src.ops.matmul` | Function | `(x1, x2)` |
| `src.ops.max` | Function | `(x, axis, keepdims, initial)` |
| `src.ops.max_pool` | Function | `(inputs, pool_size, strides, padding, data_format)` |
| `src.ops.maximum` | Function | `(x1, x2)` |
| `src.ops.mean` | Function | `(x, axis, keepdims)` |
| `src.ops.median` | Function | `(x, axis, keepdims)` |
| `src.ops.meshgrid` | Function | `(x, indexing)` |
| `src.ops.min` | Function | `(x, axis, keepdims, initial)` |
| `src.ops.minimum` | Function | `(x1, x2)` |
| `src.ops.ml_dtypes` | Object | `` |
| `src.ops.mod` | Function | `(x1, x2)` |
| `src.ops.moments` | Function | `(x, axes, keepdims, synchronized)` |
| `src.ops.moveaxis` | Function | `(x, source, destination)` |
| `src.ops.multi_hot` | Function | `(inputs, num_classes, axis, dtype, sparse, kwargs)` |
| `src.ops.multiply` | Function | `(x1, x2)` |
| `src.ops.name_scope` | Class | `(...)` |
| `src.ops.nan_to_num` | Function | `(x, nan, posinf, neginf)` |
| `src.ops.ndim` | Function | `(x)` |
| `src.ops.negative` | Function | `(x)` |
| `src.ops.nn.AdaptiveAveragePool` | Class | `(output_size, data_format, name)` |
| `src.ops.nn.AdaptiveMaxPool` | Class | `(output_size, data_format, name)` |
| `src.ops.nn.AveragePool` | Class | `(pool_size, strides, padding, data_format, name)` |
| `src.ops.nn.BatchNorm` | Class | `(axis, epsilon, name)` |
| `src.ops.nn.BinaryCrossentropy` | Class | `(from_logits, name)` |
| `src.ops.nn.CTCDecode` | Class | `(strategy, beam_width, top_paths, merge_repeated, mask_index, name)` |
| `src.ops.nn.CTCLoss` | Class | `(mask_index, name)` |
| `src.ops.nn.CategoricalCrossentropy` | Class | `(from_logits, axis, name)` |
| `src.ops.nn.Celu` | Class | `(alpha, name)` |
| `src.ops.nn.Conv` | Class | `(strides, padding, data_format, dilation_rate, name)` |
| `src.ops.nn.ConvTranspose` | Class | `(strides, padding, output_padding, data_format, dilation_rate, name)` |
| `src.ops.nn.DepthwiseConv` | Class | `(strides, padding, data_format, dilation_rate, name)` |
| `src.ops.nn.DotProductAttention` | Class | `(is_causal, flash_attention, attn_logits_soft_cap, name)` |
| `src.ops.nn.Elu` | Class | `(alpha, name)` |
| `src.ops.nn.Gelu` | Class | `(approximate, name)` |
| `src.ops.nn.Glu` | Class | `(axis, name)` |
| `src.ops.nn.HardShrink` | Class | `(threshold, name)` |
| `src.ops.nn.HardSigmoid` | Class | `(...)` |
| `src.ops.nn.HardSilu` | Class | `(...)` |
| `src.ops.nn.HardTanh` | Class | `(...)` |
| `src.ops.nn.KerasTensor` | Class | `(shape, dtype, sparse, ragged, record_history, name, kwargs)` |
| `src.ops.nn.LayerNorm` | Class | `(axis, epsilon, rms_scaling, name)` |
| `src.ops.nn.LeakyRelu` | Class | `(negative_slope, name)` |
| `src.ops.nn.LogSigmoid` | Class | `(...)` |
| `src.ops.nn.LogSoftmax` | Class | `(axis, name)` |
| `src.ops.nn.MaxPool` | Class | `(pool_size, strides, padding, data_format, name)` |
| `src.ops.nn.Moments` | Class | `(axes, keepdims, synchronized, name)` |
| `src.ops.nn.MultiHot` | Class | `(num_classes, axis, dtype, sparse, name, kwargs)` |
| `src.ops.nn.Normalize` | Class | `(axis, order, epsilon, name)` |
| `src.ops.nn.OneHot` | Class | `(num_classes, axis, dtype, sparse, name)` |
| `src.ops.nn.Operation` | Class | `(name)` |
| `src.ops.nn.PSNR` | Class | `(max_val, name)` |
| `src.ops.nn.Polar` | Class | `(...)` |
| `src.ops.nn.RMSNorm` | Class | `(axis, epsilon, name)` |
| `src.ops.nn.Relu` | Class | `(...)` |
| `src.ops.nn.Relu6` | Class | `(...)` |
| `src.ops.nn.Selu` | Class | `(...)` |
| `src.ops.nn.SeparableConv` | Class | `(strides, padding, data_format, dilation_rate, name)` |
| `src.ops.nn.Sigmoid` | Class | `(...)` |
| `src.ops.nn.Silu` | Class | `(...)` |
| `src.ops.nn.SoftShrink` | Class | `(threshold, name)` |
| `src.ops.nn.Softmax` | Class | `(axis, name)` |
| `src.ops.nn.Softplus` | Class | `(...)` |
| `src.ops.nn.Softsign` | Class | `(...)` |
| `src.ops.nn.SparseCategoricalCrossentropy` | Class | `(from_logits, axis, name)` |
| `src.ops.nn.SparsePlus` | Class | `(...)` |
| `src.ops.nn.SparseSigmoid` | Class | `(...)` |
| `src.ops.nn.Sparsemax` | Class | `(axis, name)` |
| `src.ops.nn.Squareplus` | Class | `(b, name)` |
| `src.ops.nn.TanhShrink` | Class | `(...)` |
| `src.ops.nn.Threshold` | Class | `(threshold, default_value, name)` |
| `src.ops.nn.Unfold` | Class | `(kernel_size, dilation, padding, stride, name)` |
| `src.ops.nn.adaptive_average_pool` | Function | `(inputs, output_size, data_format)` |
| `src.ops.nn.adaptive_max_pool` | Function | `(inputs, output_size, data_format)` |
| `src.ops.nn.any_symbolic_tensors` | Function | `(args, kwargs)` |
| `src.ops.nn.average_pool` | Function | `(inputs, pool_size, strides, padding, data_format)` |
| `src.ops.nn.backend` | Object | `` |
| `src.ops.nn.batch_normalization` | Function | `(x, mean, variance, axis, offset, scale, epsilon)` |
| `src.ops.nn.binary_crossentropy` | Function | `(target, output, from_logits)` |
| `src.ops.nn.categorical_crossentropy` | Function | `(target, output, from_logits, axis)` |
| `src.ops.nn.celu` | Function | `(x, alpha)` |
| `src.ops.nn.compute_conv_transpose_output_shape` | Function | `(input_shape, kernel_size, filters, strides, padding, output_padding, data_format, dilation_rate)` |
| `src.ops.nn.config` | Object | `` |
| `src.ops.nn.conv` | Function | `(inputs, kernel, strides, padding, data_format, dilation_rate)` |
| `src.ops.nn.conv_transpose` | Function | `(inputs, kernel, strides, padding, output_padding, data_format, dilation_rate)` |
| `src.ops.nn.ctc_decode` | Function | `(inputs, sequence_lengths, strategy, beam_width, top_paths, merge_repeated, mask_index)` |
| `src.ops.nn.ctc_loss` | Function | `(target, output, target_length, output_length, mask_index)` |
| `src.ops.nn.depthwise_conv` | Function | `(inputs, kernel, strides, padding, data_format, dilation_rate)` |
| `src.ops.nn.dot_product_attention` | Function | `(query, key, value, bias, mask, scale, is_causal, flash_attention, attn_logits_soft_cap)` |
| `src.ops.nn.elu` | Function | `(x, alpha)` |
| `src.ops.nn.gelu` | Function | `(x, approximate)` |
| `src.ops.nn.glu` | Function | `(x, axis)` |
| `src.ops.nn.hard_shrink` | Function | `(x, threshold)` |
| `src.ops.nn.hard_sigmoid` | Function | `(x)` |
| `src.ops.nn.hard_silu` | Function | `(x)` |
| `src.ops.nn.hard_tanh` | Function | `(x)` |
| `src.ops.nn.is_continuous_axis` | Function | `(axis)` |
| `src.ops.nn.keras_export` | Class | `(path)` |
| `src.ops.nn.layer_normalization` | Function | `(x, gamma, beta, axis, epsilon, kwargs)` |
| `src.ops.nn.leaky_relu` | Function | `(x, negative_slope)` |
| `src.ops.nn.log_sigmoid` | Function | `(x)` |
| `src.ops.nn.log_softmax` | Function | `(x, axis)` |
| `src.ops.nn.max_pool` | Function | `(inputs, pool_size, strides, padding, data_format)` |
| `src.ops.nn.moments` | Function | `(x, axes, keepdims, synchronized)` |
| `src.ops.nn.multi_hot` | Function | `(inputs, num_classes, axis, dtype, sparse, kwargs)` |
| `src.ops.nn.normalize` | Function | `(x, axis, order, epsilon)` |
| `src.ops.nn.one_hot` | Function | `(x, num_classes, axis, dtype, sparse)` |
| `src.ops.nn.operation_utils` | Object | `` |
| `src.ops.nn.polar` | Function | `(abs_, angle)` |
| `src.ops.nn.psnr` | Function | `(x1, x2, max_val)` |
| `src.ops.nn.reduce_shape` | Function | `(shape, axis, keepdims)` |
| `src.ops.nn.relu` | Function | `(x)` |
| `src.ops.nn.relu6` | Function | `(x)` |
| `src.ops.nn.rms_normalization` | Function | `(x, scale, axis, epsilon)` |
| `src.ops.nn.selu` | Function | `(x)` |
| `src.ops.nn.separable_conv` | Function | `(inputs, depthwise_kernel, pointwise_kernel, strides, padding, data_format, dilation_rate)` |
| `src.ops.nn.sigmoid` | Function | `(x)` |
| `src.ops.nn.silu` | Function | `(x)` |
| `src.ops.nn.soft_shrink` | Function | `(x, threshold)` |
| `src.ops.nn.softmax` | Function | `(x, axis)` |
| `src.ops.nn.softplus` | Function | `(x)` |
| `src.ops.nn.softsign` | Function | `(x)` |
| `src.ops.nn.sparse_categorical_crossentropy` | Function | `(target, output, from_logits, axis)` |
| `src.ops.nn.sparse_plus` | Function | `(x)` |
| `src.ops.nn.sparse_sigmoid` | Function | `(x)` |
| `src.ops.nn.sparsemax` | Function | `(x, axis)` |
| `src.ops.nn.squareplus` | Function | `(x, b)` |
| `src.ops.nn.standardize_data_format` | Function | `(data_format)` |
| `src.ops.nn.tanh_shrink` | Function | `(x)` |
| `src.ops.nn.threshold` | Function | `(x, threshold, default_value)` |
| `src.ops.nn.unfold` | Function | `(x, kernel_size, dilation, padding, stride)` |
| `src.ops.node.KerasHistory` | Class | `(...)` |
| `src.ops.node.KerasTensor` | Class | `(shape, dtype, sparse, ragged, record_history, name, kwargs)` |
| `src.ops.node.Node` | Class | `(operation, call_args, call_kwargs, outputs)` |
| `src.ops.node.SymbolicArguments` | Class | `(args, kwargs)` |
| `src.ops.node.is_keras_tensor` | Function | `(obj)` |
| `src.ops.node.tree` | Object | `` |
| `src.ops.nonzero` | Function | `(x)` |
| `src.ops.norm` | Function | `(x, ord, axis, keepdims)` |
| `src.ops.normalize` | Function | `(x, axis, order, epsilon)` |
| `src.ops.not_equal` | Function | `(x1, x2)` |
| `src.ops.np` | Object | `` |
| `src.ops.numpy.Abs` | Class | `(...)` |
| `src.ops.numpy.Absolute` | Class | `(...)` |
| `src.ops.numpy.Add` | Class | `(...)` |
| `src.ops.numpy.All` | Class | `(axis, keepdims, name)` |
| `src.ops.numpy.Amax` | Class | `(axis, keepdims, name)` |
| `src.ops.numpy.Amin` | Class | `(axis, keepdims, name)` |
| `src.ops.numpy.Angle` | Class | `(...)` |
| `src.ops.numpy.Any` | Class | `(axis, keepdims, name)` |
| `src.ops.numpy.Append` | Class | `(axis, name)` |
| `src.ops.numpy.Arange` | Class | `(dtype, name)` |
| `src.ops.numpy.Arccos` | Class | `(...)` |
| `src.ops.numpy.Arccosh` | Class | `(...)` |
| `src.ops.numpy.Arcsin` | Class | `(...)` |
| `src.ops.numpy.Arcsinh` | Class | `(...)` |
| `src.ops.numpy.Arctan` | Class | `(...)` |
| `src.ops.numpy.Arctan2` | Class | `(...)` |
| `src.ops.numpy.Arctanh` | Class | `(...)` |
| `src.ops.numpy.Argmax` | Class | `(axis, keepdims, name)` |
| `src.ops.numpy.Argmin` | Class | `(axis, keepdims, name)` |
| `src.ops.numpy.Argpartition` | Class | `(kth, axis, name)` |
| `src.ops.numpy.Argsort` | Class | `(axis, name)` |
| `src.ops.numpy.Array` | Class | `(dtype, name)` |
| `src.ops.numpy.ArraySplit` | Class | `(indices_or_sections, axis, name)` |
| `src.ops.numpy.Average` | Class | `(axis, name)` |
| `src.ops.numpy.Bartlett` | Class | `(...)` |
| `src.ops.numpy.Bincount` | Class | `(weights, minlength, sparse, name)` |
| `src.ops.numpy.BitwiseAnd` | Class | `(...)` |
| `src.ops.numpy.BitwiseInvert` | Class | `(...)` |
| `src.ops.numpy.BitwiseLeftShift` | Class | `(...)` |
| `src.ops.numpy.BitwiseNot` | Class | `(...)` |
| `src.ops.numpy.BitwiseOr` | Class | `(...)` |
| `src.ops.numpy.BitwiseRightShift` | Class | `(...)` |
| `src.ops.numpy.BitwiseXor` | Class | `(...)` |
| `src.ops.numpy.Blackman` | Class | `(...)` |
| `src.ops.numpy.BroadcastTo` | Class | `(shape, name)` |
| `src.ops.numpy.Cbrt` | Class | `(...)` |
| `src.ops.numpy.Ceil` | Class | `(...)` |
| `src.ops.numpy.Clip` | Class | `(x_min, x_max, name)` |
| `src.ops.numpy.Concatenate` | Class | `(axis, name)` |
| `src.ops.numpy.Conj` | Class | `(...)` |
| `src.ops.numpy.Conjugate` | Class | `(...)` |
| `src.ops.numpy.Copy` | Class | `(...)` |
| `src.ops.numpy.Corrcoef` | Class | `(...)` |
| `src.ops.numpy.Correlate` | Class | `(mode, name)` |
| `src.ops.numpy.Cos` | Class | `(...)` |
| `src.ops.numpy.Cosh` | Class | `(...)` |
| `src.ops.numpy.CountNonzero` | Class | `(axis, name)` |
| `src.ops.numpy.Cross` | Class | `(axisa, axisb, axisc, axis, name)` |
| `src.ops.numpy.Cumprod` | Class | `(axis, dtype, name)` |
| `src.ops.numpy.Cumsum` | Class | `(axis, dtype, name)` |
| `src.ops.numpy.Deg2rad` | Class | `(...)` |
| `src.ops.numpy.Diag` | Class | `(k, name)` |
| `src.ops.numpy.Diagflat` | Class | `(k, name)` |
| `src.ops.numpy.Diagonal` | Class | `(offset, axis1, axis2, name)` |
| `src.ops.numpy.Diff` | Class | `(n, axis, name)` |
| `src.ops.numpy.Digitize` | Class | `(...)` |
| `src.ops.numpy.Divide` | Class | `(...)` |
| `src.ops.numpy.DivideNoNan` | Class | `(...)` |
| `src.ops.numpy.Dot` | Class | `(...)` |
| `src.ops.numpy.Einsum` | Class | `(subscripts, name)` |
| `src.ops.numpy.EmptyLike` | Class | `(dtype, name)` |
| `src.ops.numpy.Equal` | Class | `(...)` |
| `src.ops.numpy.Exp` | Class | `(...)` |
| `src.ops.numpy.Exp2` | Class | `(...)` |
| `src.ops.numpy.ExpandDims` | Class | `(axis, name)` |
| `src.ops.numpy.Expm1` | Class | `(...)` |
| `src.ops.numpy.Flip` | Class | `(axis, name)` |
| `src.ops.numpy.Floor` | Class | `(...)` |
| `src.ops.numpy.FloorDivide` | Class | `(...)` |
| `src.ops.numpy.Full` | Class | `(shape, dtype, name)` |
| `src.ops.numpy.FullLike` | Class | `(dtype, name)` |
| `src.ops.numpy.Gcd` | Class | `(...)` |
| `src.ops.numpy.GetItem` | Class | `(...)` |
| `src.ops.numpy.Greater` | Class | `(...)` |
| `src.ops.numpy.GreaterEqual` | Class | `(...)` |
| `src.ops.numpy.Hamming` | Class | `(...)` |
| `src.ops.numpy.Hanning` | Class | `(...)` |
| `src.ops.numpy.Heaviside` | Class | `(...)` |
| `src.ops.numpy.Histogram` | Class | `(bins, range, name)` |
| `src.ops.numpy.Hstack` | Class | `(...)` |
| `src.ops.numpy.Hypot` | Class | `(...)` |
| `src.ops.numpy.Imag` | Class | `(...)` |
| `src.ops.numpy.Inner` | Class | `(...)` |
| `src.ops.numpy.IsIn` | Class | `(assume_unique, invert, name)` |
| `src.ops.numpy.Isclose` | Class | `(equal_nan, name)` |
| `src.ops.numpy.Isfinite` | Class | `(...)` |
| `src.ops.numpy.Isinf` | Class | `(...)` |
| `src.ops.numpy.Isnan` | Class | `(...)` |
| `src.ops.numpy.Isneginf` | Class | `(...)` |
| `src.ops.numpy.Isposinf` | Class | `(...)` |
| `src.ops.numpy.Isreal` | Class | `(...)` |
| `src.ops.numpy.Kaiser` | Class | `(beta, name)` |
| `src.ops.numpy.KerasTensor` | Class | `(shape, dtype, sparse, ragged, record_history, name, kwargs)` |
| `src.ops.numpy.Kron` | Class | `(...)` |
| `src.ops.numpy.Lcm` | Class | `(...)` |
| `src.ops.numpy.Ldexp` | Class | `(...)` |
| `src.ops.numpy.LeftShift` | Class | `(...)` |
| `src.ops.numpy.Less` | Class | `(...)` |
| `src.ops.numpy.LessEqual` | Class | `(...)` |
| `src.ops.numpy.Linspace` | Class | `(num, endpoint, retstep, dtype, axis, name)` |
| `src.ops.numpy.Log` | Class | `(...)` |
| `src.ops.numpy.Log10` | Class | `(...)` |
| `src.ops.numpy.Log1p` | Class | `(...)` |
| `src.ops.numpy.Log2` | Class | `(...)` |
| `src.ops.numpy.Logaddexp` | Class | `(...)` |
| `src.ops.numpy.Logaddexp2` | Class | `(...)` |
| `src.ops.numpy.LogicalAnd` | Class | `(...)` |
| `src.ops.numpy.LogicalNot` | Class | `(...)` |
| `src.ops.numpy.LogicalOr` | Class | `(...)` |
| `src.ops.numpy.LogicalXor` | Class | `(...)` |
| `src.ops.numpy.Logspace` | Class | `(num, endpoint, base, dtype, axis, name)` |
| `src.ops.numpy.Matmul` | Class | `(...)` |
| `src.ops.numpy.Max` | Class | `(axis, keepdims, initial, name)` |
| `src.ops.numpy.Maximum` | Class | `(...)` |
| `src.ops.numpy.Mean` | Class | `(axis, keepdims, name)` |
| `src.ops.numpy.Median` | Class | `(axis, keepdims, name)` |
| `src.ops.numpy.Meshgrid` | Class | `(indexing, name)` |
| `src.ops.numpy.Min` | Class | `(axis, keepdims, initial, name)` |
| `src.ops.numpy.Minimum` | Class | `(...)` |
| `src.ops.numpy.Mod` | Class | `(...)` |
| `src.ops.numpy.Moveaxis` | Class | `(source, destination, name)` |
| `src.ops.numpy.Multiply` | Class | `(...)` |
| `src.ops.numpy.NanToNum` | Class | `(nan, posinf, neginf, name)` |
| `src.ops.numpy.Ndim` | Class | `(...)` |
| `src.ops.numpy.Negative` | Class | `(...)` |
| `src.ops.numpy.Nonzero` | Class | `(...)` |
| `src.ops.numpy.NotEqual` | Class | `(...)` |
| `src.ops.numpy.OnesLike` | Class | `(dtype, name)` |
| `src.ops.numpy.Operation` | Class | `(name)` |
| `src.ops.numpy.Outer` | Class | `(...)` |
| `src.ops.numpy.Pad` | Class | `(pad_width, mode, name)` |
| `src.ops.numpy.Power` | Class | `(...)` |
| `src.ops.numpy.Prod` | Class | `(axis, keepdims, dtype, name)` |
| `src.ops.numpy.Quantile` | Class | `(axis, method, keepdims, name)` |
| `src.ops.numpy.Ravel` | Class | `(...)` |
| `src.ops.numpy.Real` | Class | `(...)` |
| `src.ops.numpy.Reciprocal` | Class | `(...)` |
| `src.ops.numpy.Repeat` | Class | `(repeats, axis, name)` |
| `src.ops.numpy.Reshape` | Class | `(newshape, name)` |
| `src.ops.numpy.RightShift` | Class | `(...)` |
| `src.ops.numpy.Roll` | Class | `(shift, axis, name)` |
| `src.ops.numpy.Rot90` | Class | `(k, axes, name)` |
| `src.ops.numpy.Round` | Class | `(decimals, name)` |
| `src.ops.numpy.SearchSorted` | Class | `(side, name)` |
| `src.ops.numpy.Select` | Class | `(...)` |
| `src.ops.numpy.Sign` | Class | `(...)` |
| `src.ops.numpy.Signbit` | Class | `(...)` |
| `src.ops.numpy.Sin` | Class | `(...)` |
| `src.ops.numpy.Sinh` | Class | `(...)` |
| `src.ops.numpy.Size` | Class | `(...)` |
| `src.ops.numpy.Slogdet` | Class | `(...)` |
| `src.ops.numpy.Sort` | Class | `(axis, name)` |
| `src.ops.numpy.Split` | Class | `(indices_or_sections, axis, name)` |
| `src.ops.numpy.Sqrt` | Class | `(...)` |
| `src.ops.numpy.Square` | Class | `(...)` |
| `src.ops.numpy.Squeeze` | Class | `(axis, name)` |
| `src.ops.numpy.Stack` | Class | `(axis, name)` |
| `src.ops.numpy.Std` | Class | `(axis, keepdims, name)` |
| `src.ops.numpy.Subtract` | Class | `(...)` |
| `src.ops.numpy.Sum` | Class | `(axis, keepdims, name)` |
| `src.ops.numpy.Swapaxes` | Class | `(axis1, axis2, name)` |
| `src.ops.numpy.Take` | Class | `(axis, name)` |
| `src.ops.numpy.TakeAlongAxis` | Class | `(axis, name)` |
| `src.ops.numpy.Tan` | Class | `(...)` |
| `src.ops.numpy.Tanh` | Class | `(...)` |
| `src.ops.numpy.Tensordot` | Class | `(axes, name)` |
| `src.ops.numpy.Tile` | Class | `(repeats, name)` |
| `src.ops.numpy.Trace` | Class | `(offset, axis1, axis2, name)` |
| `src.ops.numpy.Transpose` | Class | `(axes, name)` |
| `src.ops.numpy.Trapezoid` | Class | `(x, dx, axis, name)` |
| `src.ops.numpy.Tril` | Class | `(k, name)` |
| `src.ops.numpy.Triu` | Class | `(k, name)` |
| `src.ops.numpy.TrueDivide` | Class | `(...)` |
| `src.ops.numpy.Trunc` | Class | `(...)` |
| `src.ops.numpy.UnravelIndex` | Class | `(shape, name)` |
| `src.ops.numpy.Vander` | Class | `(N, increasing, name)` |
| `src.ops.numpy.Var` | Class | `(axis, keepdims, name)` |
| `src.ops.numpy.Vdot` | Class | `(...)` |
| `src.ops.numpy.View` | Class | `(dtype, name)` |
| `src.ops.numpy.Vstack` | Class | `(...)` |
| `src.ops.numpy.Where` | Class | `(...)` |
| `src.ops.numpy.ZerosLike` | Class | `(dtype, name)` |
| `src.ops.numpy.abs` | Function | `(x)` |
| `src.ops.numpy.absolute` | Function | `(x)` |
| `src.ops.numpy.add` | Function | `(x1, x2)` |
| `src.ops.numpy.all` | Function | `(x, axis, keepdims)` |
| `src.ops.numpy.amax` | Function | `(x, axis, keepdims)` |
| `src.ops.numpy.amin` | Function | `(x, axis, keepdims)` |
| `src.ops.numpy.angle` | Function | `(x)` |
| `src.ops.numpy.any` | Function | `(x, axis, keepdims)` |
| `src.ops.numpy.any_symbolic_tensors` | Function | `(args, kwargs)` |
| `src.ops.numpy.append` | Function | `(x1, x2, axis)` |
| `src.ops.numpy.arange` | Function | `(start, stop, step, dtype)` |
| `src.ops.numpy.arccos` | Function | `(x)` |
| `src.ops.numpy.arccosh` | Function | `(x)` |
| `src.ops.numpy.arcsin` | Function | `(x)` |
| `src.ops.numpy.arcsinh` | Function | `(x)` |
| `src.ops.numpy.arctan` | Function | `(x)` |
| `src.ops.numpy.arctan2` | Function | `(x1, x2)` |
| `src.ops.numpy.arctanh` | Function | `(x)` |
| `src.ops.numpy.argmax` | Function | `(x, axis, keepdims)` |
| `src.ops.numpy.argmin` | Function | `(x, axis, keepdims)` |
| `src.ops.numpy.argpartition` | Function | `(x, kth, axis)` |
| `src.ops.numpy.argsort` | Function | `(x, axis)` |
| `src.ops.numpy.array` | Function | `(x, dtype)` |
| `src.ops.numpy.array_split` | Function | `(x, indices_or_sections, axis)` |
| `src.ops.numpy.average` | Function | `(x, axis, weights)` |
| `src.ops.numpy.backend` | Object | `` |
| `src.ops.numpy.bartlett` | Function | `(x)` |
| `src.ops.numpy.bincount` | Function | `(x, weights, minlength, sparse)` |
| `src.ops.numpy.bitwise_and` | Function | `(x, y)` |
| `src.ops.numpy.bitwise_invert` | Function | `(x)` |
| `src.ops.numpy.bitwise_left_shift` | Function | `(x, y)` |
| `src.ops.numpy.bitwise_not` | Function | `(x)` |
| `src.ops.numpy.bitwise_or` | Function | `(x, y)` |
| `src.ops.numpy.bitwise_right_shift` | Function | `(x, y)` |
| `src.ops.numpy.bitwise_xor` | Function | `(x, y)` |
| `src.ops.numpy.blackman` | Function | `(x)` |
| `src.ops.numpy.broadcast_shapes` | Function | `(shape1, shape2)` |
| `src.ops.numpy.broadcast_to` | Function | `(x, shape)` |
| `src.ops.numpy.canonicalize_axis` | Function | `(axis, num_dims)` |
| `src.ops.numpy.cbrt` | Function | `(x)` |
| `src.ops.numpy.ceil` | Function | `(x)` |
| `src.ops.numpy.clip` | Function | `(x, x_min, x_max)` |
| `src.ops.numpy.concatenate` | Function | `(xs, axis)` |
| `src.ops.numpy.conj` | Function | `(x)` |
| `src.ops.numpy.conjugate` | Function | `(x)` |
| `src.ops.numpy.copy` | Function | `(x)` |
| `src.ops.numpy.corrcoef` | Function | `(x)` |
| `src.ops.numpy.correlate` | Function | `(x1, x2, mode)` |
| `src.ops.numpy.cos` | Function | `(x)` |
| `src.ops.numpy.cosh` | Function | `(x)` |
| `src.ops.numpy.count_nonzero` | Function | `(x, axis)` |
| `src.ops.numpy.cross` | Function | `(x1, x2, axisa, axisb, axisc, axis)` |
| `src.ops.numpy.cumprod` | Function | `(x, axis, dtype)` |
| `src.ops.numpy.cumsum` | Function | `(x, axis, dtype)` |
| `src.ops.numpy.deg2rad` | Function | `(x)` |
| `src.ops.numpy.diag` | Function | `(x, k)` |
| `src.ops.numpy.diagflat` | Function | `(x, k)` |
| `src.ops.numpy.diagonal` | Function | `(x, offset, axis1, axis2)` |
| `src.ops.numpy.diff` | Function | `(a, n, axis)` |
| `src.ops.numpy.digitize` | Function | `(x, bins)` |
| `src.ops.numpy.divide` | Function | `(x1, x2)` |
| `src.ops.numpy.divide_no_nan` | Function | `(x1, x2)` |
| `src.ops.numpy.dot` | Function | `(x1, x2)` |
| `src.ops.numpy.dtypes` | Object | `` |
| `src.ops.numpy.einsum` | Function | `(subscripts, operands, kwargs)` |
| `src.ops.numpy.empty` | Function | `(shape, dtype)` |
| `src.ops.numpy.empty_like` | Function | `(x, dtype)` |
| `src.ops.numpy.equal` | Function | `(x1, x2)` |
| `src.ops.numpy.exp` | Function | `(x)` |
| `src.ops.numpy.exp2` | Function | `(x)` |
| `src.ops.numpy.expand_dims` | Function | `(x, axis)` |
| `src.ops.numpy.expm1` | Function | `(x)` |
| `src.ops.numpy.eye` | Function | `(N, M, k, dtype)` |
| `src.ops.numpy.flip` | Function | `(x, axis)` |
| `src.ops.numpy.floor` | Function | `(x)` |
| `src.ops.numpy.floor_divide` | Function | `(x1, x2)` |
| `src.ops.numpy.full` | Function | `(shape, fill_value, dtype)` |
| `src.ops.numpy.full_like` | Function | `(x, fill_value, dtype)` |
| `src.ops.numpy.gcd` | Function | `(x1, x2)` |
| `src.ops.numpy.get_item` | Function | `(x, key)` |
| `src.ops.numpy.greater` | Function | `(x1, x2)` |
| `src.ops.numpy.greater_equal` | Function | `(x1, x2)` |
| `src.ops.numpy.hamming` | Function | `(x)` |
| `src.ops.numpy.hanning` | Function | `(x)` |
| `src.ops.numpy.heaviside` | Function | `(x1, x2)` |
| `src.ops.numpy.histogram` | Function | `(x, bins, range)` |
| `src.ops.numpy.hstack` | Function | `(xs)` |
| `src.ops.numpy.hypot` | Function | `(x1, x2)` |
| `src.ops.numpy.identity` | Function | `(n, dtype)` |
| `src.ops.numpy.imag` | Function | `(x)` |
| `src.ops.numpy.inner` | Function | `(x1, x2)` |
| `src.ops.numpy.isclose` | Function | `(x1, x2, rtol, atol, equal_nan)` |
| `src.ops.numpy.isfinite` | Function | `(x)` |
| `src.ops.numpy.isin` | Function | `(x1, x2, assume_unique, invert)` |
| `src.ops.numpy.isinf` | Function | `(x)` |
| `src.ops.numpy.isnan` | Function | `(x)` |
| `src.ops.numpy.isneginf` | Function | `(x)` |
| `src.ops.numpy.isposinf` | Function | `(x)` |
| `src.ops.numpy.isreal` | Function | `(x)` |
| `src.ops.numpy.kaiser` | Function | `(x, beta)` |
| `src.ops.numpy.keras_export` | Class | `(path)` |
| `src.ops.numpy.kron` | Function | `(x1, x2)` |
| `src.ops.numpy.lcm` | Function | `(x1, x2)` |
| `src.ops.numpy.ldexp` | Function | `(x1, x2)` |
| `src.ops.numpy.left_shift` | Function | `(x, y)` |
| `src.ops.numpy.less` | Function | `(x1, x2)` |
| `src.ops.numpy.less_equal` | Function | `(x1, x2)` |
| `src.ops.numpy.linspace` | Function | `(start, stop, num, endpoint, retstep, dtype, axis)` |
| `src.ops.numpy.log` | Function | `(x)` |
| `src.ops.numpy.log10` | Function | `(x)` |
| `src.ops.numpy.log1p` | Function | `(x)` |
| `src.ops.numpy.log2` | Function | `(x)` |
| `src.ops.numpy.logaddexp` | Function | `(x1, x2)` |
| `src.ops.numpy.logaddexp2` | Function | `(x1, x2)` |
| `src.ops.numpy.logical_and` | Function | `(x1, x2)` |
| `src.ops.numpy.logical_not` | Function | `(x)` |
| `src.ops.numpy.logical_or` | Function | `(x1, x2)` |
| `src.ops.numpy.logical_xor` | Function | `(x1, x2)` |
| `src.ops.numpy.logspace` | Function | `(start, stop, num, endpoint, base, dtype, axis)` |
| `src.ops.numpy.matmul` | Function | `(x1, x2)` |
| `src.ops.numpy.max` | Function | `(x, axis, keepdims, initial)` |
| `src.ops.numpy.maximum` | Function | `(x1, x2)` |
| `src.ops.numpy.mean` | Function | `(x, axis, keepdims)` |
| `src.ops.numpy.median` | Function | `(x, axis, keepdims)` |
| `src.ops.numpy.meshgrid` | Function | `(x, indexing)` |
| `src.ops.numpy.min` | Function | `(x, axis, keepdims, initial)` |
| `src.ops.numpy.minimum` | Function | `(x1, x2)` |
| `src.ops.numpy.mod` | Function | `(x1, x2)` |
| `src.ops.numpy.moveaxis` | Function | `(x, source, destination)` |
| `src.ops.numpy.multiply` | Function | `(x1, x2)` |
| `src.ops.numpy.nan_to_num` | Function | `(x, nan, posinf, neginf)` |
| `src.ops.numpy.ndim` | Function | `(x)` |
| `src.ops.numpy.negative` | Function | `(x)` |
| `src.ops.numpy.nonzero` | Function | `(x)` |
| `src.ops.numpy.not_equal` | Function | `(x1, x2)` |
| `src.ops.numpy.ones` | Function | `(shape, dtype)` |
| `src.ops.numpy.ones_like` | Function | `(x, dtype)` |
| `src.ops.numpy.operation_utils` | Object | `` |
| `src.ops.numpy.outer` | Function | `(x1, x2)` |
| `src.ops.numpy.pad` | Function | `(x, pad_width, mode, constant_values)` |
| `src.ops.numpy.power` | Function | `(x1, x2)` |
| `src.ops.numpy.prod` | Function | `(x, axis, keepdims, dtype)` |
| `src.ops.numpy.quantile` | Function | `(x, q, axis, method, keepdims)` |
| `src.ops.numpy.ravel` | Function | `(x)` |
| `src.ops.numpy.real` | Function | `(x)` |
| `src.ops.numpy.reciprocal` | Function | `(x)` |
| `src.ops.numpy.reduce_shape` | Function | `(shape, axis, keepdims)` |
| `src.ops.numpy.repeat` | Function | `(x, repeats, axis)` |
| `src.ops.numpy.reshape` | Function | `(x, newshape)` |
| `src.ops.numpy.right_shift` | Function | `(x, y)` |
| `src.ops.numpy.roll` | Function | `(x, shift, axis)` |
| `src.ops.numpy.rot90` | Function | `(array, k, axes)` |
| `src.ops.numpy.round` | Function | `(x, decimals)` |
| `src.ops.numpy.searchsorted` | Function | `(sorted_sequence, values, side)` |
| `src.ops.numpy.select` | Function | `(condlist, choicelist, default)` |
| `src.ops.numpy.shape_equal` | Function | `(shape1, shape2, axis, allow_none)` |
| `src.ops.numpy.sign` | Function | `(x)` |
| `src.ops.numpy.signbit` | Function | `(x)` |
| `src.ops.numpy.sin` | Function | `(x)` |
| `src.ops.numpy.sinh` | Function | `(x)` |
| `src.ops.numpy.size` | Function | `(x)` |
| `src.ops.numpy.slogdet` | Function | `(x)` |
| `src.ops.numpy.sort` | Function | `(x, axis)` |
| `src.ops.numpy.split` | Function | `(x, indices_or_sections, axis)` |
| `src.ops.numpy.sqrt` | Function | `(x)` |
| `src.ops.numpy.square` | Function | `(x)` |
| `src.ops.numpy.squeeze` | Function | `(x, axis)` |
| `src.ops.numpy.stack` | Function | `(x, axis)` |
| `src.ops.numpy.std` | Function | `(x, axis, keepdims)` |
| `src.ops.numpy.subtract` | Function | `(x1, x2)` |
| `src.ops.numpy.sum` | Function | `(x, axis, keepdims)` |
| `src.ops.numpy.swapaxes` | Function | `(x, axis1, axis2)` |
| `src.ops.numpy.take` | Function | `(x, indices, axis)` |
| `src.ops.numpy.take_along_axis` | Function | `(x, indices, axis)` |
| `src.ops.numpy.tan` | Function | `(x)` |
| `src.ops.numpy.tanh` | Function | `(x)` |
| `src.ops.numpy.tensordot` | Function | `(x1, x2, axes)` |
| `src.ops.numpy.tile` | Function | `(x, repeats)` |
| `src.ops.numpy.to_tuple_or_list` | Function | `(value)` |
| `src.ops.numpy.trace` | Function | `(x, offset, axis1, axis2)` |
| `src.ops.numpy.transpose` | Function | `(x, axes)` |
| `src.ops.numpy.trapezoid` | Function | `(y, x, dx, axis)` |
| `src.ops.numpy.tri` | Function | `(N, M, k, dtype)` |
| `src.ops.numpy.tril` | Function | `(x, k)` |
| `src.ops.numpy.triu` | Function | `(x, k)` |
| `src.ops.numpy.true_divide` | Function | `(x1, x2)` |
| `src.ops.numpy.trunc` | Function | `(x)` |
| `src.ops.numpy.unravel_index` | Function | `(indices, shape)` |
| `src.ops.numpy.vander` | Function | `(x, N, increasing)` |
| `src.ops.numpy.var` | Function | `(x, axis, keepdims)` |
| `src.ops.numpy.vdot` | Function | `(x1, x2)` |
| `src.ops.numpy.vectorize` | Function | `(pyfunc, excluded, signature)` |
| `src.ops.numpy.view` | Function | `(x, dtype)` |
| `src.ops.numpy.vstack` | Function | `(xs)` |
| `src.ops.numpy.where` | Function | `(condition, x1, x2)` |
| `src.ops.numpy.zeros` | Function | `(shape, dtype)` |
| `src.ops.numpy.zeros_like` | Function | `(x, dtype)` |
| `src.ops.one_hot` | Function | `(x, num_classes, axis, dtype, sparse)` |
| `src.ops.ones` | Function | `(shape, dtype)` |
| `src.ops.ones_like` | Function | `(x, dtype)` |
| `src.ops.operation.KerasSaveable` | Class | `(...)` |
| `src.ops.operation.Node` | Class | `(operation, call_args, call_kwargs, outputs)` |
| `src.ops.operation.Operation` | Class | `(name)` |
| `src.ops.operation.any_symbolic_tensors` | Function | `(args, kwargs)` |
| `src.ops.operation.auto_name` | Function | `(prefix)` |
| `src.ops.operation.backend` | Object | `` |
| `src.ops.operation.dtype_policies` | Object | `` |
| `src.ops.operation.is_nnx_enabled` | Function | `()` |
| `src.ops.operation.keras_export` | Class | `(path)` |
| `src.ops.operation.python_utils` | Object | `` |
| `src.ops.operation.traceback_utils` | Object | `` |
| `src.ops.operation.tree` | Object | `` |
| `src.ops.operation_utils.broadcast_shapes` | Function | `(shape1, shape2)` |
| `src.ops.operation_utils.canonicalize_axis` | Function | `(axis, num_dims)` |
| `src.ops.operation_utils.compute_conv_output_shape` | Function | `(input_shape, filters, kernel_size, strides, padding, data_format, dilation_rate)` |
| `src.ops.operation_utils.compute_expand_dims_output_shape` | Function | `(input_shape, axis)` |
| `src.ops.operation_utils.compute_matmul_output_shape` | Function | `(shape1, shape2)` |
| `src.ops.operation_utils.compute_pooling_output_shape` | Function | `(input_shape, pool_size, strides, padding, data_format)` |
| `src.ops.operation_utils.compute_reshape_output_shape` | Function | `(input_shape, newshape, newshape_arg_name)` |
| `src.ops.operation_utils.compute_take_along_axis_output_shape` | Function | `(input_shape, indices_shape, axis)` |
| `src.ops.operation_utils.compute_transpose_output_shape` | Function | `(input_shape, axes)` |
| `src.ops.operation_utils.get_source_inputs` | Function | `(tensor)` |
| `src.ops.operation_utils.keras_export` | Class | `(path)` |
| `src.ops.operation_utils.reduce_shape` | Function | `(shape, axis, keepdims)` |
| `src.ops.operation_utils.to_tuple_or_list` | Function | `(value)` |
| `src.ops.operation_utils.tree` | Object | `` |
| `src.ops.outer` | Function | `(x1, x2)` |
| `src.ops.pad` | Function | `(x, pad_width, mode, constant_values)` |
| `src.ops.polar` | Function | `(abs_, angle)` |
| `src.ops.power` | Function | `(x1, x2)` |
| `src.ops.prod` | Function | `(x, axis, keepdims, dtype)` |
| `src.ops.psnr` | Function | `(x1, x2, max_val)` |
| `src.ops.qr` | Function | `(x, mode)` |
| `src.ops.quantile` | Function | `(x, q, axis, method, keepdims)` |
| `src.ops.random` | Object | `` |
| `src.ops.ravel` | Function | `(x)` |
| `src.ops.re` | Object | `` |
| `src.ops.real` | Function | `(x)` |
| `src.ops.reciprocal` | Function | `(x)` |
| `src.ops.reduce_shape` | Function | `(shape, axis, keepdims)` |
| `src.ops.relu` | Function | `(x)` |
| `src.ops.relu6` | Function | `(x)` |
| `src.ops.repeat` | Function | `(x, repeats, axis)` |
| `src.ops.reshape` | Function | `(x, newshape)` |
| `src.ops.rfft` | Function | `(x, fft_length)` |
| `src.ops.right_shift` | Function | `(x, y)` |
| `src.ops.rms_normalization` | Function | `(x, scale, axis, epsilon)` |
| `src.ops.roll` | Function | `(x, shift, axis)` |
| `src.ops.rot90` | Function | `(array, k, axes)` |
| `src.ops.round` | Function | `(x, decimals)` |
| `src.ops.rsqrt` | Function | `(x)` |
| `src.ops.saturate_cast` | Function | `(x, dtype)` |
| `src.ops.scan` | Function | `(f, init, xs, length, reverse, unroll)` |
| `src.ops.scatter` | Function | `(indices, values, shape)` |
| `src.ops.scatter_update` | Function | `(inputs, indices, updates)` |
| `src.ops.searchsorted` | Function | `(sorted_sequence, values, side)` |
| `src.ops.segment_max` | Function | `(data, segment_ids, num_segments, sorted)` |
| `src.ops.segment_sum` | Function | `(data, segment_ids, num_segments, sorted)` |
| `src.ops.select` | Function | `(condlist, choicelist, default)` |
| `src.ops.selu` | Function | `(x)` |
| `src.ops.separable_conv` | Function | `(inputs, depthwise_kernel, pointwise_kernel, strides, padding, data_format, dilation_rate)` |
| `src.ops.serialization_lib` | Object | `` |
| `src.ops.shape` | Function | `(x)` |
| `src.ops.shape_equal` | Function | `(shape1, shape2, axis, allow_none)` |
| `src.ops.sigmoid` | Function | `(x)` |
| `src.ops.sign` | Function | `(x)` |
| `src.ops.signbit` | Function | `(x)` |
| `src.ops.silu` | Function | `(x)` |
| `src.ops.sin` | Function | `(x)` |
| `src.ops.sinh` | Function | `(x)` |
| `src.ops.size` | Function | `(x)` |
| `src.ops.slice` | Function | `(inputs, start_indices, shape)` |
| `src.ops.slice_along_axis` | Function | `(x, start, stop, step, axis)` |
| `src.ops.slice_update` | Function | `(inputs, start_indices, updates)` |
| `src.ops.slogdet` | Function | `(x)` |
| `src.ops.soft_shrink` | Function | `(x, threshold)` |
| `src.ops.softmax` | Function | `(x, axis)` |
| `src.ops.softplus` | Function | `(x)` |
| `src.ops.softsign` | Function | `(x)` |
| `src.ops.solve` | Function | `(a, b)` |
| `src.ops.solve_triangular` | Function | `(a, b, lower)` |
| `src.ops.sort` | Function | `(x, axis)` |
| `src.ops.sparse_categorical_crossentropy` | Function | `(target, output, from_logits, axis)` |
| `src.ops.sparse_plus` | Function | `(x)` |
| `src.ops.sparse_sigmoid` | Function | `(x)` |
| `src.ops.sparsemax` | Function | `(x, axis)` |
| `src.ops.split` | Function | `(x, indices_or_sections, axis)` |
| `src.ops.sqrt` | Function | `(x)` |
| `src.ops.square` | Function | `(x)` |
| `src.ops.squareplus` | Function | `(x, b)` |
| `src.ops.squeeze` | Function | `(x, axis)` |
| `src.ops.stack` | Function | `(x, axis)` |
| `src.ops.standardize_data_format` | Function | `(data_format)` |
| `src.ops.std` | Function | `(x, axis, keepdims)` |
| `src.ops.stft` | Function | `(x, sequence_length, sequence_stride, fft_length, window, center)` |
| `src.ops.stop_gradient` | Function | `(variable)` |
| `src.ops.subtract` | Function | `(x1, x2)` |
| `src.ops.sum` | Function | `(x, axis, keepdims)` |
| `src.ops.svd` | Function | `(x, full_matrices, compute_uv)` |
| `src.ops.swapaxes` | Function | `(x, axis1, axis2)` |
| `src.ops.switch` | Function | `(index, branches, operands)` |
| `src.ops.symbolic_arguments.KerasTensor` | Class | `(shape, dtype, sparse, ragged, record_history, name, kwargs)` |
| `src.ops.symbolic_arguments.SymbolicArguments` | Class | `(args, kwargs)` |
| `src.ops.symbolic_arguments.tree` | Object | `` |
| `src.ops.take` | Function | `(x, indices, axis)` |
| `src.ops.take_along_axis` | Function | `(x, indices, axis)` |
| `src.ops.tan` | Function | `(x)` |
| `src.ops.tanh` | Function | `(x)` |
| `src.ops.tanh_shrink` | Function | `(x)` |
| `src.ops.tensordot` | Function | `(x1, x2, axes)` |
| `src.ops.threshold` | Function | `(x, threshold, default_value)` |
| `src.ops.tile` | Function | `(x, repeats)` |
| `src.ops.to_tuple_or_list` | Function | `(value)` |
| `src.ops.top_k` | Function | `(x, k, sorted)` |
| `src.ops.trace` | Function | `(x, offset, axis1, axis2)` |
| `src.ops.traceback_utils` | Object | `` |
| `src.ops.transpose` | Function | `(x, axes)` |
| `src.ops.trapezoid` | Function | `(y, x, dx, axis)` |
| `src.ops.tree` | Object | `` |
| `src.ops.tri` | Function | `(N, M, k, dtype)` |
| `src.ops.tril` | Function | `(x, k)` |
| `src.ops.triu` | Function | `(x, k)` |
| `src.ops.true_divide` | Function | `(x1, x2)` |
| `src.ops.trunc` | Function | `(x)` |
| `src.ops.unfold` | Function | `(x, kernel_size, dilation, padding, stride)` |
| `src.ops.unravel_index` | Function | `(indices, shape)` |
| `src.ops.unstack` | Function | `(x, num, axis)` |
| `src.ops.vander` | Function | `(x, N, increasing)` |
| `src.ops.var` | Function | `(x, axis, keepdims)` |
| `src.ops.vdot` | Function | `(x1, x2)` |
| `src.ops.vectorize` | Function | `(pyfunc, excluded, signature)` |
| `src.ops.vectorized_map` | Function | `(function, elements)` |
| `src.ops.view` | Function | `(x, dtype)` |
| `src.ops.view_as_complex` | Function | `(x)` |
| `src.ops.view_as_real` | Function | `(x)` |
| `src.ops.vstack` | Function | `(xs)` |
| `src.ops.warnings` | Object | `` |
| `src.ops.where` | Function | `(condition, x1, x2)` |
| `src.ops.while_loop` | Function | `(cond, body, loop_vars, maximum_iterations)` |
| `src.ops.zeros` | Function | `(shape, dtype)` |
| `src.ops.zeros_like` | Function | `(x, dtype)` |
| `src.optimizers.ALL_OBJECTS` | Object | `` |
| `src.optimizers.ALL_OBJECTS_DICT` | Object | `` |
| `src.optimizers.Adadelta` | Class | `(learning_rate, rho, epsilon, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps, name, kwargs)` |
| `src.optimizers.Adafactor` | Class | `(learning_rate, beta_2_decay, epsilon_1, epsilon_2, clip_threshold, relative_step, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps, name, kwargs)` |
| `src.optimizers.Adagrad` | Class | `(learning_rate, initial_accumulator_value, epsilon, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps, name, kwargs)` |
| `src.optimizers.Adam` | Class | `(learning_rate, beta_1, beta_2, epsilon, amsgrad, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps, name, kwargs)` |
| `src.optimizers.AdamW` | Class | `(learning_rate, weight_decay, beta_1, beta_2, epsilon, amsgrad, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps, name, kwargs)` |
| `src.optimizers.Adamax` | Class | `(learning_rate, beta_1, beta_2, epsilon, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps, name, kwargs)` |
| `src.optimizers.Ftrl` | Class | `(learning_rate, learning_rate_power, initial_accumulator_value, l1_regularization_strength, l2_regularization_strength, l2_shrinkage_regularization_strength, beta, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps, name, kwargs)` |
| `src.optimizers.LegacyOptimizerWarning` | Class | `(args, kwargs)` |
| `src.optimizers.Lion` | Class | `(learning_rate, beta_1, beta_2, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps, name, kwargs)` |
| `src.optimizers.LossScaleOptimizer` | Class | `(inner_optimizer, initial_scale, dynamic_growth_steps, name, kwargs)` |
| `src.optimizers.Muon` | Class | `(learning_rate, adam_beta_1, adam_beta_2, adam_weight_decay, epsilon, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps, name, exclude_layers, exclude_embeddings, muon_a, muon_b, muon_c, adam_lr_ratio, momentum, ns_steps, nesterov, rms_rate, kwargs)` |
| `src.optimizers.Nadam` | Class | `(learning_rate, beta_1, beta_2, epsilon, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps, name, kwargs)` |
| `src.optimizers.Optimizer` | Class | `(...)` |
| `src.optimizers.RMSprop` | Class | `(learning_rate, rho, momentum, epsilon, centered, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps, name, kwargs)` |
| `src.optimizers.SGD` | Class | `(learning_rate, momentum, nesterov, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps, name, kwargs)` |
| `src.optimizers.adadelta.Adadelta` | Class | `(learning_rate, rho, epsilon, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps, name, kwargs)` |
| `src.optimizers.adadelta.keras_export` | Class | `(path)` |
| `src.optimizers.adadelta.ops` | Object | `` |
| `src.optimizers.adadelta.optimizer` | Object | `` |
| `src.optimizers.adafactor.Adafactor` | Class | `(learning_rate, beta_2_decay, epsilon_1, epsilon_2, clip_threshold, relative_step, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps, name, kwargs)` |
| `src.optimizers.adafactor.backend` | Object | `` |
| `src.optimizers.adafactor.keras_export` | Class | `(path)` |
| `src.optimizers.adafactor.ops` | Object | `` |
| `src.optimizers.adafactor.optimizer` | Object | `` |
| `src.optimizers.adagrad.Adagrad` | Class | `(learning_rate, initial_accumulator_value, epsilon, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps, name, kwargs)` |
| `src.optimizers.adagrad.initializers` | Object | `` |
| `src.optimizers.adagrad.keras_export` | Class | `(path)` |
| `src.optimizers.adagrad.ops` | Object | `` |
| `src.optimizers.adagrad.optimizer` | Object | `` |
| `src.optimizers.adam.Adam` | Class | `(learning_rate, beta_1, beta_2, epsilon, amsgrad, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps, name, kwargs)` |
| `src.optimizers.adam.keras_export` | Class | `(path)` |
| `src.optimizers.adam.ops` | Object | `` |
| `src.optimizers.adam.optimizer` | Object | `` |
| `src.optimizers.adamax.Adamax` | Class | `(learning_rate, beta_1, beta_2, epsilon, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps, name, kwargs)` |
| `src.optimizers.adamax.keras_export` | Class | `(path)` |
| `src.optimizers.adamax.ops` | Object | `` |
| `src.optimizers.adamax.optimizer` | Object | `` |
| `src.optimizers.adamw.AdamW` | Class | `(learning_rate, weight_decay, beta_1, beta_2, epsilon, amsgrad, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps, name, kwargs)` |
| `src.optimizers.adamw.adam` | Object | `` |
| `src.optimizers.adamw.keras_export` | Class | `(path)` |
| `src.optimizers.adamw.optimizer` | Object | `` |
| `src.optimizers.base_optimizer.BaseOptimizer` | Class | `(learning_rate, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps, name, kwargs)` |
| `src.optimizers.base_optimizer.KerasSaveable` | Class | `(...)` |
| `src.optimizers.base_optimizer.auto_name` | Function | `(prefix)` |
| `src.optimizers.base_optimizer.backend` | Object | `` |
| `src.optimizers.base_optimizer.base_optimizer_keyword_args` | Object | `` |
| `src.optimizers.base_optimizer.clip_by_global_norm` | Function | `(value_list, clip_norm)` |
| `src.optimizers.base_optimizer.global_norm` | Function | `(value_list)` |
| `src.optimizers.base_optimizer.initializers` | Object | `` |
| `src.optimizers.base_optimizer.learning_rate_schedule` | Object | `` |
| `src.optimizers.base_optimizer.ops` | Object | `` |
| `src.optimizers.base_optimizer.serialization_lib` | Object | `` |
| `src.optimizers.base_optimizer.tracking` | Object | `` |
| `src.optimizers.deserialize` | Function | `(config, custom_objects)` |
| `src.optimizers.ftrl.Ftrl` | Class | `(learning_rate, learning_rate_power, initial_accumulator_value, l1_regularization_strength, l2_regularization_strength, l2_shrinkage_regularization_strength, beta, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps, name, kwargs)` |
| `src.optimizers.ftrl.initializers` | Object | `` |
| `src.optimizers.ftrl.keras_export` | Class | `(path)` |
| `src.optimizers.ftrl.ops` | Object | `` |
| `src.optimizers.ftrl.optimizer` | Object | `` |
| `src.optimizers.get` | Function | `(identifier)` |
| `src.optimizers.keras_export` | Class | `(path)` |
| `src.optimizers.lamb.Lamb` | Class | `(learning_rate, beta_1, beta_2, epsilon, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps, name, kwargs)` |
| `src.optimizers.lamb.keras_export` | Class | `(path)` |
| `src.optimizers.lamb.ops` | Object | `` |
| `src.optimizers.lamb.optimizer` | Object | `` |
| `src.optimizers.lion.Lion` | Class | `(learning_rate, beta_1, beta_2, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps, name, kwargs)` |
| `src.optimizers.lion.keras_export` | Class | `(path)` |
| `src.optimizers.lion.ops` | Object | `` |
| `src.optimizers.lion.optimizer` | Object | `` |
| `src.optimizers.loss_scale_optimizer.LossScaleOptimizer` | Class | `(inner_optimizer, initial_scale, dynamic_growth_steps, name, kwargs)` |
| `src.optimizers.loss_scale_optimizer.backend` | Object | `` |
| `src.optimizers.loss_scale_optimizer.initializers` | Object | `` |
| `src.optimizers.loss_scale_optimizer.keras_export` | Class | `(path)` |
| `src.optimizers.loss_scale_optimizer.ops` | Object | `` |
| `src.optimizers.loss_scale_optimizer.optimizer` | Object | `` |
| `src.optimizers.loss_scale_optimizer.serialization_lib` | Object | `` |
| `src.optimizers.loss_scale_optimizer.tracking` | Object | `` |
| `src.optimizers.muon.Muon` | Class | `(learning_rate, adam_beta_1, adam_beta_2, adam_weight_decay, epsilon, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps, name, exclude_layers, exclude_embeddings, muon_a, muon_b, muon_c, adam_lr_ratio, momentum, ns_steps, nesterov, rms_rate, kwargs)` |
| `src.optimizers.muon.keras_export` | Class | `(path)` |
| `src.optimizers.muon.ops` | Object | `` |
| `src.optimizers.muon.optimizer` | Object | `` |
| `src.optimizers.nadam.Nadam` | Class | `(learning_rate, beta_1, beta_2, epsilon, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps, name, kwargs)` |
| `src.optimizers.nadam.backend` | Object | `` |
| `src.optimizers.nadam.keras_export` | Class | `(path)` |
| `src.optimizers.nadam.ops` | Object | `` |
| `src.optimizers.nadam.optimizer` | Object | `` |
| `src.optimizers.optimizer.BackendOptimizer` | Class | `(...)` |
| `src.optimizers.optimizer.Optimizer` | Class | `(...)` |
| `src.optimizers.optimizer.backend` | Object | `` |
| `src.optimizers.optimizer.base_optimizer` | Object | `` |
| `src.optimizers.optimizer.base_optimizer_keyword_args` | Object | `` |
| `src.optimizers.optimizer.keras_export` | Class | `(path)` |
| `src.optimizers.rmsprop.RMSprop` | Class | `(learning_rate, rho, momentum, epsilon, centered, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps, name, kwargs)` |
| `src.optimizers.rmsprop.keras_export` | Class | `(path)` |
| `src.optimizers.rmsprop.ops` | Object | `` |
| `src.optimizers.rmsprop.optimizer` | Object | `` |
| `src.optimizers.schedules.CosineDecay` | Class | `(initial_learning_rate, decay_steps, alpha, name, warmup_target, warmup_steps)` |
| `src.optimizers.schedules.CosineDecayRestarts` | Class | `(initial_learning_rate, first_decay_steps, t_mul, m_mul, alpha, name)` |
| `src.optimizers.schedules.ExponentialDecay` | Class | `(initial_learning_rate, decay_steps, decay_rate, staircase, name)` |
| `src.optimizers.schedules.InverseTimeDecay` | Class | `(initial_learning_rate, decay_steps, decay_rate, staircase, name)` |
| `src.optimizers.schedules.PiecewiseConstantDecay` | Class | `(boundaries, values, name)` |
| `src.optimizers.schedules.PolynomialDecay` | Class | `(initial_learning_rate, decay_steps, end_learning_rate, power, cycle, name)` |
| `src.optimizers.schedules.learning_rate_schedule.CosineDecay` | Class | `(initial_learning_rate, decay_steps, alpha, name, warmup_target, warmup_steps)` |
| `src.optimizers.schedules.learning_rate_schedule.CosineDecayRestarts` | Class | `(initial_learning_rate, first_decay_steps, t_mul, m_mul, alpha, name)` |
| `src.optimizers.schedules.learning_rate_schedule.ExponentialDecay` | Class | `(initial_learning_rate, decay_steps, decay_rate, staircase, name)` |
| `src.optimizers.schedules.learning_rate_schedule.InverseTimeDecay` | Class | `(initial_learning_rate, decay_steps, decay_rate, staircase, name)` |
| `src.optimizers.schedules.learning_rate_schedule.LearningRateSchedule` | Class | `(...)` |
| `src.optimizers.schedules.learning_rate_schedule.PiecewiseConstantDecay` | Class | `(boundaries, values, name)` |
| `src.optimizers.schedules.learning_rate_schedule.PolynomialDecay` | Class | `(initial_learning_rate, decay_steps, end_learning_rate, power, cycle, name)` |
| `src.optimizers.schedules.learning_rate_schedule.deserialize` | Function | `(config, custom_objects)` |
| `src.optimizers.schedules.learning_rate_schedule.keras_export` | Class | `(path)` |
| `src.optimizers.schedules.learning_rate_schedule.ops` | Object | `` |
| `src.optimizers.schedules.learning_rate_schedule.serialization_lib` | Object | `` |
| `src.optimizers.schedules.learning_rate_schedule.serialize` | Function | `(learning_rate_schedule)` |
| `src.optimizers.serialization_lib` | Object | `` |
| `src.optimizers.serialize` | Function | `(optimizer)` |
| `src.optimizers.sgd.SGD` | Class | `(learning_rate, momentum, nesterov, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps, name, kwargs)` |
| `src.optimizers.sgd.keras_export` | Class | `(path)` |
| `src.optimizers.sgd.ops` | Object | `` |
| `src.optimizers.sgd.optimizer` | Object | `` |
| `src.quantizers.ALL_OBJECTS` | Object | `` |
| `src.quantizers.ALL_OBJECTS_DICT` | Object | `` |
| `src.quantizers.AbsMaxQuantizer` | Class | `(axis, value_range, epsilon, output_dtype)` |
| `src.quantizers.Float8QuantizationConfig` | Class | `()` |
| `src.quantizers.Int4QuantizationConfig` | Class | `(weight_quantizer, activation_quantizer)` |
| `src.quantizers.Int8QuantizationConfig` | Class | `(weight_quantizer, activation_quantizer)` |
| `src.quantizers.QuantizationConfig` | Class | `(weight_quantizer, activation_quantizer)` |
| `src.quantizers.Quantizer` | Class | `(output_dtype)` |
| `src.quantizers.abs_max_quantize` | Function | `(inputs, axis, value_range, dtype, epsilon, to_numpy)` |
| `src.quantizers.compute_float8_amax_history` | Function | `(x, amax_history)` |
| `src.quantizers.compute_float8_scale` | Function | `(amax, scale, dtype_max, margin)` |
| `src.quantizers.deserialize` | Function | `(config, custom_objects)` |
| `src.quantizers.fake_quant_with_min_max_vars` | Function | `(inputs, min_vals, max_vals, num_bits, narrow_range, axis)` |
| `src.quantizers.get` | Function | `(identifier, kwargs)` |
| `src.quantizers.gptq.Dense` | Class | `(units, activation, use_bias, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint, lora_rank, lora_alpha, quantization_config, kwargs)` |
| `src.quantizers.gptq.EinsumDense` | Class | `(equation, output_shape, activation, bias_axes, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, kernel_constraint, bias_constraint, lora_rank, lora_alpha, gptq_unpacked_column_size, quantization_config, kwargs)` |
| `src.quantizers.gptq.GPTQ` | Class | `(layer, config)` |
| `src.quantizers.gptq.GPTQConfig` | Class | `(dataset, tokenizer, weight_bits: int, num_samples: int, per_channel: bool, sequence_length: int, hessian_damping: float, group_size: int, symmetric: bool, activation_order: bool, quantization_layer_structure: dict)` |
| `src.quantizers.gptq.GPTQQuantizer` | Class | `(config, compute_dtype)` |
| `src.quantizers.gptq.compute_quantization_parameters` | Function | `(x, bits, symmetric, per_channel, group_size, weight, compute_dtype)` |
| `src.quantizers.gptq.dequantize_with_zero_point` | Function | `(input_tensor, scale, zero)` |
| `src.quantizers.gptq.gptq_quantize_matrix` | Function | `(weights_transpose, inv_hessian, blocksize, group_size, activation_order, order_metric, compute_scale_zero)` |
| `src.quantizers.gptq.linalg` | Object | `` |
| `src.quantizers.gptq.ops` | Object | `` |
| `src.quantizers.gptq.quantize_with_zero_point` | Function | `(input_tensor, scale, zero, maxq)` |
| `src.quantizers.gptq.quantizers` | Object | `` |
| `src.quantizers.gptq_config.GPTQConfig` | Class | `(dataset, tokenizer, weight_bits: int, num_samples: int, per_channel: bool, sequence_length: int, hessian_damping: float, group_size: int, symmetric: bool, activation_order: bool, quantization_layer_structure: dict)` |
| `src.quantizers.gptq_config.QuantizationConfig` | Class | `(weight_quantizer, activation_quantizer)` |
| `src.quantizers.gptq_config.keras_export` | Class | `(path)` |
| `src.quantizers.gptq_core.DTypePolicyMap` | Class | `(default_policy, policy_map)` |
| `src.quantizers.gptq_core.Dense` | Class | `(units, activation, use_bias, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint, lora_rank, lora_alpha, quantization_config, kwargs)` |
| `src.quantizers.gptq_core.EinsumDense` | Class | `(equation, output_shape, activation, bias_axes, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, kernel_constraint, bias_constraint, lora_rank, lora_alpha, gptq_unpacked_column_size, quantization_config, kwargs)` |
| `src.quantizers.gptq_core.GPTQ` | Class | `(layer, config)` |
| `src.quantizers.gptq_core.GPTQConfig` | Class | `(dataset, tokenizer, weight_bits: int, num_samples: int, per_channel: bool, sequence_length: int, hessian_damping: float, group_size: int, symmetric: bool, activation_order: bool, quantization_layer_structure: dict)` |
| `src.quantizers.gptq_core.GPTQDTypePolicy` | Class | `(mode, source_name)` |
| `src.quantizers.gptq_core.apply_gptq_layerwise` | Function | `(dataloader, config, structure, filters)` |
| `src.quantizers.gptq_core.find_layers_in_block` | Function | `(block)` |
| `src.quantizers.gptq_core.get_dataloader` | Function | `(tokenizer, sequence_length, dataset, num_samples, strategy, seed, stride, eos_id)` |
| `src.quantizers.gptq_core.get_group_size_for_layer` | Function | `(layer, config)` |
| `src.quantizers.gptq_core.get_weight_bits_for_layer` | Function | `(layer, config)` |
| `src.quantizers.gptq_core.gptq_quantize` | Function | `(config, quantization_layer_structure, filters)` |
| `src.quantizers.gptq_core.keras_utils` | Object | `` |
| `src.quantizers.gptq_core.ops` | Object | `` |
| `src.quantizers.gptq_core.should_quantize_layer` | Function | `(layer, filters)` |
| `src.quantizers.gptq_core.stream_hessians` | Function | `(layers_map, gptq_objects)` |
| `src.quantizers.keras_export` | Class | `(path)` |
| `src.quantizers.pack_int4` | Function | `(arr, axis, dtype)` |
| `src.quantizers.quantization_config.Float8QuantizationConfig` | Class | `()` |
| `src.quantizers.quantization_config.Int4QuantizationConfig` | Class | `(weight_quantizer, activation_quantizer)` |
| `src.quantizers.quantization_config.Int8QuantizationConfig` | Class | `(weight_quantizer, activation_quantizer)` |
| `src.quantizers.quantization_config.QUANTIZATION_MODES` | Object | `` |
| `src.quantizers.quantization_config.QuantizationConfig` | Class | `(weight_quantizer, activation_quantizer)` |
| `src.quantizers.quantization_config.keras_export` | Class | `(path)` |
| `src.quantizers.quantization_config.serialization_lib` | Object | `` |
| `src.quantizers.quantization_config.validate_and_resolve_config` | Function | `(mode, config)` |
| `src.quantizers.quantize_and_dequantize` | Function | `(inputs, scale, quantized_dtype, compute_dtype)` |
| `src.quantizers.quantizers.AbsMaxQuantizer` | Class | `(axis, value_range, epsilon, output_dtype)` |
| `src.quantizers.quantizers.FakeQuantWithMinMaxVars` | Class | `(num_bits, narrow_range, axis)` |
| `src.quantizers.quantizers.GPTQConfig` | Class | `(dataset, tokenizer, weight_bits: int, num_samples: int, per_channel: bool, sequence_length: int, hessian_damping: float, group_size: int, symmetric: bool, activation_order: bool, quantization_layer_structure: dict)` |
| `src.quantizers.quantizers.GPTQQuantizer` | Class | `(config, compute_dtype)` |
| `src.quantizers.quantizers.KerasTensor` | Class | `(shape, dtype, sparse, ragged, record_history, name, kwargs)` |
| `src.quantizers.quantizers.Operation` | Class | `(name)` |
| `src.quantizers.quantizers.Quantizer` | Class | `(output_dtype)` |
| `src.quantizers.quantizers.abs_max_quantize` | Function | `(inputs, axis, value_range, dtype, epsilon, to_numpy)` |
| `src.quantizers.quantizers.adjust_and_nudge` | Function | `(min_range, max_range, num_bits, narrow_range)` |
| `src.quantizers.quantizers.any_symbolic_tensors` | Function | `(args, kwargs)` |
| `src.quantizers.quantizers.backend` | Object | `` |
| `src.quantizers.quantizers.canonicalize_axis` | Function | `(axis, num_dims)` |
| `src.quantizers.quantizers.compute_float8_amax_history` | Function | `(x, amax_history)` |
| `src.quantizers.quantizers.compute_float8_scale` | Function | `(amax, scale, dtype_max, margin)` |
| `src.quantizers.quantizers.compute_quantization_parameters` | Function | `(x, bits, symmetric, per_channel, group_size, weight, compute_dtype)` |
| `src.quantizers.quantizers.dequantize_with_sz_map` | Function | `(weights_matrix, scale, zero, g_idx)` |
| `src.quantizers.quantizers.dequantize_with_zero_point` | Function | `(input_tensor, scale, zero)` |
| `src.quantizers.quantizers.fake_quant_with_min_max_vars` | Function | `(inputs, min_vals, max_vals, num_bits, narrow_range, axis)` |
| `src.quantizers.quantizers.keras_export` | Class | `(path)` |
| `src.quantizers.quantizers.ops` | Object | `` |
| `src.quantizers.quantizers.pack_int4` | Function | `(arr, axis, dtype)` |
| `src.quantizers.quantizers.quantize_and_dequantize` | Function | `(inputs, scale, quantized_dtype, compute_dtype)` |
| `src.quantizers.quantizers.quantize_with_sz_map` | Function | `(weights_matrix, scale, zero, g_idx, maxq)` |
| `src.quantizers.quantizers.quantize_with_zero_point` | Function | `(input_tensor, scale, zero, maxq)` |
| `src.quantizers.quantizers.standardize_axis_for_numpy` | Function | `(axis)` |
| `src.quantizers.quantizers.unpack_int4` | Function | `(packed, orig_len, axis, dtype)` |
| `src.quantizers.serialization_lib` | Object | `` |
| `src.quantizers.serialize` | Function | `(initializer)` |
| `src.quantizers.to_snake_case` | Function | `(name)` |
| `src.quantizers.unpack_int4` | Function | `(packed, orig_len, axis, dtype)` |
| `src.quantizers.utils.should_quantize_layer` | Function | `(layer, filters)` |
| `src.random.SeedGenerator` | Class | `(seed, name, kwargs)` |
| `src.random.categorical` | Function | `(logits, num_samples, dtype, seed)` |
| `src.random.dropout` | Function | `(inputs, rate, noise_shape, seed)` |
| `src.random.gamma` | Function | `(shape, alpha, dtype, seed)` |
| `src.random.normal` | Function | `(shape, mean, stddev, dtype, seed)` |
| `src.random.randint` | Function | `(shape, minval, maxval, dtype, seed)` |
| `src.random.random.backend` | Object | `` |
| `src.random.random.beta` | Function | `(shape, alpha, beta, dtype, seed)` |
| `src.random.random.binomial` | Function | `(shape, counts, probabilities, dtype, seed)` |
| `src.random.random.categorical` | Function | `(logits, num_samples, dtype, seed)` |
| `src.random.random.dropout` | Function | `(inputs, rate, noise_shape, seed)` |
| `src.random.random.gamma` | Function | `(shape, alpha, dtype, seed)` |
| `src.random.random.keras_export` | Class | `(path)` |
| `src.random.random.normal` | Function | `(shape, mean, stddev, dtype, seed)` |
| `src.random.random.randint` | Function | `(shape, minval, maxval, dtype, seed)` |
| `src.random.random.shuffle` | Function | `(x, axis, seed)` |
| `src.random.random.truncated_normal` | Function | `(shape, mean, stddev, dtype, seed)` |
| `src.random.random.uniform` | Function | `(shape, minval, maxval, dtype, seed)` |
| `src.random.seed_generator.GLOBAL_SEED_GENERATOR` | Object | `` |
| `src.random.seed_generator.SeedGenerator` | Class | `(seed, name, kwargs)` |
| `src.random.seed_generator.auto_name` | Function | `(prefix)` |
| `src.random.seed_generator.backend` | Object | `` |
| `src.random.seed_generator.draw_seed` | Function | `(seed)` |
| `src.random.seed_generator.global_seed_generator` | Function | `()` |
| `src.random.seed_generator.global_state` | Object | `` |
| `src.random.seed_generator.jax_utils` | Object | `` |
| `src.random.seed_generator.keras_export` | Class | `(path)` |
| `src.random.seed_generator.make_default_seed` | Function | `()` |
| `src.random.shuffle` | Function | `(x, axis, seed)` |
| `src.random.truncated_normal` | Function | `(shape, mean, stddev, dtype, seed)` |
| `src.random.uniform` | Function | `(shape, minval, maxval, dtype, seed)` |
| `src.regularizers.ALL_OBJECTS` | Object | `` |
| `src.regularizers.ALL_OBJECTS_DICT` | Object | `` |
| `src.regularizers.L1` | Class | `(l1)` |
| `src.regularizers.L1L2` | Class | `(l1, l2)` |
| `src.regularizers.L2` | Class | `(l2)` |
| `src.regularizers.OrthogonalRegularizer` | Class | `(factor, mode)` |
| `src.regularizers.Regularizer` | Class | `(...)` |
| `src.regularizers.deserialize` | Function | `(config, custom_objects)` |
| `src.regularizers.get` | Function | `(identifier)` |
| `src.regularizers.keras_export` | Class | `(path)` |
| `src.regularizers.regularizers.L1` | Class | `(l1)` |
| `src.regularizers.regularizers.L1L2` | Class | `(l1, l2)` |
| `src.regularizers.regularizers.L2` | Class | `(l2)` |
| `src.regularizers.regularizers.OrthogonalRegularizer` | Class | `(factor, mode)` |
| `src.regularizers.regularizers.Regularizer` | Class | `(...)` |
| `src.regularizers.regularizers.keras_export` | Class | `(path)` |
| `src.regularizers.regularizers.normalize` | Function | `(x, axis, order)` |
| `src.regularizers.regularizers.ops` | Object | `` |
| `src.regularizers.regularizers.validate_float_arg` | Function | `(value, name)` |
| `src.regularizers.serialization_lib` | Object | `` |
| `src.regularizers.serialize` | Function | `(regularizer)` |
| `src.regularizers.to_snake_case` | Function | `(name)` |
| `src.saving.CustomObjectScope` | Class | `(custom_objects)` |
| `src.saving.custom_object_scope` | Object | `` |
| `src.saving.deserialize_keras_object` | Function | `(config, custom_objects, safe_mode, kwargs)` |
| `src.saving.file_editor.H5IOStore` | Class | `(path_or_io, archive, mode)` |
| `src.saving.file_editor.KerasFileEditor` | Class | `(filepath)` |
| `src.saving.file_editor.backend` | Object | `` |
| `src.saving.file_editor.display_weight` | Function | `(weight, axis, threshold)` |
| `src.saving.file_editor.get_id_counter` | Function | `()` |
| `src.saving.file_editor.get_weight_spec_of_container` | Function | `(container, spec, visited_saveables)` |
| `src.saving.file_editor.get_weight_spec_of_saveable` | Function | `(saveable, spec, visited_saveables)` |
| `src.saving.file_editor.increment_id_counter` | Function | `()` |
| `src.saving.file_editor.initialize_id_counter` | Function | `()` |
| `src.saving.file_editor.is_ipython_notebook` | Function | `()` |
| `src.saving.file_editor.keras_export` | Class | `(path)` |
| `src.saving.file_editor.naming` | Object | `` |
| `src.saving.file_editor.saving_lib` | Object | `` |
| `src.saving.file_editor.summary_utils` | Object | `` |
| `src.saving.get_custom_objects` | Function | `()` |
| `src.saving.get_registered_name` | Function | `(obj)` |
| `src.saving.get_registered_object` | Function | `(name, custom_objects, module_objects)` |
| `src.saving.keras_saveable.KerasSaveable` | Class | `(...)` |
| `src.saving.load_model` | Function | `(filepath, custom_objects, compile, safe_mode)` |
| `src.saving.object_registration.CustomObjectScope` | Class | `(custom_objects)` |
| `src.saving.object_registration.GLOBAL_CUSTOM_NAMES` | Object | `` |
| `src.saving.object_registration.GLOBAL_CUSTOM_OBJECTS` | Object | `` |
| `src.saving.object_registration.custom_object_scope` | Object | `` |
| `src.saving.object_registration.get_custom_objects` | Function | `()` |
| `src.saving.object_registration.get_registered_name` | Function | `(obj)` |
| `src.saving.object_registration.get_registered_object` | Function | `(name, custom_objects, module_objects)` |
| `src.saving.object_registration.global_state` | Object | `` |
| `src.saving.object_registration.keras_export` | Class | `(path)` |
| `src.saving.object_registration.register_keras_serializable` | Function | `(package, name)` |
| `src.saving.register_keras_serializable` | Function | `(package, name)` |
| `src.saving.saving_api.file_utils` | Object | `` |
| `src.saving.saving_api.io_utils` | Object | `` |
| `src.saving.saving_api.keras_export` | Class | `(path)` |
| `src.saving.saving_api.legacy_h5_format` | Object | `` |
| `src.saving.saving_api.load_model` | Function | `(filepath, custom_objects, compile, safe_mode)` |
| `src.saving.saving_api.load_weights` | Function | `(model, filepath, skip_mismatch, kwargs)` |
| `src.saving.saving_api.save_model` | Function | `(model, filepath, overwrite, zipped, kwargs)` |
| `src.saving.saving_api.save_weights` | Function | `(model, filepath, overwrite, max_shard_size, kwargs)` |
| `src.saving.saving_api.saving_lib` | Object | `` |
| `src.saving.saving_lib.DiskIOStore` | Class | `(root_path, archive, mode)` |
| `src.saving.saving_lib.H5IOStore` | Class | `(path_or_io, archive, mode)` |
| `src.saving.saving_lib.NpzIOStore` | Class | `(root_path, archive, mode)` |
| `src.saving.saving_lib.ObjectSharingScope` | Class | `(...)` |
| `src.saving.saving_lib.ShardedH5IOStore` | Class | `(path_or_io, max_shard_size, archive, mode)` |
| `src.saving.saving_lib.backend` | Object | `` |
| `src.saving.saving_lib.check_pydot` | Function | `()` |
| `src.saving.saving_lib.deserialize_keras_object` | Function | `(config, custom_objects, safe_mode, kwargs)` |
| `src.saving.saving_lib.dtype_utils` | Object | `` |
| `src.saving.saving_lib.file_utils` | Object | `` |
| `src.saving.saving_lib.get_attr_skipset` | Function | `(obj_type)` |
| `src.saving.saving_lib.get_temp_dir` | Function | `()` |
| `src.saving.saving_lib.global_state` | Object | `` |
| `src.saving.saving_lib.io_utils` | Object | `` |
| `src.saving.saving_lib.is_memory_sufficient` | Function | `(model)` |
| `src.saving.saving_lib.keras_version` | Object | `` |
| `src.saving.saving_lib.load_model` | Function | `(filepath, custom_objects, compile, safe_mode)` |
| `src.saving.saving_lib.load_weights_only` | Function | `(model, filepath, skip_mismatch, objects_to_skip)` |
| `src.saving.saving_lib.naming` | Object | `` |
| `src.saving.saving_lib.plot_model` | Function | `(model, to_file, show_shapes, show_dtype, show_layer_names, rankdir, expand_nested, dpi, show_layer_activations, show_trainable, kwargs)` |
| `src.saving.saving_lib.readable_memory_size` | Function | `(weight_memory_size)` |
| `src.saving.saving_lib.save_model` | Function | `(model, filepath, weights_format, zipped)` |
| `src.saving.saving_lib.save_weights_only` | Function | `(model, filepath, max_shard_size, objects_to_skip)` |
| `src.saving.saving_lib.serialize_keras_object` | Function | `(obj)` |
| `src.saving.saving_lib.weight_memory_size` | Function | `(weights)` |
| `src.saving.serialization_lib.BUILTIN_MODULES` | Object | `` |
| `src.saving.serialization_lib.KerasSaveable` | Class | `(...)` |
| `src.saving.serialization_lib.LOADING_APIS` | Object | `` |
| `src.saving.serialization_lib.ObjectSharingScope` | Class | `(...)` |
| `src.saving.serialization_lib.PLAIN_TYPES` | Object | `` |
| `src.saving.serialization_lib.SafeModeScope` | Class | `(safe_mode)` |
| `src.saving.serialization_lib.SerializableDict` | Class | `(config)` |
| `src.saving.serialization_lib.api_export` | Object | `` |
| `src.saving.serialization_lib.backend` | Object | `` |
| `src.saving.serialization_lib.deserialize_keras_object` | Function | `(config, custom_objects, safe_mode, kwargs)` |
| `src.saving.serialization_lib.enable_unsafe_deserialization` | Function | `()` |
| `src.saving.serialization_lib.get_build_and_compile_config` | Function | `(obj, config)` |
| `src.saving.serialization_lib.get_shared_object` | Function | `(obj_id)` |
| `src.saving.serialization_lib.global_state` | Object | `` |
| `src.saving.serialization_lib.in_safe_mode` | Function | `()` |
| `src.saving.serialization_lib.keras_export` | Class | `(path)` |
| `src.saving.serialization_lib.object_registration` | Object | `` |
| `src.saving.serialization_lib.python_utils` | Object | `` |
| `src.saving.serialization_lib.record_object_after_deserialization` | Function | `(obj, obj_id)` |
| `src.saving.serialization_lib.record_object_after_serialization` | Function | `(obj, config)` |
| `src.saving.serialization_lib.serialize_dict` | Function | `(obj)` |
| `src.saving.serialization_lib.serialize_keras_object` | Function | `(obj)` |
| `src.saving.serialization_lib.serialize_with_public_class` | Function | `(cls, inner_config)` |
| `src.saving.serialization_lib.serialize_with_public_fn` | Function | `(fn, config, fn_module_name)` |
| `src.saving.serialization_lib.tf` | Object | `` |
| `src.saving.serialize_keras_object` | Function | `(obj)` |
| `src.testing.TestCase` | Class | `(args, kwargs)` |
| `src.testing.jax_uses_gpu` | Function | `()` |
| `src.testing.tensorflow_uses_gpu` | Function | `()` |
| `src.testing.test_case.KerasTensor` | Class | `(shape, dtype, sparse, ragged, record_history, name, kwargs)` |
| `src.testing.test_case.Loss` | Class | `(name, reduction, dtype)` |
| `src.testing.test_case.Model` | Class | `(args, kwargs)` |
| `src.testing.test_case.TestCase` | Class | `(args, kwargs)` |
| `src.testing.test_case.backend` | Object | `` |
| `src.testing.test_case.clear_session` | Function | `(free_memory)` |
| `src.testing.test_case.create_eager_tensors` | Function | `(input_shape, dtype, sparse, ragged)` |
| `src.testing.test_case.create_keras_tensors` | Function | `(input_shape, dtype, sparse, ragged)` |
| `src.testing.test_case.distribution` | Object | `` |
| `src.testing.test_case.from_json_with_tuples` | Function | `(value)` |
| `src.testing.test_case.get_seed_generators` | Function | `(layer)` |
| `src.testing.test_case.is_float_dtype` | Function | `(dtype)` |
| `src.testing.test_case.is_shape_tuple` | Function | `(x)` |
| `src.testing.test_case.jax_uses_gpu` | Function | `()` |
| `src.testing.test_case.map_shape_dtype_structure` | Function | `(fn, shape, dtype)` |
| `src.testing.test_case.ops` | Object | `` |
| `src.testing.test_case.standardize_dtype` | Function | `(dtype)` |
| `src.testing.test_case.tensorflow_uses_gpu` | Function | `()` |
| `src.testing.test_case.to_json_with_tuples` | Function | `(value)` |
| `src.testing.test_case.torch_uses_gpu` | Function | `()` |
| `src.testing.test_case.traceback_utils` | Object | `` |
| `src.testing.test_case.tree` | Object | `` |
| `src.testing.test_case.uses_cpu` | Function | `()` |
| `src.testing.test_case.uses_gpu` | Function | `()` |
| `src.testing.test_case.uses_tpu` | Function | `()` |
| `src.testing.test_case.utils` | Object | `` |
| `src.testing.test_utils.get_test_data` | Function | `(train_samples, test_samples, input_shape, num_classes, random_seed)` |
| `src.testing.test_utils.named_product` | Function | `(args, kwargs)` |
| `src.testing.torch_uses_gpu` | Function | `()` |
| `src.testing.uses_gpu` | Function | `()` |
| `src.testing.uses_tpu` | Function | `()` |
| `src.trainers.compile_utils.CompileLoss` | Class | `(loss, loss_weights, reduction, output_names)` |
| `src.trainers.compile_utils.CompileMetrics` | Class | `(metrics, weighted_metrics, name, output_names)` |
| `src.trainers.compile_utils.KerasTensor` | Class | `(shape, dtype, sparse, ragged, record_history, name, kwargs)` |
| `src.trainers.compile_utils.MetricsList` | Class | `(metrics, name, output_name)` |
| `src.trainers.compile_utils.Tracker` | Class | `(config, exclusions)` |
| `src.trainers.compile_utils.get_loss` | Function | `(identifier, y_true, y_pred)` |
| `src.trainers.compile_utils.get_metric` | Function | `(identifier, y_true, y_pred)` |
| `src.trainers.compile_utils.get_object_name` | Function | `(obj)` |
| `src.trainers.compile_utils.is_binary_or_sparse_categorical` | Function | `(y_true, y_pred)` |
| `src.trainers.compile_utils.is_function_like` | Function | `(value)` |
| `src.trainers.compile_utils.loss_module` | Object | `` |
| `src.trainers.compile_utils.losses_module` | Object | `` |
| `src.trainers.compile_utils.metrics_module` | Object | `` |
| `src.trainers.compile_utils.ops` | Object | `` |
| `src.trainers.compile_utils.tree` | Object | `` |
| `src.trainers.data_adapters.ArrayDataAdapter` | Class | `(x, y, sample_weight, batch_size, steps, shuffle, class_weight)` |
| `src.trainers.data_adapters.GeneratorDataAdapter` | Class | `(generator)` |
| `src.trainers.data_adapters.GrainDatasetAdapter` | Class | `(dataset)` |
| `src.trainers.data_adapters.PyDatasetAdapter` | Class | `(x, class_weight, shuffle)` |
| `src.trainers.data_adapters.TFDatasetAdapter` | Class | `(dataset, class_weight, distribution)` |
| `src.trainers.data_adapters.TorchDataLoaderAdapter` | Class | `(dataloader)` |
| `src.trainers.data_adapters.array_data_adapter.ArrayDataAdapter` | Class | `(x, y, sample_weight, batch_size, steps, shuffle, class_weight)` |
| `src.trainers.data_adapters.array_data_adapter.DataAdapter` | Class | `(...)` |
| `src.trainers.data_adapters.array_data_adapter.array_slicing` | Object | `` |
| `src.trainers.data_adapters.array_data_adapter.can_convert_arrays` | Function | `(arrays)` |
| `src.trainers.data_adapters.array_data_adapter.data_adapter_utils` | Object | `` |
| `src.trainers.data_adapters.array_data_adapter.tree` | Object | `` |
| `src.trainers.data_adapters.array_slicing.ARRAY_TYPES` | Object | `` |
| `src.trainers.data_adapters.array_slicing.JaxSparseSliceable` | Class | `(...)` |
| `src.trainers.data_adapters.array_slicing.NumpySliceable` | Class | `(...)` |
| `src.trainers.data_adapters.array_slicing.PandasDataFrameSliceable` | Class | `(...)` |
| `src.trainers.data_adapters.array_slicing.PandasSeriesSliceable` | Class | `(...)` |
| `src.trainers.data_adapters.array_slicing.PandasSliceable` | Class | `(...)` |
| `src.trainers.data_adapters.array_slicing.ScipySparseSliceable` | Class | `(array)` |
| `src.trainers.data_adapters.array_slicing.Sliceable` | Class | `(array)` |
| `src.trainers.data_adapters.array_slicing.TensorflowRaggedSliceable` | Class | `(...)` |
| `src.trainers.data_adapters.array_slicing.TensorflowSliceable` | Class | `(...)` |
| `src.trainers.data_adapters.array_slicing.TensorflowSparseSliceable` | Class | `(array)` |
| `src.trainers.data_adapters.array_slicing.TensorflowSparseWrapper` | Object | `` |
| `src.trainers.data_adapters.array_slicing.TorchSliceable` | Class | `(...)` |
| `src.trainers.data_adapters.array_slicing.backend` | Object | `` |
| `src.trainers.data_adapters.array_slicing.can_slice_array` | Function | `(x)` |
| `src.trainers.data_adapters.array_slicing.convert_to_sliceable` | Function | `(arrays, target_backend)` |
| `src.trainers.data_adapters.array_slicing.data_adapter_utils` | Object | `` |
| `src.trainers.data_adapters.array_slicing.slice_tensorflow_sparse_wrapper` | Function | `(sparse_wrapper, indices)` |
| `src.trainers.data_adapters.array_slicing.tf` | Object | `` |
| `src.trainers.data_adapters.array_slicing.to_tensorflow_sparse_wrapper` | Function | `(sparse)` |
| `src.trainers.data_adapters.array_slicing.train_validation_split` | Function | `(arrays, validation_split)` |
| `src.trainers.data_adapters.array_slicing.tree` | Object | `` |
| `src.trainers.data_adapters.data_adapter.DataAdapter` | Class | `(...)` |
| `src.trainers.data_adapters.data_adapter_utils.NUM_BATCHES_FOR_TENSOR_SPEC` | Object | `` |
| `src.trainers.data_adapters.data_adapter_utils.backend` | Object | `` |
| `src.trainers.data_adapters.data_adapter_utils.check_data_cardinality` | Function | `(data)` |
| `src.trainers.data_adapters.data_adapter_utils.class_weight_to_sample_weights` | Function | `(y, class_weight)` |
| `src.trainers.data_adapters.data_adapter_utils.convert_to_tf_tensor_spec` | Function | `(keras_tensor, batch_axis_to_none)` |
| `src.trainers.data_adapters.data_adapter_utils.get_jax_iterator` | Function | `(iterable)` |
| `src.trainers.data_adapters.data_adapter_utils.get_keras_tensor_spec` | Function | `(batches)` |
| `src.trainers.data_adapters.data_adapter_utils.get_numpy_iterator` | Function | `(iterable)` |
| `src.trainers.data_adapters.data_adapter_utils.get_tensor_spec` | Function | `(batches)` |
| `src.trainers.data_adapters.data_adapter_utils.get_torch_dataloader` | Function | `(iterable)` |
| `src.trainers.data_adapters.data_adapter_utils.is_jax_array` | Function | `(value)` |
| `src.trainers.data_adapters.data_adapter_utils.is_jax_sparse` | Function | `(value)` |
| `src.trainers.data_adapters.data_adapter_utils.is_scipy_sparse` | Function | `(x)` |
| `src.trainers.data_adapters.data_adapter_utils.is_tensorflow_ragged` | Function | `(value)` |
| `src.trainers.data_adapters.data_adapter_utils.is_tensorflow_sparse` | Function | `(value)` |
| `src.trainers.data_adapters.data_adapter_utils.is_tensorflow_tensor` | Function | `(value)` |
| `src.trainers.data_adapters.data_adapter_utils.is_torch_tensor` | Function | `(value)` |
| `src.trainers.data_adapters.data_adapter_utils.jax_sparse_to_tf_sparse` | Function | `(x)` |
| `src.trainers.data_adapters.data_adapter_utils.keras_export` | Class | `(path)` |
| `src.trainers.data_adapters.data_adapter_utils.list_to_tuple` | Function | `(maybe_list)` |
| `src.trainers.data_adapters.data_adapter_utils.ops` | Object | `` |
| `src.trainers.data_adapters.data_adapter_utils.pack_x_y_sample_weight` | Function | `(x, y, sample_weight)` |
| `src.trainers.data_adapters.data_adapter_utils.scipy_sparse_to_jax_sparse` | Function | `(x)` |
| `src.trainers.data_adapters.data_adapter_utils.scipy_sparse_to_tf_sparse` | Function | `(x)` |
| `src.trainers.data_adapters.data_adapter_utils.tf_sparse_to_jax_sparse` | Function | `(x)` |
| `src.trainers.data_adapters.data_adapter_utils.tree` | Object | `` |
| `src.trainers.data_adapters.data_adapter_utils.unpack_x_y_sample_weight` | Function | `(data)` |
| `src.trainers.data_adapters.distribution_lib` | Object | `` |
| `src.trainers.data_adapters.generator_data_adapter.DataAdapter` | Class | `(...)` |
| `src.trainers.data_adapters.generator_data_adapter.GeneratorDataAdapter` | Class | `(generator)` |
| `src.trainers.data_adapters.generator_data_adapter.data_adapter_utils` | Object | `` |
| `src.trainers.data_adapters.generator_data_adapter.peek_and_restore` | Function | `(generator)` |
| `src.trainers.data_adapters.generator_data_adapter.tree` | Object | `` |
| `src.trainers.data_adapters.get_data_adapter` | Function | `(x, y, sample_weight, batch_size, steps_per_epoch, shuffle, class_weight)` |
| `src.trainers.data_adapters.grain_dataset_adapter.DataAdapter` | Class | `(...)` |
| `src.trainers.data_adapters.grain_dataset_adapter.GrainDatasetAdapter` | Class | `(dataset)` |
| `src.trainers.data_adapters.grain_dataset_adapter.data_adapter_utils` | Object | `` |
| `src.trainers.data_adapters.grain_dataset_adapter.grain` | Object | `` |
| `src.trainers.data_adapters.grain_dataset_adapter.tf` | Object | `` |
| `src.trainers.data_adapters.grain_dataset_adapter.tree` | Object | `` |
| `src.trainers.data_adapters.is_grain_dataset` | Function | `(x)` |
| `src.trainers.data_adapters.is_tf_dataset` | Function | `(x)` |
| `src.trainers.data_adapters.is_torch_dataloader` | Function | `(x)` |
| `src.trainers.data_adapters.py_dataset_adapter.DataAdapter` | Class | `(...)` |
| `src.trainers.data_adapters.py_dataset_adapter.OrderedEnqueuer` | Class | `(py_dataset, workers, use_multiprocessing, max_queue_size, shuffle)` |
| `src.trainers.data_adapters.py_dataset_adapter.PyDataset` | Class | `(workers, use_multiprocessing, max_queue_size)` |
| `src.trainers.data_adapters.py_dataset_adapter.PyDatasetAdapter` | Class | `(x, class_weight, shuffle)` |
| `src.trainers.data_adapters.py_dataset_adapter.PyDatasetEnqueuer` | Class | `(py_dataset, workers, use_multiprocessing, max_queue_size)` |
| `src.trainers.data_adapters.py_dataset_adapter.data_adapter_utils` | Object | `` |
| `src.trainers.data_adapters.py_dataset_adapter.get_index` | Function | `(uid, i)` |
| `src.trainers.data_adapters.py_dataset_adapter.get_pool_class` | Function | `(use_multiprocessing)` |
| `src.trainers.data_adapters.py_dataset_adapter.get_worker_id_queue` | Function | `()` |
| `src.trainers.data_adapters.py_dataset_adapter.init_pool_generator` | Function | `(gens, random_seed, id_queue)` |
| `src.trainers.data_adapters.py_dataset_adapter.keras_export` | Class | `(path)` |
| `src.trainers.data_adapters.raise_unsupported_arg` | Function | `(arg_name, arg_description, input_type)` |
| `src.trainers.data_adapters.tf_dataset_adapter.DataAdapter` | Class | `(...)` |
| `src.trainers.data_adapters.tf_dataset_adapter.TFDatasetAdapter` | Class | `(dataset, class_weight, distribution)` |
| `src.trainers.data_adapters.tf_dataset_adapter.data_adapter_utils` | Object | `` |
| `src.trainers.data_adapters.tf_dataset_adapter.make_class_weight_map_fn` | Function | `(class_weight)` |
| `src.trainers.data_adapters.tf_dataset_adapter.tree` | Object | `` |
| `src.trainers.data_adapters.torch_data_loader_adapter.DataAdapter` | Class | `(...)` |
| `src.trainers.data_adapters.torch_data_loader_adapter.TorchDataLoaderAdapter` | Class | `(dataloader)` |
| `src.trainers.data_adapters.torch_data_loader_adapter.data_adapter_utils` | Object | `` |
| `src.trainers.data_adapters.torch_data_loader_adapter.tree` | Object | `` |
| `src.trainers.epoch_iterator.EpochIterator` | Class | `(x, y, sample_weight, batch_size, steps_per_epoch, shuffle, class_weight, steps_per_execution)` |
| `src.trainers.epoch_iterator.config` | Object | `` |
| `src.trainers.epoch_iterator.data_adapters` | Object | `` |
| `src.trainers.trainer.CompileLoss` | Class | `(loss, loss_weights, reduction, output_names)` |
| `src.trainers.trainer.CompileMetrics` | Class | `(metrics, weighted_metrics, name, output_names)` |
| `src.trainers.trainer.LossScaleOptimizer` | Class | `(inner_optimizer, initial_scale, dynamic_growth_steps, name, kwargs)` |
| `src.trainers.trainer.Trainer` | Class | `()` |
| `src.trainers.trainer.backend` | Object | `` |
| `src.trainers.trainer.data_adapter_utils` | Object | `` |
| `src.trainers.trainer.metrics_module` | Object | `` |
| `src.trainers.trainer.model_supports_jit` | Function | `(model)` |
| `src.trainers.trainer.ops` | Object | `` |
| `src.trainers.trainer.optimizers` | Object | `` |
| `src.trainers.trainer.python_utils` | Object | `` |
| `src.trainers.trainer.serialization_lib` | Object | `` |
| `src.trainers.trainer.traceback_utils` | Object | `` |
| `src.trainers.trainer.tracking` | Object | `` |
| `src.trainers.trainer.tree` | Object | `` |
| `src.tree.assert_same_paths` | Function | `(a, b)` |
| `src.tree.assert_same_structure` | Function | `(a, b, check_types)` |
| `src.tree.dmtree_impl.ClassRegistration` | Object | `` |
| `src.tree.dmtree_impl.REGISTERED_CLASSES` | Object | `` |
| `src.tree.dmtree_impl.TypeErrorRemapping` | Class | `(...)` |
| `src.tree.dmtree_impl.assert_same_paths` | Function | `(a, b)` |
| `src.tree.dmtree_impl.assert_same_structure` | Function | `(a, b)` |
| `src.tree.dmtree_impl.backend` | Function | `()` |
| `src.tree.dmtree_impl.dmtree` | Object | `` |
| `src.tree.dmtree_impl.flatten` | Function | `(structure)` |
| `src.tree.dmtree_impl.flatten_with_path` | Function | `(structure)` |
| `src.tree.dmtree_impl.is_nested` | Function | `(structure)` |
| `src.tree.dmtree_impl.lists_to_tuples` | Function | `(structure)` |
| `src.tree.dmtree_impl.map_shape_structure` | Function | `(func, structure)` |
| `src.tree.dmtree_impl.map_structure` | Function | `(func, structures, none_is_leaf)` |
| `src.tree.dmtree_impl.map_structure_up_to` | Function | `(shallow_structure, func, structures)` |
| `src.tree.dmtree_impl.pack_sequence_as` | Function | `(structure, flat_sequence)` |
| `src.tree.dmtree_impl.register_tree_node` | Function | `(cls, flatten_func, unflatten_func)` |
| `src.tree.dmtree_impl.register_tree_node_class` | Function | `(cls)` |
| `src.tree.dmtree_impl.sorted_keys_and_values` | Function | `(d)` |
| `src.tree.dmtree_impl.traverse` | Function | `(func, structure, top_down)` |
| `src.tree.flatten` | Function | `(structure)` |
| `src.tree.flatten_with_path` | Function | `(structure)` |
| `src.tree.is_nested` | Function | `(structure)` |
| `src.tree.lists_to_tuples` | Function | `(structure)` |
| `src.tree.map_shape_structure` | Function | `(func, structure)` |
| `src.tree.map_structure` | Function | `(func, structures, none_is_leaf)` |
| `src.tree.map_structure_up_to` | Function | `(shallow_structure, func, structures)` |
| `src.tree.optree_impl.assert_same_paths` | Function | `(a, b)` |
| `src.tree.optree_impl.assert_same_structure` | Function | `(a, b)` |
| `src.tree.optree_impl.backend` | Function | `()` |
| `src.tree.optree_impl.flatten` | Function | `(structure)` |
| `src.tree.optree_impl.flatten_with_path` | Function | `(structure)` |
| `src.tree.optree_impl.is_nested` | Function | `(structure)` |
| `src.tree.optree_impl.lists_to_tuples` | Function | `(structure)` |
| `src.tree.optree_impl.map_shape_structure` | Function | `(func, structure)` |
| `src.tree.optree_impl.map_structure` | Function | `(func, structures, none_is_leaf)` |
| `src.tree.optree_impl.map_structure_up_to` | Function | `(shallow_structure, func, structures)` |
| `src.tree.optree_impl.pack_sequence_as` | Function | `(structure, flat_sequence)` |
| `src.tree.optree_impl.register_tree_node_class` | Function | `(cls)` |
| `src.tree.optree_impl.sorted_keys_and_values` | Function | `(d)` |
| `src.tree.optree_impl.traverse` | Function | `(func, structure, top_down)` |
| `src.tree.pack_sequence_as` | Function | `(structure, flat_sequence)` |
| `src.tree.register_tree_node_class` | Function | `(cls)` |
| `src.tree.torchtree_impl.assert_same_paths` | Function | `(a, b)` |
| `src.tree.torchtree_impl.assert_same_structure` | Function | `(a, b)` |
| `src.tree.torchtree_impl.flatten` | Function | `(structure)` |
| `src.tree.torchtree_impl.flatten_with_path` | Function | `(structure)` |
| `src.tree.torchtree_impl.is_nested` | Function | `(structure)` |
| `src.tree.torchtree_impl.lists_to_tuples` | Function | `(structure)` |
| `src.tree.torchtree_impl.map_shape_structure` | Function | `(func, structure)` |
| `src.tree.torchtree_impl.map_structure` | Function | `(func, structures, none_is_leaf)` |
| `src.tree.torchtree_impl.map_structure_up_to` | Function | `(shallow_structure, func, structures)` |
| `src.tree.torchtree_impl.pack_sequence_as` | Function | `(structure, flat_sequence)` |
| `src.tree.torchtree_impl.register_tree_node_class` | Function | `(cls)` |
| `src.tree.torchtree_impl.traverse` | Function | `(func, structure, top_down)` |
| `src.tree.traverse` | Function | `(func, structure, top_down)` |
| `src.tree.tree_api.MAP_TO_NONE` | Class | `(...)` |
| `src.tree.tree_api.assert_same_paths` | Function | `(a, b)` |
| `src.tree.tree_api.assert_same_structure` | Function | `(a, b, check_types)` |
| `src.tree.tree_api.backend` | Function | `()` |
| `src.tree.tree_api.dmtree` | Object | `` |
| `src.tree.tree_api.flatten` | Function | `(structure)` |
| `src.tree.tree_api.flatten_with_path` | Function | `(structure)` |
| `src.tree.tree_api.is_nested` | Function | `(structure)` |
| `src.tree.tree_api.keras_export` | Class | `(path)` |
| `src.tree.tree_api.lists_to_tuples` | Function | `(structure)` |
| `src.tree.tree_api.map_shape_structure` | Function | `(func, structure)` |
| `src.tree.tree_api.map_structure` | Function | `(func, structures, none_is_leaf)` |
| `src.tree.tree_api.map_structure_up_to` | Function | `(shallow_structure, func, structures)` |
| `src.tree.tree_api.optree` | Object | `` |
| `src.tree.tree_api.pack_sequence_as` | Function | `(structure, flat_sequence)` |
| `src.tree.tree_api.register_tree_node_class` | Function | `(cls)` |
| `src.tree.tree_api.traverse` | Function | `(func, structure, top_down)` |
| `src.tree.tree_api.tree_impl` | Object | `` |
| `src.utils.Progbar` | Class | `(target, width, verbose, interval, stateful_metrics, unit_name)` |
| `src.utils.argument_validation.standardize_padding` | Function | `(value, allow_causal)` |
| `src.utils.argument_validation.standardize_tuple` | Function | `(value, n, name, allow_zero)` |
| `src.utils.argument_validation.validate_string_arg` | Function | `(value, allowable_strings, caller_name, arg_name, allow_none, allow_callables)` |
| `src.utils.array_to_img` | Function | `(x, data_format, scale, dtype)` |
| `src.utils.audio_dataset_from_directory` | Function | `(directory, labels, label_mode, class_names, batch_size, sampling_rate, output_sequence_length, ragged, shuffle, seed, validation_split, subset, follow_links, verbose)` |
| `src.utils.audio_dataset_utils.ALLOWED_FORMATS` | Object | `` |
| `src.utils.audio_dataset_utils.audio_dataset_from_directory` | Function | `(directory, labels, label_mode, class_names, batch_size, sampling_rate, output_sequence_length, ragged, shuffle, seed, validation_split, subset, follow_links, verbose)` |
| `src.utils.audio_dataset_utils.dataset_utils` | Object | `` |
| `src.utils.audio_dataset_utils.get_dataset` | Function | `(file_paths, labels, directory, validation_split, subset, label_mode, class_names, sampling_rate, output_sequence_length, ragged, shuffle, shuffle_buffer_size, seed)` |
| `src.utils.audio_dataset_utils.get_training_and_validation_dataset` | Function | `(file_paths, labels, validation_split, directory, label_mode, class_names, sampling_rate, output_sequence_length, ragged, shuffle, shuffle_buffer_size, seed)` |
| `src.utils.audio_dataset_utils.keras_export` | Class | `(path)` |
| `src.utils.audio_dataset_utils.paths_and_labels_to_dataset` | Function | `(file_paths, labels, label_mode, num_classes, sampling_rate, output_sequence_length, ragged, shuffle, shuffle_buffer_size, seed)` |
| `src.utils.audio_dataset_utils.prepare_dataset` | Function | `(dataset, batch_size, class_names, output_sequence_length, ragged)` |
| `src.utils.audio_dataset_utils.read_and_decode_audio` | Function | `(path, sampling_rate, output_sequence_length)` |
| `src.utils.audio_dataset_utils.tf` | Object | `` |
| `src.utils.audio_dataset_utils.tfio` | Object | `` |
| `src.utils.backend_utils.DynamicBackend` | Class | `(backend)` |
| `src.utils.backend_utils.TFGraphScope` | Class | `()` |
| `src.utils.backend_utils.backend_module` | Object | `` |
| `src.utils.backend_utils.convert_tf_tensor` | Function | `(outputs, dtype)` |
| `src.utils.backend_utils.global_state` | Object | `` |
| `src.utils.backend_utils.in_grain_data_pipeline` | Function | `()` |
| `src.utils.backend_utils.in_tf_graph` | Function | `()` |
| `src.utils.backend_utils.keras_export` | Class | `(path)` |
| `src.utils.backend_utils.set_backend` | Function | `(backend)` |
| `src.utils.code_stats.count_loc` | Function | `(directory, exclude, extensions, verbose)` |
| `src.utils.config.Config` | Class | `(kwargs)` |
| `src.utils.config.keras_export` | Class | `(path)` |
| `src.utils.dataset_utils.backend` | Object | `` |
| `src.utils.dataset_utils.check_validation_split_arg` | Function | `(validation_split, subset, shuffle, seed)` |
| `src.utils.dataset_utils.file_utils` | Object | `` |
| `src.utils.dataset_utils.get_batch_size` | Function | `(dataset)` |
| `src.utils.dataset_utils.get_training_or_validation_split` | Function | `(samples, labels, validation_split, subset)` |
| `src.utils.dataset_utils.grain` | Object | `` |
| `src.utils.dataset_utils.index_directory` | Function | `(directory, labels, formats, class_names, shuffle, seed, follow_links, verbose)` |
| `src.utils.dataset_utils.index_subdirectory` | Function | `(directory, class_indices, follow_links, formats)` |
| `src.utils.dataset_utils.io_utils` | Object | `` |
| `src.utils.dataset_utils.is_batched` | Function | `(dataset)` |
| `src.utils.dataset_utils.is_grain_dataset` | Function | `(dataset)` |
| `src.utils.dataset_utils.is_tf_dataset` | Function | `(dataset)` |
| `src.utils.dataset_utils.is_torch_dataset` | Function | `(dataset)` |
| `src.utils.dataset_utils.iter_valid_files` | Function | `(directory, follow_links, formats)` |
| `src.utils.dataset_utils.keras_export` | Class | `(path)` |
| `src.utils.dataset_utils.labels_to_dataset_grain` | Function | `(labels, label_mode, num_classes)` |
| `src.utils.dataset_utils.labels_to_dataset_tf` | Function | `(labels, label_mode, num_classes)` |
| `src.utils.dataset_utils.split_dataset` | Function | `(dataset, left_size, right_size, shuffle, seed, preferred_backend)` |
| `src.utils.dataset_utils.tree` | Object | `` |
| `src.utils.default` | Function | `(method)` |
| `src.utils.disable_interactive_logging` | Function | `()` |
| `src.utils.dtype_utils.DTYPE_TO_SIZE` | Object | `` |
| `src.utils.dtype_utils.backend` | Object | `` |
| `src.utils.dtype_utils.cast_to_common_dtype` | Function | `(tensors)` |
| `src.utils.dtype_utils.dtype_size` | Function | `(dtype)` |
| `src.utils.dtype_utils.is_float` | Function | `(dtype)` |
| `src.utils.dtype_utils.ops` | Object | `` |
| `src.utils.enable_interactive_logging` | Function | `()` |
| `src.utils.file_utils.File` | Function | `(path, mode)` |
| `src.utils.file_utils.Progbar` | Class | `(target, width, verbose, interval, stateful_metrics, unit_name)` |
| `src.utils.file_utils.config` | Object | `` |
| `src.utils.file_utils.copy` | Function | `(src, dst)` |
| `src.utils.file_utils.exists` | Function | `(path)` |
| `src.utils.file_utils.extract_archive` | Function | `(file_path, path, archive_format)` |
| `src.utils.file_utils.extract_open_archive` | Function | `(archive, path)` |
| `src.utils.file_utils.filter_safe_tarinfos` | Function | `(members)` |
| `src.utils.file_utils.filter_safe_zipinfos` | Function | `(members)` |
| `src.utils.file_utils.get_file` | Function | `(fname, origin, untar, md5_hash, file_hash, cache_subdir, hash_algorithm, extract, archive_format, cache_dir, force_download)` |
| `src.utils.file_utils.gfile` | Object | `` |
| `src.utils.file_utils.hash_file` | Function | `(fpath, algorithm, chunk_size)` |
| `src.utils.file_utils.io_utils` | Object | `` |
| `src.utils.file_utils.is_link_in_dir` | Function | `(info, base)` |
| `src.utils.file_utils.is_path_in_dir` | Function | `(path, base_dir)` |
| `src.utils.file_utils.is_remote_path` | Function | `(filepath)` |
| `src.utils.file_utils.isdir` | Function | `(path)` |
| `src.utils.file_utils.join` | Function | `(path, paths)` |
| `src.utils.file_utils.keras_export` | Class | `(path)` |
| `src.utils.file_utils.listdir` | Function | `(path)` |
| `src.utils.file_utils.makedirs` | Function | `(path)` |
| `src.utils.file_utils.path_to_string` | Function | `(path)` |
| `src.utils.file_utils.remove` | Function | `(path)` |
| `src.utils.file_utils.resolve_hasher` | Function | `(algorithm, file_hash)` |
| `src.utils.file_utils.resolve_path` | Function | `(path)` |
| `src.utils.file_utils.rmtree` | Function | `(path)` |
| `src.utils.file_utils.validate_file` | Function | `(fpath, file_hash, algorithm, chunk_size)` |
| `src.utils.get_file` | Function | `(fname, origin, untar, md5_hash, file_hash, cache_subdir, hash_algorithm, extract, archive_format, cache_dir, force_download)` |
| `src.utils.grain_utils.backend` | Object | `` |
| `src.utils.grain_utils.make_batch` | Function | `(values)` |
| `src.utils.grain_utils.make_string_batch` | Function | `(values)` |
| `src.utils.grain_utils.tree` | Object | `` |
| `src.utils.image_dataset_from_directory` | Function | `(directory, labels, label_mode, class_names, color_mode, batch_size, image_size, shuffle, seed, validation_split, subset, interpolation, follow_links, crop_to_aspect_ratio, pad_to_aspect_ratio, data_format, format, verbose)` |
| `src.utils.image_dataset_utils.ALLOWLIST_FORMATS` | Object | `` |
| `src.utils.image_dataset_utils.dataset_utils` | Object | `` |
| `src.utils.image_dataset_utils.grain` | Object | `` |
| `src.utils.image_dataset_utils.image_dataset_from_directory` | Function | `(directory, labels, label_mode, class_names, color_mode, batch_size, image_size, shuffle, seed, validation_split, subset, interpolation, follow_links, crop_to_aspect_ratio, pad_to_aspect_ratio, data_format, format, verbose)` |
| `src.utils.image_dataset_utils.image_utils` | Object | `` |
| `src.utils.image_dataset_utils.keras_export` | Class | `(path)` |
| `src.utils.image_dataset_utils.make_batch` | Function | `(values)` |
| `src.utils.image_dataset_utils.paths_and_labels_to_dataset` | Function | `(image_paths, image_size, num_channels, labels, label_mode, num_classes, interpolation, data_format, crop_to_aspect_ratio, pad_to_aspect_ratio, shuffle, shuffle_buffer_size, seed, format)` |
| `src.utils.image_dataset_utils.pil_image_resampling` | Object | `` |
| `src.utils.image_dataset_utils.standardize_data_format` | Function | `(data_format)` |
| `src.utils.image_dataset_utils.tf` | Object | `` |
| `src.utils.image_utils.PIL_INTERPOLATION_METHODS` | Object | `` |
| `src.utils.image_utils.array_to_img` | Function | `(x, data_format, scale, dtype)` |
| `src.utils.image_utils.backend` | Object | `` |
| `src.utils.image_utils.img_to_array` | Function | `(img, data_format, dtype)` |
| `src.utils.image_utils.keras_export` | Class | `(path)` |
| `src.utils.image_utils.load_img` | Function | `(path, color_mode, target_size, interpolation, keep_aspect_ratio)` |
| `src.utils.image_utils.pil_image_resampling` | Object | `` |
| `src.utils.image_utils.save_img` | Function | `(path, x, data_format, file_format, scale, kwargs)` |
| `src.utils.image_utils.smart_resize` | Function | `(x, size, interpolation, data_format, backend_module)` |
| `src.utils.img_to_array` | Function | `(img, data_format, dtype)` |
| `src.utils.io_utils.ask_to_proceed_with_overwrite` | Function | `(filepath)` |
| `src.utils.io_utils.disable_interactive_logging` | Function | `()` |
| `src.utils.io_utils.enable_interactive_logging` | Function | `()` |
| `src.utils.io_utils.global_state` | Object | `` |
| `src.utils.io_utils.is_interactive_logging_enabled` | Function | `()` |
| `src.utils.io_utils.keras_export` | Class | `(path)` |
| `src.utils.io_utils.print_msg` | Function | `(message, line_break)` |
| `src.utils.io_utils.set_logging_verbosity` | Function | `(level)` |
| `src.utils.is_default` | Function | `(method)` |
| `src.utils.is_interactive_logging_enabled` | Function | `()` |
| `src.utils.jax_layer.FlaxLayer` | Class | `(module, method, variables, kwargs)` |
| `src.utils.jax_layer.JaxLayer` | Class | `(call_fn, init_fn, params, state, seed, kwargs)` |
| `src.utils.jax_layer.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.utils.jax_layer.backend` | Object | `` |
| `src.utils.jax_layer.is_float_dtype` | Function | `(dtype)` |
| `src.utils.jax_layer.jax` | Object | `` |
| `src.utils.jax_layer.jax_utils` | Object | `` |
| `src.utils.jax_layer.keras_export` | Class | `(path)` |
| `src.utils.jax_layer.serialization_lib` | Object | `` |
| `src.utils.jax_layer.standardize_dtype` | Function | `(dtype)` |
| `src.utils.jax_layer.tf` | Object | `` |
| `src.utils.jax_layer.tf_no_automatic_dependency_tracking` | Function | `(fn)` |
| `src.utils.jax_layer.tracking` | Object | `` |
| `src.utils.jax_layer.tree` | Object | `` |
| `src.utils.jax_utils.backend` | Object | `` |
| `src.utils.jax_utils.is_in_jax_tracing_scope` | Function | `(x)` |
| `src.utils.load_img` | Function | `(path, color_mode, target_size, interpolation, keep_aspect_ratio)` |
| `src.utils.model_to_dot` | Function | `(model, show_shapes, show_dtype, show_layer_names, rankdir, expand_nested, dpi, subgraph, show_layer_activations, show_trainable, kwargs)` |
| `src.utils.model_visualization.add_edge` | Function | `(dot, src, dst)` |
| `src.utils.model_visualization.check_graphviz` | Function | `()` |
| `src.utils.model_visualization.check_pydot` | Function | `()` |
| `src.utils.model_visualization.get_layer_activation_name` | Function | `(layer)` |
| `src.utils.model_visualization.io_utils` | Object | `` |
| `src.utils.model_visualization.keras_export` | Class | `(path)` |
| `src.utils.model_visualization.make_layer_label` | Function | `(layer, kwargs)` |
| `src.utils.model_visualization.make_node` | Function | `(layer, kwargs)` |
| `src.utils.model_visualization.model_to_dot` | Function | `(model, show_shapes, show_dtype, show_layer_names, rankdir, expand_nested, dpi, subgraph, show_layer_activations, show_trainable, kwargs)` |
| `src.utils.model_visualization.plot_model` | Function | `(model, to_file, show_shapes, show_dtype, show_layer_names, rankdir, expand_nested, dpi, show_layer_activations, show_trainable, kwargs)` |
| `src.utils.model_visualization.tree` | Object | `` |
| `src.utils.module_utils.LazyModule` | Class | `(name, pip_name, import_error_msg)` |
| `src.utils.module_utils.OrbaxLazyModule` | Class | `(...)` |
| `src.utils.module_utils.dmtree` | Object | `` |
| `src.utils.module_utils.gfile` | Object | `` |
| `src.utils.module_utils.grain` | Object | `` |
| `src.utils.module_utils.jax` | Object | `` |
| `src.utils.module_utils.litert` | Object | `` |
| `src.utils.module_utils.ocp` | Object | `` |
| `src.utils.module_utils.optree` | Object | `` |
| `src.utils.module_utils.scipy` | Object | `` |
| `src.utils.module_utils.tensorflow` | Object | `` |
| `src.utils.module_utils.tensorflow_io` | Object | `` |
| `src.utils.module_utils.tf2onnx` | Object | `` |
| `src.utils.module_utils.torch_xla` | Object | `` |
| `src.utils.naming.auto_name` | Function | `(prefix)` |
| `src.utils.naming.get_object_name` | Function | `(obj)` |
| `src.utils.naming.get_uid` | Function | `(prefix)` |
| `src.utils.naming.global_state` | Object | `` |
| `src.utils.naming.keras_export` | Class | `(path)` |
| `src.utils.naming.reset_uids` | Function | `()` |
| `src.utils.naming.to_snake_case` | Function | `(name)` |
| `src.utils.naming.uniquify` | Function | `(name)` |
| `src.utils.normalize` | Function | `(x, axis, order)` |
| `src.utils.numerical_utils.backend` | Object | `` |
| `src.utils.numerical_utils.build_pos_neg_masks` | Function | `(query_labels, key_labels, remove_diagonal)` |
| `src.utils.numerical_utils.encode_categorical_inputs` | Function | `(inputs, output_mode, depth, dtype, sparse, count_weights, backend_module)` |
| `src.utils.numerical_utils.keras_export` | Class | `(path)` |
| `src.utils.numerical_utils.normalize` | Function | `(x, axis, order)` |
| `src.utils.numerical_utils.tf_utils` | Object | `` |
| `src.utils.numerical_utils.to_categorical` | Function | `(x, num_classes)` |
| `src.utils.pad_sequences` | Function | `(sequences, maxlen, dtype, padding, truncating, value)` |
| `src.utils.plot_model` | Function | `(model, to_file, show_shapes, show_dtype, show_layer_names, rankdir, expand_nested, dpi, show_layer_activations, show_trainable, kwargs)` |
| `src.utils.progbar.Progbar` | Class | `(target, width, verbose, interval, stateful_metrics, unit_name)` |
| `src.utils.progbar.io_utils` | Object | `` |
| `src.utils.progbar.keras_export` | Class | `(path)` |
| `src.utils.python_utils.default` | Function | `(method)` |
| `src.utils.python_utils.func_dump` | Function | `(func)` |
| `src.utils.python_utils.func_load` | Function | `(code, defaults, closure, globs)` |
| `src.utils.python_utils.is_continuous_axis` | Function | `(axis)` |
| `src.utils.python_utils.is_default` | Function | `(method)` |
| `src.utils.python_utils.pythonify_logs` | Function | `(logs)` |
| `src.utils.python_utils.remove_by_id` | Function | `(lst, value)` |
| `src.utils.python_utils.remove_long_seq` | Function | `(maxlen, seq, label)` |
| `src.utils.python_utils.removeprefix` | Function | `(x, prefix)` |
| `src.utils.python_utils.removesuffix` | Function | `(x, suffix)` |
| `src.utils.python_utils.to_list` | Function | `(x)` |
| `src.utils.removeprefix` | Function | `(x, prefix)` |
| `src.utils.removesuffix` | Function | `(x, suffix)` |
| `src.utils.rng_utils.GLOBAL_RANDOM_SEED` | Object | `` |
| `src.utils.rng_utils.backend` | Object | `` |
| `src.utils.rng_utils.get_random_seed` | Function | `()` |
| `src.utils.rng_utils.global_state` | Object | `` |
| `src.utils.rng_utils.keras_export` | Class | `(path)` |
| `src.utils.rng_utils.seed_generator` | Object | `` |
| `src.utils.rng_utils.set_random_seed` | Function | `(seed)` |
| `src.utils.rng_utils.tf` | Object | `` |
| `src.utils.save_img` | Function | `(path, x, data_format, file_format, scale, kwargs)` |
| `src.utils.sequence_utils.keras_export` | Class | `(path)` |
| `src.utils.sequence_utils.pad_sequences` | Function | `(sequences, maxlen, dtype, padding, truncating, value)` |
| `src.utils.set_random_seed` | Function | `(seed)` |
| `src.utils.split_dataset` | Function | `(dataset, left_size, right_size, shuffle, seed, preferred_backend)` |
| `src.utils.summary_utils.backend` | Object | `` |
| `src.utils.summary_utils.bold_text` | Function | `(x, color)` |
| `src.utils.summary_utils.count_params` | Function | `(weights)` |
| `src.utils.summary_utils.dtype_utils` | Object | `` |
| `src.utils.summary_utils.format_layer_shape` | Function | `(layer)` |
| `src.utils.summary_utils.get_layer_index_bound_by_layer_name` | Function | `(layers, layer_range)` |
| `src.utils.summary_utils.highlight_number` | Function | `(x)` |
| `src.utils.summary_utils.highlight_symbol` | Function | `(x)` |
| `src.utils.summary_utils.io_utils` | Object | `` |
| `src.utils.summary_utils.print_summary` | Function | `(model, line_length, positions, print_fn, expand_nested, show_trainable, layer_range)` |
| `src.utils.summary_utils.readable_memory_size` | Function | `(weight_memory_size)` |
| `src.utils.summary_utils.tree` | Object | `` |
| `src.utils.summary_utils.weight_memory_size` | Function | `(weights)` |
| `src.utils.text_dataset_from_directory` | Function | `(directory, labels, label_mode, class_names, batch_size, max_length, shuffle, seed, validation_split, subset, follow_links, format, verbose)` |
| `src.utils.text_dataset_utils.dataset_utils` | Object | `` |
| `src.utils.text_dataset_utils.grain` | Object | `` |
| `src.utils.text_dataset_utils.keras_export` | Class | `(path)` |
| `src.utils.text_dataset_utils.make_string_batch` | Function | `(values)` |
| `src.utils.text_dataset_utils.paths_and_labels_to_dataset` | Function | `(file_paths, labels, label_mode, num_classes, max_length, shuffle, shuffle_buffer_size, seed, format)` |
| `src.utils.text_dataset_utils.text_dataset_from_directory` | Function | `(directory, labels, label_mode, class_names, batch_size, max_length, shuffle, seed, validation_split, subset, follow_links, format, verbose)` |
| `src.utils.text_dataset_utils.tf` | Object | `` |
| `src.utils.tf_utils.backend` | Object | `` |
| `src.utils.tf_utils.dense_bincount` | Function | `(inputs, depth, binary_output, dtype, count_weights)` |
| `src.utils.tf_utils.ensure_tensor` | Function | `(inputs, dtype)` |
| `src.utils.tf_utils.expand_dims` | Function | `(inputs, axis)` |
| `src.utils.tf_utils.get_tensor_spec` | Function | `(t, dynamic_batch, name)` |
| `src.utils.tf_utils.is_ragged_tensor` | Function | `(x)` |
| `src.utils.tf_utils.sparse_bincount` | Function | `(inputs, depth, binary_output, dtype, count_weights)` |
| `src.utils.tf_utils.tf` | Object | `` |
| `src.utils.tf_utils.tf_encode_categorical_inputs` | Function | `(inputs, output_mode, depth, dtype, sparse, count_weights, idf_weights)` |
| `src.utils.timeseries_dataset_from_array` | Function | `(data, targets, sequence_length, sequence_stride, sampling_rate, batch_size, shuffle, seed, start_index, end_index)` |
| `src.utils.timeseries_dataset_utils.keras_export` | Class | `(path)` |
| `src.utils.timeseries_dataset_utils.sequences_from_indices` | Function | `(array, indices_ds, start_index, end_index)` |
| `src.utils.timeseries_dataset_utils.tf` | Object | `` |
| `src.utils.timeseries_dataset_utils.timeseries_dataset_from_array` | Function | `(data, targets, sequence_length, sequence_stride, sampling_rate, batch_size, shuffle, seed, start_index, end_index)` |
| `src.utils.to_categorical` | Function | `(x, num_classes)` |
| `src.utils.torch_utils.Layer` | Class | `(activity_regularizer, trainable, dtype, autocast, name, kwargs)` |
| `src.utils.torch_utils.TorchModuleWrapper` | Class | `(module, name, output_shape, kwargs)` |
| `src.utils.torch_utils.backend` | Object | `` |
| `src.utils.torch_utils.convert_to_numpy` | Function | `(x)` |
| `src.utils.torch_utils.convert_to_tensor` | Function | `(x, dtype, sparse, ragged)` |
| `src.utils.torch_utils.in_safe_mode` | Function | `()` |
| `src.utils.torch_utils.keras_export` | Class | `(path)` |
| `src.utils.torch_utils.no_grad` | Function | `(orig_func)` |
| `src.utils.traceback_utils.backend` | Object | `` |
| `src.utils.traceback_utils.disable_traceback_filtering` | Function | `()` |
| `src.utils.traceback_utils.enable_traceback_filtering` | Function | `()` |
| `src.utils.traceback_utils.filter_traceback` | Function | `(fn)` |
| `src.utils.traceback_utils.format_argument_value` | Function | `(value)` |
| `src.utils.traceback_utils.global_state` | Object | `` |
| `src.utils.traceback_utils.include_frame` | Function | `(fname)` |
| `src.utils.traceback_utils.inject_argument_info_in_traceback` | Function | `(fn, object_name)` |
| `src.utils.traceback_utils.is_traceback_filtering_enabled` | Function | `()` |
| `src.utils.traceback_utils.keras_export` | Class | `(path)` |
| `src.utils.traceback_utils.tree` | Object | `` |
| `src.utils.tracking.DotNotTrackScope` | Class | `(...)` |
| `src.utils.tracking.TrackedDict` | Class | `(values, tracker)` |
| `src.utils.tracking.TrackedList` | Class | `(values, tracker)` |
| `src.utils.tracking.TrackedSet` | Class | `(values, tracker)` |
| `src.utils.tracking.Tracker` | Class | `(config, exclusions)` |
| `src.utils.tracking.get_global_attribute` | Function | `(name, default, set_to_default)` |
| `src.utils.tracking.is_tracking_enabled` | Function | `()` |
| `src.utils.tracking.no_automatic_dependency_tracking` | Function | `(fn)` |
| `src.utils.tracking.python_utils` | Object | `` |
| `src.utils.tracking.set_global_attribute` | Function | `(name, value)` |
| `src.utils.tracking.tree` | Object | `` |
| `src.version.keras_export` | Class | `(path)` |
| `src.version.version` | Function | `()` |
| `src.visualization.draw_bounding_boxes.backend` | Object | `` |
| `src.visualization.draw_bounding_boxes.convert_format` | Function | `(boxes, source, target, height, width, dtype)` |
| `src.visualization.draw_bounding_boxes.draw_bounding_boxes` | Function | `(images, bounding_boxes, bounding_box_format, class_mapping, color, line_thickness, text_thickness, font_scale, data_format)` |
| `src.visualization.draw_bounding_boxes.keras_export` | Class | `(path)` |
| `src.visualization.draw_bounding_boxes.ops` | Object | `` |
| `src.visualization.draw_segmentation_masks.backend` | Object | `` |
| `src.visualization.draw_segmentation_masks.draw_segmentation_masks` | Function | `(images, segmentation_masks, num_classes, color_mapping, alpha, blend, ignore_index, data_format)` |
| `src.visualization.draw_segmentation_masks.keras_export` | Class | `(path)` |
| `src.visualization.draw_segmentation_masks.ops` | Object | `` |
| `src.visualization.plot_bounding_box_gallery.backend` | Object | `` |
| `src.visualization.plot_bounding_box_gallery.draw_bounding_boxes` | Function | `(images, bounding_boxes, bounding_box_format, class_mapping, color, line_thickness, text_thickness, font_scale, data_format)` |
| `src.visualization.plot_bounding_box_gallery.keras_export` | Class | `(path)` |
| `src.visualization.plot_bounding_box_gallery.ops` | Object | `` |
| `src.visualization.plot_bounding_box_gallery.plot_bounding_box_gallery` | Function | `(images, bounding_box_format, y_true, y_pred, value_range, true_color, pred_color, line_thickness, font_scale, text_thickness, class_mapping, ground_truth_mapping, prediction_mapping, legend, legend_handles, rows, cols, data_format, kwargs)` |
| `src.visualization.plot_bounding_box_gallery.plot_image_gallery` | Function | `(images, y_true, y_pred, label_map, rows, cols, value_range, scale, path, show, transparent, dpi, legend_handles, data_format)` |
| `src.visualization.plot_image_gallery.BaseImagePreprocessingLayer` | Class | `(factor, bounding_box_format, data_format, kwargs)` |
| `src.visualization.plot_image_gallery.backend` | Object | `` |
| `src.visualization.plot_image_gallery.keras_export` | Class | `(path)` |
| `src.visualization.plot_image_gallery.ops` | Object | `` |
| `src.visualization.plot_image_gallery.plot_image_gallery` | Function | `(images, y_true, y_pred, label_map, rows, cols, value_range, scale, path, show, transparent, dpi, legend_handles, data_format)` |
| `src.visualization.plot_segmentation_mask_gallery.backend` | Object | `` |
| `src.visualization.plot_segmentation_mask_gallery.draw_segmentation_masks` | Function | `(images, segmentation_masks, num_classes, color_mapping, alpha, blend, ignore_index, data_format)` |
| `src.visualization.plot_segmentation_mask_gallery.keras_export` | Class | `(path)` |
| `src.visualization.plot_segmentation_mask_gallery.ops` | Object | `` |
| `src.visualization.plot_segmentation_mask_gallery.plot_image_gallery` | Function | `(images, y_true, y_pred, label_map, rows, cols, value_range, scale, path, show, transparent, dpi, legend_handles, data_format)` |
| `src.visualization.plot_segmentation_mask_gallery.plot_segmentation_mask_gallery` | Function | `(images, num_classes, value_range, y_true, y_pred, color_mapping, blend, alpha, ignore_index, data_format, kwargs)` |
| `src.wrappers.SKLearnClassifier` | Class | `(...)` |
| `src.wrappers.SKLearnRegressor` | Class | `(...)` |
| `src.wrappers.SKLearnTransformer` | Class | `(...)` |
| `src.wrappers.fixes.type_of_target` | Function | `(y, input_name, raise_unknown)` |
| `src.wrappers.sklearn_wrapper.BaseEstimator` | Class | `(...)` |
| `src.wrappers.sklearn_wrapper.ClassifierMixin` | Class | `(...)` |
| `src.wrappers.sklearn_wrapper.Model` | Class | `(args, kwargs)` |
| `src.wrappers.sklearn_wrapper.RegressorMixin` | Class | `(...)` |
| `src.wrappers.sklearn_wrapper.SKLBase` | Class | `(model, warm_start, model_kwargs, fit_kwargs)` |
| `src.wrappers.sklearn_wrapper.SKLearnClassifier` | Class | `(...)` |
| `src.wrappers.sklearn_wrapper.SKLearnRegressor` | Class | `(...)` |
| `src.wrappers.sklearn_wrapper.SKLearnTransformer` | Class | `(...)` |
| `src.wrappers.sklearn_wrapper.TargetReshaper` | Class | `(...)` |
| `src.wrappers.sklearn_wrapper.TransformerMixin` | Class | `(...)` |
| `src.wrappers.sklearn_wrapper.assert_sklearn_installed` | Function | `(symbol_name)` |
| `src.wrappers.sklearn_wrapper.clone_model` | Function | `(model, input_tensors, clone_function, call_function, recursive, kwargs)` |
| `src.wrappers.sklearn_wrapper.keras_export` | Class | `(path)` |
| `src.wrappers.sklearn_wrapper.type_of_target` | Function | `(y, input_name, raise_unknown)` |
| `src.wrappers.utils.BaseEstimator` | Class | `(...)` |
| `src.wrappers.utils.TargetReshaper` | Class | `(...)` |
| `src.wrappers.utils.TransformerMixin` | Class | `(...)` |
| `src.wrappers.utils.assert_sklearn_installed` | Function | `(symbol_name)` |
| `tree.MAP_TO_NONE` | Class | `(...)` |
| `tree.assert_same_paths` | Function | `(a, b)` |
| `tree.assert_same_structure` | Function | `(a, b, check_types)` |
| `tree.flatten` | Function | `(structure)` |
| `tree.flatten_with_path` | Function | `(structure)` |
| `tree.is_nested` | Function | `(structure)` |
| `tree.lists_to_tuples` | Function | `(structure)` |
| `tree.map_shape_structure` | Function | `(func, structure)` |
| `tree.map_structure` | Function | `(func, structures, none_is_leaf)` |
| `tree.map_structure_up_to` | Function | `(shallow_structure, func, structures)` |
| `tree.pack_sequence_as` | Function | `(structure, flat_sequence)` |
| `tree.traverse` | Function | `(func, structure, top_down)` |
| `utils.Config` | Class | `(kwargs)` |
| `utils.CustomObjectScope` | Class | `(custom_objects)` |
| `utils.FeatureSpace` | Class | `(features, output_mode, crosses, crossing_dim, hashing_dim, num_discretization_bins, name)` |
| `utils.Progbar` | Class | `(target, width, verbose, interval, stateful_metrics, unit_name)` |
| `utils.PyDataset` | Class | `(workers, use_multiprocessing, max_queue_size)` |
| `utils.Sequence` | Class | `(workers, use_multiprocessing, max_queue_size)` |
| `utils.array_to_img` | Function | `(x, data_format, scale, dtype)` |
| `utils.audio_dataset_from_directory` | Function | `(directory, labels, label_mode, class_names, batch_size, sampling_rate, output_sequence_length, ragged, shuffle, seed, validation_split, subset, follow_links, verbose)` |
| `utils.bounding_boxes.affine_transform` | Function | `(boxes, angle, translate_x, translate_y, scale, shear_x, shear_y, height, width, center_x, center_y, bounding_box_format)` |
| `utils.bounding_boxes.clip_to_image_size` | Function | `(bounding_boxes, height, width, bounding_box_format)` |
| `utils.bounding_boxes.compute_ciou` | Function | `(boxes1, boxes2, bounding_box_format, image_shape)` |
| `utils.bounding_boxes.compute_iou` | Function | `(boxes1, boxes2, bounding_box_format, use_masking, mask_val, image_shape)` |
| `utils.bounding_boxes.convert_format` | Function | `(boxes, source, target, height, width, dtype)` |
| `utils.bounding_boxes.crop` | Function | `(boxes, top, left, height, width, bounding_box_format)` |
| `utils.bounding_boxes.decode_deltas_to_boxes` | Function | `(anchors, boxes_delta, anchor_format, box_format, encoded_format, variance, image_shape)` |
| `utils.bounding_boxes.encode_box_to_deltas` | Function | `(anchors, boxes, anchor_format, box_format, encoding_format, variance, image_shape)` |
| `utils.bounding_boxes.pad` | Function | `(boxes, top, left, height, width, bounding_box_format)` |
| `utils.clear_session` | Function | `(free_memory)` |
| `utils.custom_object_scope` | Class | `(custom_objects)` |
| `utils.deserialize_keras_object` | Function | `(config, custom_objects, safe_mode, kwargs)` |
| `utils.disable_interactive_logging` | Function | `()` |
| `utils.enable_interactive_logging` | Function | `()` |
| `utils.get_custom_objects` | Function | `()` |
| `utils.get_file` | Function | `(fname, origin, untar, md5_hash, file_hash, cache_subdir, hash_algorithm, extract, archive_format, cache_dir, force_download)` |
| `utils.get_registered_name` | Function | `(obj)` |
| `utils.get_registered_object` | Function | `(name, custom_objects, module_objects)` |
| `utils.get_source_inputs` | Function | `(tensor)` |
| `utils.image_dataset_from_directory` | Function | `(directory, labels, label_mode, class_names, color_mode, batch_size, image_size, shuffle, seed, validation_split, subset, interpolation, follow_links, crop_to_aspect_ratio, pad_to_aspect_ratio, data_format, format, verbose)` |
| `utils.img_to_array` | Function | `(img, data_format, dtype)` |
| `utils.is_interactive_logging_enabled` | Function | `()` |
| `utils.is_keras_tensor` | Function | `(x)` |
| `utils.legacy.deserialize_keras_object` | Function | `(identifier, module_objects, custom_objects, printable_module_name)` |
| `utils.legacy.serialize_keras_object` | Function | `(instance)` |
| `utils.load_img` | Function | `(path, color_mode, target_size, interpolation, keep_aspect_ratio)` |
| `utils.model_to_dot` | Function | `(model, show_shapes, show_dtype, show_layer_names, rankdir, expand_nested, dpi, subgraph, show_layer_activations, show_trainable, kwargs)` |
| `utils.normalize` | Function | `(x, axis, order)` |
| `utils.pack_x_y_sample_weight` | Function | `(x, y, sample_weight)` |
| `utils.pad_sequences` | Function | `(sequences, maxlen, dtype, padding, truncating, value)` |
| `utils.plot_model` | Function | `(model, to_file, show_shapes, show_dtype, show_layer_names, rankdir, expand_nested, dpi, show_layer_activations, show_trainable, kwargs)` |
| `utils.register_keras_serializable` | Function | `(package, name)` |
| `utils.save_img` | Function | `(path, x, data_format, file_format, scale, kwargs)` |
| `utils.serialize_keras_object` | Function | `(obj)` |
| `utils.set_random_seed` | Function | `(seed)` |
| `utils.split_dataset` | Function | `(dataset, left_size, right_size, shuffle, seed, preferred_backend)` |
| `utils.standardize_dtype` | Function | `(dtype)` |
| `utils.text_dataset_from_directory` | Function | `(directory, labels, label_mode, class_names, batch_size, max_length, shuffle, seed, validation_split, subset, follow_links, format, verbose)` |
| `utils.timeseries_dataset_from_array` | Function | `(data, targets, sequence_length, sequence_stride, sampling_rate, batch_size, shuffle, seed, start_index, end_index)` |
| `utils.to_categorical` | Function | `(x, num_classes)` |
| `utils.unpack_x_y_sample_weight` | Function | `(data)` |
| `version` | Function | `()` |
| `visualization.draw_bounding_boxes` | Function | `(images, bounding_boxes, bounding_box_format, class_mapping, color, line_thickness, text_thickness, font_scale, data_format)` |
| `visualization.draw_segmentation_masks` | Function | `(images, segmentation_masks, num_classes, color_mapping, alpha, blend, ignore_index, data_format)` |
| `visualization.plot_bounding_box_gallery` | Function | `(images, bounding_box_format, y_true, y_pred, value_range, true_color, pred_color, line_thickness, font_scale, text_thickness, class_mapping, ground_truth_mapping, prediction_mapping, legend, legend_handles, rows, cols, data_format, kwargs)` |
| `visualization.plot_image_gallery` | Function | `(images, y_true, y_pred, label_map, rows, cols, value_range, scale, path, show, transparent, dpi, legend_handles, data_format)` |
| `visualization.plot_segmentation_mask_gallery` | Function | `(images, num_classes, value_range, y_true, y_pred, color_mapping, blend, alpha, ignore_index, data_format, kwargs)` |
| `wrappers.SKLearnClassifier` | Class | `(...)` |
| `wrappers.SKLearnRegressor` | Class | `(...)` |
| `wrappers.SKLearnTransformer` | Class | `(...)` |