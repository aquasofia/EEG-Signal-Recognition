{
  "batch_size": 1,
  "shuffle": "False",
  "max_epochs": 100,

  "model": {

    "layer_1_input_dim": 1,
    "layer_1_output_dim": 16,
    "kernel_1": 3,
    "stride_1": 2,
    "padding_1": 2,
    "pooling_1_kernel": 4,
    "pooling_1_stride": 1,

    "layer_2_input_dim": 16,
    "layer_2_output_dim": 32,
    "kernel_2": 3,
    "stride_2": 2,
    "padding_2": 2,
    "pooling_2_kernel": 4,
    "pooling_2_stride": 2,

    "classifier_output": 1,
    "input_features": 10240,
    "dropout": 0.1}

}