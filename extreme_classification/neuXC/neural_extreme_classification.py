import torch.nn as nn


def _get_sequential_module(config):
    seq_mdl = nn.Sequential()
    for i, layer in enumerate(config):
        if 'kwargs' in layer.keys():
            seq_mdl.add_module('{}_{}'.format(layer['name'], i),
                               getattr(nn, layer['name'])(**layer['kwargs']))
        else:
            seq_mdl.add_module('{}_{}'.format(layer['name'], i),
                               getattr(nn, layer['name'])())
    return seq_mdl


class NeuralXC(nn.Module):
    """
    Module for performing extreme classifcation with Autoencoders and Regressors.

    Args:
        input_encoder_config : List of dictionaries with layer types and configurations for the
                               input encoder.
        input_decoder_config : List of dictionaries with layer types and configurations for the
                               input decoder.
        output_encoder_config : List of dictionaries with layer types and configurations for the
                                output encoder.
        output_decoder_config : List of dictionaries with layer types and configurations for the
                                output decoder.
        regressor_config : List of dictionaries with layer types and configurations for the
                           regressor.

        An example of a configuration would be:
               [{'name': 'Linear',
                 'kwargs': {'in_features': 10, 'out_features': 10, 'bias': False}},
                {'name': 'ReLU',
                 'kwargs': {'inplace': True}},
                {'name': 'Linear',
                 'kwargs: {'in_features': 10, 'out_features': 10}},
               ]
        `kwargs` not specified will be considered to be the default options in the constructor
        functions of the sub-modules. For instance, in the above configuration: the last layer
        has no mention of `bias` for the `Linear` layer. This is `True` by default.

    The forward pass would return a three-tuple:
        - The first item is the input reconstructed using the input encoder
        - The second item is the output reconstructed using the output encoder
        - The third item is the decoded version of the regression output

    In addition, there are separate methods for encoding/decoding inputs/outputs
    and performing regression.
    """
    def __init__(self, input_encoder_config, input_decoder_config,
                 output_encoder_config, output_decoder_config, regressor_config):
        super(NeuralXC, self).__init__()

        # Construct input encoder
        self.input_encoder = _get_sequential_module(input_encoder_config)

        # Construct input decoder
        self.input_decoder = _get_sequential_module(input_decoder_config)

        # Construct output encoder
        self.output_encoder = _get_sequential_module(output_encoder_config)

        # Construct output encoder
        self.output_decoder = _get_sequential_module(output_decoder_config)

        # Construct regressor
        self.regressor = _get_sequential_module(regressor_config)

    def forward(self, x, y):
        """
        Function to perform the forward pass.

        Args:
            x : This is a tensor of inputs (batched)
            y : This is a tensor of multi-hot outputs (batched)

        Returns:
            As described above - 3-tuple
        """
        ret_tup = (self.decode_input(self.encode_input(x)),
                   self.decode_output(self.encode_output(y)),
                   self.predict(x))
        return ret_tup

    def encode_input(self, x):
        """
        Function to return the encoding of input using the input encoder

        Args:
            x : This is a tensor of inputs (batched)

        Returns:
            Batched encoding of inputs
        """
        return self.input_encoder(x)

    def decode_input(self, enc_x):
        """
        Function to return the decoding of encoded input using the input decoder

        Args:
            x : This is a tensor of input encodings (batched)

        Returns:
            Reconstruction of inputs
        """
        return self.input_decoder(enc_x)

    def encode_output(self, y):
        """
        Function to return the encoding of multi-hot output using the output encoder

        Args:
            x : This is a tensor of outputs (batched)

        Returns:
            Batched encoding of outputs
        """
        return self.output_encoder(y)

    def decode_output(self, enc_y):
        """
        Function to return the decoding of encoded output using the output decoder

        Args:
            x : This is a tensor of output encodings (batched)

        Returns:
            Reconstruction of multi-hot encoding of outputs
        """
        return self.output_decoder(enc_y)

    def predict(self, x):
        """
        Function to provide predictions for a batch of datapoints.

        Args:
            x : This is a tensor of batch inputs
        """
        return self.decode_output(self.regressor(self.encode_input(x)))
