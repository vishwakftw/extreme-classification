import torch.nn as nn


class Autoencoder(nn.Module):
    """
    Class representing a generic autoencoder module for providing an alternate representation
    (which is typically low-dimensional).

    Args:
        encoder_layer_config : List of dictionaries with layer types and configurations for the
                               encoder
                               Example:
                                [{'name': 'Linear',
                                  'kwargs': {'in_features': 10, 'out_features': 10, 'bias': True}
                                 },
                                 {'name': 'Tanh'}
                                ]
        decoder_layer_config : List of dictionaries with layer types and configurations for the
                               decoder
                               Example:
                               [{'name': 'Linear',
                                 'kwargs': {'in_features': 10, 'out_features': 10}, 'bias': False
                                },
                                {'name': 'ReLU', 'kwargs': {'inplace': True}
                                }
                               ]
    """
    def __init__(self, encoder_layer_config, decoder_layer_config):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential()
        for i, layer in enumerate(encoder_layer_config):
            if 'kwargs' in layer.keys():
                self.encoder.add_module('{}_{}'.format(layer['name'], i),
                                        getattr(nn, layer['name'])(**layer['kwargs']))
            else:
                self.encoder.add_module('{}_{}'.format(layer['name'], i),
                                        getattr(nn, layer['name'])())

        self.decoder = nn.Sequential()
        for i, layer in enumerate(decoder_layer_config):
            if 'kwargs' in layer.keys():
                self.decoder.add_module('{}_{}'.format(layer['name'], i),
                                        getattr(nn, layer['name'])(**layer['kwargs']))
            else:
                self.decoder.add_module('{}_{}'.format(layer['name'], i),
                                        getattr(nn, layer['name'])())

    def forward(self, input):
        """
        Function for the forward-pass.

        Args:
            input : input to the autoencoder
        """
        return self.decoder(self.encoder(input))

    def encode(self, input):
        """
        Function to produce an encoding of the input using the learned encoder.

        Args:
            input : input to be encoded
        """
        return self.encoder(input)

    def decode(self, enc_input):
        """
        Function to decode an encoded input using the learned decoder.

        Args:
            enc_input : input to be decoded
        """
        return self.decoder(enc_input)
