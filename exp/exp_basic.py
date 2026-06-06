from models import (
    anomalytransformer, memto, sub_adjacent_transformer,
    dagmm, dtaad, lstm_autoencoder, npsr, tranad,
)


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            # Full plug-in (Section 5.2, Table 2)
            "anomalytransformer":        anomalytransformer,
            "memto":                     memto,
            "sub_adjacent_transformer":  sub_adjacent_transformer,
            # Loss-only plug-in (Section 5.2, Table 1)
            "dagmm":            dagmm,
            "dtaad":            dtaad,
            "lstm_autoencoder": lstm_autoencoder,
            "npsr":             npsr,
            "tranad":           tranad,
        }
        self.model = self._build_model()

    def _build_model(self):
        raise NotImplementedError

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
