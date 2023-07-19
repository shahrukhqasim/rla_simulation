# import argh
# import pytorch_lightning as pl
# import torch
# import numpy as np
# from rlasim.lib.networks import VanillaVae
#
#
# class VaeInferenceExperiment(pl.LightningModule):
#     def __init__(self, checkpoint_path):
#         super().__init__()
#         self.model = MyModel()  # Replace with your own model class
#         self.model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
#         self.model.eval()
#
#
#
# def main():
#     vae_network = VanillaVae()
#     vae_network(torch.Tensor(np.random.normal(0, 1, (100, 9))))
#
#
#     predictions = trainer.predict(inference_module, dataloaders=dataloader)
#
#
# if __name__ == '__main__':
#     argh.dispatch_command(main)