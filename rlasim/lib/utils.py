import torch



def load_checkpoint(module, path):

    if not torch.cuda.is_available():
        # module.load_from_checkpoint(path, map_location=torch.device('cpu'))
        s = torch.load(path, map_location=torch.device('cpu'))['state_dict']
        module.load_state_dict(torch.load(path, map_location=torch.device('cpu'))['state_dict'])
    else:
        module.load_state_dict(torch.load(path)['state_dict'])



'''

        if 'checkpoint_path' in params.keys():
            if self._debug:
                print("Trying to load")
            if not torch.cuda.is_available():
                print(torch.load(params['checkpoint_path'], map_location=torch.device('cpu')))

                self.load_state_dict(torch.load(params['checkpoint_path'], map_location=torch.device('cpu'))['state_dict'])
                if self._debug:
                    print(self._parameters)
                    print(self._buffers)
                    print(self._state_dict_pre_hooks)
                    print(self._modules)
            else:
                self.load_state_dict(torch.load(params['checkpoint_path'])['state_dict'])

            print("Main", len(self.preprocessors))
'''