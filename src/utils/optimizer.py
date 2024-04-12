# GROUP19_COMP0197
import torch.optim as optim

def get_optimizer(model, params):
    match params['optimizer']:
        case "Adam":
            return optim.Adam(model.parameters(), lr=params['learning_rate'])
        case "AdamW":
            return optim.AdamW(model.parameters(), lr=params['learning_rate'])
        case "SGD":
            return optim.SGD(model.parameters(), lr=params['learning_rate'], momentum=params['momentum'])
        case _:
            print(f"Optimizer {params['optimizer']} not known")

