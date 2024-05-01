from data import get_mnist_loaders
from model import SimpleModel
import torch
import torch.nn.functional as F
import argparse
def eval(model):
    _, test_loader = get_mnist_loaders(1024)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--p', type=str, default='data/model.pth')
    model = SimpleModel()
    args = parser.parse_args()
    model.load_state_dict(torch.load(args.p))

    eval(model)
