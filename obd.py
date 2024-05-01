import torch
from model import SimpleModel
import torch.optim as optim
from data import get_mnist_loaders
import torch.nn.functional as F
from constants import PRETRAIN_MODEL_PATH
import matplotlib.pyplot as plt
import json

train_loader, test_loader = get_mnist_loaders(64)
model = SimpleModel()

TOP_K = 1000


def pretrain():
    # try to load already pre-trained model, so we don't have to train it again
    try:
        model.load_state_dict(torch.load(PRETRAIN_MODEL_PATH))
        return
    except FileNotFoundError:
        pass
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    # Train
    model.train()

    for epoch in range(1, 10):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )

    # save model
    torch.save(model.state_dict(), PRETRAIN_MODEL_PATH)


def train_epoch():
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    # Train
    model.train()
    avg_loss = 0
    # for each model param, mask_0s is a mask where 1 means that the param is not set to 0, and 0 means it is set to 0
    mask_0s = [param != 0 for param in model.parameters()]
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        avg_loss += loss.item()
        optimizer.step()
        # set 'frozen' params to 0 again in case they were updated
        with torch.no_grad():
            for param, mask_0 in zip(model.parameters(), mask_0s):
                param.data *= mask_0
    print("Trained Epoch with loss: ", avg_loss / len(train_loader.dataset))


def single_obd_pruning_step():
    batch_size_obd = 512
    random_indices = torch.randint(0, len(train_loader.dataset), (batch_size_obd,))
    xbatch = train_loader.dataset.data[random_indices].float()
    ybatch = train_loader.dataset.targets[random_indices].float()

    loss = F.cross_entropy(model(xbatch), ybatch.to(torch.int64))

    # df/dw (Jacobian-vector products)
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

    rademacher_zs = [
        ((torch.rand(g.shape) < 0.5).float() * 2 - 1) for g in grads
    ]  # rademacher random variables, because we need to sample from {-1, 1}
    # Hessian-vector product
    grads2 = torch.autograd.grad(
        grads,
        model.parameters(),
        grad_outputs=rademacher_zs,
    )

    hessian_diags = [g2 * z for g2, z in zip(grads2, rademacher_zs)]

    hessdiag = torch.cat([h.view(-1) for h in hessian_diags])

    saliencies = (
        hessdiag
        * (
            torch.cat([param.contiguous().view(-1) for param in model.parameters()])
            ** 2
        )
        / 2
    )

    # we don't want to prune weights that are already zero, so for them not to be selected, we set saliency to +inf
    saliencies[
        torch.cat([param.contiguous().view(-1) for param in model.parameters()]) == 0
    ] = float("inf")

    # get param indices with top lowest saliencies magnitude (the original paper does not mention the magnitude, but it makes sense to use it, and yields better results)
    saliencies, indices = torch.topk(saliencies.abs(), TOP_K, largest=False)

    # now, for each param in the index, make it 0
    for index in indices:
        # we need to calculate the correct index
        param_index = index
        for param in model.parameters():
            param_size = param.numel()
            if param_index >= param_size:
                param_index -= param_size
            else:
                # now, we have the correct param and index
                param.data.view(-1)[param_index] = 0
                break

    # print params set to 0
    n_params_to_0 = sum([torch.sum(param == 0).item() for param in model.parameters()])
    print(f"Pruned {TOP_K} params, now {n_params_to_0} params are set to 0")


# plot the losses and accuracies during OBD

graph = plt.figure()
ax = graph.add_subplot(111)
ax.set_title("OBD")
ax.set_xlabel("Params set to 0")
ax.set_ylabel("Accuracy")
# make it interactive
plt.ion()


def update_graph(accs, n_params_to_0):
    ax.plot(n_params_to_0, accs)
    plt.show()
    plt.pause(0.1)


def obd():
    pretrain()
    accs = []
    losses = []
    n_params_to_0 = []
    NUMBER_OBD_STEPS = 100

    for i in range(NUMBER_OBD_STEPS):
        single_obd_pruning_step()
        print(
            "Number of params set to 0: ",
            sum([torch.sum(param == 0).item() for param in model.parameters()]),
        )
        train_epoch()
        print(
            "Number of params set to 0: ",
            sum([torch.sum(param == 0).item() for param in model.parameters()]),
        )
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                test_loss += F.cross_entropy(output, target, reduction="sum").item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        acc = 100.0 * correct / len(test_loader.dataset)
        n_params_to_0_now = sum(
            [torch.sum(param == 0).item() for param in model.parameters()]
        )
        print(
            f"Completed pruning step {i}, accuracy: {acc}, loss: {test_loss}, n_params_to_0: {n_params_to_0_now}"
        )
        accs.append(acc)
        losses.append(test_loss)
        n_params_to_0.append(n_params_to_0_now)

        update_graph(accs, n_params_to_0)

        d = {
            "total_params": sum([param.numel() for param in model.parameters()]),
            "set_to_0": n_params_to_0_now,
            "accuracy": acc,
            "loss": test_loss,
        }

        d_path = PRETRAIN_MODEL_PATH + "_obd_stats.jsonl"
        with open(d_path, "a") as f:
            f.write(json.dumps(d))
            f.write("\n")
        if sum([torch.sum(param == 0).item() for param in model.parameters()]) == sum(
            [param.numel() for param in model.parameters()]
        ):
            # we have pruned all the weights
            break
        # save model
        torch.save(
            model.state_dict(),
            PRETRAIN_MODEL_PATH + "_{}_pruned.pth".format(n_params_to_0_now),
        )
    return accs, losses, n_params_to_0


if __name__ == "__main__":
    obd()
