import torch
import os
import shutil

def train(model, TRAIN_X, TRAIN_LABEL, epoch, optimizer, criterion, batch_size, log_interval):
    model.train()
    correct1 = 0
    train_num = len(TRAIN_LABEL)

    train_step = train_num // batch_size
    for step in range(train_step + 1):
        if step == train_step:

            x = TRAIN_X[-batch_size:]
            y = TRAIN_LABEL[-batch_size:]

        else:

            x = TRAIN_X[step * batch_size:(step + 1) * batch_size]
            y = TRAIN_LABEL[step * batch_size:(step + 1) * batch_size]

        data = torch.FloatTensor(x)
        target = torch.LongTensor(y)
        data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)

        # print('output_type: ', output.dtype)
        # print('target_type: ', target.dtype)

        loss.backward()
        optimizer.step()

        corr = correct(outputs, target, topk=(1,))
        correct1 += corr[0]


        if step % log_interval == 0:

            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}. '
                'Top-1 accuracy: {:.2f}%({:.2f}%). '.format(epoch, step, train_step+1,
                                                           100. * step / train_step+1, loss.item(),
                                                           100. * corr[0] / batch_size,
                                                           100. * correct1 / (batch_size * (step + 1)),))
    return loss.item(), correct1 / (batch_size * (train_step + 1))

def test(model, TEST_X, TEST_LABEL, criterion, batch_size):
    model.eval()
    test_loss = 0
    correct1 = 0
    test_num = len(TEST_LABEL)
    test_step = test_num // batch_size

    for step in range(test_step + 1):
        if step == test_step:
            x = TEST_X[step * batch_size: test_num]
            y = TEST_LABEL[step * batch_size: test_num]

        else:
            x = TEST_X[step * batch_size:(step + 1) * batch_size]
            y = TEST_LABEL[step * batch_size:(step + 1) * batch_size]

        data = torch.FloatTensor(x)
        target = torch.LongTensor(y)
        data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            corr = correct(output, target, topk=(1,))
        correct1 = correct1 + corr[0]


    test_loss = test_loss / (test_step + 1)

    # print(
    #     '\nTest set: Average loss: {:.4f}, Top1: {}/{} ({:.2f}%), '.format(test_loss, int(correct1), batch_size * (test_step + 1),
    #                                    100. * correct1 / (batch_size * (test_step + 1))))
    print(
        '\nTest set: Average loss: {:.4f}, Top1: {}/{} ({:.2f}%), '.format(test_loss, int(correct1),
                                                                           test_num,
                                                                           100. * correct1 / test_num))
    return test_loss, correct1 / test_num

def correct(output, target, topk=(1,)):
    """Computes the correct@k for the specified values of k"""
    maxk = max(topk)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t().type_as(target)
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0).item()
        res.append(correct_k)
    return res

def save_checkpoint(state, is_best, filepath='./', filename='checkpoint.pth.tar'):
    save_path = os.path.join(filepath, filename)
    best_path = os.path.join(filepath, 'model_best.pth.tar')
    torch.save(state, save_path)
    if is_best:
        shutil.copyfile(save_path, best_path)