import torch.nn as nn
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from my_function_1 import *
import os
import shutil
from kim_dense_bam_300x153_wo_first_att import KimNet
from run import train, test, save_checkpoint
from datetime import datetime
from logger import CsvLogger

os.environ["CUDA_VISIBLE_DEVICES"] = '1'


batch_size = 64
decay = 4e-5
Fold = 5
Earlystop = 50
nb_classes = 7
epochs = 300
# MAX_Q_SIZE = 160
# WORKERS = 8  #16
LEARNING_RATE = 0.0001
monitor_index = 'acc'
col = 6000
Freq = 2000
Sub_freq = 2000
Pad_to = 300
fft_number = Pad_to
NFFT = 51
noverlap = 12
Generate_List = False
# fft_number = int(Pad_to/2+1) #150  #time_steps = int(pad_to/2+1),deppend on :pad_to=299
# Pad_to = fft_number*2-1
sub_col = int(col/(Freq/Sub_freq))
step_input_size = int((sub_col-NFFT)/(NFFT-noverlap))+1 #153
best_test = 0
log_interval = 50


PATH = './keras_result/complex/stft_7act_' + str(Fold) + 'fold_'+str(NFFT)+'_' + str(noverlap) +'_Adam_'+str(batch_size)+ '_'+monitor_index +'_5_5_5_psd_sgd_1/'
List_path = './list/7act_' + str(Fold) + 'fold/'
makedir(List_path)
All_list_name = 'all_list.txt'
makedir(PATH)
# shutil.copy('kim_train.py',PATH)
# shutil.copy('my_function_1.py',PATH)
train_list = 'train_list.txt'
test_list = 'test_list.txt'
Data_Path = '/home/guoji/radar/radar_7act_data_I_Q/'
All_list_path = PATH+All_list_name
test_acc = []
results_dir = './results'

def main():
    cudnn.benchmark = True
    time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = os.path.join(results_dir, time_stamp)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if Generate_List:
        generate_radar_all_list(Data_Path=Data_Path, List_name=All_list_name, List_Save_Path=List_path)
        # set_label(All_List_Path=List_path + All_list_name)
        generate_radar_train_test_list(Path=List_path, All_List_Path=List_path + All_list_name, Train_list=train_list,
                                       Test_list=test_list,
                                       Fold=Fold, Reture_number=False)

    shutil.copy(List_path + 'all_list.txt', PATH)
    shutil.copy(List_path + 'shuff_all_data_list.txt', PATH)
    shutil.copy(List_path + 'data_distribution.txt', PATH)
    f = open(All_list_path, 'r')
    All_data_n = len(f.readlines())
    print('all data size :', All_data_n)
    f.close()

    for fnum in range(Fold):
        Sub_path = PATH + 'fold' + str(fnum) + '/'
        Sub_List_path = List_path + 'fold' + str(fnum) + '/'
        makedir(Sub_path)
        shutil.copy(Sub_List_path + train_list, Sub_path)
        shutil.copy(Sub_List_path + test_list, Sub_path)


    print('==> Preparing data ..')
    for flod_num in range(Fold):
        print('fold num: ', flod_num)
        Sub_path = PATH + 'fold' + str(flod_num) + '/'
        TRAIN_X, TRAIN_Y = generate_radar_bacth_complex(Sub_path + train_list, colum=col, nb_classes=nb_classes,
                                                        NFFT=NFFT, noverlap=noverlap, frequce=Freq, sub_freq=Sub_freq,
                                                        pad_to=Pad_to, shuffle=False, stft_form=False, stft_handle=True,
                                                        only_real=False)
        # train_num = len(TRAIN_Y)
        # TRAIN_X = np.reshape(TRAIN_X, (train_num, 1, Pad_to, step_input_size))
        TRAIN_LABEL = np.argmax(TRAIN_Y, axis=1)

        TEST_X, TEST_Y = generate_radar_bacth_complex(Sub_path + test_list, colum=col, nb_classes=nb_classes,
                                                      NFFT=NFFT, noverlap=noverlap, frequce=Freq, sub_freq=Sub_freq,
                                                      pad_to=Pad_to, shuffle=False, stft_form=False, stft_handle=True,
                                                      only_real=False)
        # test_num = len(TEST_Y)
        # TEST_X = np.reshape(TEST_X, (test_num, 1, Pad_to, step_input_size))
        TEST_LABEL = np.argmax(TEST_Y, axis=1)
        print("=> creating model...")


        model = KimNet(classes=7).cuda()
        # optimizer = torch.optim.SGD(model.parameters(), LEARNING_RATE, momentum=0.9, weight_decay=decay)
        optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE, weight_decay=decay)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=20, verbose=True, min_lr=0.00001)
        # lr_scheduler = None
        criterion = nn.CrossEntropyLoss().cuda()

        csv_logger = CsvLogger(filepath=save_path, filename='fold' + str(flod_num) + 'results.csv')
        train_network(epochs, lr_scheduler, model, TRAIN_X, TRAIN_LABEL, TEST_X, TEST_LABEL, optimizer, criterion,
                      batch_size, log_interval, best_test, save_path, csv_logger, flod_num)
    print('each flod test acc: ', test_acc)
    print('averg acc: ', np.mean(np.array(test_acc)))

def train_network(epochs, lr_scheduler, model, TRAIN_X, TRAIN_LABEL, TEST_X, TEST_LABEL, optimizer, criterion,
                  batch_size, log_interval, best_test, save_path, csv_logger, flod_num):
    for epoch in range(epochs):
        train_loss, train_accuracy1 = train(model, TRAIN_X, TRAIN_LABEL, epoch, optimizer, criterion,
                                                             batch_size, log_interval)
        test_loss, test_accuracy1 = test(model, TEST_X, TEST_LABEL, criterion, batch_size)
        lr_scheduler.step(test_accuracy1)
        lr = optimizer.param_groups[0]['lr']
        print('lr: ', lr)

        save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_prec1': best_test,
                         'optimizer': optimizer.state_dict()}, test_accuracy1 > best_test, filepath=save_path,
                        filename='fold' + str(flod_num) + 'checkpoint.pth.tar')
        if test_accuracy1 > best_test:
            best_test = test_accuracy1

        csv_logger.write({'epoch': epoch + 1, 'val_acc': test_accuracy1,
                          'val_loss': test_loss, 'train_acc': train_accuracy1,
                          'train_loss': train_loss, 'val_max': best_test})
    test_acc.append(best_test)

    print('best_test: ', best_test)


if __name__ == "__main__":
    main()