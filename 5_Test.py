from PIL import Image
import os, glob, numpy as np
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = ["HD", "HL", "SD", "SL"]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    plt.figure(figsize=(15, 12))

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    # plt.savefig('./result_image/' + name + '.png')

    return ax


# caltech_dir = "./test_set/"
names = ['BM', 'JJ']

for name in names:
    caltech_dir = "./test_set/"
    image_size = 512
    image_w = image_size
    image_h = image_size

    pixels = image_h * image_w * 3

    X = []
    filenames = []
    files = glob.glob(caltech_dir + name + "/*.*")
    for i, f in enumerate(files):
        img = Image.open(f)
        img = img.convert("RGB")
        img = img.resize((image_w, image_h))
        data = np.asarray(img)
        filenames.append(f)
        X.append(data)

    X = np.array(X)
    model = load_model('./model/multi_img_classification_' + name + '.model')

    prediction = model.predict(X)
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    cnt = 0

    list_real_label = []
    list_predict_label = []

    # 이 비교는 그냥 파일들이 있으면 해당 파일과 비교. 카테고리와 함께 비교해서 진행하는 것은 _4 파일.
    for i in prediction:
        pre_ans = i.argmax()  # 예측 레이블
        print(i)
        print(pre_ans)
        pre_ans_str = ''
        # "Happy(tension up)", "Sad(이별 및 슬픔)", "Medium(약간 잠자기 전에 듣기 좋은 노래)"
        # BM
        if pre_ans == 0:
            pre_ans_str = "HD"
        elif pre_ans == 1:
            pre_ans_str = "HL"
        elif pre_ans == 2:
            pre_ans_str = "SD"
        elif pre_ans == 3:
            pre_ans_str = "SL"

        if i[0] >= 0.8: print("해당 " + filenames[cnt].split("\\")[1] + "이미지는 " + pre_ans_str + "로 추정됩니다.")
        if i[1] >= 0.8: print("해당 " + filenames[cnt].split("\\")[1] + "이미지는 " + pre_ans_str + "로 추정됩니다.")

        # BM
        if i[2] >= 0.8: print("해당 " + filenames[cnt].split("\\")[1] + "이미지는 " + pre_ans_str + "로 추정됩니다.")
        if i[3] >= 0.8: print("해당 " + filenames[cnt].split("\\")[1] + "이미지는 " + pre_ans_str + "로 추정됩니다.")

        # real_label = filenames[cnt].split("\\")[1][0:1]
        # predict_label = pre_ans_str[0:1]

        # BM
        real_label = filenames[cnt].split("\\")[1][0:2]
        predict_label = pre_ans_str[0:2]

        print('RL : ', real_label, 'PL : ', predict_label)
        # print(predict_label)

        list_real_label.append(real_label)
        list_predict_label.append(predict_label)

        cnt += 1
        # print(i.argmax()) #얘가 레이블 [1. 0. 0.] 이런식으로 되어 있는 것을 숫자로 바꿔주는 것.
        # 즉 얘랑, 나중에 카테고리 데이터 불러와서 카테고리랑 비교를 해서 같으면 맞는거고, 아니면 틀린거로 취급하면 된다.

    # BM
    classes = ["HD", "HL", "SD", "SL"]
    np.set_printoptions(precision=4)

    # Plot non-normalized confusion matrix
    plot_confusion_matrix(list_real_label, list_predict_label, classes=classes,
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plot_confusion_matrix(list_real_label, list_predict_label, classes=classes, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()
    # plt.savefig('./result_image/' + name + '.png')
