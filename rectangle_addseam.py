

import cv2
import numpy as np

INF = 1111111111

def get_energy(img, mask):
    dx = cv2.Sobel(img, 3, 1, 0)
    dy = cv2.Sobel(img, 3, 0, 1)
    out = np.zeros((dx.shape[0], dx.shape[1]), dtype=np.float64)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(3):
                out[i, j] += np.sqrt(dx[i, j][k] * dx[i, j][k] + dy[i, j][k] * dy[i, j][k])
            if mask[i, j] != 0:
                out[i, j] += 10000000
    output = out.copy()
    return output

def update_energy(img, output, mask, st, en, to):
    W = img.shape[1]
    H = img.shape[0]
    out = output.copy()
    for i in range(st, en + 1):
        for j in range(W - 1, to[i] - 1, -1):
            if j > to[i]:
                out[i, j] = out[i, j - 1]
            else:
                z = np.array([0, 0, 0])
                l = img[i, j - 1] if j > 0 else z
                r = img[i, j + 1] if j < W - 1 else z
                u = img[i - 1, j] if i > 0 else z
                d = img[i + 1, j] if i < H - 1 else z
                val = np.sqrt((l[0] - r[0]) * (l[0] - r[0]) + (l[1] - r[1]) * (l[1] - r[1])) + np.sqrt(
                    (l[2] - r[2]) * (l[2] - r[2]) + (u[0] - d[0]) * (u[0] - d[0])) + np.sqrt(
                    (u[1] - d[1]) * (u[1] - d[1]) + (u[2] - d[2]) * (u[2] - d[2]))
                out[i, j] = val
            if mask[i, j] != 0:
                out[i, j] = 10000000
    output = out.copy()
    return output

def path(img, dir, st, en):
    H, W = img.shape[0], img.shape[1]
    if dir == 2 or dir == 3:
        t = st
        st = en
        en = t
        st = H - st - 1
        en = H - en - 1
    dp = [[0 for i in range(W)] for j in range(H)]
    for i in range(W):
        dp[st][i] = int(img.at<double>(st, i))
    for i in range(st + 1, en + 1):
        for j in range(W):
            if j == 0:
                dp[i][j] = min(dp[i - 1][j], dp[i - 1][j + 1])
            elif j == W - 1:
                dp[i][j] = min(dp[i - 1][j], dp[i - 1][j - 1])
            else:
                dp[i][j] = min(min(dp[i - 1][j - 1], dp[i - 1][j]), dp[i - 1][j + 1])
            dp[i][j] += int(img.at<double>(i, j))
    to = [0 for i in range(H)]
    minn = float('inf')
    tmp = -1
    for i in range(W):
        if dp[en][i] < minn:
            minn = dp[en][i]
            tmp = i
    to[en] = tmp
    pos = (en, tmp)
    # print(minn)
    while pos[0] > st:
        x = pos[0]
        y = pos[1]
        res = dp[x][y] - int(img.at<double>(x, y))
        if y == 0:
            if res == dp[x - 1][y]:
                pos = (x - 1, y)
            elif res == dp[x - 1][y + 1]:
                pos = (x - 1, y + 1)
            else:
                print("error")
        elif y == W - 1:
            if res == dp[x - 1][y]:
                pos = (x - 1, y)
            elif res == dp[x - 1][y - 1]:
                pos = (x - 1, y - 1)
            else:
                print("error")
        else:
            if res == dp[x - 1][y]:
                pos = (x - 1, y)
            elif res == dp[x - 1][y + 1]:
                pos = (x - 1, y + 1)
            elif res == dp[x - 1][y - 1]:
                pos = (x - 1, y - 1)
            else:
                print("error")
        to[pos[0]] = pos[1]
    return to



def add_seam(img, to, dir, mask, st, en, disimg):
    W = img.shape[1]
    H = img.shape[0]
    for i in range(st, en + 1):
        for k in range(3):
            img[i, to[i]][k] = (img[i, to[i] - 1][k] + img[i, to[i]][k]) / 2 + 0.5
        if mask[i, to[i]] == 0:
            mask[i, to[i]] = 2
        elif mask[i, to[i]] == 1:
            mask[i, to[i]] = 3
    for i in range(st, en + 1):
        for j in range(W - 1, to[i], -1):
            img[i, j] = img[i, j - 1]
            mask[i, j] = mask[i, j - 1]
            dis = [0, 0]
            if dir == 1:
                dis[0] = 0
                dis[1] = 1
            elif dir == 2:
                dis[0] = 1
                dis[1] = 0
            elif dir == 3:
                dis[0] = 0
                dis[1] = -1
            else:
                dis[0] = -1
                dis[1] = 0
            disimg[i, j] += dis
    return disimg

def rot(img, flag):
    if flag == 4:
        img = img.transpose()
        img = cv2.flip(img, 1)
    if flag == 3:
        img = cv2.flip(img, -1)
    if flag == 2:
        img = img.transpose()
        img = cv2.flip(img, 0)

def invrot(img, flag):
    if flag == 4:
        img = cv2.flip(img, 1)
        img = img.transpose()
    if flag == 3:
        img = cv2.flip(img, -1)
    if flag == 2:
        img = cv2.flip(img, 0)
        img = img.transpose()

def get_len(bor, flag, len, dir, st, en):
    dif, l, r = 0, 0, 0
    num = bor.shape[0]
    if flag == 1 or flag == 3:
        for i in range(num):
            if bor[i] == 2:
                bor[i] = 1
            elif bor[i] == 3:
                bor[i] == 1
            if i == 0:
                dif = bor[i]
            elif i == num:
                dif = -bor[i - 1]
            else:
                dif = bor[i] - bor[i - 1]
            if dif == 1:
                l = i
            if dif == -1:
                r = i - 1
                if r - l + 1 > len:
                    len = r - l + 1
                    dir = flag
                    st = l
                    en = r
    else:
        for i in range(num):
            if bor[i] == 2:
                bor[i] = 1
            elif bor[i] == 3:
                bor[i] = 1
            if i == 0:
                dif = bor[i]
            elif i == num:
                dif = -bor[i - 1]
            else:
                dif = bor[i] - bor[i - 1]
            if dif == 1:
                l = i
            if dif == -1:
                r = i - 1
                if r - l + 1 > len:
                    len = r - l + 1
                    dir = flag
                    st = l
                    en = r
    return dir

def find_dir(mask, dir, st, en):
    W, H = mask.shape[1], mask.shape[0]
    len = 0
    dir = 0
    st = 0
    en = 0
    for i in range(1, 5):
        if i == 1:
            bor = mask[:, W-1].copy()
            dir = get_len(bor, 1, len, dir, st, en)
        elif i == 2:
            bor = mask[H-1, :].copy()
            dir = get_len(bor, 2, len, dir, st, en)
        elif i == 3:
            bor = mask[:, 0].copy()
            dir = get_len(bor, 3, len, dir, st, en)
        else:
            bor = mask[0, :].copy()
            dir = get_len(bor, 4, len, dir, st, en)
    if len < 28:
        dir = 0

    return dir

def localwrap(oriimg, orimask, disimg):
    dir = 0
    st = 0
    en = 0
    mask = orimask.copy()
    img = oriimg.copy()
    grad = get_energy(img, mask)
    while True:
        dir = find_dir(mask, dir, st, en)
        if dir == 0:
            break
        img = rot(img, dir)
        grad = rot(grad, dir)
        mask = rot(mask, dir)
        disimg = rot(disimg, dir)
        to = path(grad, dir, st, en)
        img = add_seam(img, to, dir, mask, st, en, disimg)
        grad = update_energy(img, grad, mask, st, en, to)
        del to
        img = invrot(img, dir)
        grad = invrot(grad, dir)
        mask = invrot(mask, dir)
        disimg = invrot(disimg, dir)

    return img.copy()