"""
https://github.com/XiaohanYE99/Rectangling-Panoramic-Images-via-Warping/tree/master
"""
import cv2
import numpy as np
import scipy.sparse.linalg as sla

pi = np.arccos(-1.0)
#
def drawGrid(gridmask, img, outimage): ## no use
    outimage = img.copy()
    for i in range(gridmask.shape[0]):
        for j in range(gridmask.shape[1]):
            if gridmask[i,j] == 1:
                outimage[i,j] = [0,255,0]

def getvertices(yy, xx, ygrid, xgrid, vx, vy):
    vx[0,0] = xgrid[yy,xx]
    vx[0,1] = xgrid[yy,xx+1]
    vx[1,0] = xgrid[yy+1,xx]
    vx[1,1] = xgrid[yy+1,xx+1]
    vy[0,0] = ygrid[yy,xx]
    vy[0,1] = ygrid[yy,xx+1]
    vy[1,0] = ygrid[yy+1,xx]
    vy[1,1] = ygrid[yy+1,xx+1]
    return vx, vy

def calintersec(s1, s2, p, flag):
    a1 = int(s1[1,1]) - int(s1[0,1])
    a2 = int(s2[1,1]) - int(s2[0,1])
    b1 = int(s1[0,0]) - int(s1[1,0])
    b2 = int(s2[0,0]) - int(s2[1,0])
    c1 = int(s1[1,0]) * int(s1[0,1]) - int(s1[1,1]) * int(s1[0,0])
    c2 = int(s2[1,0]) * int(s2[0,1]) - int(s2[1,1]) * int(s2[0,0])
    ab_matrix = np.array([[a1,b1],[a2,b2]]) + np.finfo(float).eps
    C_matrix = np.array([[-c1],[-c2]])

    try: ##Hui add
        p = np.linalg.inv(ab_matrix).dot(C_matrix)
        p = p.T
        if (p[0,0] - s1[0,0]) * (p[0,0] - s1[1,0]) <= 0:
            if (p[0,0] - s2[0,0]) * (p[0,0] - s2[1,0]) <= 0:
                if (p[0,1] - s1[0,1]) * (p[0,1] - s1[1,1]) <= 0:
                    if (p[0,1] - s2[0,1]) * (p[0,1] - s2[1,1]) <= 0:
                        flag = 1
    except:
        pass

    return flag, p

def checkIsIn(vy, vx, pstartx, pstarty, pendx, pendy):
  min_x = min(pstartx, pendx)
  min_y = min(pstarty, pendy)
  max_x = max(pstartx, pendx)
  max_y = max(pstarty, pendy)
  if (min_x < vx[0, 0] and min_x < vx[1, 0]) or (max_x > vx[0, 1] and max_x > vx[1, 1]):
    return 0
  elif (min_y < vy[0, 0] and min_y < vy[0, 1]) or (max_y > vy[1, 0] and max_y > vy[1, 1]):
    return 0
  else:
    return 1

def trans_mat(vx, vy, p, TP): ## no use
  quan = np.zeros((2, 4))
  quan[0, 0] = vx[0, 0]
  quan[1, 0] = vy[0, 0]
  quan[0, 1] = vx[0, 1]
  quan[1, 1] = vy[0, 1]
  quan[0, 2] = vx[1, 0]
  quan[1, 2] = vy[1, 0]
  quan[0, 3] = vx[1, 1]
  quan[1, 3] = vy[1, 1]
  zz = np.eye(4)
  z = np.zeros((2, 2))
  z1 = np.zeros((4, 1))
  tmp1 = np.hstack((zz, quan.T))
  tmp2 = np.hstack((quan, z))
  tmp1 = np.vstack((tmp1, tmp2))
  tmp = np.vstack((z1, p))
  x = np.linalg.inv(tmp1) @ tmp
  TT = x[:4]
  if np.linalg.norm(quan @ TT - p) > 0.0001:
    print("error")
  T = np.zeros((2, 8))
  T[0, 0] = TT[0]
  T[1, 0] = 0
  T[0, 1] = 0
  T[1, 1] = TT[0]
  T[0, 2] = TT[1]
  T[1, 2] = 0
  T[0, 3] = 0
  T[1, 3] = TT[1]
  T[0, 4] = TT[2]
  T[1, 4] = 0
  T[0, 5] = 0
  T[1, 5] = TT[2]
  T[0, 6] = TT[3]
  T[1, 6] = 0
  T[0, 7] = 0
  T[1, 7] = TT[3]
  TP = T.copy()
  return TP


def getLinTrans(pstart_y, pstart_x, yVq, xVq, T, flag):
    V = np.zeros((8,1))
    V[0,0] = xVq[0,0]
    V[1,0] = yVq[0,0]
    V[2,0] = xVq[0,1]
    V[3,0] = yVq[0,1]
    V[4,0] = xVq[1,0]
    V[5,0] = yVq[1,0]
    V[6,0] = xVq[1,1]
    V[7,0] = yVq[1,1]
    v1 = np.zeros((2,1))
    v2 = np.zeros((2,1))
    v3 = np.zeros((2,1))
    v4 = np.zeros((2,1))
    v1[0,0] = xVq[0,0]
    v1[1,0] = yVq[0,0]
    v2[0,0] = xVq[0,1]
    v2[1,0] = yVq[0,1]
    v3[0,0] = xVq[1,0]
    v3[1,0] = yVq[1,0]
    v4[0,0] = xVq[1,1]
    v4[1,0] = yVq[1,1]
    v21 = v2 - v1
    v31 = v3 - v1
    v41 = v4 - v1
    p = np.zeros((2,1))
    p[0,0] = pstart_x
    p[1,0] = pstart_y
    p1 = p - v1
    a1 = v31[0,0]
    a2 = v21[0,0]
    a3 = v41[0,0] - v31[0,0] - v21[0,0]
    b1 = v31[1,0]
    b2 = v21[1,0]
    b3 = v41[1,0] - v31[1,0] - v21[1,0]
    px = p1[0,0]
    py = p1[1,0]
    tvec = np.zeros((2,1))
    mat_t = np.zeros((2,2))
    t1n = 0
    t2n = 0
    a = 0
    b = 0
    c = 0
    if a3 == 0 and b3 == 0:
        mat_t = np.hstack((v31,v21))
        tvec = np.linalg.inv(mat_t)*p1
        t1n = tvec[0,0]
        t2n = tvec[1,0]
    else:
        a = (b2*a3 - a2*b3)
        b = (-a2*b1 + b2*a1 + px*b3 - a3*py)
        c = px*b1 - py*a1
        if a == 0:
            t2n = -c/b
        else:
            if (b*b - 4*a*c) > 0:
                t2n = (-b - np.sqrt(b*b - 4*a*c))/(2*a)
            else:
                t2n = (-b - 0)/(2*a)
        if abs(a1 + t2n*a3) <= 0.0000001:
            t1n = (py - t2n*b2)/(b1 + t2n*b3)
        else:
            t1n = (px - t2n*a2)/(a1 + t2n*a3)
    m1 = v1 + t1n*(v3 - v1)
    m4 = v2 + t1n*(v4 - v2)
    ptest = m1 + t2n*(m4 - m1)
    v1w = 1 - t1n - t2n + t1n*t2n
    v2w = t2n - t1n*t2n
    v3w = t1n - t1n*t2n
    v4w = t1n*t2n
    out = np.zeros((2,8))
    out[0,0] = v1w
    out[1,0] = 0
    out[0,1] = 0
    out[1,1] = v1w
    out[0,2] = v2w
    out[1,2] = 0
    out[0,3] = 0
    out[1,3] = v2w
    out[0,4] = v3w
    out[1,4] = 0
    out[0,5] = 0
    out[1,5] = v3w
    out[0,6] = v4w
    out[1,6] = 0
    out[0,7] = 0
    out[1,7] = v4w
    T = out.copy()
    if np.linalg.norm(T.dot(V) - p) > 0.01:
        flag = 1
        #print(np.linalg.norm(T.dot(V) - p))

    return flag

def blkdiag(input1, input2, output):
    if input1.dtype == np.uint8:
        out = np.zeros((input1.shape[0] + input2.shape[0], input1.shape[1] + input2.shape[1]), dtype=np.uint8)
        for i in range(input1.shape[0]):
            for j in range(input1.shape[1]):
                out[i, j] = input1[i, j]
        for i in range(input2.shape[0]):
            for j in range(input2.shape[1]):
                out[i + input1.shape[0], j + input1.shape[1]] = input2[i, j]
        output = out
    elif input1.dtype == np.float32:
        out = np.zeros((input1.shape[0] + input2.shape[0], input1.shape[1] + input2.shape[1]), dtype=np.float32)
        for i in range(input1.shape[0]):
            for j in range(input1.shape[1]):
                out[i, j] = input1[i, j]
        for i in range(input2.shape[0]):
            for j in range(input2.shape[1]):
                out[i + input1.shape[0], j + input1.shape[1]] = input2[i, j]
        output = out
    elif input1.dtype == np.int32:
        out = np.zeros((input1.shape[0] + input2.shape[0], input1.shape[1] + input2.shape[1]), dtype=np.int32)
        for i in range(input1.shape[0]):
            for j in range(input1.shape[1]):
                out[i, j] = input1[i, j]
        for i in range(input2.shape[0]):
            for j in range(input2.shape[1]):
                out[i + input1.shape[0], j + input1.shape[1]] = input2[i, j]
        output = out
    return output

def drawGridmask(ygrid, xgrid, rows,cols, gridmask): ## no use
    xgridN = ygrid.shape[1]
    ygridN = ygrid.shape[0]
    outmask = np.zeros((rows, cols), dtype=np.float32)
    m = 0
    for y in range(ygridN):
        for x in range(xgridN):
            if y != 0:
                if ygrid[y, x] != ygrid[y - 1, x]:
                    for i in range(int(ygrid[y - 1, x]), int(ygrid[y, x]) + 1):
                        m = (xgrid[y, x] - xgrid[y - 1, x]) / (ygrid[y, x] - ygrid[y - 1, x])
                        outmask[i, int(xgrid[y - 1, x] + int(m * (i - ygrid[y - 1, x])))] = 1
            if x != 0:
                if xgrid[y, x] != xgrid[y, x - 1]:
                    for j in range(int(xgrid[y, x - 1]), int(xgrid[y, x]) + 1):
                        m = (ygrid[y, x] - ygrid[y, x - 1]) / (xgrid[y, x] - xgrid[y, x - 1])
                        outmask[int(ygrid[y, x - 1] + int(m * (j - xgrid[y, x - 1]))), j] = 1
        
    gridmask = outmask.copy()
    return gridmask
  


def global_warp(img, disimg, mask, output):
    cols = img.shape[1]
    rows = img.shape[0]
    x_num = 30
    y_num = 20
    # rectangle grid
    xgrid = np.zeros((y_num, x_num), dtype=np.float32)
    ygrid = np.zeros((y_num, x_num), dtype=np.float32)
    x = 0
    y = 0
    for i in np.arange(0, rows, (rows-1)/(y_num-1)):
        for j in np.arange(0, cols, (cols-1)/(x_num-1)):
            xgrid[x, y] = int(j)
            ygrid[x, y] = int(i)
            y += 1
        x += 1
        y = 0
    
    #/***********warp grid********/
    warp_xgrid = xgrid.copy()
    warp_ygrid = ygrid.copy()
	#print(warp_xgrid.shape)
    for i in range(warp_xgrid.shape[0]):
        for j in range(warp_xgrid.shape[1]):
            warp_xgrid[i,j] = xgrid[i,j] - disimg[int(ygrid[i,j]),int(xgrid[i,j])][1]
            warp_ygrid[i,j] = ygrid[i,j] - disimg[int(ygrid[i,j]),int(xgrid[i,j])][0]

	#gridmask1, imageGrided1 = drawGridmask(warp_ygrid, warp_xgrid, rows, cols)
	#imageGrided1 = drawGrid(gridmask1, img)
	#cv2.imshow("imageGrided1", imageGrided1)
	#cv2.waitKey(1000)



    # /**********shape reserve mat*************/

    gridrows = y_num - 1
    gridcols = x_num - 1
    Ses = [[0 for x in range(gridcols)] for y in range(gridrows)] 
    Aq = np.zeros((8,4), dtype=np.float32)
    tmp = np.zeros((4,2), dtype=np.float32)
    for i in range(gridrows):
        for j in range(gridcols):
            tmp[0,0] = warp_xgrid[i,j]
            tmp[0,1] = warp_ygrid[i,j]
            tmp[1,0] = warp_xgrid[i,j+1]
            tmp[1,1] = warp_ygrid[i,j+1]
            tmp[2,0] = warp_xgrid[i+1,j]
            tmp[2,1] = warp_ygrid[i+1,j]
            tmp[3,0] = warp_xgrid[i+1,j+1]
            tmp[3,1] = warp_ygrid[i+1,j+1]
            for k in range(4):
                Aq[k*2,0] = tmp[k,0]
                Aq[k*2,1] = -tmp[k,1]
                Aq[k*2,2] = 1
                Aq[k*2,3] = 0
                Aq[2*k+1,0] = tmp[k,1]
                Aq[2*k+1,1] = tmp[k,0]
                Aq[2*k+1,2] = 0
                Aq[2*k+1,3] = 1
            I = np.eye(8,8, dtype=np.float32)
            Ses[i][j] = np.dot(Aq, np.dot(np.linalg.inv(np.dot(Aq.T, Aq)), Aq.T)) - I
            #print(np.sum(Ses[i][j])) #true
            #print(np.sum(Aq))

    #/***************line cut*********************/

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ls = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines = ls.detect(img_gray)[0]
    drawnLines = ls.drawSegments(img_gray, lines)

    if 0:
        cv2.imshow("Standard refinement", drawnLines)
        cv2.waitKey()
        cv2.destroyAllWindows()

    line_num = len(lines)
    num = np.zeros((y_num, x_num))
    #lineSeg = [[0 for x in range(x_num)] for y in range(y_num)] ##Hui: note delete -1
    #lineSeg = np.zeros((y_num, x_num, 5)) ##Bug
    lineSeg = []

    for i in range(line_num):
        aline = np.zeros((2,2), dtype=np.int32)
        aline[0,1] = lines[i][0][0]
        aline[0,0] = lines[i][0][1]
        aline[1,1] = lines[i][0][2]
        aline[1,0] = lines[i][0][3]
        if (mask[aline[0,0], aline[0,1]] == 1) or (mask[aline[1,0], aline[1,1]] == 1):
            continue
        
        outy1 = aline[0,0] + disimg[aline[0,0], aline[0,1]][1]
        outx1 = aline[0,1] + disimg[aline[0,0], aline[0,1]][0]
        gw = (img.shape[1] - 1) / (gridcols - 1)
        gh = (img.shape[0] - 1) / (gridrows - 1)
        stgrid_y = int(outy1 / gh)
        stgrid_x = int(outx1 / gw)
        now_x = stgrid_x
        now_y = stgrid_y
        vx = np.zeros((2,2), dtype=np.float32)
        vy = np.zeros((2,2), dtype=np.float32)
        dir = [[1,0], [0,1], [-1,0], [0,-1]]
        pst = np.zeros((1,2), dtype=np.float32)
        pen = np.zeros((1,2), dtype=np.float32)
        pnow = np.zeros((1,2), dtype=np.float32)
        pst[0,0] = aline[0,0]
        pst[0,1] = aline[0,1]
        pen[0,0] = aline[1,0]
        pen[0,1] = aline[1,1]
        pnow = pen.copy()
        flag = 0
        p = np.zeros((2,2), dtype=np.float32)
        last = -1
        tt = 0

        while True:
            tt += 1
            if tt > 5:
                break
            ok = 0
            if now_y >= y_num - 1 or now_x >= x_num - 1 or now_x < 0 or now_y < 0:
                break
            vx, vy = getvertices(now_y, now_x, warp_ygrid, warp_xgrid, vx, vy)
            isin = checkIsIn(vy, vx, pst[0,1], pst[0,0], pen[0,1], pen[0,0])
            if isin == 0:
                quad = np.array([[vy[0,1], vx[0,1]], [vy[1,1], vx[1,1]], [vy[1,0], vx[1,0]], [vy[0,0], vx[0,0]]])
                line_now = np.vstack((pst, pen))
            for k in range(4):
                if abs(last - k) == 2:
                    continue
                quad1 = np.array([[quad[k][0], quad[k][1]], [quad[(k+1)%4][0], quad[(k+1)%4][1]]])
                flag, p = calintersec(quad1, line_now, p, flag)
                if flag == 1:
                    last = k
                    ok = 1
                    now_x += dir[k][0]
                    now_y += dir[k][1]
                    pnow = p.copy()
                    break
            mat_t = np.r_[pst[0], pen[0], np.zeros(1)] ## array of lenth 5
            if now_x > x_num or now_y > y_num or now_x < 0 or now_y < 0:
                break
            if num[now_y][now_x] == 0:
                lineSeg[now_y][now_x] = mat_t.copy()
                num[now_y][now_x] += 1
            else:
                lineSeg[now_y][now_x] = np.vstack((lineSeg[now_y][now_x], mat_t)) ##Bug
            pst = pnow.copy()
            pnow = pen.copy()
            if isin == 1:
                break
            if ok == 0:
                break

    #/****************shape mat***********************************/
    quadrows = y_num - 1
    quadcols = x_num - 1
    quadID = 0
    topleftverterID = 0
    Q = np.zeros((8*quadrows*quadcols, 2*y_num*x_num))
    for i in range(quadrows):
        for j in range(quadcols):
            quadID = (i*quadcols + j) * 8
            topleftverterID = (i*x_num + j) * 2
            Q[quadID][topleftverterID] = 1
            Q[quadID][topleftverterID + 1] = 0
            Q[quadID + 1][topleftverterID] = 0
            Q[quadID + 1][topleftverterID+1] = 1
            Q[quadID + 2][topleftverterID + 2] = 1
            Q[quadID + 2][topleftverterID + 3] = 0
            Q[quadID + 3][topleftverterID + 2] = 0
            Q[quadID + 3][topleftverterID + 3] = 1
            Q[quadID + 4][topleftverterID + x_num * 2] = 1
            Q[quadID + 4][topleftverterID + x_num * 2 + 1] = 0
            Q[quadID + 5][topleftverterID + x_num * 2] = 0
            Q[quadID + 5][topleftverterID + x_num * 2 + 1] = 1
            Q[quadID + 6][topleftverterID + x_num * 2 + 2] = 1
            Q[quadID + 6][topleftverterID + x_num * 2 + 3] = 0
            Q[quadID + 7][topleftverterID + x_num * 2 + 2] = 0
            Q[quadID + 7][topleftverterID + x_num * 2 + 3] = 1
    S = None
    S_flag = 0
    Si_flag = 0
    for i in range(quadrows):
        Si = None
        Si_flag = 0
        for j in range(quadcols):
            if Si_flag == 0:
                Si = Ses[i][j]
                Si_flag += 1
            else:
                Si = blkdiag(Si, Ses[i][j], Si)
        if S_flag == 0:
            S = Si
            S_flag += 1
        else:
            S = blkdiag(S, Si, S)
    print(np.transpose(Q) @ Q)

    # #/************get theta**********************/
    delta = np.pi/49
    quad_theta = [[0 for x in range(quadcols)] for y in range(quadrows)] 
    quad_bin = [[0 for x in range(quadcols)] for y in range(quadrows)] 
    for i in range(quadrows):
        for j in range(quadcols):
            quadseg = lineSeg[i][j]
            lineN = quadseg.shape[0]
            quad_bin[i][j] = []
            quad_theta[i][j] = []
            for k in range(lineN):
                pst_x = quadseg[k,1]
                pst_y = quadseg[k,0]
                pen_x = quadseg[k,3]
                pen_y = quadseg[k,2]
                angle = np.pi/2 if pst_x == pen_x else np.arctan(float(pst_y-pen_y)/(pst_x-pen_x))
                theta = int((angle+np.pi/2)/delta)
                quad_theta[i][j].append(angle)
                quad_bin[i][j].append(theta)

    #/*************boundary mat**********************************/
    total = x_num * y_num
    B = np.zeros((total * 2, 1), dtype=np.float32)
    BI = np.zeros((total * 2, 1), dtype=np.float32)
    for i in range(0, total * 2, x_num * 2):
        B[i, 0] = 1
        BI[i, 0] = 1
    for i in range(1, x_num * 2, 2):
        B[i, 0] = 1
        BI[i, 0] = 1
    for i in range(x_num * 2 - 2, total * 2, x_num * 2):
        B[i, 0] = img.shape[1]
        BI[i, 0] = 1
    for i in range(total * 2 - x_num * 2 + 1, total * 2, 2):
        B[i, 0] = img.shape[0]
        BI[i, 0] = 1
    Dg = np.diag(BI.flatten())




    #/*********************optimization loop*********************/
    R = np.zeros((2,2), dtype=np.float32)
    e = np.zeros((2,2), dtype=np.float32)
    pst = np.zeros((2,1), dtype=np.float32)
    pen = np.zeros((2,1), dtype=np.float32)
    vx = np.zeros((2,2), dtype=np.float32)
    vy = np.zeros((2,2), dtype=np.float32)
    Cmatrixes = [[0 for x in range(quadcols)] for y in range(quadrows)] 
    iterN = 1
    NL = 0
    bad = [[[0 for x in range(110)] for y in range(quadcols)] for z in range(quadrows)]
    new_xgrid = np.zeros((y_num,x_num), dtype=np.float32)
    new_ygrid = np.zeros((y_num,x_num), dtype=np.float32)
    for it in range(iterN): #1mins/iter
        NL = 0
        bad = [[[0 for x in range(110)] for y in range(quadcols)] for z in range(quadrows)]
        Cmatrixes_flag = [[0 for x in range(quadcols+100)] for y in range(quadrows+100)]
        TT = [[[] for x in range(quadcols+100)] for y in range(quadrows+100)]
        #/*********************line mat*********************/
        for i in range(quadrows):
            for j in range(quadcols):
                lineN = lineSeg[i][j].shape[0]
                NL += lineN
                for k in range(lineN):
                    vx, vy = getvertices(i, j, warp_ygrid, warp_xgrid, vx, vy)
                    pst[0, 0] = lineSeg[i][j][k, 0]
                    pst[1, 0] = lineSeg[i][j][k, 1]
                    pen[0, 0] = lineSeg[i][j][k, 2]
                    pen[1, 0] = lineSeg[i][j][k, 3]
                    T1, T2 = None, None
                    # trans_mat(vx, vy, pst, T1)
                    # trans_mat(vx, vy, pen, T2)
                    flgg = 0
                    flgg = getLinTrans(pst[0, 0], pst[1, 0], vy, vx, T1, flgg)
                    flgg = getLinTrans(pen[0, 0], pen[1, 0], vy, vx, T2, flgg)
                    TT[i][j].append(T1)
                    TT[i][j].append(T2)
                    theta = lineSeg[i][j][k, 4]
                    R[0, 0] = np.cos(theta)
                    R[0, 1] = -np.sin(theta)
                    R[1, 0] = np.sin(theta)
                    R[1, 1] = np.cos(theta)
                    e[0, 0] = pen[1, 0] - pst[1, 0]
                    e[1, 0] = pen[0, 0] - pst[0, 0]
                    I = np.eye(2, 2, dtype=np.float32)
                    C = (R * e * np.linalg.inv(e.T * e) * e.T * R.T - I) * (T2 - T1)  # C*V
                    # print(Cmatrixes_flag[i][j])
                    # print(C)
                    if Cmatrixes_flag[i][j] == 0:
                        Cmatrixes[i][j] = C
                        Cmatrixes_flag[i][j] += 1
                    else:
                        Cmatrixes[i][j] = np.vstack((Cmatrixes[i][j], C))

        L = None
        L_flag = 0
        Li_flag = 0
        n = 0
        m = 0
        for i in range(quadrows):
            Li_flag = 0
            n = 0
            Li = None
            for j in range(quadcols):
                lineN = lineSeg[i][j].shape[0]
                if lineN == 0:
                    if Li_flag != 0:
                        x = np.zeros((Li.shape[0], 8), dtype=np.float32)
                        Li = np.hstack((Li, x))
                    else:
                        n = n + 8
                else:
                    if Li_flag == 0:
                        if n != 0:
                            Li = np.zeros((Cmatrixes[i][j].shape[0], n), dtype=np.float32)
                            Li = np.hstack((Li, Cmatrixes[i][j]))
                        else:
                            Li = Cmatrixes[i][j].copy()
                        Li_flag += 1
                    else:
                        Li = blkdiag(Li, Cmatrixes[i][j], Li)
            if L_flag == 0 and Li_flag == 0:
                m = m + n
            elif L_flag == 0 and Li_flag != 0:
                if m != 0:
                    L = np.zeros((Li.shape[0], m), dtype=np.float32)
                    L = np.hstack((L, Li))
                else:
                    L = Li
                L_flag += 1
            else:
                if Li_flag == 0:
                    Li = np.zeros((L.shape[0], n), dtype=np.float32)
                    L = np.hstack((L, Li))
                else:
                    L = blkdiag(L, Li, L)
        
        Nq = quadrows*quadcols
        lambl = 1
        lambB = 1e8
        S_matrix, Q_matrix, L_matrix,Dg_matrix = cv2.cv2eigen(S, Q, L, Dg)
        S1 = S_matrix.sparseView()
        Q1 = Q_matrix.sparseView()
        L1 = L_matrix.sparseView()
        Dg1 = Dg_matrix.sparseView()
        x1_matrix = (1.0/Nq)*S1*Q1
        x2_matrix = (lambl/NL)*L1*Q1
        x3_matrix = lambB*Dg1
        x1, x2, x3 = cv2.eigen2cv(x1_matrix, x2_matrix, x3_matrix)
        K = np.vstack((x1, x2))
        K = np.vstack((K, x3))
        BA = np.vstack((np.zeros((K.shape[0] - B.shape[0], 1)), lambB*B))
        K_matrix,BA_matrix = cv2.cv2eigen(K, BA)
        K1 = K_matrix.sparseView()
        BA1 = BA_matrix.sparseView()
        A_matrix = K1.transpose()*K1
        b_matrix = K1.transpose()*BA1
        A = A_matrix.sparseView()

        # solver = SparseLU(A) ##from eign in C ++
        solver = sla.slpu(A) ## Hui changed

        # print(solver.info()) ##Hui: need to check

        if solver.info() != Success:
            print("Compute Matrix is error")
            return
        
        x_matrix = solver.solve(b_matrix)
        x = cv2.eigen2cv(x_matrix)
        cnt = 0
        for i in range(y_num):
            for j in range(x_num):
                new_xgrid[i,j] = int(x[cnt,0]) - 1
                new_ygrid[i,j] = int(x[cnt+1,0]) - 1
                cnt += 2

        bin_num = [0] * 55
        rot_sum = [0] * 55
        for i in range(quadrows):
            for j in range(quadcols):
                lineN = lineSeg[i][j].shape[0]
                vx, vy = getvertices(i, j, new_ygrid, new_xgrid, vx, vy)
                for k in range(lineN):
                    if bad[i][j][k]:
                        continue
                    T1 = TT[i][j][k*2]
                    T2 = TT[i][j][k*2+1]
                    V = np.array([[vx[0,0], vy[0,0], vx[0,1], vy[0,1], vx[1,0], vy[1,0], vx[1,1], vy[1,1]]], dtype=np.float32)
                    st = np.dot(T1, V)
                    en = np.dot(T2, V)
                    oritheta = quad_theta[i][j][k]
                    theta = np.arctan2((st[1,0] - en[1,0]), (st[0,0] - en[0,0]))
                    delta_theta = theta - oritheta
                    if np.isnan(delta_theta):
                        continue
                    if delta_theta > np.pi/2:
                        delta_theta -= np.pi
                    if delta_theta < -np.pi/2:
                        delta_theta += np.pi
                    bin = quad_bin[i][j][k]
                    bin_num[bin] += 1
                    rot_sum[bin] += delta_theta
        for i in range(50):
            if bin_num[i] == 0:
                rot_sum[i] = 0
            else:
                rot_sum[i] = rot_sum[i] / bin_num[i]
        for i in range(quadrows):
            for j in range(quadcols):
                lineN = lineSeg[i][j].shape[0]
                for k in range(lineN):
                    lineSeg[i][j][k,4] = rot_sum[quad_bin[i][j][k]]

    vx1 = np.zeros((2,2), dtype=np.float32)
    vx2 = np.zeros((2,2), dtype=np.float32)
    vy1 = np.zeros((2,2), dtype=np.float32)
    vy2 = np.zeros((2,2), dtype=np.float32)
    outimg = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.int32)
    cnt = np.zeros((img.shape[0], img.shape[1]), dtype=np.int32)
    sx = 0
    sy = 0
    for i in range(quadrows):
        for j in range(quadcols):
            vx1, vy1 = getvertices(i, j, new_ygrid, new_xgrid, vx1, vy1)
            V1 = np.array([vx1[0,0], vy1[0,0], vx1[0,1], vy1[0,1], vx1[1,0], vy1[1,0], vx1[1,1], vy1[1,1]], dtype=np.float32).reshape(8,1)
            vx2, vy2 = getvertices(i, j, warp_ygrid, warp_xgrid, vx2, vy2)
            V2 = np.array([vx2[0,0], vy2[0,0], vx2[0,1], vy2[0,1], vx2[1,0], vy2[1,0], vx2[1,1], vy2[1,1]], dtype=np.float32).reshape(8,1)
            minx = min(min(V1[0,0], V1[2,0]), min(V1[4,0], V1[6,0]))
            maxx = max(max(V1[0,0], V1[2,0]), max(V1[4,0], V1[6,0]))
            miny = min(min(V1[1,0], V1[3,0]), min(V1[5,0], V1[7,0]))
            maxy = max(max(V1[1,0], V1[3,0]), max(V1[5,0], V1[7,0]))
            lenx = maxx - minx
            leny = maxy - miny
            sx += (img.shape[1] - 1) / (x_num - 1) / lenx
            sy += (img.shape[0] - 1) / (y_num - 1) / leny
            tx = 1.0 / (2 * lenx)
            ty = 1.0 / (2 * leny)
            for y in np.arange(0, 1, ty):
                for x in np.arange(0, 1, tx):
                    k1 = 1 - y - x + y * x
                    k2 = x - y * x
                    k3 = y - y * x
                    k4 = y * x
                    T = np.array([[k1, 0, k2, 0, k3, 0, k4, 0], [0, k1, 0, k2, 0, k3, 0, k4]], dtype=np.float32)
                    pout = np.matmul(T, V1)
                    ppre = np.matmul(T, V2)
                    x1 = int(pout[0,0])
                    y1 = int(pout[1,0])
                    x2 = int(ppre[0,0])
                    y2 = int(ppre[1,0])
                    if y1 < 0 or x1 < 0 or y2 < 0 or x2 < 0:
                        continue
                    if y1 >= img.shape[0] or x1 >= img.shape[1] or y2 >= img.shape[0] or x2 >= img.shape[1]:
                        continue
                    outimg[y1,x1] += img[y2,x2]
                    cnt[y1,x1] += 1
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if cnt[i,j] == 0:
                continue
            outimg[i,j] /= cnt[i,j]
    sx /= (1.0 * quadcols * quadrows)
    sy /= (1.0 * quadcols * quadrows)
    coll = int(outimg.shape[1] * sx)
    roww = int(outimg.shape[0] * sy)
    outimg = outimg.astype(np.uint8)
    outimg = cv2.resize(outimg, (coll, roww))
    output = outimg.copy()

    return output