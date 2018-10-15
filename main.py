import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

""" ============================== Matrix Inversion Methods ============================= """

def transposeMatrix(m):
    t = map(list, zip(*m))
    #rint(t)
    return t


def getMatrixMinor(m,i,j):
    min = [row[:j] + row[j+1:] for row in (m[:i]+m[i+1:])]
    #print(min)
    return min


def getMatrixDeternminant(m):
    #base case for 2x2 matrix
    if len(m) == 2:
        return m[0][0]*m[1][1]-m[0][1]*m[1][0]

    determinant = 0
    for c in range(len(m)):
        determinant += ((-1)**c)*m[0][c]*getMatrixDeternminant(getMatrixMinor(m,0,c))
    return determinant


def getMatrixInverse(m):
    determinant = getMatrixDeternminant(m)
    #special case for 2x2 matrix:
    if len(m) == 2:
        return [[m[1][1]/determinant, -1*m[0][1]/determinant],
                [-1*m[1][0]/determinant, m[0][0]/determinant]]

    #find matrix of cofactors
    cofactors = []
    for r in range(len(m)):
        cofactorRow = []
        for c in range(len(m)):
            minor = getMatrixMinor(m,r,c)
            cofactorRow.append(((-1)**(r+c)) * getMatrixDeternminant(minor))
        cofactors.append(cofactorRow)

    cofactors = transposeMatrix(cofactors)

    for r in range(len(cofactors)):
        for c in range(len(cofactors)):
            cofactors[r][c] = cofactors[r][c]/determinant
    return cofactors

""" ====================================================================================== """


""" ================== Funções para execução do Algoritmo Least Square =================== """

def transpose(matrix_X):
    Xt = np.transpose(matrix_X)
    return Xt


def weight(X):
    return [1 for i in X[0]]


def xWithbias2D(matrix_X):
    matrix_Xb = [[],[]]

    #print("Matrix x: ", matrix_X)

    for i in matrix_X:
        matrix_Xb[0].append(i)
        matrix_Xb[1].append(1)
    return matrix_Xb


def xWithbias3D(matrix_x1, matrix_x2):
    matrix_Xb = [[], [], []]

    for i in range(0, len(matrix_x1)):
        matrix_Xb[0].append(matrix_x1[i])
        matrix_Xb[1].append(matrix_x2[i])
        matrix_Xb[2].append(1)

    return matrix_Xb


def xWithBiasQuadratic2D(matrix_x):
    matrix_xb = [[], [], []]

    for i in matrix_x:
        matrix_xb[0].append(i)
        matrix_xb[1].append(i**2)
        matrix_xb[2].append(1)

    return matrix_xb


def xwithBiasQuadratic3D(matrix_x1, matrix_x2):
    matrix_xb = [[], [], [], [], []]

    for i in range(0, len(matrix_x1)):
        matrix_xb[0].append(matrix_x1[i])
        matrix_xb[1].append(matrix_x2[i])
        matrix_xb[2].append(matrix_x1[i]**2)
        matrix_xb[3].append(matrix_x2[i]**2)
        matrix_xb[4].append(1)

    return matrix_xb


def leastSquare(X,  y, w):
    print('funcao para descobrir a mastriz de minimos quadrados')
    W = []

    """ 
    Verificação do vetor de peso
    
    checa se o vetor de peso esta vazio, nesse caso o vetor é iniciado com valores 1 
    caso contrario o vetor de peso da função recebe o vetor passado 
    """
    if not w:
        W = weight(X)
        #print(W)
    else:
        W = w

    """ Transposição da Matrix X """
    Xt = transpose(X)
    #print(Xt)

    """ Multiplicação do vetor de peso pela matriz X """
    Mtemp = []
    for i in range(0, len(X)):
        Mtemp.append((Xt[:, i] * W))


    #Mtemp = np.matmul(W, Xt)
    #print("Mtemp ", Mtemp)

    """
    Primenra Parte da equação onde é multiplicada a Matriz X pela sua transposta e 
    assim obter uma matriz quadrática 
    """

    Mtemp2 = np.matmul(Mtemp, Xt)
    #print(Mtemp2)

    """ Inversão da matrix quadradica obtida acima """

    # Usando Metodo local para inverter a matriz
    #Xi = getMatrixInverse(Xtemp)
    #print(Xi)


    # Usando Metodo do numpy p/ inverter a matriz
    Xin = np.linalg.inv(Mtemp2)
    #print(Xin)

    """ Terceira parte da equação onde multiplicamos a matrix y pela transposta de X """
    # multiplicação do vetor de peso pela matriz y
    Mtemp3 = W * y

    Mtemp4 = np.matmul(Mtemp3, Xt)
    #print(Mtemp4)

    """ Quarta parte da equação onde obtemos as coodenadas que descrevem a reta """
    ls = np.matmul(Xin, Mtemp4)
    print(ls)

    x = []
    Y = []
    for i in X[0]:
        x.append(ls[0]*float(i)+ls[1])
    for i in y:
        Y.append(ls[0] * float(i) + ls[1])

    return (ls)


def wheightedLeastSquare(X, y, w):
    ls = leastSquare(X, y, w)

    W = []
    for i in range(0, len(X[0])):
        #print(abs(1/ ( (  y[i] - (ls[0] + X[0][i] * ls[1]  )))))
        W.append(abs(1/ ( (  y[i] - (ls[0] + X[0][i] * ls[1]  )))))
    print(W)

    return leastSquare(X, y, W)

def func(x, b, a, c):
    ae = []
    #print("X is :", x)
    for i in x:
        ae.append( a * i ** 2 + b * i + c)
    #print(ae)
    return ae

""" ======================================================================================= """

''' ====================== Funções para formar o Menu do programa ========================= '''

def basesMenu():
    print("==================================")
    print("1 - Water Alps Base")
    print("2 - Books, Attend and Grades Base")
    print("3 - US Census")
    print("4 - Height and Shoes")
    print("5 - exemple Base")
    print("==================================")


def waterOption(option):
    water = r'./bases/alpswater.xlsx'

    aw = pd.read_excel(water)

    x = aw["BPt"]
    y = aw["Pressure"]
    w = []

    if option == '1':
        X = xWithbias2D(x)
        ls = leastSquare(X, y, w)

        z = ls
        print("Z: ", z)

        p = np.poly1d(z)

        plt.scatter(X[0], y)
        plt.plot(X[0], p(X[0]), "g--",  label='fit: a=%5.3f, b=%5.3f' % tuple(z))

        plt.legend()
        plt.xlabel('Pressure')
        plt.ylabel('Bpt')
        plt.show()


    elif option == '2':

        Xq = xWithBiasQuadratic2D(x)
        ls = leastSquare(Xq, y, w)

        print(Xq)

        z = ls
        print("Z: ", z)

        plt.scatter(Xq[0], y)

        plt.plot(Xq[0], func(Xq[0], *z), "g--", label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(z))

        plt.xlabel('Pressure')
        plt.ylabel('Bpt')
        plt.legend()
        plt.show()


    elif option == '3':
        print('er')
        X = xWithbias2D(x)
        ls = wheightedLeastSquare(X, y, w)

        z = ls
        print("Z: ", z)

        p = np.poly1d(z)


        plt.scatter(X[0], y)
        plt.plot(X[0], p(X[0]), "g--", label='fit: a=%5.3f, b=%5.3f' % tuple(z))

        plt.legend()
        plt.xlabel('Pressure')
        plt.ylabel('Bpt')
        plt.show()


def booksOption(option):
    books = r'./bases/Books_attend_grade.xls'
    bag = pd.read_excel(books)

    x1 = bag["BOOKS"]
    x2 = bag["ATTEND"]
    y = bag["GRADE"]
    w = []

    if option == '1':
        X = xWithbias3D(x1, x2)
        leastSquare(X, y, w)

    elif option == '2':
        X = xwithBiasQuadratic3D(x1, x2)
        leastSquare(X, y, w)

    elif option == '3':
        X = xWithbias3D(x1, x2)
        wheightedLeastSquare(X, y, w)


def censusOption(option):
    census = r'./bases/us_census.xlsx'
    uc = pd.read_excel(census)

    x = uc["year"]
    y = uc["number"]
    w = []

    if option == '1':
        X = xWithbias2D(x)
        ls = leastSquare(X, y, w)

        z = ls
        print("Z: ", z)

        p = np.poly1d(z)

        plt.scatter(X[0], y)
        plt.plot(X[0], p(X[0]), "g--", label='fit: a=%5.3f, b=%5.3f' % tuple(z))

        plt.legend()
        plt.xlabel('Pressure')
        plt.ylabel('Bpt')
        plt.show()

    elif option == '2':

        Xq = xWithBiasQuadratic2D(x)
        ls = leastSquare(Xq, y, w)

        print(Xq)

        z = ls
        print("Z: ", z)

        plt.scatter(Xq[0], y)

        plt.plot(Xq[0], func(Xq[0], *z), "g--", label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(z))

        plt.xlabel('Pressure')
        plt.ylabel('Bpt')
        plt.legend()
        plt.show()


    elif option == '3':
        print('er')
        X = xWithbias2D(x)
        ls = wheightedLeastSquare(X, y, w)

        z = ls
        print("Z: ", z)

        p = np.poly1d(z)


        plt.scatter(X[0], y)
        plt.plot(X[0], p(X[0]), "g--", label='fit: a=%5.3f, b=%5.3f' % tuple(z))

        plt.legend()
        plt.xlabel('Pressure')
        plt.ylabel('Bpt')
        plt.show()


def shoesOption(option):
    height = r'./bases/height-shoes.xlsx'
    hs = pd.read_excel(height)

    x = hs["height"]
    y = hs["shoes"]
    w = []

    if option == '1':
        X = xWithbias2D(x)
        ls = leastSquare(X, y, w)

        z = ls
        print("Z: ", z)

        p = np.poly1d(z)

        plt.scatter(X[0], y)
        plt.plot(X[0], p(X[0]), "g--", label='fit: a=%5.3f, b=%5.3f' % tuple(z))

        plt.legend()
        plt.xlabel('Pressure')
        plt.ylabel('Bpt')
        plt.show()


    elif option == '2':

        print('X: ', x)
        print('Y: ', y)
        Xq = xWithBiasQuadratic2D(x)
        ls = leastSquare(Xq, y, w)

        print("X with default bias: ", Xq)

        z = ls
        print("Z: ", z)

        print("x2: ",func(Xq[0], *z) )

        plt.scatter(Xq[0], y)

        plt.scatter(Xq[0], func(Xq[0], *z))

        #plt.plot(Xq[0], func(Xq[0], *z), "r--", label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(z))

        plt.xlabel('Pressure')
        plt.ylabel('Bpt')
        plt.legend()
        plt.show()


    elif option == '3':
        print('er')
        X = xWithbias2D(x)
        ls = wheightedLeastSquare(X, y, w)

        z = ls
        print("Z: ", z)

        p = np.poly1d(z)

        plt.scatter(X[0], y)
        plt.plot(X[0], p(X[0]), "g--", label='fit: a=%5.3f, b=%5.3f' % tuple(z))

        plt.legend()
        plt.xlabel('Pressure')
        plt.ylabel('Bpt')
        plt.show()



def exempleOption(option):
    exemple = r'./bases/exemple.xlsx'
    ex = pd.read_excel(exemple)

    x = ex["x"]
    y = ex["y"]
    w = []

    if option == '1':
        X = xWithbias2D(x)
        leastSquare(X, y, w)

    elif option == '2':
        X = xWithBiasQuadratic2D(x)
        ls = leastSquare(X, y, w)

        print(X)

        z = ls
        print("Z: ", z)
        a1 = z[1]
        b1 = z[0]

        z1  = []

        z1.append(a1)
        z1.append(b1)
        z1.append(z[2])

        print("Z1: ", z1)


        # p = np.poly1d(z)
        # print("P: ",p)

        # axes = plt.axis()

        # axes = plt.add_subplot(111)

        print(func(X[0], * z1))

        a = []
        b = []
        # y=0
        # x=-50

        # for x3 in range(-50, 50, 1):
        #     y3 = x3 ** 2 + 2 * x3 + 2
        #     a.append(x3)
        #     b.append(y3)
        #     # x= x+1

        plt.scatter(X[0], y)

        plt.plot(X[0], func(X[0], *z1), "r--", label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(z1))

        # plt.plot(x, x, 'r--')
        # plt.plot(a, b, 'g-')
        plt.xlabel('Pressure')
        plt.ylabel('Bpt')
        plt.legend()
        plt.show()

    elif option == '3':
        X = xWithbias2D(x)
        wheightedLeastSquare(X, y, w)


def Menu():
    print("==================================")
    print("1 - metodo linear simples")
    print("2 - metodo quadratico")
    print("3 - metodo com peso")
    print("==================================")


def linar():
    basesMenu()

    imp = input("Type your the Database option: ")

    dataBaseChoice(imp, '1')


def quadratico():
    basesMenu()

    imp = input("Type your the Database option: ")

    dataBaseChoice(imp, '2')


def weighted():
    basesMenu()

    imp = input("Type your the Database option: ")

    dataBaseChoice(imp, '3')


def methodChoice(inp):

    switcher = {
        "1": linar,
        "2": quadratico,
        "3": weighted
    }

    func = switcher.get(inp, lambda: "This option do not exist !")

    return func()


def dataBaseChoice(inp, option):
    switcher = {
        "1": waterOption,
        "2": booksOption,
        "3": censusOption,
        "4": shoesOption,
        "5": exempleOption
    }

    func = switcher.get(inp, lambda: "This option do not exist !")

    return func(option)


''' ======================================================================================== '''


if __name__ == '__main__':

    Menu()

    inp = input("Thype your option: ")
    methodChoice(inp)


