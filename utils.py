import numpy as np
import scipy.linalg as ln
import matplotlib.pyplot as plt


def array_vector_product(A, x):
    """
    Perform the product of an array and a vector using the definition.

    Parameters
    ----------
    A : numpy.ndarray
        The array to be multiplied by the vector. Must have shape (m, n).
    x : numpy.ndarray
        The vector to be multiplied by the array. Must have shape (n,).

    Returns
    -------
    numpy.ndarray
        The result of the matrix-vector product. Has shape (m,).

    Raises
    ------
    ValueError
        If the shapes of the input arrays are not compatible for matrix-vector multiplication.
    """

    # Check that the shapes of the input arrays are compatible for matrix-vector multiplication.
    if A.shape[1] != x.shape[0]:
        raise ValueError("Shapes of input arrays are not compatible for matrix-vector multiplication")

    # Compute the matrix-vector product using the definition.
    m = A.shape[0]
    result = np.zeros(m)
    for i in range(m):
        for j in range(A.shape[1]):
            result[i] += A[i, j] * x[j]
    return result


def matriz_x_matrix(A, B):
    # esta funcion calcula el producto de dos matrices
    # usando la definicion
    # A y B son matrices de numpy
    # A tiene dimensiones m x n
    # B tiene dimensiones n x p     <--- ya no, se usa dot
    # C tiene dimensiones m x p
    # C = AB
    # usando la funcion array_vector_product
    # calculamos el producto de cada fila de A por cada columna de B
    # y guardamos el resultado en la matriz C

    return np.dot(A, B)
    m = A.shape[0]
    p = B.shape[1]
    C = []
    for j in range(p):
        C.append(array_vector_product(A, B[:, j]))
    return np.array(C).T


def pseudoinv(A):
    '''
    esta funcion calcula la pseudo inversa de una matriz

    :param A: matriz a invertir
    :return:  matriz pseudo inversa
    '''
    return np.linalg.pinv(A)


def kronecker_product(A, B):
    # the Kronecker product of two matrices A and B is
    # the matrix C such that C(j_1-1)N'_2+j_2;(i_1-1)N_2+i_2 = Aj_1;i_1*Bj_2;i_2 .
    # where N_1 and N_2 are the number of rows of A and B respectively
    # and N'_1 and N'_2 are the number of columns of A and B respectively
    # we define the function kronecker_product(A,B) that returns the Kronecker product of A and B
    # we assume that A and B are NumPy arrays

    # reshape arrays as needed
    A = np.atleast_2d(A)
    B = np.atleast_2d(B)

    # we initialize the matrix C
    C = np.zeros((np.shape(A)[0] * np.shape(B)[0], np.shape(A)[1] * np.shape(B)[1]))
    # we iterate over the rows of A
    for i in range(np.shape(A)[0]):
        # we iterate over the columns of A
        for j in range(np.shape(A)[1]):
            # we iterate over the rows of B
            for k in range(np.shape(B)[0]):
                # we iterate over the columns of B
                for l in range(np.shape(B)[1]):
                    C[i * np.shape(B)[0] + k, j * np.shape(B)[1] + l] = A[i, j] * B[k, l]
    return C


# ----------------------------------------------------------------------------------------------------------------------

def kronecker_product_l(list):
    # returns kronecker product for a list of matrices
    if len(list) == 1:
        return list[0]
    else:
        return kronecker_product(list[0], kronecker_product_l(list[1:]))


# ----------------------------------------------------------------------------------------------------------------------
def sum_matrices(list):
    # returns sum of matrices in list
    if len(list) == 1:
        return list[0]
    else:
        return list[0] + sum_matrices(list[1:])


def separated_product(A, B, N):
    # this function returns the product of two matrices A and B
    # that are given in separated representation with the dimensions N
    C = []
    for A_sub in A:
        for B_sub in B:
            C.append(kronecker_product_l([np.dot(A_sub[i], B_sub[i]) for i in range(len(A_sub))]))

    return sum_matrices(C)
def conforma_matriz(A, N):
    result = np.zeros((np.prod(N), np.prod(N)))
    for i in range(len(A)):
        result += kronecker_product_l(A[i])
    return result

# ----------------------------------------------------------------------------------------------------------------------
def error_als3(x, a, d):
    # esta función calcula el error del problema
    # como el productorio de la diferencia entre cada
    # componente de x y a
    error = 1
    for i in range(d):
        error = error * np.linalg.norm(x[i] - a[i])
        # print(error)
    return error


# ----------------------------------------------------------------------------------------------------------------------
# ***** definimos ahora el algoritmo ALS de mínimos cuadrados alternados *****
# ----------------------------------------------------------------------------------------------------------------------

def ALS3(f, A, itera_max, tol, N):
    try:
        d = len(N)
        # esta función devuelve el vector x que minimiza el problema de mínimos cuadrados
        # Ax = f
        # mediante el algoritmo ALS3
        # inicializamos el vector x

        x = [np.random.rand(N[i], 1) for i in range(d)]
        # inicializamos el contador de iteraciones
        itera = 0
        # iteramos hasta alcanzar el número máximo de iteraciones o hasta que el error sea menor que la tolerancia
        while itera < itera_max:
            # movemos el actual vector x al vector a
            a = x.copy()
            if itera == 0:
                x = a.copy()
            # iteramos sobre cada componente del vector x
            for k in range(d):
                # calculamos la matriz Z que es la matriz con
                # el producto de kronecker de las componentes
                # del vector x excepto la k-ésima. A partir
                # de la k-ésima componente, usamos las componentes
                # del vector a
                Z = np.eye(N[k])
                for i in range(d):
                    if i < k:
                        # se premultiplica por x[i]
                        Z = kronecker_product(x[i], Z)
                    elif i > k:
                        # se posmultiplica por a[i]
                        Z = kronecker_product(Z, a[i])

                    else:
                        pass
                # sale del producto de kronecker correspondiente a la componente k
                # luego la premultiplicamos por la matriz A

                Z = matriz_x_matrix(A, Z)
                # producto con A correspondiente a la componente k
                # calculamos la pseudo inversa de la matriz Z
                # Z_pinv = pseudoinv2(Z)
                Z_pinv = pseudoinv(Z)
                # actualizamos el nuevo vector x_k
                x[k] = matriz_x_matrix(Z_pinv, f)
                x[k] = x[k].reshape(N[k], 1)

            # comprobamos si el error es menor que la tolerancia
            if error_als3(x, a, d) < tol:
                print(itera, 'iteraciones realizadas')
                return x
            itera += 1
        print(itera, 'iteraciones realizadas')
        return x
    except KeyboardInterrupt:
        print(itera, 'iteraciones realizadas')
        return x


# ----------------------------------------------------------------------------------------------------------------------
#    DEFINIMOS EL ALGORITMO ALS4   -- MÍNIMOS CUADRADOS ALTERNADOS CON REPRESENTACIÓN SEPARADA
# ----------------------------------------------------------------------------------------------------------------------

def ALS4(f, A, itera_max, tol, N):
    '''
    The function ALS4 uses the Alternating Least Squares algorithm with Representation Separation of Coefficients and Variables to approximate the solution of the linear system Ax = f.
    The solution x is represented as a list of d vectors of sizes [N_1, N_2, ..., N_d], where d is the dimension of the problem.
    The matrices A are represented as a list of r_A matrices of sizes [[N_1, N_2, ..., N_d], [N_1, N_2, ..., N_d], ..., [N_1, N_2, ..., N_d]].

    Parameters:
        - f: a vector of size n = prod(N), the right-hand side of the linear system Ax = f.
        - A: a list of matrices of sizes [[N_1, N_2, ..., N_d], [N_1, N_2, ..., N_d], ..., [N_1, N_2, ..., N_d]], the matrices that define the linear system.
        - itera_max: maximum number of iterations of the algorithm.
        - tol: tolerance of error.
        - N: a vector of sizes of the components of x and matrices of A, that is, N=[N_1, N_2, ..., N_d].

    Returns:
        - a list of vectors x=[x_1, x_2, ..., x_d] of sizes [N_1, N_2, ..., N_d], representing the solution of the problem when kronecker-multiplied each of the components of x.
    '''
    try:
        d = len(N)  # number of dimensions of the problem
        r_A = len(A)  # number of lists of d matrices involved in the definition of A
        x = [np.random.rand(N[i], 1) for i in range(d)]  # initial guess for the solution x
        itera = 0  # iteration counter
        while itera < itera_max:  # loop until maximum number of iterations is reached
            a = x.copy()  # copy of the previous solution x
            # print(itera, 'iteraciones realizadas')
            if itera == 0:
                x = a.copy()  # this is for dimensionality purposes
            for k in range(d):  # loop over the components of x
                for j in range(r_A):  # loop over the matrices in A
                    Z_aux = A[j][k]  # auxiliary matrix Z, initialized with the corresponding matrix in A     dim x dim
                    # print('Z_aux.shape', Z_aux.shape)
                    for i in range(d):  # loop over the components of x and matrices in A
                        if i < k:  # for i<k, kronecker product is applied to the left of Z_aux
                            # A_aux = A[j][i]@x[i]  # n_i x n_i  dot  n_i x 1 --> n_i x 1
                            A_aux = matriz_x_matrix(A[j][i], x[i])  # n_i x n_i  dot  n_i x 1 --> n_i x 1
                            # print('A_aux.shape', A_aux.shape) #           --- en primera iteracion ----
                            Z_aux = kronecker_product(A_aux, Z_aux)  # n_i x 1   kr_dot  dim x dim --> (n_i)(dim) x dim
                        elif i > k:  # for i>k, kronecker product is applied to the right of Z_aux
                            A_aux = matriz_x_matrix(A[j][i], a[i])  # n_i x n_i  dot  n_i x 1 --> n_i x 1
                            # print('A_aux.shape', A_aux.shape)
                            Z_aux = kronecker_product(Z_aux, A_aux)
                        else:  # for i=k, do nothing
                            pass
                        # print('Z_aux.shape', Z_aux.shape)
                    # aquí Z_aux ya tiene el tamaño N x dim
                    if j == 0:
                        Z = Z_aux  # if this is the first matrix in A, Z is initialized with Z_aux
                    else:
                        Z = Z + Z_aux  # if this is not the first matrix in A, Z is updated with Z_aux
                        # print('Z.shape', Z.shape)  #
                # if range(Z) != min(Z.shape):
                #     print('Z No tiene rango completo')
                if np.linalg.matrix_rank(Z) != min(Z.shape):
                    print(np.linalg.matrix_rank(Z), min(Z.shape))
                #                print('determinante de Z', np.linalg.det(Z))
                Z_pinv = pseudoinv(Z)  # pseudo-inverse of Z
                # print('fin cálculo de pseudoinversa de Z, con Z_pinv.shape', Z_pinv.shape)
                x[k] = matriz_x_matrix(Z_pinv, f)  # solution for the k-th component of x
                x[k] = x[k].reshape(N[k], 1)  # reshape the solution to a column vector
            if error_als3(x, a, d) < tol:  # check if the relative error between the current iterate and the
                print(itera, 'iteraciones realizadas')  # previous one is below the tolerance tol
                return x  # if yes, return the current iterate
            itera += 1
        print(itera, 'iteraciones realizadas')  # if the maximum number of iterations is reached before the tolerance
        return x  # is reached, return the current iterate
    except KeyboardInterrupt:
        print(itera, 'iteraciones realizadas')  # if the user interrupts the execution of the algorithm, return the
        return x  # current iterate
    # except LinAlgError("Singular matrix"):
    #     print('máximo: ', max(Z), 'minimo', min(Z))


# ----------------------------------------------------------------------------------------------------------------------
# ***** definimos ahora el algoritmo GROU *****
# ----------------------------------------------------------------------------------------------------------------------

def GROU(f, A, e, tol, rank_max, itera_max, N, inner_procedure):
    """
    Solve linear system Ax = f using GROU algorithm with separated representation of the unknown vector.

    Parameters
    ----------
    f : numpy.ndarray
        The right-hand side of the linear system.
    A : numpy.ndarray
        The matrix of the linear system.
    e : float
        The tolerance for the norm of the residual.
    tol : float
        The tolerance for the change in the norm of the residual.
    rank_max : int
        The maximum rank of the separated representation.
    itera_max : int
        The maximum number of iterations for the ALS algorithm.

    Returns
    -------
    tuple
        If the algorithm converges, it returns a tuple (u, r_norm) where u is
         the separated representation of the unknown vector and r_norm is the norm of the residual.
        If the algorithm does not converge, it returns a tuple (i, u, r_norm) where i is the number
        of iterations performed before the algorithm stopped.
    """
    try:
        # if A is a list of matrices, the procedure used is ALS4
        # if A is a matrix, the procedure used is ALS3 <--- this goes by parameter
        if inner_procedure == ALS4:
            print('procedimiento ALS4')
            A_exp = conforma_matriz(A, N)
        elif inner_procedure == ALS3:
            print('procedimiento ALS3')
            A_exp = A
        else:
            raise TypeError('A must be a list or a numpy.ndarray / numpy.array')
        # Initialize the separated representation and the residual.
        # print('f empieza por : ', f[:9], '...')
        r = [0, f.copy()]
        u = np.zeros(np.shape(f)[0])
        # we reshape u to be a column vector
        u = u.reshape(np.shape(u)[0], 1)
        OK = 0
        MAL = 0
        ESTABLE = 0
        # Loop over the rank of the separated representation.
        for i in range(rank_max):
            # Update the residual.
            # here we swap the two elements of the list r so that r[0] is the residual of the previous iteration
            r[0], r[1] = r[1], r[0]
            # here we compute the new residual -----------------
            y = kronecker_product_l(inner_procedure(r[0], A, itera_max, tol, N))
            # we reshape y to be a column vector
            # y = y.reshape(np.shape(y)[0], 1)

            r[1] = r[0] - array_vector_product(A_exp, y)

            # Print the norm of the residual and check for convergence.
            print('(', i + 1, ') ', 'norma del residuo: ', ln.norm(r[1]))
            if ln.norm(r[1]) < ln.norm(r[0]):
                OK += 1
                # print ok if the residual norm has decreased
                print('iteración ', i + 1, 'OK')
                u = u + y
                print('u', u.shape)
            elif ln.norm(r[1]) == ln.norm(r[0]):
                print('iteración ', i + 1, 'NO HAY CAMBIO')
                MAL += 1
            else:
                print('iteración ', i + 1, 'Mal')
                # si ha ido mal la actualización por ALS* entonces se intenta otra vez
                # por si no se hubiera llegado a un mínimo lo suficientemente bajo
                r[0], r[1] = r[1], r[0]
                MAL += 1
                if MAL > 10:
                    # pero si no hay mejora en unas cuantas iteraciones, entonces se
                    # supone que ya no va a haberla en la práctica
                    return u, ln.norm(r[1])
            if ln.norm(r[1]) < e or abs(ln.norm(r[1]) - ln.norm(r[0])) < tol:
                print(OK, MAL)
                return u, ln.norm(r[1])

        # If the algorithm did not converge,
        # return the separated representation, and the residual norm.
        print(OK, MAL)
        return u, ln.norm(r[1])
    # on keyboard interrupt (Ctrl + C), print the number of improvements and degradations in the residual norm
    except KeyboardInterrupt:
        print('Detectado KeyboardInterrupt.')
        print('Número de mejoras en residuo: ', OK)
        print('Número de empeoramientos en residuo: ', MAL)
        return 'Interrupted'

# define the function f
def f(x,y,z):
    sin_x = np.sin(2*np.pi*x - np.pi)
    sin_y = np.sin(2*np.pi*y - np.pi)
    sin_z = np.sin(2*np.pi*z - np.pi)
    return 3*((2*np.pi)**2)*sin_x*sin_y*sin_z

# define the analytical solution
def u(x,y,z):
    sin_x = np.sin(2*np.pi*x - np.pi)
    sin_y = np.sin(2*np.pi*y - np.pi)
    sin_z = np.sin(2*np.pi*z - np.pi)
    return sin_x*sin_y*sin_z

# this function constructs matrices A and B
def construct_matrices(N):
    # first construct A
    h = 1.0/(N-1)
    A = np.zeros((N,N))
    A[0][0] = 2.0/h
    A[0][1] = -1.0/h
    A[N-1][N-2] = -1.0/h
    A[N-1][N-1] = 2.0/h
    for i in range(1,N-1):
        A[i][i-1] = -1.0/h
        A[i][i] = 2.0/h
        A[i][i+1] = -1.0/h
    # then construct B
    B = np.zeros((N,N))
    B[0][0] = 2.0*h/3.0
    B[0][1] = h/6.0
    B[N-1][N-2] = h/6.0
    B[N-1][N-1] = 2.0*h/3.0
    for i in range(1,N-1):
        B[i][i-1] = h/6.0
        B[i][i] = 2.0*h/3.0
        B[i][i+1] = h/6.0
    return A, B
def construye_f(N):
    h = 1.0/(N-1)
    f_vec = np.zeros(N**3)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                f_vec[i + j*N + k*N**2] = f(k*h, j*h, i*h)
    return f_vec

def construye_u(N):
    h = 1.0/(N-1)
    u_vec = np.zeros(N**3)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                u_vec[i + j*N + k*N**2] = u(k*h, j*h, i*h)
    return u_vec
# this function solves the Poisson equation
def solve_poisson_linalg(N):
    # construct matrices A and B
    A, B = construct_matrices(N)
    # construct the left hand side
    LHS = kronecker_product_l([A,B,B]) + kronecker_product_l([B,A,B]) + kronecker_product_l([B,B,A])
    # construct the right hand side
    h = 1.0/(N-1)
    f_vec = construye_f(N)
    # solve the linear system
    u_vec = np.linalg.solve(LHS, f_vec)
    return u_vec

def relative_error(u_val, result_val):
    print(np.linalg.norm(u_val,2), 'norm of u')
    print(np.linalg.norm(result_val,2), 'norm of result')

    a = np.linalg.norm(u_val - result_val)/np.linalg.norm(u_val)
    print(a, 'relative error')
    return a
def compare_analytical(result, N):
    h = 1.0/(N-1)
    u_vec = construye_u(N)

    return relative_error(np.array(u_vec), np.array(result))

def plot_error():
    N_list = [4,5,6,7,8,9,10,11,12,13,14,15]
    error_list = []
    for N in N_list:
        result = solve_poisson_linalg(N)
        error_list.append(compare_analytical(result,N))
    plt.loglog(N_list, error_list, 'o-')
    plt.xlabel('N')
    plt.ylabel('Relative error')
    plt.title('Relative error vs. N')
    plt.show()

def solve_poisson_grou(N):
    # construct matrices A and B
    A, B = construct_matrices(N)
    # construct the left hand side
    LHS = kronecker_product_l([A,B,B]) + kronecker_product_l([B,A,B]) + kronecker_product_l([B,B,A])

    # construct the right hand side
    f_vec = construye_f(N)
    itera_max = 20
    tol = 2.22E-18
    rank_max = 1000
    e = 5.0E-200
    # solve the linear system
    u_vec, resto = GROU(f_vec, LHS, e, tol, rank_max, itera_max, N=[N,N,N], inner_procedure=ALS3)
    return u_vec, resto, LHS



def plot_error_grou():
    N_list = [4,5,6,7,8,9,10,11,12]
    error_list = []

    for N in N_list:
        f_vec = construye_f(N).reshape(N**3,1)
        result, r, LHS = solve_poisson_grou(N)
        error_list.append(np.linalg.norm(np.dot(LHS, result) - f_vec))
    plt.semilogy(N_list, error_list, 'o-')
    plt.xlabel('N')
    plt.ylabel('Relative error')
    plt.title('Relative error vs. N')
    plt.show()


def plot_error_comp():
    N_list = [10]
    error_list = []
    error_list_grou = []
    for N in N_list:
        A, B = construct_matrices(N)
        # construct the left hand side
        LHS = kronecker_product_l([A, B, B]) + kronecker_product_l([B, A, B]) + kronecker_product_l([B, B, A])

        result = solve_poisson_linalg(N)
        error_list.append(compare_analytical(np.dot(LHS,result), N))
        try:
            result_grou = solve_poisson_grou(N)[0]
            error_list_grou.append(compare_analytical(np.dot(LHS,result_grou), N))
        except:
            print('GROU failed')
            error_list_grou.append(0)
    plt.loglog(N_list, error_list, 'o-', label='ALS3')
    plt.loglog(N_list, error_list_grou, 'o-', label='GROU')
    plt.xlabel('N')
    plt.ylabel('error')
    plt.title('error vs. N')
    plt.legend()
    plt.show()



def plot_error_grou_sep(lista_N):
    N_list = lista_N
    error_list = []

    for N in N_list:
        f_vec = construye_f(N).reshape(N**3,1)
        result, r, LHS = solve_poisson_grou_sep(N)
        LHS_e = conforma_matriz(LHS, [N,N,N])
        error_list.append(np.linalg.norm(np.dot(LHS_e, result) - f_vec))
    plt.semilogy(N_list, error_list, 'o-')
    plt.xlabel('N')
    plt.ylabel('Relative error')
    plt.title('Relative error vs. N')
    plt.show()

def imprime_LHS(LHS):
    LHS_e = separated_product(LHS)
    plt.spy(LHS_e)
    plt.show()
    return None


def solve_poisson_grou_sep(N):
    # construct matrices A and B
    A, B = construct_matrices(N)
    # construct the left hand side
    LHS = [[A,B,B],[B,A,B],[B,B,A]]
    imprime_LHS(LHS)
    # construct the right hand side
    f_vec = construye_f(N)
    itera_max = 10
    tol = 2.22E-18
    rank_max = 1000
    e = 5.0E-200
    # solve the linear system
    u_vec, resto = GROU_sep(f_vec, LHS, e, tol, rank_max, itera_max, N=[N, N, N], inner_procedure=ALS4)
    return u_vec, resto



