import numpy as np
from typing import Union, List

class Matrix:
    """Representación algebraica de matrices booleanas para teoría de relaciones.

    Proporciona operaciones matriciales especializadas para análisis de grafos
    y relaciones binarias, con implementación basada en NumPy para eficiencia.

    Attributes:
        _matrix (np.ndarray): Matriz NumPy subyacente.
        shape (tuple): Dimensiones (filas, columnas).

    Example:
        >>> m = Matrix([[1, 0], [1, 1]])
        >>> m.transitiveClosure().toNumpy()
        array([[1, 0],
               [1, 1]])
    """


    def __init__(self, data: Union[np.ndarray, List[List[int]], List[List[bool]]]):
        """
        Crea una nueva matriz desde una lista de listas o un np.array.

        Args:
            data (Union[np.ndarray, List[List[int]], List[List[bool]]]): 
                Matriz inicial (np.array o listas de listas).
        """
        if isinstance(data, np.ndarray):
            self._matrix = data.astype(int)
        else:
            self._matrix = np.array(data, dtype=int)
        self.shape = self._matrix.shape

    def __str__(self) -> str:
        """Devuelve la representación en string de la matriz."""
        return str(self._matrix)

    def __add__(self, other: 'Matrix') -> 'Matrix':
        """Suma elemento a elemento de dos matrices."""
        return Matrix(self._matrix + other._matrix)

    def __matmul__(self, other: 'Matrix') -> 'Matrix':
        """Multiplicación matricial (usando el operador @)."""
        return Matrix(self._matrix @ other._matrix)

    def __mul__(self, other: Union['Matrix', int]) -> 'Matrix':
        """
        Multiplicación punto a punto entre matrices o por escalar.

        Args:
            other (Matrix | int): Otra matriz o escalar.

        Returns:
            Matrix: Resultado de la operación.
        """
        if isinstance(other, Matrix):
            return Matrix(self._matrix * other._matrix)
        else:
            return Matrix(self._matrix * other)

    def __eq__(self, other: 'Matrix') -> bool:
        """Compara si dos matrices son exactamente iguales."""
        return np.array_equal(self._matrix, other._matrix)

    def toNumpy(self) -> np.ndarray:
        """Devuelve la matriz interna como np.array."""
        return self._matrix

    def reflexiveClosure(self) -> 'Matrix':
        """
        Calcula la clausura reflexiva agregando unos en la diagonal principal.

        Returns:
            Matrix: Clausura reflexiva.
        """
        closure = self._matrix.copy()
        np.fill_diagonal(closure, 1)
        return Matrix(closure)

    def transitiveClosure(self, maxIter: int = 1000) -> 'Matrix':
        """Calcula la clausura transitiva usando el algoritmo de Warshall.

        Args:
            maxIter: Límite de iteraciones para prevenir no convergencia.

        Returns:
            Nueva Matrix que representa la clausura transitiva.

        Raises:
            Exception: Si no converge en maxIter iteraciones.

        Note:
            Complejidad computacional: O(n³) donde n es el tamaño de la matriz.
            Usa multiplicación matricial booleana para eficiencia.

        Example:
            >>> m = Matrix([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
            >>> m.transitiveClosure().toNumpy()
            array([[1, 1, 0],
                   [0, 1, 0],
                   [0, 0, 1]])
        """
        P_R = self._matrix.copy()
        closure = self._matrix.copy()
        for i in range(maxIter):
            lastClosure = closure.copy()
            closure = ((closure @ P_R) + closure).astype(bool).astype(int)
            if np.array_equal(closure, lastClosure):
                return Matrix(closure)
        raise Exception("No converge: max iterations reached")

    def fromRelation(self, relation: List[tuple]) -> 'Matrix':
        """
        Rellena la matriz con una relación dada (lista de pares ordenados).

        Args:
            relation (List[tuple]): Lista de pares (i, j).

        Returns:
            Matrix: Referencia a sí mismo (para encadenar).
        """
        self._matrix = np.zeros((self.shape[0], self.shape[1]), dtype=int)
        for i, j in relation:
            self._matrix[i, j] = 1
        return self

    def toRelation(self) -> List[tuple]:
        """
        Convierte la matriz a una lista de pares (i, j) con valor 1.

        Returns:
            List[tuple]: Lista de relaciones activas.
        """
        return [(i, j) for i in range(self.shape[0]) for j in range(self.shape[1]) if self._matrix[i, j] == 1]

    @staticmethod
    def identity(size: int) -> 'Matrix':
        """Crea una matriz identidad de tamaño dado."""
        return Matrix(np.identity(size, dtype=int))

    @staticmethod
    def zero(rows: int, cols: int = None) -> 'Matrix':
        """
        Crea una matriz de ceros.

        Args:
            rows (int): Número de filas.
            cols (int): Número de columnas. Si no se indica, se crea cuadrada.

        Returns:
            Matrix: Matriz de ceros.
        """
        if cols is None:
            cols = rows
        return Matrix(np.zeros((rows, cols), dtype=int))

    def isSymmetric(self) -> bool:
        """Verifica si la matriz es simétrica."""
        return np.array_equal(self._matrix, self._matrix.T)

    def isReflexive(self) -> bool:
        """Verifica si la matriz es reflexiva (unos en la diagonal)."""
        return np.all(np.diag(self._matrix) == 1)

    def exportToCSV(self, filename: str):
        """
        Guarda la matriz en un archivo CSV.

        Args:
            filename (str): Nombre del archivo destino.
        """
        np.savetxt(filename, self._matrix, fmt='%d', delimiter=',')
