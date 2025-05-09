from typing import List, Tuple, Union, Any, Dict, Set
from relationutils.matrix import Matrix
import matplotlib.pyplot as plt
import matplotlib_venn as venn

class Relation:
    """Representación matemática de relaciones binarias con operaciones de análisis.

    Implementa relaciones sobre conjuntos finitos con:
    - Representación matricial y por pares
    - Operaciones de clausura
    - Verificación de propiedades algebraicas
    - Visualización gráfica

    Attributes:
        A (List[Any]): Dominio de la relación.
        B (List[Any]): Codominio (igual a A para relaciones homogéneas).
        R (Set[Tuple[Any, Any]]): Conjunto de pares de la relación.
        matrix (Matrix): Representación matricial.

    Example:
        >>> rel = Relation(['a', 'b'], [('a', 'b')])
        >>> rel.isReflexive()
        False
        >>> rel.reflexiveClosure()
        >>> rel.isReflexive()
        True
    """

    def __init__(self, elements: List[Any], pairs: List[Tuple[Any, Any]] = None):
        """
        Inicializa la relación con un conjunto de elementos y pares opcional.

        Args:
            elements (List[Any]): Lista de elementos únicos.
            pairs (List[Tuple[Any, Any]], optional): Lista de tuplas que representan relaciones.
        """
        self.A = elements
        self.B = elements  # Para relaciones homogéneas A x A
        self.indexMap: Dict[Any, int] = {elem: idx for idx, elem in enumerate(elements)}
        self.reverseMap: Dict[int, Any] = {idx: elem for idx, elem in enumerate(elements)}
        self.R: Set[Tuple[Any, Any]] = set()
        self.matrix = Matrix.zero(len(elements))
        if pairs:
            self.addPairs(pairs)

    def __getitem__(self, x: Any) -> Set[Any]:
        """
        Acceso a los seguidores de un elemento mediante corchetes.

        Args:
            x (Any): Elemento origen.

        Returns:
            Set[Any]: Seguidores de x.
        """
        return self.getFollowers(x)

    def __add__(self, pair: Tuple[Any, Any]) -> 'Relation':
        """
        Agrega un par a la relación utilizando el operador +.

        Args:
            pair (Tuple[Any, Any]): Par a agregar.

        Returns:
            Relation: Relación resultante.
        """
        return self.addPair(pair)

    def __repr__(self) -> str:
        """
        Representación legible del conjunto A, B y R.

        Returns:
            str: Representación de la instancia.
        """
        return f"A = {self.A}, B = {self.B}, R = {self.R}"

    def addPair(self, pair: Tuple[Any, Any]) -> 'Relation':
        """
        Agrega un par (a, b) a la relación y actualiza la matriz.

        Args:
            pair (Tuple[Any, Any]): Par ordenado.

        Returns:
            Relation: La instancia actual para encadenamiento.
        """
        a, b = pair
        self.R.add((a, b))
        i, j = self.indexMap[a], self.indexMap[b]
        self.matrix._matrix[i, j] = 1
        return self

    def addPairs(self, pairs: List[Tuple[Any, Any]]):
        """
        Agrega múltiples pares a la relación.

        Args:
            pairs (List[Tuple[Any, Any]]): Lista de pares (a, b).
        """
        for pair in pairs:
            self.addPair(pair)

    def toPairs(self) -> List[Tuple[Any, Any]]:
        """
        Devuelve la relación como lista de pares.

        Returns:
            List[Tuple[Any, Any]]: Relaciones actuales.
        """
        return list(self.R)

    def reflexiveClosure(self):
        """
        Aplica la clausura reflexiva: ∀a ∈ A, se asegura que (a, a) ∈ R.
        """
        self.matrix = self.matrix.reflexiveClosure()
        for e in self.A:
            self.R.add((e, e))

    def transitiveClosure(self):
        """Representación matemática de relaciones binarias con operaciones de análisis.

        Implementa relaciones sobre conjuntos finitos con:
        - Representación matricial y por pares
        - Operaciones de clausura
        - Verificación de propiedades algebraicas
        - Visualización gráfica

        Attributes:
            A (List[Any]): Dominio de la relación.
            B (List[Any]): Codominio (igual a A para relaciones homogéneas).
            R (Set[Tuple[Any, Any]]): Conjunto de pares de la relación.
            matrix (Matrix): Representación matricial.

        Example:
            >>> rel = Relation(['a', 'b'], [('a', 'b')])
            >>> rel.isReflexive()
            False
            >>> rel.reflexiveClosure()
            >>> rel.isReflexive()
            True
        """
        self.matrix = self.matrix.transitiveClosure()
        newR = set()
        for i, row in enumerate(self.matrix.toNumpy()):
            for j, val in enumerate(row):
                if val:
                    newR.add((self.reverseMap[i], self.reverseMap[j]))
        self.R = newR

    def isSymmetric(self) -> bool:
        """
        Verifica si R es simétrica: ∀(a, b) ∈ R ⇒ (b, a) ∈ R.

        Returns:
            bool: True si la relación es simétrica.
        """
        return all((b, a) in self.R for (a, b) in self.R)

    def isReflexive(self) -> bool:
        """
        Verifica si R es reflexiva: ∀a ∈ A, (a, a) ∈ R.

        Returns:
            bool: True si la relación es reflexiva.
        """
        return all((a, a) in self.R for a in self.A)

    def isTransitive(self) -> bool:
        """
        Verifica si R es transitiva: ∀(a, b), (b, c) ∈ R ⇒ (a, c) ∈ R.

        Returns:
            bool: True si la relación es transitiva.
        """
        for a, b in self.R:
            for c, d in self.R:
                if b == c and (a, d) not in self.R:
                    return False
        return True

    def isFunction(self) -> bool:
        """
        Verifica si cada elemento de A tiene a lo sumo una salida.
        Matemáticamente, R es función ⟺ ∀a ∈ A, ∃≤1 b ∈ B tal que (a, b) ∈ R.

        Returns:
            bool: True si es una función.
        """
        seen = dict()
        for a, b in self.R:
            if a in seen and seen[a] != b:
                return False
            seen[a] = b
        return True

    def getFollowers(self, a: Any) -> Set[Any]:
        """
        Devuelve todos los elementos relacionados desde 'a'.

        Args:
            a (Any): Nodo de partida.

        Returns:
            Set[Any]: Seguidores de 'a'.
        """
        return {b for (x, b) in self.R if x == a}

    def getParents(self, b: Any) -> Set[Any]:
        """
        Devuelve todos los elementos relacionados hacia 'b'.

        Args:
            b (Any): Nodo destino.

        Returns:
            Set[Any]: Padres de 'b'.
        """
        return {a for (a, y) in self.R if y == b}

    def getSiblings(self, a: Any) -> Set[Any]:
        """
        Devuelve los hermanos de 'a', es decir, aquellos que comparten al menos un padre.

        Args:
            a (Any): Elemento base.

        Returns:
            Set[Any]: Hermanos de 'a'.
        """
        parents = self.getParents(a)
        siblings = set()
        for p in parents:
            siblings.update(self.getFollowers(p))
        siblings.discard(a)
        return siblings

    def areParents(self, a: Any, b: Any) -> bool:
        """
        Verifica si a y b comparten al menos un hijo.

        Args:
            a (Any): Primer elemento.
            b (Any): Segundo elemento.

        Returns:
            bool: True si comparten descendencia.
        """
        return len(self.getFollowers(a) & self.getFollowers(b) - {a, b}) > 0

    def areSiblings(self, a: Any, b: Any) -> bool:
        """
        Verifica si a y b comparten al menos un padre.

        Args:
            a (Any): Primer elemento.
            b (Any): Segundo elemento.

        Returns:
            bool: True si tienen padre(s) en común.
        """
        return len(self.getParents(a) & self.getParents(b) - {a, b}) > 0

    def show(self):
        """
        Imprime la matriz de adyacencia y los pares de la relación.
        """
        print("Matriz:")
        print(self.matrix)
        print("Relaciones:")
        for a, b in sorted(self.R):
            print(f"{a} -> {b}")

    def showVenn(self, a: Any = None, b: Any = None):
        """Visualiza relaciones mediante diagramas de Venn.

        Args:
            a: Primer elemento para comparación (opcional).
            b: Segundo elemento para comparación (opcional).

        Note:
            Sin argumentos, muestra dominio vs. codominio.
            Con argumentos, compara seguidores de ambos elementos.

        Example:
            >>> rel = Relation(['x', 'y', 'z'], [('x', 'y'), ('x', 'z')])
            >>> rel.showVenn('x', 'y')  # Muestra diagrama comparativo
        """
        if a is not None and b is not None:
            set1 = self.getFollowers(a)
            set2 = self.getFollowers(b)
            labels = (str(a), str(b))
        else:
            domain = {x for (x, _) in self.R}
            codomain = {y for (_, y) in self.R}
            set1 = domain
            set2 = codomain
            labels = ("Dominio", "Codominio")

        venn.venn2([set1, set2], set_labels=labels)
        plt.title("Diagrama de Venn de la Relación")
        plt.show()