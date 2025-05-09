from typing import Any, Dict, List, Set, Tuple, Optional
from relationutils.relation import Relation
from relationutils.matrix import Matrix
from collections import defaultdict, deque

class Graph:
    """Implementación de grafos dirigidos y no dirigidos para análisis estructural.

    Esta clase proporciona operaciones fundamentales de teoría de grafos con
    representación interna basada en relaciones y matrices. Soporta:

    - Grafos dirigidos y no dirigidos
    - Operaciones algebraicas gráficas
    - Algoritmos de búsqueda y recorrido
    - Análisis de propiedades estructurales

    Attributes:
        nodes (Set[Any]): Conjunto de nodos del grafo.
        edges (List[Tuple[Any, Any, Optional[Any]]): Lista de aristas con etiquetas opcionales.
        directed (bool): Indica si el grafo es dirigido.
        relation (Optional[Relation]): Representación relacional interna.

    Example:
        >>> g = Graph(directed=True)
        >>> g.addEdge('A', 'B')
        >>> g.addEdge('B', 'C')
        >>> g.findPath('A', 'C')
        ['A', 'B', 'C']
    """

    def __init__(self, directed: bool = True):
        """Inicializa un grafo vacío.

        Args:
            directed: Si es True, crea un grafo dirigido (default). Si es False,
                crea un grafo no dirigido (las aristas son bidireccionales).
        """
        self.nodes: Set[Any] = set()
        self.edges: List[Tuple[Any, Any, Optional[Any]]] = []  # a, b, etiqueta
        self.directed = directed
        self.relation: Optional[Relation] = None

    def addNode(self, node: Any):
        self.nodes.add(node)

    def addEdge(self, a: Any, b: Any, label: Any = None):
        """Añade una arista entre dos nodos.

        Args:
            a: Nodo origen.
            b: Nodo destino.
            label: Etiqueta opcional para la arista.

        Note:
            Para grafos no dirigidos, añade automáticamente la arista inversa.

        Example:
            >>> g = Graph(directed=False)
            >>> g.addEdge('X', 'Y', 'conexion')
            >>> len(g.edges)
            2
        """
        self.nodes.update([a, b])
        self.edges.append((a, b, label))
        if not self.directed:
            self.edges.append((b, a, label))

    def buildRelation(self):
        pure_edges = [(a, b) for a, b, _ in self.edges]
        self.relation = Relation(list(self.nodes), pure_edges)

    def getNeighbors(self, node: Any) -> Set[Any]:
        if not self.relation:
            self.buildRelation()
        return self.relation.getFollowers(node)

    def getEndpoints(self) -> Tuple[Any, Any]:
        """Retorna un nodo con grado impar de entrada/salida para caminos Eulerianos abiertos."""
        indeg, outdeg = defaultdict(int), defaultdict(int)
        for a, b, _ in self.edges:
            outdeg[a] += 1
            indeg[b] += 1
        endpoints = [n for n in self.nodes if abs(indeg[n] - outdeg[n]) == 1]
        return tuple(endpoints) if len(endpoints) == 2 else (None, None)

    def findEulerianPath(self) -> Optional[List[Any]]:
        """Encuentra un camino euleriano si existe.

        Returns:
            Lista de nodos en el camino euleriano, o None si no existe.

        Raises:
            ValueError: Si el grafo no está conectado.

        Note:
            Implementa el algoritmo de Hierholzer. Un grafo tiene camino euleriano si:
            - Dirigido: Todos los nodos tienen igual grado de entrada/salida,
              o exactamente 2 nodos con diferencia 1 (-1 y +1)
            - No dirigido: Todos los nodos tienen grado par,
              o exactamente 2 nodos con grado impar

        Example:
            >>> g = Graph(directed=False)
            >>> g.addEdge('A', 'B')
            >>> g.addEdge('B', 'C')
            >>> g.findEulerianPath() is None
            True
        """
        if not self.isEulerianCycle():
            start, _ = self.getEndpoints()
            if not start:
                return None
        else:
            start = next(iter(self.nodes))

        temp_edges = self.edges.copy()
        graph = defaultdict(list)
        for a, b, _ in temp_edges:
            graph[a].append(b)

        path, stack = [], [start]
        while stack:
            v = stack[-1]
            if graph[v]:
                u = graph[v].pop()
                stack.append(u)
            else:
                path.append(stack.pop())
        return path[::-1]

    def fromRelation(self, relation: Relation):
        self.nodes = set(relation.A)
        self.edges = [(a, b, None) for a, b in relation.toPairs()]
        self.relation = relation

    def toRelation(self) -> Relation:
        self.buildRelation()
        return self.relation

    def fromMatrix(self, matrix: Matrix, labels: List[Any]):
        self.nodes = set(labels)
        self.edges.clear()
        data = matrix.toNumpy()
        for i, a in enumerate(labels):
            for j, b in enumerate(labels):
                if data[i][j]:
                    self.edges.append((a, b, None))
        self.buildRelation()

    def toMatrix(self) -> Matrix:
        self.buildRelation()
        return self.relation.matrix

    # Métodos existentes
    def isConnected(self) -> bool:
        if not self.relation:
            self.buildRelation()
        closure = Relation(self.relation.A, list(self.relation.R))
        closure.transitiveClosure()
        return all((a, b) in closure.R for a in self.nodes for b in self.nodes)

    def hasCycle(self) -> bool:
        visited = set()

        def dfs(v, path):
            visited.add(v)
            path.add(v)
            for neighbor in self.getNeighbors(v):
                if neighbor in path or (neighbor not in visited and dfs(neighbor, path)):
                    return True
            path.remove(v)
            return False

        return any(dfs(v, set()) for v in self.nodes if v not in visited)

    def findPath(self, start: Any, end: Any, visited=None) -> Optional[List[Any]]:
        if visited is None:
            visited = set()
        visited.add(start)
        if start == end:
            return [start]
        for neighbor in self.getNeighbors(start):
            if neighbor not in visited:
                path = self.findPath(neighbor, end, visited)
                if path:
                    return [start] + path
        return None

    def findSimplePath(self, start: Any, end: Any, path=None) -> Optional[List[Any]]:
        if path is None:
            path = [start]
        if start == end:
            return path
        for neighbor in self.getNeighbors(start):
            if neighbor not in path:
                new_path = self.findSimplePath(neighbor, end, path + [neighbor])
                if new_path:
                    return new_path
        return None

    def isEulerianCycle(self) -> bool:
        if not self.isConnected():
            return False
        degree = defaultdict(int)
        for a, b, _ in self.edges:
            degree[a] += 1
            if not self.directed:
                degree[b] += 1
        if self.directed:
            indeg = defaultdict(int)
            outdeg = defaultdict(int)
            for a, b, _ in self.edges:
                outdeg[a] += 1
                indeg[b] += 1
            return all(indeg[n] == outdeg[n] for n in self.nodes)
        else:
            return all(degree[n] % 2 == 0 for n in self.nodes)

    def getComponents(self) -> List[Set[Any]]:
        visited = set()
        components = []

        def dfs(v, group):
            group.add(v)
            visited.add(v)
            for u in self.getNeighbors(v):
                if u not in visited:
                    dfs(u, group)

        for node in self.nodes:
            if node not in visited:
                group = set()
                dfs(node, group)
                components.append(group)
        return components

    def getGraphType(self) -> str:
        type_info = []
        if self.directed:
            type_info.append("Dirigido")
        else:
            type_info.append("No Dirigido")
        seen = set()
        has_multiedges = False
        has_loops = False
        for a, b, _ in self.edges:
            if (a, b) in seen:
                has_multiedges = True
            seen.add((a, b))
            if a == b:
                has_loops = True
        if has_multiedges:
            type_info.append("Multigrafo")
        if has_loops:
            type_info.append("Pseudografo")
        return ", ".join(type_info)

    def show(self):
        """Imprime una representación textual del grafo.
        
        Muestra:
            - Lista de nodos ordenados
            - Lista de aristas con etiquetas (si existen)
            - Tipo de grafo (dirigido/no dirigido)
            - Matriz de adyacencia (si está disponible)
        
        Example:
            >>> g = Graph(directed=False)
            >>> g.addEdge('A', 'B')
            >>> g.show()
            Nodos: ['A', 'B']
            Aristas:
            A -> B
            B -> A
            Tipo de grafo: No Dirigido
        """
        print(f"Nodos: {sorted(self.nodes)}")
        print(f"Aristas:")
        for a, b, lbl in self.edges:
            if lbl is not None:
                print(f"  {a} -> {b} [etiqueta: {lbl}]")
            else:
                print(f"  {a} -> {b}")
        print(f"Tipo de grafo: {self.getGraphType()}")
        if self.relation:
            print("\nRepresentación como relación:")
            self.relation.show()