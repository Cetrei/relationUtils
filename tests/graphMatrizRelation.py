import pytest
from relationutils.matrix import Matrix
from relationutils.relation import Relation
from relationutils.graph import Graph

__author__ = "Cetrei#0636"
__copyright__ = "Cetrei#0636"
__license__ = "MIT"

def test_integrated_operations():
    # ====================
    # Parte 1: Pruebas de Matrix
    # ====================
    
    # Crear matriz de prueba
    data = [
        [1, 0, 1],
        [0, 0, 1],
        [0, 1, 0]
    ]
    m = Matrix(data)
    
    # Operaciones básicas
    m2 = Matrix.identity(3)
    assert (m + m2).toNumpy().tolist() == [[2, 0, 1], [0, 1, 1], [0, 1, 1]]
    
    # Multiplicación matricial
    result = (m @ m).toNumpy().tolist()
    assert result == [[1, 1, 1], [0, 1, 0], [0, 0, 1]]
    
    # Clausuras
    reflexive = m.reflexiveClosure().toNumpy().tolist()
    assert reflexive == [[1, 0, 1], [0, 1, 1], [0, 1, 1]]
    
    transitive = m.transitiveClosure().toNumpy().tolist()
    assert transitive == [[1, 1, 1], [0, 1, 1], [0, 1, 1]]
    
    # Conversión a relación
    relation_pairs = m.toRelation()
    assert set(relation_pairs) == {(0, 0), (0, 2), (1, 2), (2, 1)}
    
    # ====================
    # Parte 2: Pruebas de Relation
    # ====================
    
    # Crear relación con elementos etiquetados
    elements = ['a', 'b', 'c']
    pairs = [('a', 'a'), ('a', 'c'), ('b', 'c'), ('c', 'b')]
    rel = Relation(elements, pairs)
    
    # Propiedades básicas
    assert not rel.isReflexive()  # Falta (b,b) y (c,c)
    assert not rel.isSymmetric()  # (a,c) no tiene inversa
    assert not rel.isTransitive()  # (c,b) y (b,c) deberían implicar (c,c)
    
    # Clausuras
    rel.reflexiveClosure()
    assert rel.isReflexive()
    
    rel.transitiveClosure()
    assert rel.isTransitive()
    
    # Operaciones de consulta
    assert rel.getFollowers('a') == {'a', 'b', 'c'}
    assert rel.getParents('c') == {'a', 'b', 'c'}
    assert rel.areSiblings('a', 'b') == True
    
    # ====================
    # Parte 3: Pruebas de Graph
    # ====================
    
    # Crear grafo dirigido
    g = Graph(directed=True)
    for node in ['a', 'b', 'c']:
        g.addNode(node)
    
    g.addEdge('a', 'a')
    g.addEdge('a', 'c')
    g.addEdge('b', 'c')
    g.addEdge('c', 'b')
    
    # Propiedades del grafo
    assert g.getGraphType() == "Dirigido, Pseudografo"
    assert not g.isConnected()  # No hay camino de 'a' a otros
    
    # Encontrar caminos
    assert g.findPath('b', 'c') == ['b', 'c']
    assert g.findSimplePath('c', 'a') == None  # No hay camino
    
    # Componentes conexas
    components = g.getComponents()
    assert len(components) == 2
    assert {'a'} in components
    assert {'b', 'c'} in components
    
    # Convertir a/desde Relation
    rel_from_graph = g.toRelation()
    assert rel_from_graph.isReflexive() == False  # Solo 'a' tiene bucle
    
    new_g = Graph()
    new_g.fromRelation(rel_from_graph)
    assert new_g.nodes == {'a', 'b', 'c'}
    
    # Grafo no dirigido
    undir_g = Graph(directed=False)
    undir_g.addEdge('a', 'b')
    undir_g.addEdge('b', 'c')
    assert undir_g.isConnected()
    assert undir_g.findEulerianPath() == None  # No es euleriano

if __name__ == "__main__":
    test_integrated_operations()
    print("Todas las pruebas pasaron exitosamente!")