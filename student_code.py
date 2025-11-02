'''VersatileDigraph and BinaryGraph and SortingTree'''

class VersatileDigraph:
    '''This is the versatile digraph class'''

    def __init__(self):
        self.__edge_weights = {}
        self.__edge_names = {}
        self.__edge_head = {}
        self.__node_values = {}

    def add_edge(self, tail, head, **vararg):
        '''
        Adds an edge to the graph
        O(1) average per insertion 
        (dict lookups + inserts are amortized O(1)).
        If tail/head are new, add_node is also O(1).
        '''
        try:
            edge_weight = vararg.get("edge_weight", 0)
            if edge_weight < 0:
                raise ValueError("Edge weight must be non-negative")
        except TypeError as type_error:
            raise ValueError("Edge weight must be a number") from type_error

        if not tail in self.get_nodes():
            self.add_node(tail, vararg.get("start_node_value", 0))
        if not head in self.get_nodes():
            self.add_node(head, vararg.get("end_node_value", 0))
        edge_name = vararg.get("edge_name", tail + " to " + head)
        self.__edge_names[tail][head] = edge_name
        self.__edge_head[tail][edge_name] = head
        self.__edge_weights[tail][head] = edge_weight

    def get_nodes(self):
        '''
        Returns a list of nodes in the graph
        O(1) (dict_keys view creation).
        Iterating through it costs O(V)
        '''
        return self.__node_values.keys()

    def add_node(self, node_id, node_value=0):
        '''
        Adds a node to the graph
        O(1) average (dict insertions)
        '''
        if not isinstance(node_value, (int, float)):
            raise TypeError("Node value must be a number")
        if node_id in self.get_nodes():
            raise ValueError("Node already exists")

        self.__node_values[node_id] = node_value
        self.__edge_weights[node_id] = {}
        self.__edge_names[node_id] = {}
        self.__edge_head[node_id] = {}

    def get_edge_weight(self, tail, head):
        '''
        Return the weight of an edge
        O(1) average (dict lookup)
        '''
        try:
            edge_weight = self.__edge_weights[tail][head]
        except KeyError as key_error:
            raise KeyError("One or both nodes do not exist") from key_error
        return edge_weight

    def get_edge_name(self, tail, head):
        '''
        Return the name of an edge
        O(1) average (dict lookup)
        '''
        try:
            edge_name = self.__edge_names[tail][head]
        except KeyError as key_error:
            raise KeyError("One or both nodes do not exist") from key_error
        return edge_name

    def get_node_value(self, node):
        '''
        Return the value of a node
        O(1) average (dict lookup)
        '''
        return self.__node_values[node]

    def print_graph(self):
        '''Prints sentences describing the graph'''
        for tail in self.get_nodes():
            print("Node: " + str(tail) +
                  " has a value of " + str(self.get_node_value(tail)) + ".")
            for head in self.__edge_weights[tail]:
                print("There is an edge from node " + str(tail) +
                      " to node " + str(head) +
                      " of weight " + str(self.get_edge_weight(tail, head)) +
                      " and name " + self.__edge_names[tail][head] + ".")

    def predecessors(self, node):
        '''
        Given a node, return a list of nodes that immediately preceed that node
        '''
        if node not in self.get_nodes():
            raise KeyError("Node does not exist")

        list_of_predecessors = \
            [key for (key, value) in self.__edge_weights.items() if node in value.keys()]
        return list_of_predecessors

    def successors(self, node):
        '''
        Given a node, return a list of nodes that immediately succeed that node
        '''
        return list(self.__edge_weights[node].keys())

    def successor_on_edge(self, node, edge_name):
        '''
        Given a node and an edge name,identify the successor of the node on the
        edge with the provided name
        '''
        return self.__edge_head[node][edge_name]

    def in_degree(self, node):
        '''
        Given a node, return the number of edges that lead to that node
        '''
        return len(self.predecessors(node))

    def out_degree(self, node):
        '''
        Given a node, return the number of edges that lead from that node
        '''
        return len(self.successors(node))

    def plot_graph(self):
        '''Plot the graph using graphviz'''
        try:
            import graphviz # pylint: disable=import-outside-toplevel
        except ImportError as import_error:
            raise ImportError("graphviz package not found") from import_error
        dot = graphviz.Digraph(comment='A graphviz Digraph')

        # Add a title for the graph
        dot.graph_attr['label'] = "Versatile Directed Graph"
        dot.graph_attr['labelloc'] = "t"
        dot.graph_attr['fontsize'] = "20"

        nodes = self.get_nodes()
        for node in nodes:
            node_name = f"{node}:{self.get_node_value(node)}"
            dot.node(str(node), label=node_name)
        for tail in nodes:
            for head in self.successors(tail):
                edge_name = self.get_edge_name(tail, head)
                edge_weight = self.get_edge_weight(tail, head)
                dot.edge(str(tail), str(head), label=f"{edge_name}:{edge_weight}")

        dot.render('versatile_digraph', view=True)

    def plot_edge_weights(self):
        '''Plot the edge weights using bokeh'''
        try:
            from bokeh.plotting import figure, show # pylint: disable=import-outside-toplevel
        except ImportError as import_error:
            raise ImportError("bokeh package not found") from import_error
        edge_weights = []
        edge_names = []
        for tail in self.get_nodes():
            for head in self.successors(tail):
                edge_weight = self.get_edge_weight(tail, head)
                edge_name = self.get_edge_name(tail, head)
                if edge_name in edge_names:
                    continue
                edge_weights.append(edge_weight)
                edge_names.append(edge_name)
        print(edge_weights)
        print(edge_names)

        p = figure(x_range=edge_names, height=350, title="Edge Weights by Edge Name",
                   toolbar_location=None, tools="")

        p.vbar(x=edge_names, top=edge_weights, width=0.9)

        # Add axis titles
        p.xaxis.axis_label = "Edges"
        p.yaxis.axis_label = "Weight"

        p.xgrid.grid_line_color = None
        p.y_range.start = 0

        show(p)



class BinaryGraph(VersatileDigraph):
    '''Binary Tree class'''
    def __init__(self, root_id="Root", root_value=0):
        super().__init__()
        self.add_node(root_id, root_value)

    def add_node_left(self, child_id, child_value, parent_id="Root"):
        '''Adds a new left child node to a specified existing node'''
        if parent_id not in self.get_nodes():
            raise KeyError("Parent node does not exist")
        left_child = self.get_node_left(parent_id)
        if left_child is not None:
            raise ValueError("Left child already exists")
        self.add_edge(
            parent_id,
            child_id,
            edge_name="left",
            end_node_value=child_value,
            edge_weight=0
        )

    def add_node_right(self, child_id, child_value, parent_id="Root"):
        '''Adds a new right child node to a specified existing node'''
        if parent_id not in self.get_nodes():
            raise KeyError("Parent node does not exist")
        right_child = self.get_node_right(parent_id)
        if right_child is not None:
            raise ValueError("Right child already exists")
        self.add_edge(
            parent_id,
            child_id,
            edge_name="right",
            end_node_value=child_value,
            edge_weight=0
        )

    def get_node_left(self, parent_id):
        '''Returns the ID of the left child node for a specified node'''
        if parent_id not in self.get_nodes():
            raise KeyError("Parent node does not exist")
        for head in self.successors(parent_id):
            if self.get_edge_name(parent_id, head) == "left":
                return head
        return None

    def get_node_right(self, parent_id):
        '''Returns the ID of the right child node for a specified node'''
        if parent_id not in self.get_nodes():
            raise KeyError("Parent node does not exist")
        for head in self.successors(parent_id):
            if self.get_edge_name(parent_id, head) == "right":
                return head
        return None

    def plot_graph(self):
        '''Plot the binary tree using graphviz'''
        try:
            import graphviz # pylint: disable=import-outside-toplevel
        except ImportError as import_error:
            raise ImportError("graphviz package not found") from import_error
        dot = graphviz.Digraph(
            comment='A binary tree',
            node_attr={'shape': 'circle'},
            graph_attr={
                'label': 'A binary tree',
                'labelloc': 't',
                'fontsize': '20',
                'splines': 'false',
                'layout': 'dot'
            },
            edge_attr={'arrowhead': 'none'},
        )

        nodes = self.get_nodes()
        for node in nodes:
            node_name = self.get_node_value(node)
            dot.node(str(node), label=str(node_name))
        for tail in nodes:
            left_child = self.get_node_left(tail)
            right_child = self.get_node_right(tail)
            if left_child is None and right_child is None:
                continue

            if left_child is not None:
                dot.edge(str(tail), str(left_child), tailport='sw', headport='n')
            else:
                dot.node(f"na_{tail}_l", label='', style='invisible')
                dot.edge(str(tail), f"na_{tail}_l", tailport='sw', headport='n', style='invisible')

            if right_child is not None:
                dot.edge(str(tail), str(right_child), tailport='se', headport='n')
            else:
                dot.node(f"na_{tail}_r", label='', style='invisible')
                dot.edge(str(tail), f"na_{tail}_r", tailport='se', headport='n', style='invisible')

        dot.render('binary_tree', view=True)

class SortingTree(BinaryGraph):
    '''A binary tree that sorts values'''
    def __init__(self, root_value=0, root_id="Root",):
        super().__init__(root_id, root_value)

    def insert(self, value, node_id=None, current_node_id="Root"):
        '''Insert a value into the sorting tree'''
        if not isinstance(value, (int, float)):
            raise TypeError("Value must be a number")
        if node_id is None:
            node_id = f"Node_{len(self.get_nodes())}_th"
        if node_id in self.get_nodes():
            raise ValueError("Node ID already exists")

        current_value = self.get_node_value(current_node_id)

        # if value == current_value:
        #     raise ValueError("Value already exists in the tree")

        if value < current_value:
            left_child = self.get_node_left(current_node_id)
            if left_child is None:
                self.add_node_left(node_id, value, current_node_id)
            else:
                self.insert(value, node_id, left_child)
        else:
            right_child = self.get_node_right(current_node_id)
            if right_child is None:
                self.add_node_right(node_id, value, current_node_id)
            else:
                self.insert(value, node_id, right_child)

    def traverse(self, node_id="Root"):
        '''Traverse the tree and return a sorted list of values'''
        if node_id not in self.get_nodes():
            raise KeyError("Node does not exist")

        result = []
        left_child = self.get_node_left(node_id)
        if left_child is not None:
            result.extend(self.traverse(left_child))

        result.append(self.get_node_value(node_id))

        right_child = self.get_node_right(node_id)
        if right_child is not None:
            result.extend(self.traverse(right_child))

        if node_id == "Root":
            print(" ".join(map(str, result)), end=" ")

        return result

class SortableDigraph(VersatileDigraph):
    '''A versatile digraph that can be sorted topologically'''
    def top_sort(self):
        '''Topological sort of the digraph using Kahn's algorithm'''
        # The in-degree for each node
        count = dict((u, 0) for u in self.get_nodes())
        # Count every in-edge
        for u in self.get_nodes():
            for v in self.successors(u):
                count[v] += 1
        # Valid initial nodes
        q = [u for u in self.get_nodes() if count[u] == 0]
        # The result
        s = []
        # While we have start nodes...
        while q:
            # Pick one
            u = q.pop()
            # Use it as first of the rest
            s.append(u)
            # "Uncount" its out-edges
            for v in self.successors(u):
                count[v] -= 1
                # New valid start nodes?
                if count[v] == 0:
                    q.append(v)
        # Deal with them next
        if len(s) != len(self.get_nodes()):
            raise ValueError("Graph has at least one cycle")
        return s

class TraversableDigraph(SortableDigraph):
    '''A sortable digraph that can be traversed'''
    def dfs(self, start_node):
        '''Depth-First Search traversal of the digraph'''
        s, q = set(), list(self.successors(start_node))
        while q:
            u = q.pop()
            if u in s:
                continue
            s.add(u)
            q.extend(self.successors(u))
            yield u

    def bfs(self, start_node):
        '''Breadth-First Search traversal of the digraph'''
        try:
            from collections import deque # pylint: disable=import-outside-toplevel
        except ImportError as import_error:
            raise ImportError("collections package not found") from import_error
        s, q = set(), deque(self.successors(start_node))
        while q:
            u = q.popleft()
            if u in s:
                continue
            s.add(u)
            for v in self.successors(u):
                q.append(v)
            yield u

class DAG(TraversableDigraph):
    '''A traversable digraph that is a directed acyclic graph'''
    def add_edge(self, tail, head, **vararg):
        '''Adds an edge to the DAG ensuring no cycles are created'''
        # Check if adding the edge would create a cycle
        for node in self.dfs(head):
            if node == tail:
                raise ValueError("Adding this edge would create a cycle")
        super().add_edge(tail, head, **vararg)
