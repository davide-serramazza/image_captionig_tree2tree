import numpy as np

n_err=0

class Tree:
    """
    class tree
    """
    def __init__(self):
        self.childs = []
        self.data = None
        self.label = None
        self.internal_nodes = set()
        self.leafs = set()
        self.parent = None
        self._current_tree = None
        self.isLeaf = False


    def get_leafs(self):
        """
        fucntion to return current leaf
        :return:
        """
        tmp = []
        current = self._current_tree
        if current != None:
            for child in current.childs:
                if child.isLeaf == True:
                    tmp.append(int(child.label))
        else:
            return -1
        return tmp


    def get_string_from_tree(self):
        """
        recursive function that write tree as a string
        :return:
        """
        string=""
        #write root
        #info = extract_string_fromTensor( str(self.data) )
        info = "["
        for el in self.data:
            info+=(str(el.detach().numpy())+", ")
        info=info[:-2]+"]"
        string = string + str(self.label) +  " data:" + info + ",("

        #first loop for leaf
        for child in self.childs:
            if child.isLeaf == True:
                info = extract_string_fromTensor( child.data )
                string = string + str(child.label) + " data:" + info +","

        #second loop for child that are internal node
        for child in self.childs:
            if child.isLeaf == False:
                string = string + child.get_string_from_tree()

        string = string[:-1] + "),"
        return string


    def find_node(self,root):
        """
        function to find a specific node in the tree
        :param root:
        :return:
        """

        to_search = []
        to_search.append(self)
        while len(to_search) > 0:
            tree = to_search.pop(0)
            if tree.label == root:
                return tree
            else:
                for i in range(0,len(tree.childs)):
                    if isinstance(tree.childs[i], Tree):
                        to_search.append(tree.childs[i])
                        if tree.childs[i].label == root:
                            return tree.childs[i]




def extract_string_fromTensor(data):
    """
    trsfomr tensor representation into string representation
    :param info:
    :return:
    """
    info=""
    for el in data:
        info+=(str(el.detach().numpy())+", ")
    return "[" + info[:-2] + "])"


def read_file(file):
    """
    function that read form file the representation of the tree
    """
    f=open( file, "r")
    file =f.read()
    lines = file.split("\n")
    lines = lines[:-1]
    for i in range(0,len(lines)):
        lines[i] = lines[i].split(" ")
        assert lines[i][1] == ":"
    #remoe "root" from the first node
    lines[0][0] = lines[0][0][:-4]
    return lines


def max_in_str(list):
    """
    function to find max element in a lis of str representing int
    """
    max=0
    for string in list:
        v = (int(string))
        if v >  max:
            max = v
    return max


def create_tree(file):
    lines = read_file(file)

    #create tree with only root
    tree = Tree()
    tree.label = lines[0][0]
    tree.internal_nodes.add(lines[0][0])

    #temporany dictionary used later in the function
    temp_dict = dict()

     #fill the tree
    for l in lines:

        root = l[0]
        #after have read the current subtree's root, extract this subtree from the whole tree
        if (root in tree.internal_nodes):
            actual_tree = tree.find_node(root)
        else:
            try:
                parent = tree(temp_dict[root])
                actual_tree = parent(root)
                actual_tree.parent = parent
            except:
                print("error with ", file, root)
                global n_err
                n_err = n_err+1

        #add leaf i.e. final segments or internal nodes
        for node in  l[2:]:
            if node.endswith("leaf"):
                new_leaf = Tree()
                new_leaf.parent = actual_tree
                new_leaf.label = int (node[:-4])
                new_leaf.isLeaf = True
                actual_tree.childs.append(new_leaf)
                tree.leafs.add(node[:-4])
            else:
                new_tree = Tree()
                new_tree.label = node
                actual_tree.childs.append(new_tree)
                new_tree.parent = actual_tree
                temp_dict[node] = root
                tree.internal_nodes.add(node)

    tree._current_tree = actual_tree
    #check if every final segment are been analyzed
    max = max_in_str(tree.leafs)
    assert  max == len(tree.leafs)
    return tree
