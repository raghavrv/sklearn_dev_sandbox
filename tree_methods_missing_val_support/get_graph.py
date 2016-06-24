import matplotlib.pyplot as plt
import pygraphviz as pgv
import networkx as nx
import pygraphviz
import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np
from value_dropper import drop_values

from sklearn.tree import *
from StringIO import StringIO
from io import BytesIO
import numpy as np

def get_graph(dtc, feat_names=None, size=[7, 7], max_depth=10):
    n_classes = dtc.n_classes_

    # Get the dot graph of our decision tree
    export_graphviz(dtc, out_file="_dotfile.dot", feature_names=feat_names,
                    rounded=True, filled=True,
                    special_characters=False,
                    show_missing_dir=True,
                    class_names=map(str, range(1, n_classes+1)),
                    max_depth=max_depth)
   
    # Convert this dot graph into an image
    g = pygraphviz.AGraph("_dotfile.dot")
    g.layout('dot')
    g.draw(path="_image.png")

    # Plot it

    from IPython.display import Image
    return Image("_image.png")
